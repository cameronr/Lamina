"""Tests for setup.py's --vision gpu[:GPU]: the GPU it picks, the config's `vision` section, the encoder's build and its
place beside the server, without building or running anything.

    python -m unittest tools.test_setup_vision
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

MAIN = setup.Gpu(0, "NVIDIA GeForce RTX 3090", "GPU-aaaa", 24576, 86, False)
TIER = setup.Gpu(1, "NVIDIA GeForce RTX 3090", "GPU-bbbb", 24576, 86, True)
SMALL = setup.Gpu(2, "NVIDIA GeForce RTX 5060 Ti", "GPU-cccc", 16311, 120, False)
ALL = [MAIN, TIER, SMALL]


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


class PickVisionGpu(unittest.TestCase):
    def test_the_one_named(self):
        self.assertEqual(setup.pick_vision_gpu("1", ALL, MAIN), TIER)
        self.assertEqual(setup.pick_vision_gpu("GPU-cccc", ALL, MAIN), SMALL)
        self.assertEqual(setup.pick_vision_gpu("5060", ALL, MAIN), SMALL)
        self.assertEqual(setup.pick_vision_gpu("0", ALL, MAIN), MAIN)   # the main GPU when asked for

    def test_a_name_no_gpu_has(self):
        self.assertIsNone(setup.pick_vision_gpu("4090", ALL, MAIN))

    def test_default_the_least_vram_besides_the_main_gpu(self):
        self.assertEqual(setup.pick_vision_gpu(None, ALL, MAIN), SMALL)
        self.assertEqual(setup.pick_vision_gpu(None, [MAIN, TIER], MAIN), TIER)

    def test_default_on_one_gpu_is_that_gpu(self):
        self.assertEqual(setup.pick_vision_gpu(None, [MAIN], MAIN), MAIN)

    def test_default_passes_over_the_desktops_card(self):
        # the development PC: the 5070 Ti drives the desktop and has 8 MiB less than the 5060 Ti
        desk = setup.Gpu(0, "NVIDIA GeForce RTX 5070 Ti", "GPU-dddd", 16303, 120, True)
        tier = setup.Gpu(1, "NVIDIA GeForce RTX 5060 Ti", "GPU-eeee", 16311, 120, False)
        main = setup.Gpu(2, "NVIDIA GeForce RTX 3090", "GPU-ffff", 24576, 86, False)
        self.assertEqual(setup.pick_vision_gpu(None, [desk, tier, main], main), tier)
        self.assertEqual(setup.pick_vision_gpu(None, [desk, main], main), desk)   # the only other one
        self.assertEqual(setup.pick_vision_gpu("0", [desk, tier, main], main), desk)   # when asked for


class VisionSection(unittest.TestCase):
    EXAMPLE = {"mmproj": "m.gguf", "model": "x.gguf", "gpu": False, "threads": 16, "max_tokens": 1024}

    def test_the_cpu_encoder(self):
        v = setup.vision_section(self.EXAMPLE, None)
        self.assertEqual(v["threads"], os.cpu_count() or 8)
        self.assertFalse(v["gpu"])
        self.assertNotIn("cuda_device", v)

    def test_the_cuda_encoder_on_a_card(self):
        v = setup.vision_section(self.EXAMPLE, SMALL)
        self.assertIs(v["gpu"], True)
        self.assertEqual(v["cuda_device"], "GPU-cccc")
        self.assertNotIn("threads", v)
        self.assertEqual((v["mmproj"], v["model"], v["max_tokens"]), ("m.gguf", "x.gguf", 1024))

    def test_the_example_is_not_changed(self):
        before = dict(self.EXAMPLE)
        setup.vision_section(self.EXAMPLE, SMALL)
        self.assertEqual(self.EXAMPLE, before)

    def test_no_key_names_the_encoder(self):   # the server takes the lamina-vision in its own folder
        for gpu in (None, SMALL):
            self.assertNotIn("exe", setup.vision_section(self.EXAMPLE, gpu))


class WriteConfig(unittest.TestCase):
    def written(self, vision_gpu):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "ROOT", Path(d)):
            (Path(d) / "examples").mkdir()
            shutil.copy(ROOT / "examples" / "ud-q4_k_xl.json", Path(d) / "examples")
            path = setup.write_config("ud-q4_k_xl", Path(d) / "data", "NVIDIA GeForce RTX 3090", 8082, True, None, None,
                                      vision_gpu)
            return json.loads(path.read_text(encoding="utf-8"))

    def test_the_examples_section_on_a_gpu(self):
        v = self.written(SMALL)["vision"]
        self.assertEqual((v["gpu"], v["cuda_device"]), (True, "GPU-cccc"))
        self.assertNotIn("threads", v)

    def test_without_a_gpu_as_before(self):
        v = self.written(None)["vision"]
        self.assertEqual((v["gpu"], v["threads"]), (False, os.cpu_count() or 8))

    def test_the_examples_name_no_encoder(self):
        for vision_gpu in (None, SMALL):
            self.assertNotIn("exe", self.written(vision_gpu)["vision"])


class WriteLauncher(unittest.TestCase):
    def launcher(self, progs):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "ROOT", Path(d)):
            p = setup.write_launcher("ud-q4_k_xl", setup.MODELS["ud-q4_k_xl"], Path(d) / "lamina-ud-q4_k_xl.json", 8082,
                                     progs)
            return p.read_bytes().decode()

    def test_the_built_server(self):
        server = "build\\lamina-server.exe" if setup.WIN else "exec build/lamina-server"
        self.assertIn(server + " --config lamina-ud-q4_k_xl.json", self.launcher("build"))

    def test_a_releases_server_in_bin(self):
        server = "bin\\lamina-server.exe" if setup.WIN else "exec bin/lamina-server"
        self.assertIn(server + " --config lamina-ud-q4_k_xl.json", self.launcher("bin"))


class BuildVision(unittest.TestCase):
    TC = setup.Toolchain(None, None, "/opt/cuda/bin/nvcc", (13, 4), None, None)

    def test_the_cpu_build_as_before(self):
        with mock.patch.object(setup, "cmake_build") as cb, mock.patch.object(setup, "place_vision", return_value=True) as pv:
            quiet(setup.build_vision, self.TC, {})
        _, src, b, target, defs, _ = cb.call_args.args
        self.assertEqual((src, b, target), (setup.ROOT / "tools" / "vision", setup.ROOT / "build-vision", "lamina-vision"))
        self.assertIn("-DLAMINA_VISION_CUDA=OFF", defs)
        pv.assert_called_once_with(False)   # the CPU build, beside the server

    def test_the_cuda_build(self):
        with mock.patch.object(setup, "cmake_build") as cb, \
                mock.patch.object(setup, "cuda_defs", return_value=["-DCMAKE_CUDA_COMPILER=nvcc"]), \
                mock.patch.object(setup, "place_vision", return_value=True) as pv:
            quiet(setup.build_vision, self.TC, {}, "86;120")
        _, src, b, target, defs, _ = cb.call_args.args
        self.assertEqual((b, target), (setup.ROOT / "build-vision-cuda", "lamina-vision"))
        for d in ("-DLAMINA_VISION_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=86;120", "-DCMAKE_CUDA_COMPILER=nvcc"):
            self.assertIn(d, defs)
        pv.assert_called_once_with(True)   # the CUDA build, beside the server


class PlaceVision(unittest.TestCase):
    def placed(self, cuda, built):
        """Whether place_vision(cuda) placed an encoder, and which build's is in build/, with one built in each of
        `built`'s folders."""
        name = "lamina-vision" + setup.EXE
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "ROOT", Path(d)):
            (Path(d) / "build").mkdir()
            for folder in built:
                (Path(d) / folder / "bin").mkdir(parents=True)
                (Path(d) / folder / "bin" / name).write_text(folder)
            done = setup.place_vision(cuda)
            copy = Path(d) / "build" / name
            return done, copy.read_text() if copy.exists() else None

    def test_the_cpu_build_beside_the_server(self):
        self.assertEqual(self.placed(False, ["build-vision", "build-vision-cuda"]), (True, "build-vision"))

    def test_the_cuda_build_beside_the_server(self):
        self.assertEqual(self.placed(True, ["build-vision", "build-vision-cuda"]), (True, "build-vision-cuda"))

    def test_a_build_that_is_not_there(self):
        self.assertEqual(self.placed(True, ["build-vision"]), (False, None))


if __name__ == "__main__":
    unittest.main()
