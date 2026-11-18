"""Tests for tools/mtp_rt.py: the Q2_0 experts' blob layout and its inverse, a runtime folder written as the layer's
GGUF and checked against it, and another draft vocabulary over a written file.  Small layers of random bytes (two
experts), no model, no GPU.

    python -m unittest tools.test_mtp_rt
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtp_rt as R  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402

RNG = np.random.default_rng(7)


def rt_folder(d: Path, experts_txt: str | None = None) -> None:
    """A runtime folder of the earlier layout: three dense tensors (one of each kind, at 256-byte offsets), two
    experts, a draft vocabulary."""
    dense = [("self_attn.q_proj.weight", "q8_0", 4, 64), ("mlp.gate.weight", "bf16", 2, 8), ("x.norm.weight", "f32", 1, 8)]
    blob, lines = bytearray(), []
    for name, kind, rows, cols in dense:
        n = rows * R.row_bytes(R.KINDS[kind], cols)
        blob += b"\0" * ((-len(blob)) % 256)
        lines.append(f"{name} {kind} {rows} {cols} {len(blob)} {n}")
        blob += RNG.integers(0, 256, n, dtype=np.uint8).tobytes()
    (d / "dense.bin").write_bytes(bytes(blob))
    (d / "dense.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if experts_txt:
        tg, td = (R.T(int(x)) for x in experts_txt.split())
        per = 2 * R.FF * R.row_bytes(tg, R.H) + R.H * R.row_bytes(td, R.FF)
        (d / "experts.txt").write_text(experts_txt + "\n", encoding="utf-8")
    else:
        per = R.BLOB
    RNG.integers(0, 256, 2 * per, dtype=np.uint8).tofile(d / "experts.bin")
    np.array([5, 1, 9], dtype=np.int32).tofile(d / "draft_vocab.bin")


class BlobLayout(unittest.TestCase):
    def test_blob_of_and_fused_of_are_inverses(self):
        blobs = RNG.integers(0, 256, (2, R.BLOB), dtype=np.uint8)
        gu, dn = R.fused_of(blobs)
        self.assertEqual((gu.shape, dn.shape), ((2, 2 * R.FF, 720), (2, R.H, 180)))
        self.assertTrue(np.array_equal(R.blob_of(gu, dn), blobs))

    def test_a_block_moves_whole(self):
        """Up row 0's first block: its scale and codes land where the engine's blob keeps gate/up row 1's."""
        gu = np.zeros((1, 2 * R.FF, 720), dtype=np.uint8)
        gu[0, R.FF, :18] = np.arange(1, 19, dtype=np.uint8)
        blob = R.blob_of(gu, np.zeros((1, R.H, 180), dtype=np.uint8))[0]
        codes, scales = R.H // 4, 2 * R.FF * R.H // 4 + R.H * R.FF // 4   # row 1 of the codes; the scales' plane
        self.assertEqual(blob[codes:codes + 16].tolist(), list(range(3, 19)))
        self.assertEqual(blob[scales + 2 * (R.H // 64):scales + 2 * (R.H // 64) + 2].tolist(), [1, 2])


class Files(unittest.TestCase):
    def convert(self, rt: Path, out: Path):
        R.write(R.from_rt(rt), out, "test")
        R.check_rt(out, rt)   # exits on a difference
        return GGUFFile(out)

    def test_a_folder_as_the_layers_file(self):
        with tempfile.TemporaryDirectory() as t:
            rt, out = Path(t), Path(t) / "layer.gguf"
            rt_folder(rt)
            g = self.convert(rt, out)
            self.assertEqual(g.metadata["general.architecture"], R.ARCH)
            self.assertEqual(g.metadata["lamina.mtp.version"], R.VERSION)
            tens = {x.name: x for x in g.tensors}
            self.assertEqual([x.name for x in g.tensors][:3], ["self_attn.q_proj.weight", "mlp.gate.weight", "x.norm.weight"])
            self.assertEqual((tens["self_attn.q_proj.weight"].type_name, tens["self_attn.q_proj.weight"].shape), ("Q8_0", [64, 4]))
            self.assertEqual((tens["x.norm.weight"].type_name, tens["x.norm.weight"].shape), ("F32", [8]))
            self.assertEqual((tens[R.GU].type_name, tens[R.GU].shape), ("Q2_0", [R.H, 2 * R.FF, 2]))
            self.assertEqual((tens[R.DN].type_name, tens[R.DN].shape), ("Q2_0", [R.FF, R.H, 2]))
            self.assertEqual((tens[R.VOCAB].type_name, tens[R.VOCAB].shape), ("I32", [3]))

    def test_experts_in_gguf_blocks(self):
        with tempfile.TemporaryDirectory() as t:
            rt, out = Path(t), Path(t) / "layer.gguf"
            rt_folder(rt, experts_txt="8 8")
            tens = {x.name: x for x in self.convert(rt, out).tensors}
            self.assertEqual((tens[R.GU].type_name, tens[R.DN].type_name), ("Q8_0", "Q8_0"))

    def test_another_draft_vocabulary_over_a_file(self):
        with tempfile.TemporaryDirectory() as t:
            rt = Path(t)
            rt_folder(rt)
            first, second = rt / "a.gguf", rt / "b.gguf"
            R.write(R.from_rt(rt), first, "test")
            lay = R.from_gguf(first)
            lay.vocab = np.array([2, 4], dtype=np.int32)
            R.write(lay, second, "test")
            a, b = R.from_gguf(first), R.from_gguf(second)
            self.assertEqual(b.vocab.tolist(), [2, 4])
            for (na, ta, ra), (nb, tb, rb) in zip(a.dense + a.experts, b.dense + b.experts):
                self.assertEqual((na, ta), (nb, tb))
                self.assertTrue(np.array_equal(np.asarray(ra), np.asarray(rb)))


if __name__ == "__main__":
    unittest.main()
