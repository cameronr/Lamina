#!/usr/bin/env python3
"""Lamina's CI (.github/workflows/build.yml), runnable on a PC the same way: the CUDA toolkit from NVIDIA's
redistributable archives, the engine with its tests and the CPU image encoder built by setup.py's own functions, the
tests that need no GPU, and the release archives.

    python .github/ci.py toolkit --dest DIR     the toolkit for this OS and CPU into DIR (in GitHub Actions also
                                                 CUDA_PATH and PATH for the later steps; elsewhere LAMINA_NVCC names it)
    python .github/ci.py build                  llama.cpp at the pinned commit, build/ (every program, the release's CUDA
                                                 architectures, for other PCs) and build-vision/ (the CPU encoder)
    python .github/ci.py test                   the tests that need no GPU: ctest -L cpu, the tools' unit tests
    python .github/ci.py package [--version V]  dist/lamina-V-bin-<os>-cuda-<x.y>-<cpu>.zip (.tar.gz on Linux): the
                                                 programs, and dist/cudart-lamina-bin-<os>-cuda-<x.y>-<cpu>.zip: the CUDA
                                                 runtime and cuBLAS, both laid out as in a checkout (build/, build-vision/)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import setup as S  # noqa: E402  (the checkout's setup.py)

# NVIDIA's redistributable CUDA 13.4.1 (redistrib_13.4.1.json): nvcc 13.4.59, cuBLAS 13.7.0.27, the toolkit build.cmd
# uses on the PC Lamina is developed on.  The parts the engine's build takes, nothing else.
CUDA_REDIST = "13.4.1"
REDIST_URL = "https://developer.download.nvidia.com/compute/cuda/redist"
COMPONENTS = ("cuda_nvcc", "cuda_crt", "libnvvm", "cuda_cudart", "libcublas", "cccl")
SKIP = {"libcublas_static.a", "libcublasLt_static.a"}   # ~1 GB the engine does not link
# The release's GPUs: RTX 30 (86), 40 (89) and 50 (120, with its PTX for newer ones), as the Dockerfile's
CUDA_ARCHS = "86-real;89-real;120-real;120-virtual"


def redist_platform() -> str:
    if S.WIN:
        return "windows-arm64" if S.ARM else "windows-x86_64"
    return "linux-sbsa" if S.ARM else "linux-x86_64"


def export(name: str, value: str):
    """`name` for this process and, in GitHub Actions, for the job's later steps."""
    os.environ[name] = value
    if os.environ.get("GITHUB_ENV"):
        with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")


def extract(arc: Path, dest: Path):
    """An archive's files into dest, without its top folder (<component>-<platform>-<version>-archive)."""
    if arc.name.endswith(".zip"):
        with zipfile.ZipFile(arc) as z:
            for m in z.infolist():
                rel = m.filename.split("/", 1)[1] if "/" in m.filename else ""
                if not rel or m.is_dir() or Path(rel).name in SKIP:
                    continue
                out = dest / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, open(out, "wb") as f:
                    shutil.copyfileobj(src, f)
        return
    with tarfile.open(arc) as t:
        for m in t.getmembers():
            rel = m.name.split("/", 1)[1] if "/" in m.name else ""
            if not rel or Path(rel).name in SKIP:
                continue
            m.name = rel
            if m.islnk():
                m.linkname = m.linkname.split("/", 1)[1]
            t.extract(m, dest, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))


def toolkit(dest: Path, version: str):
    plat = redist_platform()
    mark, key = dest / ".lamina-redist", f"{version} {plat} {' '.join(COMPONENTS)}"
    if mark.exists() and mark.read_text().strip() == key:
        S.ok(f"CUDA {version} ({plat}) in {dest}")
    else:
        with urllib.request.urlopen(f"{REDIST_URL}/redistrib_{version}.json", timeout=60) as r:
            manifest = json.loads(r.read())
        dest.mkdir(parents=True, exist_ok=True)
        for c in COMPONENTS:
            e = manifest[c][plat]
            arc = dest.parent / Path(e["relative_path"]).name
            h = S.get(f"{REDIST_URL}/{e['relative_path']}", arc, f"{c} {manifest[c]['version']}", int(e["size"]),
                      hashlib.sha256())
            if h.hexdigest() != e["sha256"]:
                arc.unlink()
                S.fail(f"{arc.name}: not the archive NVIDIA's manifest names (SHA-256)")
            extract(arc, dest)
            arc.unlink()
        if not S.WIN and not (dest / "lib64").exists():   # nvcc.profile links from lib64, the archives hold lib
            (dest / "lib64").symlink_to("lib")
        mark.write_text(key + "\n")
        S.ok(f"CUDA {version} ({plat}) in {dest}")
    export("CUDA_PATH", str(dest))
    nvcc = dest / "bin" / ("nvcc" + S.EXE)
    if os.environ.get("GITHUB_PATH"):
        with open(os.environ["GITHUB_PATH"], "a", encoding="utf-8") as f:
            f.write(f"{dest / 'bin'}\n")
    else:
        S.say(f"  for the next steps: LAMINA_NVCC={nvcc}")


def checked_toolchain() -> S.Toolchain:
    tc = S.toolchain()
    problems = tc.problems() + ([] if tc.cmake and tc.ninja else ["CMake and Ninja (pip install -r requirements-build.txt)"])
    if problems:
        S.fail("missing for the build: " + "; ".join(problems))
    return tc


def engine_version() -> str:
    """The version the programs report (CMakeLists.txt's project VERSION, LAMINA_VERSION)."""
    m = re.search(r"^project\(lamina VERSION ([0-9.]+)", (S.ROOT / "CMakeLists.txt").read_text(encoding="utf-8-sig"),
                  re.MULTILINE)
    return m.group(1) if m else "?"


def build():
    tag = os.environ.get("GITHUB_REF_NAME", "") if os.environ.get("GITHUB_REF_TYPE") == "tag" else ""
    if tag and tag != f"v{engine_version()}":   # before an hour of building
        S.fail(f"the tag {tag} is not the version the programs report, v{engine_version()}",
               "set project(lamina VERSION ...) in CMakeLists.txt to the release's and tag that commit")
    tc = checked_toolchain()
    S.say(f"CUDA {tc.cuda[0]}.{tc.cuda[1]} ({tc.nvcc}), {CUDA_ARCHS}")
    S.fetch_llama()
    env = S.vs_environment(tc.vcvars) if S.WIN else S.clean_env()
    ccache = shutil.which("ccache")
    if ccache:   # read by CMake at a build folder's first configure, the image encoder's too
        for lang in ("C", "CXX", "CUDA"):
            env[f"CMAKE_{lang}_COMPILER_LAUNCHER"] = ccache
    # Linux: the programs look for the CUDA runtime beside them (the runtime archive), not in this machine's toolkit
    extra = () if S.WIN else ("-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON", "-DCMAKE_INSTALL_RPATH=$ORIGIN")
    S.build_engine(tc, CUDA_ARCHS, True, env, tests=True, extra=extra)
    S.build_vision(tc, env)
    if ccache:
        subprocess.run([ccache, "--show-stats"])


def toolkit_root(tc: S.Toolchain) -> Path:
    return Path(tc.nvcc).resolve().parent.parent


def test():
    tc = checked_toolchain()
    env = S.clean_env()
    tk = toolkit_root(tc)
    if S.WIN:   # build/ holds the CUDA DLLs (build_engine), the toolkit's are the same
        env["PATH"] = os.pathsep.join([str(tk / "bin" / ("arm64" if S.ARM else "x64")), env.get("PATH", "")])
    else:       # the programs' RUNPATH is their own folder ($ORIGIN, build())
        lib = S.cuda_lib_dir(tc.nvcc) or str(tk / "lib")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(x for x in (lib, env.get("LD_LIBRARY_PATH", "")) if x)
    ctest = Path(tc.cmake).with_name("ctest" + S.EXE)
    S.run([ctest, "--test-dir", S.ROOT / "build", "-L", "cpu", "--output-on-failure", "--timeout", "900"], env=env)
    # the tools' tests (requirements.txt), from the checkout's folder as CONTRIBUTING.md runs them
    S.say("  $ python -m unittest discover -s tools -p test_*.py")
    if subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tools", "-p", "test_*.py"], cwd=S.ROOT,
                      env=S.clean_env()).returncode != 0:
        S.fail("the tools' tests failed (above)")


def write_archive(base: Path, files: dict) -> Path:
    """files ({name in the archive: file}) into base.zip on Windows, base.tar.gz elsewhere (programs executable)."""
    if S.WIN:
        out = base.with_name(base.name + ".zip")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for name, src in files.items():
                z.write(src, name)
    else:
        out = base.with_name(base.name + ".tar.gz")
        with tarfile.open(out, "w:gz") as t:
            for name, src in files.items():
                info = t.gettarinfo(str(src), name)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with open(src, "rb") as f:
                    t.addfile(info, f)
    S.ok(f"{out.relative_to(S.ROOT)}: {out.stat().st_size / 1e6:.0f} MB")
    return out


def package(version: str):
    tc = checked_toolchain()
    tk = toolkit_root(tc)
    name = f"{'win' if S.WIN else 'linux'}-cuda-{tc.cuda[0]}.{tc.cuda[1]}-{'arm64' if S.ARM else 'x64'}"
    dist = S.ROOT / "dist"
    dist.mkdir(exist_ok=True)
    progs = [f"build/lamina{S.EXE}", f"build/lamina-server{S.EXE}", f"build-vision/bin/lamina-vision{S.EXE}"]
    files = {p: S.ROOT / p for p in progs}
    files |= {"build/licenses/LICENSE": S.ROOT / "LICENSE",
              "build/licenses/LICENSE-llama.cpp": S.LLAMA / "LICENSE",
              "build/licenses/LICENSE-cpp-httplib": S.LLAMA / "vendor" / "cpp-httplib" / "LICENSE",
              "build/licenses/LICENSE-jsonhpp": S.LLAMA / "licenses" / "LICENSE-jsonhpp",
              "build/licenses/OFL.txt": S.ROOT / "src" / "server" / "web" / "fonts" / "OFL.txt"}
    missing = [str(f) for f in files.values() if not f.is_file()]
    if missing:
        S.fail("not built or not there: " + ", ".join(missing))
    write_archive(dist / f"lamina-{version}-bin-{name}", files)
    if S.WIN:
        runtime = {f"build/{d.name}": d for d in S.cuda_dlls(tc.nvcc)}
    else:   # the sonames the programs ask for, as files
        lib = Path(S.cuda_lib_dir(tc.nvcc) or tk / "lib")
        runtime = {f"build/{n}": (lib / n).resolve()
                   for n in (f"libcudart.so.{tc.cuda[0]}", f"libcublas.so.{tc.cuda[0]}", f"libcublasLt.so.{tc.cuda[0]}")}
    if len(runtime) != 3 or not all(f.is_file() for f in runtime.values()):
        S.fail(f"the CUDA runtime and cuBLAS of {tk}: {sorted(runtime)}")
    if (tk / "LICENSE").is_file():
        runtime["build/licenses/LICENSE-cuda.txt"] = tk / "LICENSE"
    write_archive(dist / f"cudart-lamina-bin-{name}", runtime)


def default_version() -> str:
    """A tag's name, else the commit's (8 digits)."""
    if os.environ.get("GITHUB_REF_TYPE") == "tag":
        return os.environ["GITHUB_REF_NAME"]
    return S.out(["git", "-C", S.ROOT, "rev-parse", "--short=8", "HEAD"]).strip() or "local"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 epilog=__doc__.split("\n\n", 1)[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("toolkit")
    t.add_argument("--dest", required=True)
    t.add_argument("--version", default=CUDA_REDIST, help=f"NVIDIA's redistrib_<version>.json (default {CUDA_REDIST})")
    sub.add_parser("build")
    sub.add_parser("test")
    p = sub.add_parser("package")
    p.add_argument("--version", help="the archives' version (default: the tag, else the commit)")
    a = ap.parse_args()
    if a.cmd == "toolkit":
        toolkit(Path(a.dest).resolve(), a.version)
    elif a.cmd == "build":
        build()
    elif a.cmd == "test":
        test()
    else:
        package(a.version or default_version())
    return 0


if __name__ == "__main__":
    sys.exit(main())
