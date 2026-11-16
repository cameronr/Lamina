# Setup

What a PC needs: [the README](../README.md#what-you-need). This page covers the build from the source, a model file
run with `lamina-server -m`, the release binaries, the config, the GPUs, the engine options worth knowing, and
Qwen3.6-35B-A3B and Qwen3.8-27B.

## Building

From a clone of this repository, with the CUDA toolkit 13.0 or newer and, on Windows, Visual Studio 2022's C++ tools
(its free Build Tools with the workload "Desktop development with C++" do), on Linux GCC 11 or newer, CMake 3.22 or
newer and Ninja (Ubuntu 22.04's own packages do):

```
build.bat                        (Windows)
./build.sh                       (Linux)
```

They build the preset `native` of `CMakePresets.json` into `build/`: `lamina-server`, `lamina` and the image encoder
`lamina-vision`, on Windows with the CUDA runtime's DLLs beside them (on Linux they take the toolkit's), for this PC's
CPU and its GPUs of the RTX 30 series and newer, as `nvidia-smi` names them at each build (after a card is added, build
again). The programs run from there: `build\lamina-server -m MODEL.gguf` ([below](#a-model-file)).
`build.bat portable` (`./build.sh portable`) builds them for other PCs into `build-portable/`, as the releases are: RTX
30, 40 and 50 GPUs (compute capability 8.6, 8.9 and 12.0, with PTX for newer ones), any x64 CPU with AVX2 or arm64 CPU
with Armv8.2's dot product and FP16 instructions, the C++ runtime linked in. CMake options after the preset's name go
to its configure (`build.bat native -DLAMINA_BUILD_TESTS=ON`, `--fresh` for a new start of the build folder). The
presets work with CMake alone too: `cmake --preset native`, `cmake --build --preset native` (on Windows in a Visual
Studio developer prompt).

- **llama.cpp**: the first configure downloads it at the commit the engine is built against, checks its SHA-256 and
  unpacks it into `third_party/llama.cpp`, without llama.cpp's web app (`tools/ui`, whose deep paths pass Windows'
  260-character limit). `-DLAMINA_GGML_DIR=<folder>` names another checkout.
- **The tools**: `build.bat` finds Visual Studio 2022 (else 2019 or 2026) itself, or takes a developer prompt's, and
  its CMake and Ninja unless the PATH has others; on Windows on Arm it compiles C and C++ with LLVM's clang-cl (Visual
  Studio's "C++ Clang Compiler for Windows" or LLVM's own: ggml's CPU code for Arm does not build with MSVC). nvcc is
  the one on the PATH, in `CUDA_PATH` or in `/usr/local/cuda`; `CUDACXX` names another, `CXX` the C++ compiler (on
  Linux nvcc's host compiler too: `CXX=g++-14` when the GCC is newer than the toolkit takes).
- **The checks**: a CUDA toolkit older than 13.0 stops the configure (engines built with CUDA 12.8 crashed in long
  prompts on RTX 50 GPUs), as do 13.2.0 and 13.2.1 for RTX 50 GPUs (their nvcc compiled some of the kernels wrong: the
  answers were garbage), and GPUs older than the RTX 30 series are left out.
- **Jobs**: half the logical cores compile and link at once, so that the PC stays usable, fewer when the free RAM holds
  fewer (1.5 GB a job: nvcc's largest files take up to 0.9 GB); `-DLAMINA_JOBS=N` sets another number. A build that
  stops is tried once more: CUDA's ptxas now and then fails to read a file it has just written.
- **What it builds**: the three programs. `-DLAMINA_BUILD_TESTS=ON` adds the tests (`ctest --test-dir build`), the
  parity checks and the tools, which otherwise build by name (`cmake --build build --target verify_parity`);
  `-DLAMINA_BUILD_VISION=OFF` leaves the image encoder out.

`cmake --install build --prefix <folder>` puts the programs into a folder of their own, flat, as a release's two
downloads extract: the programs, `download-draft-layer.bat` (`.sh` on Linux) and the licenses (the component
`lamina`), and the CUDA runtime and cuBLAS the programs load, with NVIDIA's license (`cudart`; `--component` installs
one). Installed on Linux, the programs load the libraries in their own folder.

**A CPU-only image encoder.** `tools/vision` builds one without CUDA by default: 5 MB instead of the CUDA build's
~92 MB, in seconds instead of minutes, as fast on the CPU. From the clone (on Windows in a Visual Studio developer
prompt; it downloads llama.cpp as the engine's build does when `third_party/llama.cpp` is not there):

```
cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-vision --target lamina-vision
```

Copy `build-vision/bin/lamina-vision` (`.exe`) beside `lamina-server`, over the one there, and keep `"gpu": false` in
the configs' `vision` sections: with `"gpu": true` such a build refuses to start, and the server says so. The engine's
build with `-DLAMINA_BUILD_VISION=OFF` leaves it there; a build with the encoder writes its CUDA one over it.

## A model file

`lamina-server -m MODEL.gguf` runs a model file as it lies, a split one by any of its shards:

```
lamina-server -m D:\Models\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
```

The server builds a config for the file and says in its window what it took:

- **The preset**, one of `examples/*.json`, built into the server and picked by the file's header: ISTA-DASLab's
  Qwen3.8-Flash-Next files take `iq3_s`, the family's other files (unsloth's quants, Agention's Gyro-L) `ud-q4_k_xl`,
  Qwen3.6-35B-A3B's `qwen36-ud-q4_k_xl`, Qwen3.8-27B's `qwen38-27b-ud-q6_k_m`. Its engine options, sampling and port
  (8081 to 8084) are the config's; the model's name in the API is the file's, without the shard number.
- **The GPU** it runs on: of the cards of the RTX 30 series and newer, the one with the most VRAM that drives no
  display (the lower number on a tie; `--main-gpu` names another), every other card as `--second-gpu auto` takes it.
- **Flash-Next's draft layer**, `Lamina-Qwen3.8-Flash-Next-MTP.gguf`, beside the model, else beside the server
  ([the README](../README.md#the-draft-layer)).
- **Pictures**: an image projector made for the model beside it (an `mmproj*.gguf` whose width is the model's; of
  several the smallest; Qwen3.6's and Qwen3.8-27B's repositories have theirs, `mmproj-F16.gguf`), read by the
  `lamina-vision` beside the server on the CPU's logical cores, where it hides the GPUs so that they keep their VRAM for
  experts. With `--vision-gpu` it runs on a GPU (~1.3 GB of its VRAM): of the cards besides the main one, the one with
  the least VRAM that drives no display, else any, else the main one; `--vision-cuda-device <UUID>` names another
  (`nvidia-smi -L` lists the UUIDs). `--mmproj FILE` names a projector, `--no-mmproj` leaves pictures out; without a
  projector or the encoder the window says why there are none. For Qwen3.8-Flash-Next on the CPU, download Lamina's Q8_0
  projector into the model's folder: `mmproj-Qwen3.8-Flash-Next-Q8_0.gguf` (0.6 GB, ISTA-DASLab's BF16 one in Q8_0), in
  [eddoursul/Lamina-Qwen3.8-Flash-Next-MTP](https://huggingface.co/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP) beside the
  draft layer. The CPU encodes a picture with it in ~20% less time than with the BF16 one and 280 MiB less RAM, with the
  same answers.
- **The expert profile**, Qwen3.8-Flash-Next's, built into the programs (`--expert-profile builtin`, the engine's
  default under `--serve`).
- **What a run writes**, beside the server: the engine's log (`<model>.log`) and what the run learned about the
  experts (`<model>-learned.bin`, [below](#the-config)), which the next start begins from.

The engine also runs Flash-Next's other GGUF files, most of unsloth's quants and
[Agention's Gyro-L](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) among them. Of unsloth's Dynamic
3.1 quants it refuses UD-IQ1_S (a layer with its gate and up in two formats) and BF16, saying so at load. A quant's
n-gram table can be one of unsloth's engram tables (BF16, Q8_0, Q5_1 or IQ4_NL): `--ple-gguf <the table's file>`, on
the command line or in a config's `args`, or the file renamed over the quant's table shard as unsloth's README says.

**The draft layer.** `download-draft-layer.bat FOLDER` (`./download-draft-layer.sh FOLDER` on Linux; a release's archive
has its platform's script beside the programs) downloads `Lamina-Qwen3.8-Flash-Next-MTP.gguf` from
[eddoursul/Lamina-Qwen3.8-Flash-Next-MTP](https://huggingface.co/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP) into a folder,
at a pinned revision, checked by its SHA-256 and a broken download resumed. A config whose `args` name no `--mtp` takes
`Lamina-Qwen3.8-Flash-Next-MTP.gguf` from beside the model, else from beside the server, and the server's window says
which; elsewhere the `args` name it: `"--mtp", "<folder>/Lamina-Qwen3.8-Flash-Next-MTP.gguf"`. A folder of the layer's
earlier layout (`dense.txt`, `dense.bin`, `experts.bin`, `draft_vocab.bin`), as earlier versions downloaded it to
`mtp/rt-distilled`, still works as `--mtp`. Flash-Next's server starts without the layer too, when none is found or the
named file or folder is not there, and warns in its window with the script's command. It then generates a token a verify
window, or prompt lookup's drafts (`--spec-lookup`). On the development PC, new text came at 44-61% of IQ3_S's speed
with the layer and 50-68% of UD-Q4_K_XL's, an edit that quotes its prompt at 92-96%. The layer's file holds its draft
vocabulary, the 40,525 tokens it proposes from; `tools/mtp_rt.py --gguf <the file> --draft-vocab <ids> --out <new file>`
writes one over another (`data/draft_vocab_fr.bin`, `draft_vocab_cyrillic.bin`, `tools/draft_vocab.py`).

Every key of a config is an option of the command line too ([USING.md](USING.md#the-config)), and a value given there
takes the place of the config file's, the file's that of the preset: `lamina-server -m MODEL.gguf --port 8090
--temperature 0.6`. A config file can be as short as `{"model": "D:/Models/x.gguf", "port": 8090}`: without `"args"`
the server builds the rest from the model's preset, as with `-m`, and the file's keys take the place of the preset's;
a config with `"args"` stays as it is, `-m` or its `"model"` then naming the file in place of their `--native`. What a
run writes (the log, the learned profile) goes beside that config file. An engine option on the command line, with the
values after it, takes the place of the preset's or the config's: `lamina-server -m MODEL.gguf --max-context 65536
--kv q4_0`. The engine's own parser reads those options first, for the model's family, and the server stops on one it
does not take ("unknown argument: ...", a value out of range); `--native`, `--serve` and `--vision` are the server's to
set.

`--write-config FILE` writes the config the command line runs to FILE (the preset's, the config file's and the command
line's settings in one) and stops; `lamina-server --config FILE` then runs it, and the web app's Model settings card
saves into it (a server started with `-m` alone has no file: the card's changes last until it stops). The log and the
learned profile then go beside FILE. An earlier FILE is kept as `FILE.bak`, and its keys outside the preset (an
`api_key`, an `images` section) carry over.

`--check` says what this PC has (the CPU, the RAM, the GPUs) and, with `-m` or `--config`, the model's file, the preset,
the GPUs it runs on, the RAM its experts take, the draft layer, the pictures, the port and the engine's command line,
then whether it would start; it loads nothing and exits with 1 when the model would not start (no usable GPU, too
little RAM, a file the engine does not read, a missing encoder or projector).

`--bench` measures the model the way it is used and writes a report to share: `lamina-server -m MODEL.gguf --bench`
(or `--config FILE --bench`) loads it on a port of its own, sends a fixed suite through the server's API, three runs of
each test, and stops: three short questions (decode speed, time to the first token), fresh prompts of ~4K and ~16K
tokens (prompt speed), and a coding agent's session of six turns with four tools (what the conversation cache
reused), 3-10 minutes after loading. Speeds are the engine's own timings, first-token times the client's clock;
thinking is off. The report goes into `bench-results/<date>-community-<GPU>-<model>/` beside the program:
`README.md`, `results.json` with every request, and the engine's log lines, with home folders, user and host names,
private addresses and API keys replaced. `--bench-clients N` adds N requests at once, `--bench-long` a ~32K prompt,
`--bench-compare OLD` puts an earlier report beside this one (`--bench-compare OLD NEW`: two reports, nothing run),
and `--bench-url http://127.0.0.1:8081` measures a server that runs instead. A Lamina server on the config's port
stops it before anything loads: the two would share the GPUs.

## Release binaries

Each [release](https://github.com/eddoursul/Lamina/releases) has the programs built for Windows and Linux, on x64 and
arm64, in two downloads a platform:

- `lamina-<version>-bin-<os>-cuda-<x.y>-<cpu>`: the server with the engine (`lamina-server`), the engine alone
  (`lamina`) and the image encoder (`lamina-vision`, on the CPU or a GPU as a config says), with their licenses, and
  the script that downloads Flash-Next's draft layer (`download-draft-layer.bat`, `.sh` on Linux);
- `cudart-lamina-bin-<os>-cuda-<x.y>-<cpu>`: the CUDA runtime and cuBLAS they load, beside them (the same in every
  release of that CUDA version: one download serves the next releases).

Both are flat: extract them into one folder, and `lamina-server -m` runs a model file from there
([above](#a-model-file)). The programs are built for RTX 30, 40 and 50 GPUs (compute capability 8.6, 8.9 and 12.x, the
GB10's among them, with PTX for newer ones), any x64 CPU with AVX2 (the AVX-512 kernels where the CPU has them) or arm64
CPU with Armv8.2's dot product and FP16 instructions, and a driver for CUDA 13; the Linux ones for glibc 2.34 and
OpenSSL 3, as Ubuntu 22.04, Debian 12, RHEL 9 (and Rocky Linux, AlmaLinux), Amazon Linux 2023, openSUSE Leap 15.6 and
newer systems have them (the C++ runtime is linked in), and without the runtime archive they take the CUDA 13 libraries
of the system (`LD_LIBRARY_PATH`). Other GPUs (an A100's 8.0, a GH200's 9.0) and a native build for the PC's CPU need a
build from the source ([above](#building)). The arm64 programs are built and their CPU tests run on each release, not
yet run with a GPU.

The releases come from the GitHub workflow `.github/workflows/build.yml` at a `v*` tag; it also builds every push to
`main` and every pull request, and runs the tests that need no GPU. Its steps are `.github/ci.py`'s, which runs on a PC
the same way (`toolkit`, `build`, `test`, `package`).

## The config

A config holds the engine's command line and the server's settings; `--write-config` writes one from a preset
([above](#a-model-file)). Its keys:

| Key | |
| --- | --- |
| `args` | the engine's options (below and `build/lamina --help`) |
| `cwd`, `lib_dirs` | the folder the engine runs in, where the config's relative paths start (the folder the server starts in when it has none); on Linux, folders put first on the library path |
| `model_name` | the name the API lists |
| `port`, `log` | the server's port; the engine's log file |
| `sampling` | the defaults for requests that send no sampling fields ([USING.md](USING.md#sampling)) |
| `vision` | the image encoder's files and settings: the projector, the model, `threads` (the CPU's logical cores), or `gpu` and `cuda_device` (the card it runs on, by its UUID: the server does not start when that card is not there; without one the server picks a card as `--vision-gpu` does). The encoder itself is the `lamina-vision` in the server's folder |
| `expert_profile_save`, `expert_profile_save_every` | what the run learned about the experts (below) |

The server's other keys (the network, API keys, idle unload, MCP servers, image generation):
[USING.md](USING.md#the-config).

**The learned expert profile.** The GPUs' expert caches start from a ranking of the experts by how often the model
routes to them (`data/expert-profile.bin`, built into the programs). With `expert_profile_save`, the engine also counts
this PC's routing and writes a profile (`<model>-learned.bin` beside the config file, beside the server with `-m` alone)
at exit, on Ctrl+C and between requests at most once a minute (`expert_profile_save_every`); the next start fills the
caches from it. It ranks the experts by this PC's routing and the profile it started from at equal weight. On the
development PC, after a session of code and reasoning requests, a round of fixed text took 9-14% less time on code, -3.8
to +0.9% on thinking and edits. Remove both keys to start from the shipped profile every time.

## The GPUs

**`--main-gpu`** names the GPU the model runs on. Nothing else may use it: the engine synchronizes the GPU with its host
loop through spin-waits, and a display or another program on the same GPU makes the driver time-slice it, which slows
generation to a few tokens per second. Name it by its name or a part only its name has (`"RTX 3090"`), its UUID or the
start of it (`nvidia-smi -L` lists them), its PCI bus id, or its number as `nvidia-smi` numbers the GPUs (a number
moves when a card is added). The engine leaves `--vram-reserve-mib` (default 700) of it free and fills the rest with
experts (`--expert-cache auto`).

**`--second-gpu auto`** (the examples' setting) makes every other GPU a further expert tier, up to three, the fastest
link first: a short copy test at the start measures each GPU's rate from the host, and the log shows it. Each tier
holds the next experts of the profile and of the conversation and computes them beside the CPU; the first one also
computes the experts of a prompt that the main GPU lacks and part of the output head. A tier may drive the display: one
that does keeps 2048 MiB free, a headless one 512 (`--second-gpu-reserve-mib`). To fix the tiers and their order, name
them: `--second-gpu "RTX 5070 Ti,RTX 5060 Ti"`; `--second-gpu -1` uses none. Only the first tier takes part in
prompts. The others help generation, where an expert crosses the link once and stays, so a GPU on a slow link (an x1
or x4 slot) still pays as a later tier. Such a tier gets only short pieces of work, and a driver may lower its clocks
between them until its results come late: `--second-gpu-clocks` (below) locks them while the engine generates.

**Each GPU is checked** before the engine uses it: one older than the RTX 30 series (compute capability below 8.0), or
one the programs have no kernels for, stops the start as the main GPU and is left out as a tier or a layer split's GPU,
the log saying why. A build from the source is for this PC's GPUs: after a card is added, build again. The releases'
programs run on the RTX 30, 40 and 50 series.

## Engine options

The example configs' options, and the ones that pay on some PCs. `build/lamina --help` lists every one.

| Option | |
| --- | --- |
| `--max-context N` | the context the K/V is sized for (the examples: 262,144, the trained context) |
| `--kv-grow` | the K/V takes VRAM only for the cells the requests reach; the expert cache holds the rest and gives it back as a conversation grows |
| `--kv fp16\|int8\|k8v4\|q4_0` | the K/V's format: 8-bit (`int8`), 8-bit keys and 4-bit values (`k8v4`), 4-bit both; smaller formats leave more VRAM to experts, at a cost in precision |
| `--spec T`, `--spec-min-p P`, `--spec-lookup N` | verify windows of up to T tokens (the last one and T-1 drafts), a draft kept while its probability under the draft layer stays above P (a sampled request's by the number its position is drawn with: [HOW_IT_WORKS.md](HOW_IT_WORKS.md#drafts)), and drafts from earlier text that repeats (16 tokens of it) |
| `--prefill N` | the prompt chunk (the examples: 16,384 tokens) |
| `--feed-max N` | prompt parts of up to N tokens go through verify windows instead of the prompt path |
| `--hc-q8` | the hyper-connection projections in 8 bits only: half their VRAM, read ~30% faster |
| `--prompt-shift F` | on a second GPU, a share of the experts the main GPU holds is computed beside it in long prompts |
| `--pcie-mode kernel --pcie-frac 0` | the experts no GPU holds are all computed by the CPU, none read by the main GPU over its link (the examples' setting) |
| `--prompt-split L[,M]` | off by default: a prompt of more than one chunk runs in two stages, the second GPU taking each chunk's layers L and up while the main GPU runs the next chunk; `L,M` with two or more tiers: three stages, the tier after the first taking the layers L to M-1 (with one tier: two stages, split halfway between L and M) |
| `--prompt-split-from N` | off by default: with `--prompt-split`, a prompt part of N tokens or more that fits one chunk is read as two chunks of half its size through the stages (a shorter one runs on the main GPU's stage alone) |
| `--prompt-tiers 2` | off by default: with two or more tiers, the tier after the first computes the experts it holds in prompts too, where they otherwise stream to the first tier |
| `--arena-place` | off by default: for RAM with slower regions (unpaired DIMMs), the experts the CPU reads are kept out of them |
| `--no-file-cache` | off by default: the experts are read past the OS file cache, which then holds no second copy of the model |
| `--second-gpu-clocks C[,..]` | off by default: per tier, its graphics and memory clocks locked through NVML while the engine generates, `<graphics min>-<graphics max>/<memory>` in MHz, the card's own (`0`: that tier's stay the driver's); needs administrator rights (root on Linux) |

| `--second-gpu-clocks-hold S` | with `--second-gpu-clocks` under `--serve`: the locks stay S seconds after a request (5 by default); a locked card draws its awake power until then |
| `--second-gpu-clocks-lead S` | off by default (the locks are asked for from a batched prompt part's last chunk on): they are asked for S seconds before the part ends, by its progress; a lock is at speed ~0.6 s after it is asked for |
| `--second-gpu-sleep-us T[,..]` | off by default: per tier after the first, the tier sleeps (its experts to the CPU, its lock released) while the CPU pool would need less than T microseconds a layer without it, averaged over a few seconds of generation; trades speed for the card's awake power, T by model file |

The last nine are measured on the development PC in [examples/](../examples/README.md); measure them on yours before
keeping them.

## Control vectors

A control vector in llama.cpp's format (a GGUF of `direction.<layer>` tensors, `general.architecture`
`controlvector`) steers Qwen3.8-Flash-Next at every token: after each layer of `--control-vector-layer-range A B`
(default 1-47, inclusive), on every hyper-connection stream h, as llama.cpp applies it with `build_cvec`:

| `--cvec-mode` | |
| --- | --- |
| `add` (the default, llama.cpp's) | h + S d |
| `project` (the [cvec-hooks](https://github.com/Cudila/llama.cpp/tree/cvec-hooks) fork's, which Strata applies) | h - \|S d\| (h . v) v, v = d / \|d\|: a unit direction at S 1 is removed |

`--control-vector-scaled FILE:S` takes a file at scale S (`--control-vector FILE`: 1); several files add up per layer,
and `--cvec-dir single:L` (project) applies layer L's direction in every layer of the range.

**The refusal direction of huihui-ai's abliterated releases**,
[Huihui-Qwen3.8-Flash-Next-refusal-direction-r-v2.gguf](https://huggingface.co/Pragmatism0220/Huihui-v2-Qwen3.8-Flash-Next-GSQ-RCO-Strata-abliterated-direction-GGUF),
takes out at runtime, from any quant of the base model, the direction their weights edit (their v1 removes it from what
each layer writes, v2 inverts it at 1.78 times that). In a config's `args`, as Strata's speed projection applied its
own vector:

```
"--control-vector-scaled", "<folder>/Huihui-Qwen3.8-Flash-Next-refusal-direction-r-v2.gguf:1.0",
"--control-vector-layer-range", "4", "44", "--cvec-mode", "project"
```

On the development PC, IQ3_S declined none of eight requests that it declined every one of without it, and began four
harmless answers word for word the same. That removes a safety behaviour: what the model writes with it is yours to
answer for. On ordinary text it moves the next-token distributions about as much as rounding does. Generation is
0.2-1% slower, a long prompt 0.4% (the prompt path applies the vector in the pass that writes the layer's output).
The vector holds for every request of the process: Strata's per-request switch is not there.

## Qwen3.6-35B-A3B and Qwen3.8-27B

The engine runs Qwen3.6-35B-A3B (GGUF architecture `qwen35moe`) and Qwen3.8-27B (`qwen35`, a dense model of the same
family) on a path of its own (`src/qwen35`): `lamina` switches to it from the model file's header, so their configs
name the file with `--native` like the others. It reads the GGUF as it is (no expert profile; the drafts come from the
file's own MTP block). K and V are fp16 (~22 KB a cell for Qwen3.6, ~68 KB for the 27B), or with `--kv q8_0` 8 bits
in llama.cpp's q8_0 rounding, about half: the 27B's full 262,144 cells then take ~9 GiB.

The drafts: `--mtp N` at most a step; `--spec-min-p P` ends a step's drafts before one whose joint probability with the
drafts before it, under the draft head, is below P; `--draft-vocab N` lets the draft head score only the first N token
ids (a draft outside them is a rejected draft, nothing else). The 27B's example takes 7, 0.3 and 65,536: up to 8
tokens a step on text that repeats its prompt, fewer where the drafts are unsure.

With `--second-gpu` (GPUs named as for `--main-gpu`, comma-separated, or `auto`: every other GPU in CUDA's order; the
examples have `auto`) the path splits the layers over the fewest of those GPUs, in order, that hold the whole model:
each takes the layers that fit beside their K/V and its buffers, the last the rest with the output head and the MTP
block, and each pass hands its activations from one GPU to the next. `--layer-split N,N,..` sets each GPU's layer
count. A prompt of several chunks (`--prefill`, the 27B's example 1,536 tokens) runs on the GPUs at once: the first
takes the next chunk while the second runs this one. Split, every Qwen3.6 expert is in VRAM and none takes RAM: on the
development PC, over the RTX 3090 and the RTX 5070 Ti (~4.7 GiB on the second), it generates 1.6-1.7x as fast and
reads prompts 1.9-3.3x as fast as on the 3090 alone ([COMPARISON.md](COMPARISON.md#qwen36-35b-a3b)). Where no split
holds every expert, or on one GPU, the main GPU keeps the experts that fit in its VRAM and the CPU computes the rest:
on a 24 GB GPU 212 of 256 a layer at the example's 65,536 cells, every one at 8,192 (`--context` sets another).
Qwen3.8-27B needs a split where one GPU does not hold it: its full 262,144 cells with K and V in q8_0 take ~36 GB of
VRAM (on the development PC the 5070 Ti first with 26 layers, then the 3090 with 38 and the head:
[COMPARISON.md](COMPARISON.md#qwen38-27b)).

Not on this path yet: the prompt cache's checkpoints (a follow-up turn whose history differs from what the engine holds
is read again from the start), `/v1/choose` and the GPU tiers. The config's `parallel` (batch slots) runs on one GPU.

## Updating

`git pull` and the build again update a clone's programs; the configs and the model files stay where they are. With
release binaries, extract the new release's archives over the old ones.

**A clone an earlier version's setup made.** `setup.bat` and `setup.sh` are gone: `build.bat` (`./build.sh`) builds the
programs into the same `build/`, where the `run-<model>.bat` (`.sh`) and `lamina-<model>.json` setup wrote run them
as before. The configs need no change: their `vision` section's `exe`, which named setup's encoder in
`build-vision/bin` or `build-vision-cuda/bin`, is not read (the server starts the `lamina-vision` the build writes
beside it, so a build of `lamina-server` alone, or with `-DLAMINA_BUILD_VISION=OFF`, leaves such a config unable to
start), and the draft layer's folder `mtp/rt-distilled` still loads.

- **CMake and Ninja**: setup installed them into `.venv`. `build.bat` takes Visual Studio's ("C++ CMake tools for
  Windows", part of the workload "Desktop development with C++"), `build.sh` the PATH's (`sudo apt install cmake
  ninja-build`). `build/` keeps the Ninja of `.venv` it was configured with: once `.venv` is deleted, its configure
  stops ("Running '...ninja' '--version' failed") until `build.bat --fresh` (`./build.sh --fresh`).
- **The CUDA toolkit**: `build/` keeps the nvcc setup chose; `CUDACXX=<path to nvcc>` with `--fresh` takes another
  ([above](#building)).
- **What setup did otherwise**: `setup --check` is `lamina-server --config lamina-<model>.json --check`; `BENCH.bat`
  and `bench.sh` are `lamina-server --config lamina-<model>.json --bench`, and `--bench-compare <an earlier report's
  folder>` reads the reports they wrote; a model setup did not download runs with `lamina-server -m`
  ([above](#a-model-file)), which downloads nothing. The image generator `tools/image/install.py` built
  (`build-image-cuda`) and the config's `images` section keep working; it is built again by hand
  ([USING.md](USING.md#image-generation)).
- **What nothing uses now**: `build-vision/` and `build-vision-cuda/` (setup's encoders) and, after a build with
  `--fresh`, `.venv` (unless the Python tools' packages, `requirements.txt`, went into it). `third_party/llama.cpp` is
  the build's.
