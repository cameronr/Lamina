# Lamina

Lamina runs [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), a 125-billion-parameter
mixture-of-experts model, on one PC with NVIDIA GPUs and plenty of RAM, and
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) and
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on one GPU or a few. It serves them through the OpenAI and
Anthropic APIs and its own web app, for chat, coding agents and pictures.

The experts live in RAM and run where it is fastest: the GPUs keep the ones the model uses most and the CPU computes
the rest beside them. Each step decodes several tokens at once: a draft layer proposes them and one pass of the model
checks them all, so the text is the model's own. How it works: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

Lamina is a fork of [Strata](https://github.com/Niko1221/Strata) by Niko1221 and its contributors. Since 2026-10-29 it
follows its own course: NVIDIA GPUs only, extra GPUs as further expert tiers, and a draft layer trained on the engine's
own output. Its speeds against llama.cpp and Strata on one PC are in [docs/COMPARISON.md](docs/COMPARISON.md).

## What you need

- **An NVIDIA GPU** of compute capability 8.0 or newer (RTX 30, 40 or 50 series, or the workstation cards of those
  generations), with a driver for CUDA 13. The model runs on one GPU that nothing else uses: no display, no other
  program. Up to three more GPUs become further expert tiers, and these may drive the display. The engine is developed
  on an RTX 3090 (24 GB), with an RTX 5070 Ti (16 GB) and a second RTX 3090 as tiers. A smaller card holds fewer
  experts and runs slower. Qwen3.6-35B-A3B and Qwen3.8-27B take more GPUs differently: their layers are split over the
  fewest that hold the whole model. Qwen3.8-27B, a dense model, needs ~36 GB of VRAM at its full context (262,144
  tokens, K and V in 8 bits): two GPUs, such as a 24 GB and a 16 GB one.
- **RAM** for the experts, which are pinned while the model runs:

  | Model file | RAM taken | Download |
  | --- | --- | --- |
  | Qwen3.8-Flash-Next, [ISTA-DASLab's GSQ-RCO IQ3_S](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | ~53 GB | ~86 GB |
  | Qwen3.8-Flash-Next, [unsloth's UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | ~81 GB | ~114 GB |
  | Qwen3.6-35B-A3B, [unsloth's UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) | ~22 GB on one GPU, ~2 GB split over GPUs that hold it | ~24 GB |
  | Qwen3.8-27B, [unsloth's UD-Q6_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | ~2 GB | ~24 GB |

  The engine also runs unsloth's other Flash-Next quants, all but UD-IQ1_S and BF16: UD-IQ1_M, UD-Q2_0, UD-IQ2_XS,
  UD-Q2_K_XL, UD-IQ3_XXS, UD-Q3_K_XL, UD-IQ4_XS, UD-Q5_K_XL, UD-Q6_K_XL and Q8_0, whose experts take ~35 GB (UD-IQ1_M)
  to ~128 GB (Q8_0) of RAM, and [Agention's Gyro-L](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF).
  `lamina-server -m` runs them as it runs the table's ([Run](#run)). The engine compares the experts with the PC's
  RAM before it loads them: it stops when they do not fit, and warns when little would be left.

- **An SSD**: Flash-Next reads a 28.8 GB table of per-layer embeddings from the model file at every token.
- **A 64-bit x86 CPU with AVX2** (AVX-512 is used when it is there), or a 64-bit Arm CPU with Armv8.2's dot product
  and FP16 instructions, where the experts run on ggml's dot products (built and its CPU tests run, not yet run with a
  GPU). The CPU computes the experts no GPU holds, so its cores and the RAM's bandwidth set much of the speed.
- **Windows 10 / 11 or Linux** (a release's programs: glibc 2.34 and OpenSSL 3, as Ubuntu 22.04, Debian 12, RHEL 9
  and newer have them).
- **To build:** the CUDA toolkit 13.0 or newer, and Visual Studio 2022's C++ tools on Windows (their CMake and Ninja
  serve the build; on Arm also LLVM's clang-cl), or GCC 11, CMake 3.22 and Ninja or newer on Linux. The
  [releases](https://github.com/eddoursul/Lamina/releases) have the programs built for RTX 30, 40 and 50 GPUs.

## Install

**A release:** extract both archives of a [release](https://github.com/eddoursul/Lamina/releases) for your platform
into one folder: the programs (`lamina-server`, `lamina`, `lamina-vision`) and the CUDA runtime they load. Download a
model's GGUF files into a folder of their own: one of the table's above (every shard of a split one), with an image
projector of its repository (an `mmproj-*.gguf`) for pictures and, for Qwen3.8-Flash-Next, Lamina's draft layer
([below](#the-draft-layer)). [Run](#run) then starts it.

**The source:** from a clone, `build.bat` (Windows) or `./build.sh` (Linux) builds the programs for this PC's GPUs
into `build`, where they run as a release's do ([Run](#run)):

```
git clone https://github.com/eddoursul/Lamina
cd Lamina
build.bat                        (Windows)
./build.sh                       (Linux)
```

The build downloads llama.cpp at the commit the engine is built against and checks it; it installs no system software
and names what is missing. `build.bat portable` builds the programs for other PCs, as the releases are, and `cmake
--install build --prefix <folder>` puts them into a folder of their own: [docs/SETUP.md](docs/SETUP.md#building).

### The draft layer

Qwen3.8-Flash-Next needs one thing no model file holds: Lamina's draft layer, one file of 0.8 GB,
`Lamina-Qwen3.8-Flash-Next-MTP.gguf`, in
[eddoursul/Lamina-Qwen3.8-Flash-Next-MTP](https://huggingface.co/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP). Whoever
installs Lamina, a person or an agent, should download it into the model's folder:

```
download-draft-layer.bat MODEL_FOLDER      (Windows)
./download-draft-layer.sh MODEL_FOLDER     (Linux)
```

The server takes the file from beside the model, or from beside the server, when the config's `args` name no
`--mtp`; elsewhere name it there: `"--mtp", "<folder>/Lamina-Qwen3.8-Flash-Next-MTP.gguf"` (a folder of the layer's
earlier layout of four files, as earlier versions downloaded it, still works there too). The scripts check the
file's SHA-256 and go on with a download that broke off. Without the layer the server still starts and says so in its
window, with this command: the model then writes new text at roughly half to two thirds of its speed, and an edit
that quotes its prompt loses under a tenth. Qwen3.6-35B-A3B and Qwen3.8-27B draft with their own files' layer and need
nothing more.

## Run

```
lamina-server -m D:\Models\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf     (a model file)
lamina-server --config iq3_s.json                                                     (a config: --write-config)
```

`-m` takes a model file the engine runs, a split one by any of its shards. The server builds its config from the
preset for it (the file's header picks one of [`examples/*.json`](examples/README.md), built into the server), this
PC's GPUs and the files beside the model, and its window says what it took. `--check` says beforehand what the model
needs against what this PC has, and loads nothing. Every key of a config is an option too (`lamina-server --help`),
whose value takes the place of the file's and the preset's (`--port 8090 --temperature 0.6`); an engine option there
takes the place of the preset's (`--max-context 65536`); `--write-config FILE` writes the config for
`lamina-server --config FILE`: [docs/SETUP.md](docs/SETUP.md#a-model-file).

The model loads in a minute or so. Then, on the preset's port (the window's `ready` line names it: 8081 for
ISTA-DASLab's Flash-Next files, 8082 for the family's others, 8083 for Qwen3.6-35B-A3B, 8084 for Qwen3.8-27B):

- **http://127.0.0.1:8081** is the web app: chat with pictures, the monitor, settings.
- **http://127.0.0.1:8081/v1** is the API, for any OpenAI- or Anthropic-compatible client, coding agents such as
  Claude Code, Codex CLI, OpenCode and pi included. Chat completions, Messages, Responses and the models list are
  there, with tool calls, streaming and pictures.

How to connect clients, the config's keys, images, image generation, MCP tools and serving on a network:
[docs/USING.md](docs/USING.md).

## A fixed chat template

The server renders prompts with the chat template in the model's file. Lamina also carries
[froggeric's fixed template for Qwen 3.5, 3.6 and 3.8](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
(v22.5) in `data/chat-template-froggeric-v22.5.jinja`. To enable it, add this key to the model's config and start the
server again (or give it `--chat-template-file` on the command line):

```json
"chat_template_file": "data/chat-template-froggeric-v22.5.jinja"
```

A relative path starts at the config's `cwd`, else at the folder the server starts in. The server's log then says
`[lamina] chat template: ...data/chat-template-froggeric-v22.5.jinja`. Remove the key to go back to the model's own
template. A conversation's history renders as it does under the model's template but for its system turn and the warnings
below, and a cached conversation is read again once. It has been tested on Qwen3.8-Flash-Next, on both model files.

What it changes:

- **The default effort.** A request that names no reasoning effort runs at `medium` instead of `xhigh`, with no effort
  sentence in the prompt, and thinks less. `high` still means `xhigh`. A config's `"reasoning_effort": "high"` keeps
  `xhigh` as the default.
- **The tool instructions.** The system turn says more about how to call tools. Under a coding agent's prompt the model
  invents fewer instructions when a conversation opens with a greeting, though not on every prompt
  (`"short_opening_tokens"` in [docs/USING.md](docs/USING.md#behaviour) covers that case on its own).
- **Tool errors.** A tool result the template takes for an error (from its first 120 characters: `"error":`,
  `command not found`, a non-zero exit code, ...) gets a `⚠️ SYSTEM WARNING` line telling the model to retry differently.
  A tool that prints such text without failing gets the line too.
- **Tags in a message.** `<|think_off|>`, `<|think_on|>`, `<|think_low|>`, `<|think_medium|>` and `<|think_xhigh|>` in a
  message set the thinking for that turn and the later ones, and are removed from the text.

## Documentation

- [docs/SETUP.md](docs/SETUP.md): the build, a model file, the release binaries, the configs, choosing the GPUs,
  Qwen3.6 and Qwen3.8-27B.
- [docs/USING.md](docs/USING.md): the server, its API and web app, clients, pictures, image generation, MCP tools.
- [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md): the engine.
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md): when something goes wrong.
- [docs/COMPARISON.md](docs/COMPARISON.md): speeds, memory and agreement with llama.cpp, against llama.cpp and
  Strata.
- [examples/](examples/README.md): the configs of the PC Lamina is developed on.
- [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md): reporting problems, pull requests, what the
  server exposes.

## License and credits

Lamina is under the MIT license ([LICENSE](LICENSE)), as Strata is. Some parts carry their own: `third_party/ggml`
(MIT, llama.cpp's), the web app's font, [Outfit](https://github.com/Outfitio/Outfit-Fonts) (SIL Open Font License 1.1,
`src/server/web/fonts/OFL.txt`), froggeric's chat template (Apache-2.0, unmodified,
`data/chat-template-froggeric-v22.5.LICENSE`), and what lamina-server takes from llama.cpp's checkout (its Jinja engine,
cpp-httplib, nlohmann/json, stb_image: MIT). It builds on:

- [Strata](https://github.com/Niko1221/Strata) by Niko1221 and its contributors: the engine Lamina grew from, with ideas
  Strata took from [Splash](https://github.com/incoai/splash), [ninfer](https://github.com/Neroued/ninfer) and
  [HyperQwen](https://github.com/syv-ai/HyperQwen).
- [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp) (MIT): the CPU and CUDA kernels it shares, the prompt
  path's matrix products, the image encoder, the GGUF tools. The build downloads it at a pinned commit.
- The models: [Qwen](https://huggingface.co/Qwen)'s, under their licenses, in the quantizations of
  [ISTA-DASLab](https://huggingface.co/ISTA-DASLab) and [unsloth](https://huggingface.co/unsloth).
- The draft layer, [eddoursul/Lamina-Qwen3.8-Flash-Next-MTP](https://huggingface.co/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP):
  Qwen's MTP layer, its dense part trained on this engine's output, under the Qwen license.
