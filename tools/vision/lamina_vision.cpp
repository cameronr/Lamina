// tools/vision/lamina_vision.cpp - the image half of Lamina's multimodal path, as the program the server starts.
//
// Turns an image into the embeddings the text model reads in place of its image pad tokens, from the model's mmproj
// file (vision encoder + projector): the engine's own encoder on ggml (include/lamina/vision/encoder.hpp: Qwen's
// projectors and Hy4-flash's).  The engine does the rest: it places the rows at the image's pad tokens and gives them
// their positions, 2-D M-RoPE ones (see --serve GENI in generate.cpp) or, for a model that reads a picture at text
// positions (Hy4-flash's file: a grid of one row), consecutive ones.
//
//   lamina-vision --mmproj <mmproj.gguf> [--gpu] [--threads N] [--max-tokens N] [--min-tokens N]
//                 [--flash-attn on|off|auto]
//   lamina-vision -v | --version
// Flash attention defaults to auto, and to off on the CPU when ggml is built with AVX-512 (encoder.hpp).
// --model <file> is taken and not read: the encoder needs no vocabulary (the servers before it name the text model).
//
// Resident: prints "READY <n_embd>", then per stdin line
//   ENC <image path> <output path>   ->  "OK <n_tokens> <nx> <ny> <ms>"  or  "ERR <message>"
//   QUIT
// The output file is  int32 {0x31455653 'SVE1', n_tokens, nx, ny, n_embd}  then float32 [n_tokens][n_embd],
// row i at grid position (x = i % nx, y = i / nx).
#include "lamina/vision/encoder.hpp"

#include <algorithm>
#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

// "ENC <image> <out>": paths may contain spaces, so the image path ends at the last " " before the output path;
// the server always writes paths without spaces, but a user calling the tool by hand might not
bool parse_enc(const std::string& line, std::string& img, std::string& out) {
    if (line.rfind("ENC ", 0) != 0) return false;
    const std::string rest = line.substr(4);
    const size_t sp = rest.rfind(' ');
    if (sp == std::string::npos || sp == 0) return false;
    img = rest.substr(0, sp);
    out = rest.substr(sp + 1);
    return !img.empty() && !out.empty();
}

}  // namespace

int main(int argc, char** argv) {
    using lamina::vision::FlashAttention;
    lamina::vision::EncoderParams params;
    bool fa_given = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--mmproj") params.mmproj = next();
        else if (a == "--model") (void) next();   // not read
        else if (a == "--gpu") params.gpu = true;
        else if (a == "-v" || a == "--version") {
            std::printf("lamina-vision " LAMINA_VERSION "\n");
            return 0;
        }
        else if (a == "--threads") params.threads = std::atoi(next().c_str());
        else if (a == "--max-tokens") params.max_tokens = std::atoi(next().c_str());
        else if (a == "--min-tokens") params.min_tokens = std::atoi(next().c_str());   // #767
        else if (a == "--flash-attn") {   // FA keeps K and V in FP16; off = the attention in FP32
            const std::string v = next();
            params.flash_attention = v == "on" ? FlashAttention::on : v == "off" ? FlashAttention::off
                                                                                 : FlashAttention::automatic;
            fa_given = true;
        }
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (params.mmproj.empty()) {
        std::fprintf(stderr, "usage: lamina-vision --mmproj <mmproj.gguf> [--gpu] [--threads N] [--max-tokens N] "
                             "[--min-tokens N] [--flash-attn on|off|auto]\n");
        return 2;
    }
#if !LAMINA_VISION_CUDA
    // Built without CUDA there is no GPU to run on: said, not done on the CPU in its place (16.6 s at 1,024 tokens on
    // an 8-core CPU, against 0.3-0.4 s on a GPU).
    if (params.gpu) {
        std::printf("ERR this lamina-vision was built without CUDA and runs on the CPU alone: set the config's vision "
                    "\"gpu\" to false, or put a CUDA build of lamina-vision beside lamina-server\n");
        std::fflush(stdout);
        return 2;
    }
#endif
    // On the CPU the GPU stays unseen: a CUDA build otherwise opens a context there (measured: 0.4-0.7 GB of VRAM,
    // 150-260 expert slots less for the engine beside it).  Before anything reaches the CUDA runtime.
    if (!params.gpu) {
#ifdef _WIN32
        _putenv_s("CUDA_VISIBLE_DEVICES", "-1");
#else
        setenv("CUDA_VISIBLE_DEVICES", "-1", 1);
#endif
    }
    // --gpu where no GPU is visible (the config's "cuda_device" names a card that is not there: the server hands it to
    // CUDA as CUDA_VISIBLE_DEVICES; or the driver gives none): said, not done on the CPU in its place.
    if (params.gpu && !lamina::vision::gpu_visible()) {
        const char* vis = std::getenv("CUDA_VISIBLE_DEVICES");
        const std::string shown = vis != nullptr && *vis ? std::string(" (CUDA_VISIBLE_DEVICES=") + vis + ")" : "";
        std::printf("ERR lamina-vision sees no GPU%s: name a card that is there in the config's vision \"cuda_device\" "
                    "(nvidia-smi -L lists them), or set \"gpu\" to false\n", shown.c_str());
        std::fflush(stdout);
        return 2;
    }

    const auto load_t0 = std::chrono::steady_clock::now();
#if !defined(_WIN32)
    // ask the OS for the projector file up front (128 KiB steps) so the loader's sequential reads find it cached;
    // LAMINA_READ_AHEAD=0 turns it off, as in the engine
    if (const char* ra = std::getenv("LAMINA_READ_AHEAD"); ra == nullptr || std::atoi(ra) != 0) {
        const int fd = open(params.mmproj.c_str(), O_RDONLY);
        if (fd >= 0) {
            const off_t n = lseek(fd, 0, SEEK_END);
            for (off_t at = 0; at < n; at += (off_t) (128 << 10))
                (void) posix_fadvise(fd, at, std::min<off_t>((off_t) (128 << 10), n - at), POSIX_FADV_WILLNEED);
            close(fd);
        }
    }
#endif
    // on the CPU without --threads: one per core, at most 16 (a 256-thread server does not need 128 encoder threads)
    if (params.threads <= 0 && !params.gpu)
        params.threads = (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency() / 2));
    // on the CPU "auto" turns flash attention on, which a build of ggml with AVX-512 does slowly (encoder.hpp)
    if (!params.gpu && !fa_given && lamina::vision::cpu_has_avx512()) params.flash_attention = FlashAttention::off;
    std::string err;
    std::unique_ptr<lamina::vision::Encoder> enc = lamina::vision::open_encoder(params, err);
    if (!enc) {
        std::printf("ERR cannot load the vision encoder %s (%s)\n", params.mmproj.c_str(), err.c_str());
        std::fflush(stdout);
        return 1;
    }
    std::fprintf(stderr, "lamina-vision: model files loaded in %.1f s\n",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - load_t0).count());
    // Warm up at the LARGEST picture before READY: the encoder's GPU work buffers are allocated now, not at the first
    // real picture.  The server starts this process before the engine, so the engine sizes its expert cache from
    // what is really left; allocating ~1 GB later, on a GPU the engine has filled, made Windows page GPU memory and
    // the engine crawl to a standstill.  (A square image well above any cap; the encoder scales it to its limit.)
    // On the CPU there is no VRAM to reserve, and the warm-up would only delay the engine's start by one encode
    // (~6 s at 1,024 tokens).
    if (!params.gpu) std::fprintf(stderr, "lamina-vision: on the CPU, %d threads, no warm-up\n", params.threads);
    else {
        const auto warm_t0 = std::chrono::steady_clock::now();
        lamina::vision::Rgb grey;
        grey.w = grey.h = 2048;
        grey.px.assign((size_t) grey.w * grey.h * 3, 128);
        lamina::vision::Rows rows;
        std::string why;
        if (!enc->encode(grey, rows, why)) rows.n = 0;
        std::fprintf(stderr, "lamina-vision: warmed up at %d image tokens in %.1f s%s%s\n", rows.n,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - warm_t0).count(),
                     why.empty() ? "" : ": ", why.c_str());
    }
    std::printf("READY %d\n", enc->n_embd());
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line == "QUIT") break;
        std::string img, out;
        if (!parse_enc(line, img, out)) { std::printf("ERR expected: ENC <image> <output>\n"); std::fflush(stdout); continue; }
        const auto t0 = std::chrono::steady_clock::now();
        lamina::vision::Rgb picture;
        lamina::vision::Rows rows;
        if (!lamina::vision::decode_file(img, picture, err)) {
            std::printf("ERR cannot read the image %s\n", img.c_str());
            std::fflush(stdout);
            continue;
        }
        err.clear();
        if (!enc->encode(picture, rows, err) && err.empty()) err = "the vision encoder failed";
        if (err.empty()) {
            FILE* f = std::fopen(out.c_str(), "wb");
            if (!f) err = "cannot write " + out;
            if (f) {
                const int32_t hdr[5] = {0x31455653, rows.n, rows.nx, rows.ny, rows.n_embd};
                const bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1 &&
                                std::fwrite(rows.data.data(), sizeof(float) * (size_t) rows.n_embd, (size_t) rows.n, f) ==
                                    (size_t) rows.n;
                std::fclose(f);
                if (!ok) err = "short write to " + out;
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (ok) std::printf("OK %d %d %d %.0f\n", rows.n, rows.nx, rows.ny, ms);
            }
        }
        if (!err.empty()) std::printf("ERR %s\n", err.c_str());
        std::fflush(stdout);
    }
    return 0;
}
