// include/lamina/vision/encoder.hpp - the image encoder: a picture's rows, the embeddings the text model reads in place
// of its image pad tokens, from the model's projector file (mmproj: a vision tower and its adapter).
//
// Two projector types, the ones the engine's models have, on one tower shape (a ViT of pre-norm blocks, GELU FFNs,
// 16x16 patches):
//   qwen3vl_merger  Qwen3.8-Flash-Next, Qwen3.6-35B-A3B, Qwen3.8-27B: two patch kernels summed (a still picture is
//                   both frames of a pair), the learned position table resized bilinearly, 2-D rotary positions in
//                   every block, the patches in 2x2 groups from the start, each group's four concatenated and passed
//                   through two linear layers.  A picture's rows form a grid: row i is the cell (i % nx, i / nx).
//   hy_v4_flash     Hy4-flash: the position table resampled on the host (HunyuanVL's sampling), no rotary positions,
//                   an RMS norm, a 2x2 convolution of stride 2, GELU (erf), a 1x1 convolution, and a learned newline
//                   row after each row of cells.  The text model reads the rows at text positions: one row of n.
// A picture is first sized as the projector's reference processor sizes it (image.hpp): for Qwen's that is
// transformers' way, a direct resize, where mtmd at the pinned commit letterboxes (LAMINA_VISION_LETTERBOX=1).
//
// The arithmetic is llama.cpp's mtmd's at the pinned commit, operation for operation on ggml (tools/mtmd, MIT; Hy4's
// projector as its pull request 32517 has it), so on the CPU the rows are the ones that library gives: Qwen's, with
// its letterbox, the same bytes.  One departure, on a GPU: the attention's heads of 72 values are zero-padded to 80, a size ggml's
// tensor-core flash attention kernel takes, which is faster and nearer the FP32 attention than the kernel it has for
// 72 (ggml_encoder.cpp, GgmlEncoder::open; LAMINA_VISION_HEAD_PAD=0 gives mtmd's bytes there too).
#pragma once

#include "lamina/vision/image.hpp"

#include <memory>
#include <string>
#include <vector>

namespace lamina::vision {

enum class FlashAttention {
    automatic,   ///< on where the backend has the operation
    on,
    off,         ///< the attention as two products in FP32
};

struct EncoderParams {
    std::string mmproj;
    bool gpu = false;      ///< on the first GPU ggml sees (CUDA_VISIBLE_DEVICES names it), the CPU for what it lacks
    int threads = 0;       ///< the CPU's threads; 0: 4
    int max_tokens = 0;    ///< a picture's tokens at most; 0: the projector's own limit
    int min_tokens = 0;    ///< a picture's tokens at least; 0: the projector's own
    FlashAttention flash_attention = FlashAttention::automatic;
};

/// A picture's rows: n = nx * ny rows of n_embd floats.
struct Rows {
    int n = 0, nx = 0, ny = 0, n_embd = 0;
    std::vector<float> data;
};

class Encoder {
public:
    virtual ~Encoder() = default;
    /// the rows' width: the text model's
    virtual int n_embd() const = 0;
    /// The picture sized, normalized and encoded.  False with `err`.
    virtual bool encode(const Rgb& picture, Rows& out, std::string& err) = 0;
};

/// The encoder on ggml's backends.  nullptr with `err`: a file that is not a projector of the two types, weights that
/// do not fit the device, --gpu without a GPU.
std::unique_ptr<Encoder> open_encoder(const EncoderParams& params, std::string& err);

/// A GPU that ggml's backends see (CUDA_VISIBLE_DEVICES narrows them; none in a build without a GPU backend).
bool gpu_visible();
/// ggml's CPU code is built with AVX-512: its fast flash attention kernel then misses this tower's heads of 72 (it
/// takes heads that are a multiple of the vector's 16 floats), so on the CPU the attention as two products is the
/// faster one there (a 1024x1024 picture on a Ryzen 7 7700X, 8 threads: AVX2 8-10 s with flash attention, 14-15 s
/// without; AVX-512 44 s with it and 26% off the FP32 attention's output, 13 s without).
bool cpu_has_avx512();

}  // namespace lamina::vision
