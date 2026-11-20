// include/lamina/vision/encoder.hpp - the image encoder: a picture's rows, the embeddings the text model reads in place
// of its image pad tokens, from the model's projector file (mmproj: a vision tower and its adapter).
//
// Two projector types, the ones the engine's models have, on one tower shape (a ViT of pre-norm blocks, GELU FFNs,
// 16x16 patches, heads of 72 values):
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
// Two implementations of one arithmetic, llama.cpp's mtmd's at the pinned commit (tools/mtmd, MIT; Hy4's projector as
// its pull request 32517 has it):
//   the CPU's    a ggml graph on ggml's CPU backend, mtmd's operation for operation (ggml_encoder.cpp): Qwen's rows,
//                with its letterbox, are that library's bytes on a CPU without AVX-512 (with it the attention's heads
//                are padded, GgmlEncoder::open)
//   a GPU's      the engine's own CUDA kernels (cuda_encoder.cu): cuBLAS products with FP16 inputs and FP32 sums, MMQ
//                for a Q8_0 projector's matrices as llama.cpp multiplies them on a GPU, and a tensor-core attention
//                kernel.  Its rows stand as near the CPU's FP32 attention as any GPU arithmetic does (the tower
//                amplifies a rounding to percents of some rows): 1.5e-2 of a picture's rows on average with an
//                unquantized projector, where ggml's CUDA backend stood 2.4e-2 off.
#pragma once

#include "lamina/vision/image.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lamina::vision {

enum class FlashAttention {
    automatic,   ///< on
    on,
    off,         ///< the attention as two products in FP32
};

struct EncoderParams {
    std::string mmproj;
    bool gpu = false;      ///< on a GPU: the CUDA device `device` of this process's
    int device = 0;
    int threads = 0;       ///< the CPU's threads; 0: one a core, at most 16
    int max_tokens = 0;    ///< a picture's tokens at most; 0: the projector's own limit
    int min_tokens = 0;    ///< a picture's tokens at least; 0: the projector's own
    FlashAttention flash_attention = FlashAttention::automatic;   ///< the CPU's attention
    /// A GPU, a Q8_0 projector: each matrix's values as FP16 and the products as the other files' (FP16 in, FP32
    /// sums), in place of MMQ over its blocks (the activations rounded to q8_1, llama.cpp's arithmetic for the file).
    bool q8_fp16 = false;
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
    /// a GPU's encoder: its weights and buffers there, in bytes (the CPU's: 0)
    virtual uint64_t gpu_bytes() const { return 0; }
    /// The picture sized, normalized and encoded.  False with `err`.  One call at a time; a GPU's encoder makes its
    /// device the calling thread's current one.
    virtual bool encode(const Rgb& picture, Rows& out, std::string& err) = 0;
};

/// The rows as the record the engine reads a picture from (program/serve_io.hpp): int32 {0x31455653 'SVE1', n, nx,
/// ny, n_embd}, then the n x n_embd floats.  False when the file cannot be written whole.
bool write_rows(const std::string& path, const Rows& rows);

/// The encoder: a GPU's (the weights and the buffers for the largest picture taken on params.device now) or the
/// CPU's.  nullptr with `err`: a file that is not a projector of the two types, weights that do not fit, no such
/// GPU.
std::unique_ptr<Encoder> open_encoder(const EncoderParams& params, std::string& err);

}  // namespace lamina::vision
