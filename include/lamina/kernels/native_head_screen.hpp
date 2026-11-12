// include/lamina/kernels/native_head_screen.hpp - the head screen's measurement kernels (phase 3, M1).
//
// The verify window reads every head row to sample or argmax, but the sampler's pick depends on at most its
// exact top-k shortlist (`sampler.cu`'s `sampled_k`, 64 for every config). A Cauchy-Schwarz bound on the
// dequantized weight row against the q8_1-quantized input (`|w.x| <= ||w|| * ||x||`, the very vectors the
// mmvq kernels multiply, one safety epsilon for accumulation order) can certify "this row is not in the
// top-k" and skip its read. M1 measures the pruning before any of it changes behavior: norms, the real
// k-th largest logit per column, and the histogram of rows the bound would let through.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace lamina::kernels {

/// Whether the head's rows can be normed (the type `dequant_f32` decodes).
bool head_screen_supported(int ggml_type) noexcept;

/// `norms[r]` = the f32 L2 norm of dequantized row r, for `rows` rows of `cols` starting at row 0, streamed
/// through `scratch` (`chunk_rows * cols` floats). One boot pass; the decode is the reference `dequant_f32`.
void head_screen_row_norms(int ggml_type, const void* weights, int64_t rows, int64_t cols, int64_t chunk_rows,
                           float* norms, float* scratch, cudaStream_t stream);

/// `norms[t]` = the L2 norm of token t's q8_1-quantized input (the vector the mmvq kernels dot against).
/// `x_q8_1` is the standard (`native_quantize_q8_1` / `_il`'s first output) layout: `n_in / 32` blocks per token.
void head_screen_input_norms(const void* x_q8_1, float* norms, int n_in, int ncols, cudaStream_t stream);

/// `tau[c]` = the k-th largest of `n_rows` logits in column c (one column per `c` of the T x n_rows `logits`).
/// NaNs rank above every finite value (they are never prunable). Exact, bit-level radix select.
void head_screen_kth(const float* logits, int ncols, int64_t n_rows, int k, float* tau, cudaStream_t stream);

/// The screen simulation's counter: rows whose bound clears `tau`, per column; the launch's LAST block writes
/// one histogram sample per column and resets `cur[c]` / `done`. `hist[c * kHeadScreenBuckets + b]`: bucket b
/// counts totals in [2^(b+5), 2^(b+6)), bucket 0 counts below 32; a NaN tau or norm keeps every row.
constexpr int kHeadScreenBuckets = 14;
void head_screen_count(const float* norms, int64_t n_rows, const float* xnorms, int ncols, const float* tau,
                       unsigned long long* cur, unsigned* done, unsigned long long* hist, cudaStream_t stream);

}  // namespace lamina::kernels
