// src/kernels/cuda/native_head_screen.cu - see include/lamina/kernels/native_head_screen.hpp.
//
// M1's measurement set. Nothing here decides what runs; the counters feed a histogram that says whether a
// conservative bound could have pruned the head read before it is ever built on.

#include "lamina/kernels/native_head_screen.hpp"

#include "lamina/kernels/dequant_bf16.hpp"

#include <cuda_fp16.h>

#include <cmath>

namespace lamina::kernels {
namespace {

// the q8_1 block, the pinned layout (same bytes as `native_mmvq.cu`'s Q81Block: one half scalar d, 32 int8s)
struct ScreenQ81 {
    half2 ds;
    int8_t qs[32];
};

/// floats ordered as uint32: ascending `key` is ascending float, with NaNs above every number (never prunable).
__device__ __forceinline__ unsigned key_of(float v) {
    const unsigned b = __float_as_uint(v);
    return b ^ ((b & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u);
}

__device__ __forceinline__ float float_of_key(unsigned key) {
    return __uint_as_float(key ^ ((key & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu));
}

__device__ __forceinline__ void block_sum256(float* sh, float acc) {
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
}

__global__ void input_norms_kernel(const ScreenQ81* __restrict__ x, float* __restrict__ norms, int nb) {
    const ScreenQ81* p = x + (size_t) blockIdx.x * nb;
    float acc = 0.0f;
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        const float d = __low2float(p[b].ds);
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const float q = (float) p[b].qs[i];
            s += q * q;
        }
        acc += d * d * s;
    }
    __shared__ float sh[256];
    block_sum256(sh, acc);
    if (threadIdx.x == 0) norms[blockIdx.x] = sqrtf(sh[0]);
}

__global__ void row_norms_kernel(const float* __restrict__ rows, float* __restrict__ norms, int cols) {
    const float* p = rows + (size_t) blockIdx.x * cols;
    float acc = 0.0f;
    bool nan = false;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        if (std::isnan(p[i])) {  // a NaN weight makes every bound NaN: the row can never be certified away
            nan = true;
            break;
        }
        acc += p[i] * p[i];
    }
    __shared__ float sh[256];
    block_sum256(sh, acc);
    if (threadIdx.x == 0) norms[blockIdx.x] = nan ? INFINITY : sqrtf(sh[0]);
}

// 4-pass radix select, 8 bits a pass, one block per column. The k-th largest is the element with exactly
// k - 1 keys above the narrowed prefix; exact down to the bits. `hist` is 256 unsigned of scratch.
__global__ void kth_kernel(const float* __restrict__ logits, int64_t n, int k, float* __restrict__ tau) {
    const float* row = logits + (size_t) blockIdx.x * n;
    __shared__ unsigned sh[256];
    __shared__ unsigned sh_above;
    __shared__ int sh_target;
    if (threadIdx.x == 0) sh_above = 0;
    __syncthreads();
    unsigned prefix = 0;
    int kept = 0;
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        for (int i = (int) threadIdx.x; i < 256; i += blockDim.x) sh[i] = 0;
        __syncthreads();
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            const unsigned key = key_of(row[i]);
            if (kept > 0 && (key >> (32 - kept)) != (prefix >> (32 - kept))) continue;
            atomicAdd(&sh[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned cum = sh_above;
            int target = 0;
            for (int b = 255; b >= 0; --b) {
                if (cum + sh[b] >= (unsigned) k) {
                    target = b;
                    break;
                }
                cum += sh[b];
            }
            sh_above = cum;
            sh_target = target;
        }
        __syncthreads();
        prefix |= (unsigned) sh_target << shift;
        kept += 8;
        __syncthreads();
    }
    if (threadIdx.x == 0) tau[blockIdx.x] = float_of_key(prefix);
}

/// Each block counts, per column, the rows its slice leaves uncertifiable, then the block that finishes last
/// turns the totals into one histogram sample per column and resets them (all blocks in a launch at once).
__global__ void count_kernel(const float* __restrict__ norms, int64_t n, const float* __restrict__ xnorms,
                             int ncols, const float* __restrict__ tau, unsigned long long* __restrict__ cur,
                             unsigned* __restrict__ done, unsigned long long* __restrict__ hist) {
    __shared__ unsigned long long sh[8];
    __shared__ unsigned ticket;
    for (int c = (int) threadIdx.x; c < ncols; c += blockDim.x) sh[c] = 0;
    __syncthreads();
    for (int64_t r = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; r < n; r += (int64_t) gridDim.x * blockDim.x) {
        const float nw = norms[r];
        for (int c = 0; c < ncols; ++c) {
            const float t = tau[c];
            const bool keep = std::isnan(t) || std::isnan(nw) || nw * xnorms[c] * 1.0001f + 1e-4f >= t;
            if (keep) atomicAdd(&sh[c], 1ULL);
        }
    }
    __syncthreads();
    if (threadIdx.x < ncols) atomicAdd(&cur[threadIdx.x], sh[threadIdx.x]);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) ticket = atomicAdd(done, 1u);
    __syncthreads();
    if (ticket != gridDim.x - 1 || threadIdx.x >= ncols) return;
    const unsigned long long total = atomicAdd(&cur[threadIdx.x], 0ULL);
    cur[threadIdx.x] = 0;
    if (threadIdx.x == 0) *done = 0;
    const int bucket = total < 32 ? 0 : 63 - __builtin_clzll(total) - 4;
    atomicAdd(&hist[(size_t) threadIdx.x * kHeadScreenBuckets +
                    (bucket < kHeadScreenBuckets - 1 ? bucket : kHeadScreenBuckets - 1)],
              1ULL);
}

}  // namespace

bool head_screen_supported(int ggml_type) noexcept { return dequant_bf16_supported(ggml_type); }

void head_screen_row_norms(int ggml_type, const void* weights, int64_t rows, int64_t cols, int64_t chunk_rows,
                           float* norms, float* scratch, cudaStream_t stream) {
    for (int64_t row0 = 0; row0 < rows; row0 += chunk_rows) {
        const int64_t n = chunk_rows < rows - row0 ? chunk_rows : rows - row0;
        dequant_f32(ggml_type, weights, row0, n, cols, scratch, stream);
        row_norms_kernel<<<(unsigned) n, 256, 0, stream>>>(scratch, norms + row0, (int) cols);
    }
}

void head_screen_input_norms(const void* x_q8_1, float* norms, int n_in, int ncols, cudaStream_t stream) {
    input_norms_kernel<<<ncols, 256, 0, stream>>>((const ScreenQ81*) x_q8_1, norms, n_in / 32);
}

void head_screen_kth(const float* logits, int ncols, int64_t n_rows, int k, float* tau, cudaStream_t stream) {
    kth_kernel<<<ncols, 1024, 0, stream>>>(logits, n_rows, k, tau);
}

void head_screen_count(const float* norms, int64_t n_rows, const float* xnorms, int ncols, const float* tau,
                       unsigned long long* cur, unsigned* done, unsigned long long* hist, cudaStream_t stream) {
    int blocks = (int) ((n_rows + 4095) / 4096);
    if (blocks > 1024) blocks = 1024;
    count_kernel<<<blocks, 256, 0, stream>>>(norms, n_rows, xnorms, ncols, tau, cur, done, hist);
}

}  // namespace lamina::kernels
