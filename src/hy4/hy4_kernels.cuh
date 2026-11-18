// src/hy4/hy4_kernels.cuh - Hy4-flash's kernels beside the engine's shared ones (native_mmvq, the native experts, the
// BF16 GEMV): the iHC streams, the norms, RoPE in consecutive pairs, the MLA's per-head projections, the latent
// attention with sinks, the indexer, Engram's block and the sigmoid router.  Included by hy4_engine.cu alone.
//
// Layouts (T rows, one a token): the streams X [T][hc][N]; a row's MLA query [H][576] (512 absorbed + 64 roped); a
// cached cell one latent row of 576 halves ([RMSNorm(latent) 512 | roped k_pe 64]); q8_1 blocks are llama.cpp's (half2
// d / sum, 32 int8).  Arithmetic follows llama.cpp's PR #32517 graph (src/models/hy-v4-flash.cpp): f32 activations,
// q8_1-quantized inputs to the Q8_0 matrices, the rope of rope.cu's rope_norm.
#pragma once

#include "lamina/kernels/warp.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace hy4k {

using lamina::kernels::warp_max;
using lamina::kernels::warp_sum;

constexpr int kThreads = 256;
constexpr int kAttnChunk = 32;   // cells a block of the latent attention takes (a warp's lanes)
constexpr int kKvRow = 576;      // halves of a cached latent row (kv_lora 512 + n_rot 64): the kernels' fixed width
constexpr int kKvRowPad = 578;   // the row in shared memory: 289 words, so lane c's word d / 2 is bank-conflict free
constexpr int kLatent = 512;

__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }
__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }

/// The sum of v over the block (blockDim.x a multiple of 32, at most 1024): every thread gets it.
__device__ __forceinline__ float block_sum(float v, float* sh) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[w] = v;
    __syncthreads();
    float s = 0.f;
    for (int i = 0; i < nw; ++i) s += sh[i];
    return s;
}

/// y[r] = x[r] / sqrt(mean(x[r]^2) + eps) (* w when non-null), a block per row of n; rows `ld` floats apart.
__global__ void k_rms(const float* x, int ldx, const float* w, float* y, int ldy, int n, float eps) {
    __shared__ float sh[32];
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float* yr = y + (size_t) blockIdx.x * ldy;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = w ? xr[i] * r * w[i] : xr[i] * r;
}

/// The embedding row t into the hc streams: X[t][i] = e[t] for every stream i.
__global__ void k_streams_init(const float* e, float* X, int hc, int n) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = e[(size_t) t * n + i];
        for (int s = 0; s < hc; ++s) X[((size_t) t * hc + s) * n + i] = v;
    }
}

/// iHC's pre-mix of row t (a block each, 256 threads): mixes = fn . rms(X[t] flat) (fn F32 [m][hc*n], m = 2 hc for a
/// layer, hc for the head); pre = sigmoid(mix[0:hc] * scale[0] + base[0:hc]) + eps; for a layer also post =
/// sigmoid(mix[hc:2hc] * scale[1] + base[hc:2hc]) * mag + eps into post[t][hc]; xh[t] = sum_i pre_i X[t][i] (in stream
/// order); with `wn`, xn[t] = rms(xh[t]) * wn (the sublayer's input norm, eps `eps_n`).
__global__ void __launch_bounds__(256) k_hc_pre(const float* X, const float* fn, const float* scale, const float* base,
                                                int hc, int n, int m, float eps_rms, float hc_eps, float mag,
                                                float* post, float* xh, const float* wn, float* xn, float eps_n) {
    __shared__ float sh[32];
    __shared__ float mix[16];
    __shared__ float wpre[8];
    const int t = blockIdx.x;
    const int flat = hc * n;
    const float* xr = X + (size_t) t * flat;
    float s = 0.f;
    for (int i = threadIdx.x; i < flat; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) flat + eps_rms);
    // one warp per mix (8 warps; m <= 8)
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int j = warp; j < m; j += blockDim.x >> 5) {
        const float* wr = fn + (size_t) j * flat;
        float acc = 0.f;
        for (int i = lane; i < flat; i += 32) acc += wr[i] * (xr[i] * r);
        acc = warp_sum(acc);
        if (lane == 0) mix[j] = acc;
    }
    __syncthreads();
    if (threadIdx.x < hc) {
        const int i = threadIdx.x;
        wpre[i] = sigm(mix[i] * scale[0] + base[i]) + hc_eps;
        if (m > hc) post[(size_t) t * hc + i] = sigm(mix[hc + i] * scale[1] + base[hc + i]) * mag + hc_eps;
    }
    __syncthreads();
    float s2 = 0.f;
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        float v = xr[c] * wpre[0];
        for (int i = 1; i < hc; ++i) v = v + xr[(size_t) i * n + c] * wpre[i];
        xh[(size_t) t * n + c] = v;
        s2 += v * v;
    }
    if (wn == nullptr) return;
    s2 = block_sum(s2, sh);
    const float r2 = rsqrtf(s2 / (float) n + eps_n);
    for (int c = threadIdx.x; c < n; c += blockDim.x) xn[(size_t) t * n + c] = xh[(size_t) t * n + c] * r2 * wn[c];
}

/// iHC's post: X[t][i] += y[t] * post[t][i] for every stream (y already post-normed).
__global__ void k_hc_post(float* X, const float* y, const float* post, int hc, int n) {
    const int t = blockIdx.x;
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        const float v = y[(size_t) t * n + c];
        for (int i = 0; i < hc; ++i) {
            float& x = X[((size_t) t * hc + i) * n + c];
            x = x + v * post[(size_t) t * hc + i];
        }
    }
}

/// y = rms(y) * w in place on T rows of n, then (when `add` is non-null) y += add (the routed + shared sum is made first
/// by the caller); a block per row.
__global__ void k_rms_inplace(float* y, const float* w, int n, float eps) {
    __shared__ float sh[32];
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += yr[i] * yr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = yr[i] * r * w[i];
}

/// RoPE in consecutive pairs (ggml's ROPE_TYPE_NORM, rope.cu's rope_norm without YaRN): `n_rot` values from `off` in
/// each of `heads` heads (`ld` floats apart) of row t, at position pos[t] (or pos0 + t when pos is null).
__global__ void k_rope_norm(float* x, int heads, int ld, int off, int n_rot, const int32_t* pos, int pos0,
                            float theta_scale) {
    const int t = blockIdx.y;
    const int h = blockIdx.x;
    const int i = threadIdx.x;   // a pair
    if (h >= heads || 2 * i >= n_rot) return;
    const float p = (float) (pos ? pos[t] : pos0 + t);
    const float theta = p * powf(theta_scale, (float) i);
    const float c = cosf(theta), s = sinf(theta);
    float* v = x + (size_t) t * heads * ld + (size_t) h * ld + off + 2 * i;
    const float x0 = v[0], x1 = v[1];
    v[0] = x0 * c - x1 * s;
    v[1] = x0 * s + x1 * c;
}

/// q [T][H][k_head] -> its nope parts as contiguous rows [T*H][nope] (the k_b input) and its rope parts into the query
/// rows qabs [T][H][576] at 512.. (roped there after).
__global__ void k_q_split(const float* q, float* qnope, float* qabs, int H, int k_head, int nope) {
    const int t = blockIdx.y, h = blockIdx.x;
    const float* src = q + ((size_t) t * H + h) * k_head;
    float* dn = qnope + ((size_t) t * H + h) * nope;
    float* dq = qabs + ((size_t) t * H + h) * kKvRow + kLatent;
    for (int i = threadIdx.x; i < k_head; i += blockDim.x) {
        if (i < nope) dn[i] = src[i];
        else dq[i - nope] = src[i];
    }
}

/// Per-head Q8_0 x q8_1 products (llama.cpp's batched MMVQ arithmetic, vec_dot_q8_0_q8_1: d_w d_x sum(int8 * int8) a
/// block): y[t][h][j] = W_h[j] . x[t][h] for j < n_out; W_h's rows at w + (h * n_out + j) * n_in / 32 blocks of 34
/// bytes, x[t][h] at xq + (t * H + h) * n_in / 32 blocks of 36; y rows at y + ((t * H + h) * ldy + j).  A thread per
/// output row.
__global__ void k_head_q8_0(const uint8_t* w, const uint8_t* xq, float* y, int H, int n_in, int n_out, int ldy,
                            int ld_head_y) {
    const int t = blockIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;   // h * n_out + j
    if (r >= H * n_out) return;
    const int h = r / n_out, j = r % n_out;
    const int nb = n_in / 32;
    const uint8_t* wr = w + (size_t) r * nb * 34;
    const uint8_t* xr = xq + ((size_t) t * H + h) * nb * 36;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* wb = wr + b * 34;
        const uint8_t* xb = xr + b * 36;
        const float dw = __half2float(*reinterpret_cast<const __half*>(wb));
        const float dx = __half2float(*reinterpret_cast<const __half*>(xb));
        int sumi = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            int a, c;
            memcpy(&a, wb + 2 + 4 * k, 4);
            memcpy(&c, xb + 4 + 4 * k, 4);
            sumi = __dp4a(a, c, sumi);
        }
        acc += dw * dx * (float) sumi;
    }
    y[(size_t) t * ldy + (size_t) h * ld_head_y + j] = acc;
}

/// The cached latent row of token t: [rms(kva[0:512]) * norm | roped kva[512:576]] as halves into the cache's row for
/// position pos[t]: pos % ring in a ring, else pos - base (a buffer whose row 0 is position `base`).  Rope as
/// k_rope_norm (one head of n_rot at 512).
__global__ void k_kv_store(const float* kva, const float* norm, __half* cache, const int32_t* pos, int ring, int base,
                           float theta_scale, float eps) {
    __shared__ float sh[32];
    const int t = blockIdx.x;
    const float* r = kva + (size_t) t * kKvRow;
    float s = 0.f;
    for (int i = threadIdx.x; i < kLatent; i += blockDim.x) s += r[i] * r[i];
    s = block_sum(s, sh);
    const float inv = rsqrtf(s / (float) kLatent + eps);
    __half* dst = cache + (size_t) (ring ? pos[t] % ring : pos[t] - base) * kKvRow;
    for (int i = threadIdx.x; i < kLatent; i += blockDim.x) dst[i] = __float2half(r[i] * inv * norm[i]);
    const int i = threadIdx.x;
    if (i < (kKvRow - kLatent) / 2) {
        const float p = (float) pos[t];
        const float theta = p * powf(theta_scale, (float) i);
        const float c = cosf(theta), sn = sinf(theta);
        const float x0 = r[kLatent + 2 * i], x1 = r[kLatent + 2 * i + 1];
        dst[kLatent + 2 * i] = __float2half(x0 * c - x1 * sn);
        dst[kLatent + 2 * i + 1] = __float2half(x0 * sn + x1 * c);
    }
}

/// A window layer's cells for row t: the positions max(0, p - n_swa + 1) .. p in the ring of `ring` cells.
__global__ void k_swa_cells(const int32_t* pos, int n_swa, int ring, int32_t* cells, int32_t* n) {
    const int t = blockIdx.x;
    const int p = pos[t], a = max(0, p - n_swa + 1);
    for (int i = threadIdx.x; i <= p - a; i += blockDim.x) cells[(size_t) t * n_swa + i] = (a + i) % ring;
    if (threadIdx.x == 0) n[t] = p - a + 1;
}

/// The latent attention of T rows over their cell lists (cells[t][0 .. n[t]) of the cache), split in chunks of
/// kAttnChunk cells: block (chunk, t) scores its cells for every head (s = q_h . K_c * scale), and writes per head the
/// chunk's max m, sum l of exp(s - m) and acc[512] = sum exp(s - m) V_c (V the latent, a row's first 512) to part[t]
/// [chunk][h] (514 floats).  A warp per head in turn, lane = cell.
__global__ void __launch_bounds__(256) k_mla_chunk(const float* q, const __half* cache, const int32_t* cells,
                                                    const int32_t* n, int ld_cells, int H, float scale, float* part,
                                                    int n_chunks) {
    extern __shared__ __half kt[];   // [kAttnChunk][kKvRowPad]
    const int t = blockIdx.y, ch = blockIdx.x;
    const int cnt = n[t];
    const int c0 = ch * kAttnChunk;
    float* pt = part + ((size_t) t * n_chunks + ch) * H * (kLatent + 2);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (c0 >= cnt) {   // an empty chunk: m = -inf, l = 0 for every head
        for (int h = warp; h < H; h += nw)
            if (lane == 0) { pt[(size_t) h * (kLatent + 2)] = -INFINITY; pt[(size_t) h * (kLatent + 2) + 1] = 0.f; }
        return;
    }
    const int nc = min(kAttnChunk, cnt - c0);
    // the chunk's rows into shared memory, 4 bytes at a time
    for (int i = threadIdx.x; i < kAttnChunk * (kKvRow / 2); i += blockDim.x) {
        const int c = i / (kKvRow / 2), d2 = i % (kKvRow / 2);
        __half2 v = __float2half2_rn(0.f);
        if (c < nc) v = reinterpret_cast<const __half2*>(cache + (size_t) cells[(size_t) t * ld_cells + c0 + c] * kKvRow)[d2];
        reinterpret_cast<__half2*>(kt + (size_t) c * kKvRowPad)[d2] = v;
    }
    __syncthreads();
    for (int h = warp; h < H; h += nw) {
        const float* qh = q + ((size_t) t * H + h) * kKvRow;
        float s = -INFINITY;
        if (lane < nc) {
            const __half2* kr = reinterpret_cast<const __half2*>(kt + (size_t) lane * kKvRowPad);
            const float2* q2 = reinterpret_cast<const float2*>(qh);
            float acc = 0.f;
#pragma unroll 8
            for (int d = 0; d < kKvRow / 2; ++d) {
                const float2 kv = __half22float2(kr[d]);
                const float2 qq = q2[d];
                acc = fmaf(qq.x, kv.x, acc);
                acc = fmaf(qq.y, kv.y, acc);
            }
            s = acc * scale;
        }
        const float m = warp_max(s);
        const float p = lane < nc ? expf(s - m) : 0.f;
        const float l = warp_sum(p);
        float* ph = pt + (size_t) h * (kLatent + 2);
        if (lane == 0) { ph[0] = m; ph[1] = l; }
        float acc[kLatent / 32];
#pragma unroll
        for (int j = 0; j < kLatent / 32; ++j) acc[j] = 0.f;
        for (int c = 0; c < nc; ++c) {
            const float pc = __shfl_sync(0xffffffffu, p, c);
            const __half* vr = kt + (size_t) c * kKvRowPad;
#pragma unroll
            for (int j = 0; j < kLatent / 32; ++j) acc[j] = fmaf(pc, __half2float(vr[lane + 32 * j]), acc[j]);
        }
#pragma unroll
        for (int j = 0; j < kLatent / 32; ++j) ph[2 + lane + 32 * j] = acc[j];
    }
}

/// The chunks merged with the sink: out[t][h] = (sum_k acc_k e^(m_k - M)) / (e^(sink_h - M) + sum_k l_k e^(m_k - M)),
/// M = max(sink_h, m_k).  A block per (h, t), 128 threads (4 values of 512 each).
__global__ void k_mla_merge(const float* part, const float* sinks, float* out, int H, int n_chunks) {
    const int h = blockIdx.x, t = blockIdx.y;
    const float* p0 = part + ((size_t) t * n_chunks * H + h) * (kLatent + 2);
    const size_t step = (size_t) H * (kLatent + 2);
    float M = sinks[h];
    for (int k = 0; k < n_chunks; ++k) M = fmaxf(M, p0[k * step]);
    float L = expf(sinks[h] - M);
    for (int k = 0; k < n_chunks; ++k) {
        const float m = p0[k * step];
        if (m != -INFINITY) L += p0[k * step + 1] * expf(m - M);
    }
    const float inv = 1.f / L;
    for (int d = threadIdx.x; d < kLatent; d += blockDim.x) {
        float acc = 0.f;
        for (int k = 0; k < n_chunks; ++k) {
            const float m = p0[k * step];
            if (m != -INFINITY) acc += p0[k * step + 2 + d] * expf(m - M);
        }
        out[((size_t) t * H + h) * kLatent + d] = acc * inv;
    }
}

/// o[i] *= sigmoid(g[i]) for n values.
__global__ void k_gate_mul(float* o, const float* g, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = o[i] * sigm(g[i]);
}
/// h = silu(g) * u (the dense FFN and the shared expert: no clamp).
__global__ void k_swiglu(const float* g, const float* u, float* h, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = silu(g[i]) * u[i];
}
/// y += a
__global__ void k_add(float* y, const float* a, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = y[i] + a[i];
}
/// y[t][i] = W[i] . x[t] for an F32 W [n_out][n_in] (n_in a multiple of 4), a warp a row (grid (ceil(n_out / 8), T)),
/// times `mul`.
__global__ void k_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out, float mul) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int o = blockIdx.x * 8 + warp;
    if (o >= n_out) return;
    const float4* wr = reinterpret_cast<const float4*>(W + (size_t) o * n_in);
    const float4* xr = reinterpret_cast<const float4*>(x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) blockIdx.y * n_out + o] = acc * mul;
}

/// The router of row t (a block of n_expert <= 1024 threads): p = sigmoid(logit); the top k of p + bias (ties: the
/// lower expert); their p normalized by their sum (clamped to 2^-14 from below) when `norm`, then * scale.
__global__ void k_route(const float* logits, const float* bias, int n_expert, int k, bool norm, float scale,
                        int32_t* ids, float* w) {
    __shared__ float sel[1024];
    __shared__ float prob[1024];
    __shared__ float wv[32];
    __shared__ int wi[32];
    __shared__ int pick[64];
    const int t = blockIdx.x, e = threadIdx.x;
    const float* lg = logits + (size_t) t * n_expert;
    if (e < n_expert) {
        const float p = sigm(lg[e]);
        prob[e] = p;
        sel[e] = p + (bias ? bias[e] : 0.f);
    }
    __syncthreads();
    const int lane = e & 31, warp = e >> 5, nw = (blockDim.x + 31) >> 5;
    for (int j = 0; j < k; ++j) {
        float v = e < n_expert ? sel[e] : -INFINITY;
        int id = e;
        for (int o = 16; o > 0; o >>= 1) {
            const float v2 = __shfl_xor_sync(0xffffffffu, v, o);
            const int i2 = __shfl_xor_sync(0xffffffffu, id, o);
            if (v2 > v || (v2 == v && i2 < id)) { v = v2; id = i2; }
        }
        if (lane == 0) { wv[warp] = v; wi[warp] = id; }
        __syncthreads();
        if (e == 0) {
            float bv = wv[0];
            int bi = wi[0];
            for (int i = 1; i < nw; ++i)
                if (wv[i] > bv || (wv[i] == bv && wi[i] < bi)) { bv = wv[i]; bi = wi[i]; }
            pick[j] = bi;
            sel[bi] = -INFINITY;
        }
        __syncthreads();
    }
    if (e == 0) {
        float sum = 0.f;
        for (int j = 0; j < k; ++j) sum += prob[pick[j]];
        sum = fmaxf(sum, 6.103515625e-5f);
        for (int j = 0; j < k; ++j) {
            float v = prob[pick[j]];
            if (norm) v = v / sum;
            ids[(size_t) t * k + j] = pick[j];
            w[(size_t) t * k + j] = v * scale;
        }
    }
}

/// The routed experts' rows combined: y[t] = sum_j w[t][j] * parts[t * k + j] (each product rounded, summed in pick
/// order: ggml_mul then the add chain).
__global__ void k_moe_sum(const float* parts, const float* w, float* y, int k, int n) {
    const int t = blockIdx.x;
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        float acc = parts[((size_t) t * k) * n + c] * w[(size_t) t * k];
        for (int j = 1; j < k; ++j) {
            const float p = parts[((size_t) t * k + j) * n + c] * w[(size_t) t * k + j];
            acc = acc + p;
        }
        y[(size_t) t * n + c] = acc;
    }
}

/// The routed entries (n_ent = T * K, ids[t * K + j]) whose expert holds a VRAM slot (slot_of[x] >= 0) grouped per
/// expert in ascending id: group g's blob at slots + slot * stride, its entries [grp_start[g], grp_start[g + 1]) writing
/// row ent_dst (the entry) from token ent_tok.  The others are the CPU pool's (k_merge_cpu).  One block of 512.
__global__ void k_group_slots(const int32_t* ids, int n_ent, int K_, int n_expert, const int32_t* slot_of,
                              const uint8_t* slots, unsigned long long stride, unsigned long long* grp_ptr,
                              int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok) {
    __shared__ int cnt[1024], off[1024];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x)
        if (slot_of[ids[e]] >= 0) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (slots + (size_t) slot_of[x] * stride);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            if (slot_of[ids[e]] < 0) continue;
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// The CPU pool's rows into the entries' rows: parts[e] = cpu[e] where miss[e].  A block per entry.
__global__ void k_merge_cpu(float* parts, const float* cpu, const int32_t* miss, int n) {
    const int e = blockIdx.x;
    if (!miss[e]) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) parts[(size_t) e * n + i] = cpu[(size_t) e * n + i];
}

// ------------------------------------------------------------------------------------------------ Engram
/// OCP FP8 E4M3 -> f32 (ggml_f8_e4m3_to_fp32: 0x7F / 0xFF NaN, subnormals M * 2^-9)
__device__ __forceinline__ float e4m3(uint8_t x) {
    const int ax = x & 0x7F;
    if (ax == 0x7F) return __int_as_float(0x7fc00000);
    const int ex = (ax >> 3) & 0xF, man = ax & 0x7;
    const float v = ex == 0 ? ldexpf((float) man, -9) : ldexpf(1.0f + (float) man / 8.0f, ex - 7);
    return (x & 0x80) ? -v : v;
}

/// Engram's rows of row t: raw [t][heads][dim] FP8 bytes -> out[t][heads * dim]: each row RMS-normed (eps) and scaled
/// by its head's lookup_norm [heads][dim] and the row's validity (valid[t]: 0 or 1).  A block per (head, t).
__global__ void k_engram_rows(const uint8_t* raw, const float* lnorm, const float* valid, float* out, int heads, int dim,
                              float eps) {
    __shared__ float sh[32];
    __shared__ float v[256];
    const int h = blockIdx.x, t = blockIdx.y;
    const uint8_t* r = raw + ((size_t) t * heads + h) * dim;
    float s = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        v[i] = e4m3(r[i]);
        s += v[i] * v[i];
    }
    s = block_sum(s, sh);
    const float inv = rsqrtf(s / (float) dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        out[((size_t) t * heads + h) * dim + i] = v[i] * inv * lnorm[(size_t) h * dim + i] * valid[t];
}

/// Engram's mix of row t into its streams (a block each, 256 threads): kv[t] [hc + 1][n] (+ bias [hc + 1][n]) holds the
/// hc keys and the value; dot_i = sum_c rms(X_i)_c rms(key_i)_c gate[i][c] / sqrt(n); g_i = sigmoid(sign(dot_i)
/// sqrt(max(|dot_i|, 1e-6))); X_i += rms(value) * gamma * (g_i + hc_scale[i]) when the row has a valid n-gram (any[t]).
/// rms at `eps` (the reference's FLT_EPSILON).
__global__ void __launch_bounds__(256) k_engram_mix(float* X, const float* kv, const float* bias, const float* gate,
                                                     const float* gamma, const float* hc_scale, const float* any,
                                                     int hc, int n, float eps) {
    __shared__ float sh[32];
    __shared__ float g[8];
    const int t = blockIdx.x;
    float* xs = X + (size_t) t * hc * n;
    const float* kr = kv + (size_t) t * (hc + 1) * n;
    for (int i = 0; i < hc; ++i) {
        float sx = 0.f, sk = 0.f;
        for (int c = threadIdx.x; c < n; c += blockDim.x) {
            const float x = xs[(size_t) i * n + c];
            const float k = kr[(size_t) i * n + c] + bias[(size_t) i * n + c];
            sx += x * x;
            sk += k * k;
        }
        sx = block_sum(sx, sh);
        sk = block_sum(sk, sh);
        const float rx = rsqrtf(sx / (float) n + eps), rk = rsqrtf(sk / (float) n + eps);
        float d = 0.f;
        for (int c = threadIdx.x; c < n; c += blockDim.x) {
            const float x = xs[(size_t) i * n + c] * rx;
            const float k = (kr[(size_t) i * n + c] + bias[(size_t) i * n + c]) * rk;
            d += x * k * gate[(size_t) i * n + c];
        }
        d = block_sum(d, sh);
        if (threadIdx.x == 0) {
            const float dot = d * (1.0f / sqrtf((float) n));
            const float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
            const float sign = dot < 0.f ? -1.f : 1.f;
            g[i] = sigm(mag * sign);
        }
        __syncthreads();
    }
    const float* vr = kr + (size_t) hc * n;
    const float* vb = bias + (size_t) hc * n;
    float sv = 0.f;
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        const float v = vr[c] + vb[c];
        sv += v * v;
    }
    sv = block_sum(sv, sh);
    const float rv = rsqrtf(sv / (float) n + eps);
    const float a = any[t];
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        const float signal = (vr[c] + vb[c]) * rv * gamma[c];
        for (int i = 0; i < hc; ++i) {
            const float delta = signal * (g[i] + hc_scale[(size_t) i * n + c]);
            float& x = xs[(size_t) i * n + c];
            x = x + delta * a;
        }
    }
}

// ------------------------------------------------------------------------------------------------ the indexer
/// The indexer's value and gate rows of row t into its tail cell (pos % tail): [cell][value | gate]
__global__ void k_tail_store(const float* v, const float* gt, float* tail, const int32_t* pos, int n_tail, int dim) {
    const int t = blockIdx.x;
    float* dst = tail + (size_t) (pos[t] % n_tail) * 2 * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        dst[i] = v[(size_t) t * dim + i];
        dst[dim + i] = gt[(size_t) t * dim + i];
    }
}

/// Row t's group pooled when the row completes it ((pos + 1) % pool == 0): k_pool's arithmetic on the tail's cells of
/// the group, into pooled + group * dim.  A block of dim threads per row.
__global__ void k_pool_rows(const float* tail, int n_tail, int dim, int pool, const float* ape, const float* norm,
                            float eps, float theta_scale, int n_rot, const int32_t* pos, float* pooled) {
    __shared__ float sh[32];
    __shared__ float kk[256];
    const int p = pos[blockIdx.x], c = threadIdx.x;
    if ((p + 1) % pool) return;
    const int grp = p / pool;
    const float* src = tail + (size_t) ((grp * pool) % n_tail) * 2 * dim;
    if (c < dim) {
        float mx = -INFINITY;
        for (int m = 0; m < pool; ++m) mx = fmaxf(mx, src[((size_t) m * 2 + 1) * dim + c] + ape[(size_t) m * dim + c]);
        float den = 0.f, num = 0.f;
        for (int m = 0; m < pool; ++m) {
            const float e = expf(src[((size_t) m * 2 + 1) * dim + c] + ape[(size_t) m * dim + c] - mx);
            den += e;
            num += e * src[((size_t) m * 2) * dim + c];
        }
        kk[c] = num / den;
    }
    float s = c < dim ? kk[c] * kk[c] : 0.f;
    s = block_sum(s, sh);
    const float inv = rsqrtf(s / (float) dim + eps);
    if (c < dim) kk[c] = kk[c] * inv * norm[c];
    __syncthreads();
    const int nope = dim - n_rot;
    if (c < dim) {
        float v = kk[c];
        if (c >= nope) {
            const int i = (c - nope) / 2;
            const float theta = (float) (grp * pool) * powf(theta_scale, (float) i);
            const float cs = cosf(theta), sn = sinf(theta);
            const float x0 = kk[nope + 2 * i], x1 = kk[nope + 2 * i + 1];
            v = ((c - nope) & 1) == 0 ? x0 * cs - x1 * sn : x0 * sn + x1 * cs;
        }
        pooled[(size_t) grp * dim + c] = v;
    }
}

/// The indexer's scores of row t over its visible groups 0 .. pos / pool when they outnumber `reach` (else none: the
/// row attends every cell): s_g = sum_h w[t][h] relu(q[t][h] . k_g).  A warp a group, in a grid-stride loop over
/// `gridDim.x` blocks of 8 warps (a fixed grid, for the graphs).
__global__ void __launch_bounds__(256) k_index_scores_pos(const float* q, const float* w, const float* pooled,
                                                           const int32_t* pos, int pool, int reach, int heads, int dim,
                                                           float* scores, int ld_scores) {
    extern __shared__ float qs[];   // [heads][dim], then w [heads]
    const int t = blockIdx.y;
    const int vis = pos[t] / pool;
    if (vis <= reach) return;
    for (int i = threadIdx.x; i < heads * dim; i += blockDim.x) qs[i] = q[(size_t) t * heads * dim + i];
    for (int i = threadIdx.x; i < heads; i += blockDim.x) qs[heads * dim + i] = w[(size_t) t * heads + i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int gidx = blockIdx.x * 8 + warp; gidx < vis; gidx += gridDim.x * 8) {
        const float* kr = pooled + (size_t) gidx * dim;
        float kv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) kv[j] = lane + 32 * j < dim ? kr[lane + 32 * j] : 0.f;
        float sc = 0.f;
        for (int h = 0; h < heads; ++h) {
            float d = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if (lane + 32 * j < dim) d += qs[h * dim + lane + 32 * j] * kv[j];
            d = warp_sum(d);
            sc += qs[heads * dim + h] * fmaxf(d, 0.f);
        }
        if (lane == 0) scores[(size_t) t * ld_scores + gidx] = sc;
    }
}

__device__ __forceinline__ uint32_t f2key(float f) {   // larger float, larger key
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/// A sparse layer's cells for row t (a block of 1024 per row): every position 0 .. p while its visible groups (pos /
/// pool) are at most `reach`, else the cells of the `reach` top-scoring groups (a radix select over the scores' keys,
/// ties by ascending group) in ascending group order, then its own group's cells up to itself.  cells [T][ld].
__global__ void __launch_bounds__(1024) k_select_cells(const float* scores, int ld_scores, const int32_t* pos, int pool,
                                                       int reach, int32_t* cells, int ld, int32_t* n) {
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_prefix, s_mask, s_need;
    __shared__ int s_counts[1024];
    const int t = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const int p = pos[t], vis = p / pool;
    int32_t* row = cells + (size_t) t * ld;
    if (vis <= reach) {
        for (int i = tid; i <= p; i += nt) row[i] = i;
        if (tid == 0) n[t] = p + 1;
        return;
    }
    const float* sc = scores + (size_t) t * ld_scores;
    if (tid == 0) { s_prefix = 0; s_mask = 0; s_need = (uint32_t) reach; }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += nt) hist[i] = 0;
        __syncthreads();
        const uint32_t prefix = s_prefix, mask = s_mask;
        for (int i = tid; i < vis; i += nt) {
            const uint32_t k = f2key(sc[i]);
            if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t need = s_need, acc = 0;
            int b = 255;
            for (; b > 0; --b) {
                if (acc + hist[b] >= need) break;
                acc += hist[b];
            }
            s_need = need - acc;   // of bin b, the largest still to take
            s_prefix = prefix | ((uint32_t) b << shift);
            s_mask = mask | (0xFFu << shift);
        }
        __syncthreads();
    }
    // the threshold key: every key above it, and the first s_need equal to it in group order; each thread a contiguous
    // range of groups, so the order is the groups'
    const uint32_t thr = s_prefix, need_eq = s_need;
    const int per = (vis + nt - 1) / nt, a = min(vis, tid * per), b = min(vis, a + per);
    int eq_n = 0;
    for (int i = a; i < b; ++i) eq_n += f2key(sc[i]) == thr;
    // the equal keys before each thread's range (an exclusive scan): the first need_eq of them in group order are taken
    s_counts[tid] = eq_n;
    __syncthreads();
    if (tid == 0) {
        int run = 0;
        for (int i = 0; i < nt; ++i) {
            const int v = s_counts[i];
            s_counts[i] = run;
            run += v;
        }
    }
    __syncthreads();
    // the picks in group order: thread i's above-threshold groups and its share of the equal ones; their positions in
    // `row` follow from the counts of the picks before them, which the two scans give per kind but not interleaved:
    // the picks are written per thread in its order, then the list is ascending as a whole because a later thread's
    // groups are all larger - so a pick's slot is (its thread's picks before it) + (picks of earlier threads)
    __shared__ int s_picked[1024];
    int my_eq_base = s_counts[tid];
    int mine = 0;
    for (int i = a; i < b; ++i) {
        const uint32_t k = f2key(sc[i]);
        if (k > thr || (k == thr && (uint32_t) (my_eq_base++) < need_eq)) ++mine;
    }
    s_picked[tid] = mine;
    __syncthreads();
    if (tid == 0) {
        int run = 0;
        for (int i = 0; i < nt; ++i) {
            const int v = s_picked[i];
            s_picked[i] = run;
            run += v;
        }
    }
    __syncthreads();
    int slot = s_picked[tid];
    my_eq_base = s_counts[tid];
    for (int i = a; i < b; ++i) {
        const uint32_t k = f2key(sc[i]);
        if (k > thr || (k == thr && (uint32_t) (my_eq_base++) < need_eq)) {
            for (int c = 0; c < pool; ++c) row[slot * pool + c] = i * pool + c;
            ++slot;
        }
    }
    const int tail0 = vis * pool;
    for (int i = tid; i <= p - tail0; i += nt) row[reach * pool + i] = tail0 + i;
    if (tid == 0) n[t] = reach * pool + (p - tail0 + 1);
}

/// Group g's pooled key from its `pool` cells' value and gate rows (vg + gsrc[g] * pool * 2 * dim: [pool][2][dim],
/// value then gate, the members in order): p_m = softmax over the members of (gate_m + ape[m]) per channel, k = sum_m
/// p_m value_m, RMS-normed * norm, its last n_rot values roped at position gpos[g] (rope_norm) -> out + gdst[g] * dim.
/// A block of dim threads per group.
__global__ void k_pool(const float* vg, int dim, int pool, const float* ape, const float* norm, float eps,
                       float theta_scale, int n_rot, const int32_t* gpos, const int32_t* gsrc, const int32_t* gdst,
                       float* out) {
    __shared__ float sh[32];
    __shared__ float kk[256];
    const int g = blockIdx.x, c = threadIdx.x;
    const float* src = vg + (size_t) gsrc[g] * pool * 2 * dim;
    if (c < dim) {
        float mx = -INFINITY;
        for (int m = 0; m < pool; ++m) mx = fmaxf(mx, src[((size_t) m * 2 + 1) * dim + c] + ape[(size_t) m * dim + c]);
        float den = 0.f, num = 0.f;
        for (int m = 0; m < pool; ++m) {
            const float e = expf(src[((size_t) m * 2 + 1) * dim + c] + ape[(size_t) m * dim + c] - mx);
            den += e;
            num += e * src[((size_t) m * 2) * dim + c];
        }
        kk[c] = num / den;
    }
    float s = c < dim ? kk[c] * kk[c] : 0.f;
    s = block_sum(s, sh);
    const float inv = rsqrtf(s / (float) dim + eps);
    if (c < dim) kk[c] = kk[c] * inv * norm[c];
    __syncthreads();
    const int nope = dim - n_rot;
    if (c < dim) {
        float v = kk[c];
        if (c >= nope) {
            const int i = (c - nope) / 2;
            const float p = (float) gpos[g];
            const float theta = p * powf(theta_scale, (float) i);
            const float cs = cosf(theta), sn = sinf(theta);
            const float x0 = kk[nope + 2 * i], x1 = kk[nope + 2 * i + 1];
            v = ((c - nope) & 1) == 0 ? x0 * cs - x1 * sn : x0 * sn + x1 * cs;
        }
        out[(size_t) gdst[g] * dim + c] = v;
    }
}

/// The indexer's scores of row t over its visible groups 0 .. ng[t]: s_g = sum_h w[t][h] relu(q[t][h] . k_g), the keys
/// pooled [g][dim] f32.  A warp per group, 8 a block; q in shared memory.
__global__ void __launch_bounds__(256) k_index_scores(const float* q, const float* w, const float* pooled,
                                                       const int32_t* ng, int heads, int dim, float* scores,
                                                       int ld_scores) {
    extern __shared__ float qs[];   // [heads][dim], then w [heads]
    const int t = blockIdx.y;
    const int n = ng[t];
    for (int i = threadIdx.x; i < heads * dim; i += blockDim.x) qs[i] = q[(size_t) t * heads * dim + i];
    for (int i = threadIdx.x; i < heads; i += blockDim.x) qs[heads * dim + i] = w[(size_t) t * heads + i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gidx = blockIdx.x * 8 + warp;
    if (gidx >= n) return;
    const float* kr = pooled + (size_t) gidx * dim;
    float kv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) kv[j] = lane + 32 * j < dim ? kr[lane + 32 * j] : 0.f;
    float sc = 0.f;
    for (int h = 0; h < heads; ++h) {
        float d = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (lane + 32 * j < dim) d += qs[h * dim + lane + 32 * j] * kv[j];
        d = warp_sum(d);
        sc += qs[heads * dim + h] * fmaxf(d, 0.f);
    }
    if (lane == 0) scores[(size_t) t * ld_scores + gidx] = sc;
}

}  // namespace hy4k
