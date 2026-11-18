// src/hy4/hy4_engine.cu - the engine path of Hy4-flash (GGUF architecture hy_v4_flash), a driver of its own beside
// Flash-Next's and Qwen3.6's (no layer is theirs), built from the engine's kernels and src/hy4/hy4_kernels.cuh:
//
//   * the model as lamina/hy4/geometry.hpp reads it (llama.cpp's PR #32517 is the reference: done 2026-11-18 Hy4-flash
//     support assessed); dense weights on the main GPU in their GGUF blocks (Q8_0 through native_mmvq, the BF16 head
//     through the MMVF GEMV, F32 vectors and small matrices as they are); the routed experts in host memory as native
//     blobs [gate | up | down] for the CPU pool (ggml-cpu's arithmetic, llama.cpp's CPU backend's);
//   * T <= kMaxT rows at consecutive positions through every layer: the four iHC streams, Engram's rows (the table
//     read from the SSD by PleReader) before its layers, gated MLA with sinks over the window's ring of cells or, on
//     the sparse layers, over the cells the indexer picks, the dense FFN of layer 0, the latent MoE elsewhere; the
//     head after the streams' reduce;
//   * generate mode: --tokens IDS [--n N] [--force IDS] [--dump-full-logits F] (bench/parity.py's interface through
//     scratch/q36/logits_to_parity.py).
#include "hy4_kernels.cuh"

#include "lamina/artifact/gguf_reader.hpp"
#include "lamina/artifact/gguf_split.hpp"
#include "lamina/core/cuda_host.hpp"
#include "lamina/hy4/engine.hpp"
#include "lamina/hy4/geometry.hpp"
#include "lamina/kernels/bf16_gemv.hpp"
#include "lamina/kernels/cpu/native_expert.hpp"
#include "lamina/kernels/cpu/pool.hpp"
#include "lamina/kernels/iq_kernels.hpp"
#include "lamina/kernels/native_mmvq.hpp"
#include "lamina/ngram/ple_reader.hpp"
#include "lamina/program/host_sampling.hpp"
#include "lamina/program/serve_io.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

using lamina::program::serve_print;

namespace {

namespace K = lamina::kernels;
namespace C = lamina::kernels::cpu;
using lamina::core::ms_since;
using Clock = std::chrono::steady_clock;

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "lamina: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            throw Fatal{1};                                                                                            \
        }                                                                                                              \
    } while (0)

/// The run's end on an error: its message is printed, run() returns `code` (the engine may run in lamina-server's
/// process, which an exit would end).
struct Fatal {
    int code;
};

/// What a run allocates, freed when it ends: device, pinned and heap memory, streams, events.
struct Owned {
    std::mutex mu;
    std::vector<void*> dev, pinned, heap;
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> events;
    std::vector<cudaGraphExec_t> graphs;
    ~Owned() {
        cudaDeviceSynchronize();
        for (cudaGraphExec_t g : graphs) cudaGraphExecDestroy(g);
        for (cudaEvent_t e : events) cudaEventDestroy(e);
        for (cudaStream_t st : streams) cudaStreamDestroy(st);
        for (void* d : dev) cudaFree(d);
        for (void* h : pinned) cudaFreeHost(h);
        for (void* h : heap) std::free(h);
        (void) cudaGetLastError();
    }
};
Owned* g_owned = nullptr;
template <class V, class T> void own(std::vector<V> Owned::*list, T v) {
    if (g_owned == nullptr) return;
    std::lock_guard<std::mutex> lk(g_owned->mu);
    (g_owned->*list).push_back((V) v);
}
template <class T> T* dalloc(size_t n) {
    void* p = nullptr;
    CK(cudaMalloc(&p, std::max<size_t>(n * sizeof(T), 256)));
    own(&Owned::dev, p);
    return (T*) p;
}
template <class T> T* halloc(size_t n) {   // pinned
    void* p = nullptr;
    CK(cudaHostAlloc(&p, std::max<size_t>(n * sizeof(T), 256), cudaHostAllocDefault));
    own(&Owned::pinned, p);
    return (T*) p;
}
/// pinned host memory the GPU reads through the mapping: the host pointer, the device's in *dev
template <class T> T* hmapped(size_t n, T** dev) {
    void* p = nullptr;
    CK(cudaHostAlloc(&p, std::max<size_t>(n * sizeof(T), 256), cudaHostAllocMapped));
    own(&Owned::pinned, p);
    std::memset(p, 0, std::max<size_t>(n * sizeof(T), 256));
    CK(cudaHostGetDevicePointer((void**) dev, p, 0));
    return (T*) p;
}
void* heap_new(size_t n) {
    void* h = std::malloc(n);
    if (h != nullptr) own(&Owned::heap, h);
    return h;
}

constexpr int kMaxT = 8;              // rows a forward takes at most (native_mmvq's columns)
constexpr int kTail = 16;             // the indexer's per-cell value / gate rows kept (four groups of 4)
constexpr int kScoreBlocks = 128;     // the indexer scores' grid (a grid-stride loop over the groups)
constexpr float kEngramEps = 1.1920929e-07f;   // the reference's unset RMSNorm eps: FLT_EPSILON

// ------------------------------------------------------------------------------------------------ the model
struct Mat {
    const uint8_t* d = nullptr;   // device, GGUF blocks
    int type = -1;
    int n_in = 0, n_out = 0;
};

struct Layer {
    int il = 0;
    bool mtp = false, swa = false, has_idx = false, dense = false;
    int eng = -1;   // its Engram module, or -1
    int H = 0;
    float theta = 0;   // rope theta_scale = base^(-2 / n_rot)
    const float *attn_norm = nullptr, *sinks = nullptr, *q_a_norm = nullptr, *kv_a_norm = nullptr,
                *post_attn = nullptr, *ffn_norm = nullptr, *post_ffw = nullptr;
    Mat q_a, q_b, kv_a, k_b, v_b, wo, gate;
    const float *hca_fn = nullptr, *hca_base = nullptr, *hca_scale = nullptr;
    const float *hcf_fn = nullptr, *hcf_base = nullptr, *hcf_scale = nullptr;
    // the indexer
    Mat iq_b, ik, icv, icg;
    const float *ik_norm = nullptr, *ape = nullptr, *ipool_norm = nullptr, *iproj = nullptr;
    // Engram
    Mat e_kv;
    const float *e_kv_b = nullptr, *e_gate = nullptr, *e_lnorm = nullptr, *e_gamma = nullptr, *e_hcs = nullptr;
    // the dense FFN (layer 0)
    Mat f_gate, f_up, f_down;
    // the latent MoE
    const float *router = nullptr, *bias = nullptr;
    Mat r_down, r_up, s_gate, s_up, s_down;
    C::NativeFmt fmt;
    uint8_t* arena = nullptr;   // n_expert blobs, `stride` bytes apart (host)
    size_t stride = 0;
    // VRAM slots on the main GPU: the experts the routing uses most (Engine::place_slots, rebalance)
    K::NativeExpertLayout gl;          // the GPU's view of a blob
    uint8_t* slots = nullptr;          // n_slots blobs, `stride` apart (device)
    int n_slots = 0;
    std::vector<int32_t> slot_of;      // [n_expert]: its slot or -1 (host)
    std::vector<int32_t> slot_expert;  // [n_slots]: its expert or -1
    int32_t* slot_of_d = nullptr;      // slot_of on the device
    std::vector<float> use;            // routing counts, halved at each rebalance
    // caches
    __half* kv = nullptr;       // latent rows: the context's cells (sparse) or the window's ring
    int64_t kv_cells = 0;
    float* pooled = nullptr;    // the indexer's pooled keys [ctx / pool][idx_dim]
    float* tail = nullptr;      // the indexer's last kTail cells' value and gate rows [kTail][2][idx_dim]
};

struct Model {
    lamina::hy4::Geometry g;
    std::vector<Layer> layers;   // the trunk (the MTP block is loaded by the draft path)
    const uint8_t* embd = nullptr;
    int embd_type = -1;
    size_t embd_row = 0;
    const float *out_norm = nullptr, *ohc_fn = nullptr, *ohc_base = nullptr, *ohc_scale = nullptr;
    const uint16_t* head = nullptr;   // BF16 [n_vocab][n_embd]
    std::string table_path;
    uint64_t table_off = 0;
    double arena_gib = 0;
    int32_t eos = -1;   // tokenizer.ggml.eos_token_id: the end of an answer
};

struct Loader {
    const lamina::GgufModel& gm;
    std::string err;
    const lamina::TensorInfo* find(const std::string& name, size_t* s) {
        const lamina::TensorInfo* t = gm.find(name, s);
        if (!t) {
            std::fprintf(stderr, "lamina: missing %s\n", name.c_str());
            throw Fatal{1};
        }
        return t;
    }
    const uint8_t* host(const std::string& name, const lamina::TensorInfo** ti = nullptr) {
        size_t s = 0;
        const lamina::TensorInfo* t = find(name, &s);
        if (ti) *ti = t;
        return gm.shard(s).tensor_data(*t);
    }
    /// the tensor's GGUF bytes on the device
    const uint8_t* raw(const std::string& name, const lamina::TensorInfo** ti = nullptr) {
        const lamina::TensorInfo* t = nullptr;
        const uint8_t* src = host(name, &t);
        if (ti) *ti = t;
        const uint64_t n = lamina::tensor_payload_bytes(*t);
        uint8_t* d = dalloc<uint8_t>(n);
        CK(cudaMemcpy(d, src, n, cudaMemcpyHostToDevice));
        return d;
    }
    /// a Q8_0 matrix [n_in, n_out...]: its rows on the device
    Mat mat(const std::string& name) {
        const lamina::TensorInfo* t = nullptr;
        Mat m;
        m.d = raw(name, &t);
        m.type = (int) t->type;
        m.n_in = (int) t->shape[0];
        uint64_t rows = 1;
        for (size_t i = 1; i < t->shape.size(); ++i) rows *= t->shape[i];
        m.n_out = (int) rows;
        if (m.type != 8) {
            std::fprintf(stderr, "lamina: %s is %s; this path reads its dense matrices as Q8_0\n", name.c_str(),
                         t->type_name());
            throw Fatal{1};
        }
        return m;
    }
    /// an F32 tensor on the device (BF16 / F16 widened)
    const float* f32(const std::string& name) {
        const lamina::TensorInfo* t = nullptr;
        const uint8_t* src = host(name, &t);
        const size_t n = (size_t) t->elements();
        std::vector<float> h(n);
        if (t->type == 0) std::memcpy(h.data(), src, n * 4);
        else if (t->type == 30)
            for (size_t i = 0; i < n; ++i) {
                const uint32_t u = (uint32_t) (src[2 * i] | (src[2 * i + 1] << 8)) << 16;
                std::memcpy(&h[i], &u, 4);
            }
        else {
            std::fprintf(stderr, "lamina: %s is %s, expected F32 or BF16\n", name.c_str(), t->type_name());
            throw Fatal{1};
        }
        float* d = dalloc<float>(n);
        CK(cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice));
        return d;
    }
};

bool load(const std::string& path, Model& m, int threads, std::string& err) {
    const auto t0 = Clock::now();
    const lamina::GgufModel gm = lamina::GgufModel::open(path);
    if (std::string e = lamina::hy4::read_geometry(gm.meta(), m.g); !e.empty()) { err = e; return false; }
    if (std::string e = lamina::hy4::check_layout(gm, m.g); !e.empty()) { err = e; return false; }
    if (const lamina::MetaValue* v = gm.meta().get("tokenizer.ggml.eos_token_id"); v && v->is_num())
        m.eos = (int32_t) v->u;
    const auto& g = m.g;
    Loader L{gm, {}};
    {
        size_t s = 0;
        const lamina::TensorInfo* t = gm.find("per_layer_token_embd.weight", &s);
        if (t->type != 43) { err = "per_layer_token_embd.weight is " + std::string(t->type_name()) + ", expected F8_E4M3"; return false; }
        m.table_path = gm.shard(s).path();
        m.table_off = gm.shard(s).data_start() + t->offset;
    }
    const lamina::TensorInfo* te = nullptr;
    m.embd = L.raw("token_embd.weight", &te);
    m.embd_type = (int) te->type;
    m.embd_row = (size_t) (lamina::tensor_payload_bytes(*te) / te->shape[1]);
    if (!K::iq_supported(m.embd_type)) { err = std::string("token_embd.weight is ") + te->type_name() + ", which iq_embed_rows does not read"; return false; }
    m.out_norm = L.f32("output_norm.weight");
    m.ohc_fn = L.f32("output_hc_fn.weight");
    m.ohc_base = L.f32("output_hc_base.weight");
    m.ohc_scale = L.f32("output_hc_scale.weight");
    {
        const lamina::TensorInfo* th = nullptr;
        m.head = reinterpret_cast<const uint16_t*>(L.raw("output.weight", &th));
        if (th->type != 30) { err = std::string("output.weight is ") + th->type_name() + ", expected BF16"; return false; }
    }
    m.layers.resize((size_t) g.n_layers);
    for (int64_t l = 0; l < g.n_layers; ++l) {
        Layer& y = m.layers[(size_t) l];
        const std::string p = "blk." + std::to_string(l) + ".";
        y.il = (int) l;
        y.swa = g.swa[(size_t) l];
        y.has_idx = g.has_indexer(l);
        if (!y.swa && !y.has_idx) { err = p + ": a sparse layer without an indexer, which this path does not run"; return false; }
        y.dense = l < g.n_dense;
        y.eng = g.engram_index(l);
        y.H = (int) g.n_head[(size_t) l];
        y.theta = powf((float) g.base(l), -2.0f / (float) g.n_rot);
        y.attn_norm = L.f32(p + "attn_norm.weight");
        y.sinks = L.f32(p + "attn_sinks.weight");
        y.q_a = L.mat(p + "attn_q_a.weight");
        y.q_a_norm = L.f32(p + "attn_q_a_norm.weight");
        y.q_b = L.mat(p + "attn_q_b.weight");
        y.kv_a = L.mat(p + "attn_kv_a_mqa.weight");
        y.kv_a_norm = L.f32(p + "attn_kv_a_norm.weight");
        y.k_b = L.mat(p + "attn_k_b.weight");
        y.v_b = L.mat(p + "attn_v_b.weight");
        y.wo = L.mat(p + "attn_output.weight");
        y.gate = L.mat(p + "attn_gate.weight");
        y.post_attn = L.f32(p + "post_attention_norm.weight");
        y.ffn_norm = L.f32(p + "ffn_norm.weight");
        y.post_ffw = L.f32(p + "post_ffw_norm.weight");
        y.hca_fn = L.f32(p + "hc_attn_fn.weight");
        y.hca_base = L.f32(p + "hc_attn_base.weight");
        y.hca_scale = L.f32(p + "hc_attn_scale.weight");
        y.hcf_fn = L.f32(p + "hc_ffn_fn.weight");
        y.hcf_base = L.f32(p + "hc_ffn_base.weight");
        y.hcf_scale = L.f32(p + "hc_ffn_scale.weight");
        if (y.has_idx) {
            y.iq_b = L.mat(p + "indexer.attn_q_b.weight");
            y.ik = L.mat(p + "indexer.attn_k.weight");
            y.ik_norm = L.f32(p + "indexer.k_norm.weight");
            y.icv = L.mat(p + "indexer_compressor_value.weight");
            y.icg = L.mat(p + "indexer_compressor_gate.weight");
            y.ape = L.f32(p + "indexer_compressor_ape.weight");
            y.ipool_norm = L.f32(p + "indexer_compressor_norm.weight");
            y.iproj = L.f32(p + "indexer.proj.weight");
        }
        if (y.eng >= 0) {
            y.e_kv = L.mat(p + "engram_kv.weight");
            y.e_kv_b = L.f32(p + "engram_kv_b.weight");
            y.e_gate = L.f32(p + "engram_gate.weight");
            y.e_lnorm = L.f32(p + "engram_lookup_norm.weight");
            y.e_gamma = L.f32(p + "engram_pre_gamma.weight");
            y.e_hcs = L.f32(p + "engram_hc_scale.weight");
        }
        if (y.dense) {
            y.f_gate = L.mat(p + "ffn_gate.weight");
            y.f_up = L.mat(p + "ffn_up.weight");
            y.f_down = L.mat(p + "ffn_down.weight");
        } else {
            y.router = L.f32(p + "ffn_gate_inp.weight");
            y.bias = L.f32(p + "exp_probs_b.bias");
            y.r_down = L.mat(p + "ffn_routed_down.weight");
            y.r_up = L.mat(p + "ffn_routed_up.weight");
            y.s_gate = L.mat(p + "ffn_gate_shexp.weight");
            y.s_up = L.mat(p + "ffn_up_shexp.weight");
            y.s_down = L.mat(p + "ffn_down_shexp.weight");
        }
    }
    CK(cudaDeviceSynchronize());
    const double dense_s = ms_since(t0) / 1000.0;
    // the routed experts: native blobs [gate rows | up rows | down rows] in host memory, copied by `threads` threads
    struct Src {
        Layer* y;
        const uint8_t *g, *u, *d;
        size_t gsz, dsz;
    };
    std::vector<Src> src;
    size_t total = 0;
    for (Layer& y : m.layers) {
        if (y.dense) continue;
        const std::string p = "blk." + std::to_string(y.il) + ".";
        const lamina::TensorInfo *tg = nullptr, *tu = nullptr, *td = nullptr;
        Src s{&y, L.host(p + "ffn_gate_exps.weight", &tg), L.host(p + "ffn_up_exps.weight", &tu),
              L.host(p + "ffn_down_exps.weight", &td), 0, 0};
        if (tg->type != tu->type) { err = p + "ffn_gate_exps and ffn_up_exps differ in type"; return false; }
        if (!C::native_fmt((int) tg->type, (int) td->type, g.n_latent, g.n_ff_exp, y.fmt, err)) {
            err = p + "routed experts: " + err;
            return false;
        }
        y.fmt.swiglu_limit = g.swiglu_limit[(size_t) y.il];
        s.gsz = lamina::tensor_payload_bytes(*tg) / (size_t) g.n_expert;
        s.dsz = lamina::tensor_payload_bytes(*td) / (size_t) g.n_expert;
        if (s.gsz != y.fmt.up_off || 2 * s.gsz + s.dsz != y.fmt.bytes) { err = p + "routed experts: unexpected blob layout"; return false; }
        y.stride = (y.fmt.bytes + 63) & ~(size_t) 63;
        y.arena = (uint8_t*) heap_new(y.stride * (size_t) g.n_expert);
        if (!y.arena) { err = "out of host memory for the routed experts"; return false; }
        total += y.stride * (size_t) g.n_expert;
        src.push_back(s);
    }
    m.arena_gib = (double) total / (double) (1ull << 30);
    std::atomic<size_t> next{0};
    const size_t jobs = src.size() * (size_t) g.n_expert;
    auto copy = [&] {
        for (size_t j; (j = next.fetch_add(1)) < jobs;) {
            const Src& s = src[j / (size_t) g.n_expert];
            const size_t e = j % (size_t) g.n_expert;
            uint8_t* dst = s.y->arena + e * s.y->stride;
            std::memcpy(dst, s.g + e * s.gsz, s.gsz);
            std::memcpy(dst + s.gsz, s.u + e * s.gsz, s.gsz);
            std::memcpy(dst + 2 * s.gsz, s.d + e * s.dsz, s.dsz);
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < std::max(1, threads); ++i) th.emplace_back(copy);
    for (auto& t : th) t.join();
    std::fprintf(stderr, "lamina: Hy4-flash: %lld layers (+%lld MTP), dense weights on the GPU in %.1f s, the routed "
                 "experts (%.1f GiB) in host memory in %.1f s\n", (long long) g.n_layers, (long long) g.n_mtp, dense_s,
                 m.arena_gib, ms_since(t0) / 1000.0 - dense_s);
    return true;
}

// ------------------------------------------------------------------------------------------------ Engram's hash
/// The 16 table rows of each Engram module for the token at `pos` of `seq` (llama.cpp's engram_hash): ids [modules]
/// [16], `valid` 0 for a token outside the map (an image position), else 1.
void engram_ids(const lamina::hy4::Geometry& g, const std::vector<int32_t>& seq, int64_t pos, uint32_t* ids,
                float& valid) {
    const int64_t n_gram = g.ngram, n_head = g.heads_per_ngram, n_cols = g.eng_heads();
    const int64_t pad = g.eng_map[(size_t) g.eng_pad];
    auto cid = [&](int64_t p) -> int64_t {
        if (p < 0) return -1;
        const int32_t t = seq[(size_t) p];
        return t < 0 || t >= (int32_t) g.eng_map.size() ? -1 : g.eng_map[(size_t) t];
    };
    const int64_t c0 = cid(pos);
    valid = c0 < 0 ? 0.f : 1.f;
    uint64_t toks[8];
    toks[0] = (uint64_t) (c0 < 0 ? pad : c0);
    for (int64_t s = 1; s < n_gram; ++s) {
        const int64_t c = cid(pos - s);
        toks[s] = (uint64_t) (c < 0 ? pad : c);
    }
    for (size_t e = 0; e < g.eng_layers.size(); ++e) {
        const uint64_t* mult = g.eng_mult.data() + e * (size_t) n_gram;
        const uint64_t* prime = g.eng_prime.data() + e * (size_t) n_cols;
        const uint64_t* off = g.eng_off.data() + e * (size_t) n_cols;
        uint64_t rolling = toks[0] * mult[0];
        for (int64_t o = 1; o < n_gram; ++o) {
            rolling ^= toks[o] * mult[o];
            for (int64_t h = 0; h < n_head; ++h) {
                const int64_t col = (o - 1) * n_head + h;
                ids[e * (size_t) n_cols + (size_t) col] = (uint32_t) (rolling % prime[col] + off[col]);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ the engine
struct Engine {
    Model* m = nullptr;
    int64_t ctx = 0;     // cells of the sparse layers' caches
    int64_t ring = 0;    // cells of a window layer's ring
    cudaStream_t st = nullptr;
    std::unique_ptr<C::ExpertPool> pool;
    lamina::ngram::PleReader reader;
    std::vector<int32_t> seq;   // the tokens in the state, in order

    // buffers (kMaxT rows)
    float *X = nullptr, *xh = nullptr, *xn = nullptr, *y = nullptr, *sy = nullptr, *emb = nullptr, *post = nullptr;
    uint8_t *xq = nullptr, *qrq = nullptr, *qnq = nullptr, *attq = nullptr, *oq = nullptr, *hq = nullptr,
            *moeq = nullptr, *erq = nullptr, *ikq = nullptr;
    float *qa = nullptr, *q = nullptr, *qnope = nullptr, *qabs = nullptr, *kva = nullptr, *gt = nullptr,
          *part = nullptr, *att = nullptr, *o = nullptr;
    float *fg = nullptr, *fu = nullptr, *fh = nullptr;
    float *logits_r = nullptr, *rin = nullptr, *sg = nullptr, *su = nullptr, *sh = nullptr, *parts = nullptr,
          *moe = nullptr, *wts = nullptr;
    int32_t* ids = nullptr;
    uint8_t* eraw = nullptr;
    float *erows = nullptr, *ekv = nullptr, *evalid = nullptr;
    float *iq = nullptr, *ik = nullptr, *ivg = nullptr, *iw = nullptr, *scores = nullptr;
    int32_t *cells_swa = nullptr, *n_swa = nullptr, *cells_sel = nullptr, *n_sel = nullptr;
    float* logits = nullptr;
    int32_t *tok_d = nullptr, *pos_d = nullptr;
    // host (pinned; h_out and h_miss mapped: the next segment reads the CPU's rows through the mapping)
    float *h_rin = nullptr, *h_out = nullptr, *h_out_d = nullptr, *h_valid = nullptr;
    int32_t *h_ids = nullptr, *h_tok = nullptr, *h_pos = nullptr;
    uint8_t* h_raw = nullptr;
    uint32_t* h_rows = nullptr;
    uint8_t* acts = nullptr;
    std::vector<C::ExpertJobMulti> jobs;
    cudaEvent_t ev = nullptr;
    // the GPU's share of the routed experts: the entries grouped by slot, their input as q8_1, the kernels' scratch
    unsigned long long* grp_ptr = nullptr;
    int32_t *grp_start = nullptr, *n_groups = nullptr, *ent_dst = nullptr, *ent_tok = nullptr;
    int32_t *h_miss = nullptr, *h_miss_d = nullptr;
    uint8_t *rinq = nullptr, *escr = nullptr;
    int64_t hits = 0, lookups = 0;
    std::string profile_path;
    // counters
    double cpu_ms = 0, table_ms = 0;
    int64_t forwards = 0;

    int n_chunks_max() const { return std::max<int>((int) (m->g.idx_top_k / hy4k::kAttnChunk), (int) ((m->g.n_swa + hy4k::kAttnChunk - 1) / hy4k::kAttnChunk)); }

    void init(Model& model, int64_t max_ctx, int threads) {
        m = &model;
        const auto& g = m->g;
        ctx = (max_ctx + g.idx_pool - 1) / g.idx_pool * g.idx_pool;
        ring = g.n_swa + kMaxT;
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        own(&Owned::streams, st);
        CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        own(&Owned::events, ev);
        const size_t T = kMaxT, N = (size_t) g.n_embd, hc = (size_t) g.hc;
        size_t hmax = 0;
        for (const Layer& ly : m->layers) hmax = std::max<size_t>(hmax, (size_t) ly.H);
        const size_t QH = hmax * (size_t) g.k_head, VH = hmax * (size_t) g.v_head;
        const size_t q81 = 36;   // bytes of a q8_1 block of 32
        const size_t widest = std::max<size_t>({N, (size_t) g.n_ff, QH, VH});
        X = dalloc<float>(T * hc * N);
        xh = dalloc<float>(T * N);
        xn = dalloc<float>(T * N);
        y = dalloc<float>(T * N);
        sy = dalloc<float>(T * N);
        emb = dalloc<float>(T * N);
        post = dalloc<float>(T * hc);
        xq = dalloc<uint8_t>(T * widest / 32 * q81);
        qa = dalloc<float>(T * (size_t) g.q_lora);
        qrq = dalloc<uint8_t>(T * (size_t) g.q_lora / 32 * q81);
        q = dalloc<float>(T * QH);
        qnope = dalloc<float>(T * hmax * (size_t) g.nope());
        qnq = dalloc<uint8_t>(T * hmax * (size_t) g.nope() / 32 * q81);
        qabs = dalloc<float>(T * hmax * (size_t) hy4k::kKvRow);
        kva = dalloc<float>(T * (size_t) g.kv_row());
        gt = dalloc<float>(T * VH);
        part = dalloc<float>(T * (size_t) n_chunks_max() * hmax * (hy4k::kLatent + 2));
        att = dalloc<float>(T * hmax * hy4k::kLatent);
        attq = dalloc<uint8_t>(T * hmax * hy4k::kLatent / 32 * q81);
        o = dalloc<float>(T * VH);
        oq = dalloc<uint8_t>(T * VH / 32 * q81);
        fg = dalloc<float>(T * (size_t) g.n_ff);
        fu = dalloc<float>(T * (size_t) g.n_ff);
        fh = dalloc<float>(T * (size_t) g.n_ff);
        hq = dalloc<uint8_t>(T * (size_t) std::max(g.n_ff, g.n_ff_exp) / 32 * q81);
        logits_r = dalloc<float>(T * (size_t) g.n_expert);
        ids = dalloc<int32_t>(T * (size_t) g.top_k);
        wts = dalloc<float>(T * (size_t) g.top_k);
        rin = dalloc<float>(T * (size_t) g.n_latent);
        sg = dalloc<float>(T * (size_t) g.n_ff_exp);
        su = dalloc<float>(T * (size_t) g.n_ff_exp);
        sh = dalloc<float>(T * (size_t) g.n_ff_exp);
        parts = dalloc<float>(T * (size_t) g.top_k * (size_t) g.n_latent);
        moe = dalloc<float>(T * (size_t) g.n_latent);
        moeq = dalloc<uint8_t>(T * (size_t) g.n_latent / 32 * q81);
        const size_t n_mod = g.eng_layers.size(), eh = (size_t) g.eng_heads(), ed = (size_t) g.eng_dim;
        eraw = dalloc<uint8_t>(n_mod * T * eh * ed);
        erows = dalloc<float>(T * eh * ed);
        erq = dalloc<uint8_t>(T * eh * ed / 32 * q81);
        ekv = dalloc<float>(T * (hc + 1) * N);
        evalid = dalloc<float>(T);
        const size_t ih = (size_t) g.idx_heads, idd = (size_t) g.idx_dim;
        iq = dalloc<float>(T * ih * idd);
        ik = dalloc<float>(T * idd);
        ikq = dalloc<uint8_t>(T * idd / 32 * q81);
        ivg = dalloc<float>(T * 2 * idd);
        iw = dalloc<float>(T * ih);
        scores = dalloc<float>(T * (size_t) (ctx / g.idx_pool));
        cells_swa = dalloc<int32_t>(T * (size_t) g.n_swa);
        cells_sel = dalloc<int32_t>(T * (size_t) g.idx_top_k);
        n_swa = dalloc<int32_t>(T);
        n_sel = dalloc<int32_t>(T);
        logits = dalloc<float>(T * (size_t) g.n_vocab);
        tok_d = dalloc<int32_t>(T);
        pos_d = dalloc<int32_t>(T);
        h_rin = halloc<float>(T * (size_t) g.n_latent);
        h_out = hmapped<float>(T * (size_t) g.top_k * (size_t) g.n_latent, &h_out_d);
        h_valid = halloc<float>(T);
        h_ids = halloc<int32_t>(T * (size_t) g.top_k);
        h_tok = halloc<int32_t>(T);
        h_pos = halloc<int32_t>(T);
        for (size_t li = 0; li < m->layers.size(); ++li)
            if (!m->layers[li].dense) moe_layers.push_back((int) li);
        h_raw = halloc<uint8_t>(n_mod * T * eh * ed);
        h_rows = halloc<uint32_t>(n_mod * T * eh);
        acts = (uint8_t*) heap_new(T * C::kNativeActBytes);
        const size_t n_ent = T * (size_t) g.top_k;
        grp_ptr = dalloc<unsigned long long>(n_ent + 1);
        grp_start = dalloc<int32_t>(n_ent + 2);
        n_groups = dalloc<int32_t>(1);
        ent_dst = dalloc<int32_t>(n_ent);
        ent_tok = dalloc<int32_t>(n_ent);
        h_miss = hmapped<int32_t>(n_ent, &h_miss_d);
        rinq = dalloc<uint8_t>(T * (size_t) g.n_latent / 32 * q81);
        escr = dalloc<uint8_t>(K::native_expert_scratch_bytes((int64_t) n_ent, g.n_ff_exp));
        // the caches
        size_t kv_bytes = 0;
        for (Layer& l : m->layers) {
            l.kv_cells = l.swa ? ring : ctx;
            l.kv = dalloc<__half>((size_t) l.kv_cells * hy4k::kKvRow);
            kv_bytes += (size_t) l.kv_cells * hy4k::kKvRow * 2;
            if (l.has_idx) {
                l.pooled = dalloc<float>((size_t) (ctx / g.idx_pool) * idd);
                l.tail = dalloc<float>((size_t) kTail * 2 * idd);
                kv_bytes += (size_t) (ctx / g.idx_pool) * idd * 4;
            }
        }
        std::fprintf(stderr, "lamina: %lld cells of context: K/V and pooled keys %.2f GiB\n", (long long) ctx,
                     (double) kv_bytes / (double) (1ull << 30));
        pool = std::make_unique<C::ExpertPool>(threads);
        std::string err;
        if (!reader.open(m->table_path, m->table_off, m->g.eng_rows, 256, (uint64_t) 1 << 20, err, true,
                         (uint32_t) m->g.eng_dim)) {
            std::fprintf(stderr, "lamina: the Engram table: %s\n", err.c_str());
            throw Fatal{1};
        }
        reader.set_keepalive(100.0, 30.0);
    }

    int64_t pos() const { return (int64_t) seq.size(); }

    // ---- going back.  The sparse layers' cells and the pooled keys are by position: what a shorter state keeps is
    // already right.  The window layers' rings and the indexer's tail hold the last cells only: going back up to
    // kMaxT tokens leaves them whole (the ring has kMaxT cells to spare, the tail 16); further back takes a snapshot of
    // both taken at or before the point (snapshot(): at each prompt's end), else the state starts over.
    struct Snap {
        int64_t pos = 0;
        std::vector<uint8_t> bytes;   // every window layer's ring, then every indexer layer's tail
        std::vector<float> logits;    // the last row's (the next token's distribution there)
    };
    std::vector<Snap> snaps;
    static constexpr size_t kSnaps = 8;
    size_t snap_bytes() const {
        size_t n = 0;
        for (const Layer& l : m->layers) {
            if (l.swa) n += (size_t) l.kv_cells * hy4k::kKvRow * 2;
            if (l.has_idx) n += (size_t) kTail * 2 * (size_t) m->g.idx_dim * 4;
        }
        return n;
    }
    void snapshot(const std::vector<float>& logits_now) {
        if (!snaps.empty() && snaps.back().pos == pos()) return;
        if (snaps.size() == kSnaps) snaps.erase(snaps.begin());
        Snap s;
        s.pos = pos();
        s.logits = logits_now;
        s.bytes.resize(snap_bytes());
        size_t at = 0;
        for (const Layer& l : m->layers) {
            if (l.swa) {
                const size_t n = (size_t) l.kv_cells * hy4k::kKvRow * 2;
                CK(cudaMemcpyAsync(s.bytes.data() + at, l.kv, n, cudaMemcpyDeviceToHost, st));
                at += n;
            }
            if (l.has_idx) {
                const size_t n = (size_t) kTail * 2 * (size_t) m->g.idx_dim * 4;
                CK(cudaMemcpyAsync(s.bytes.data() + at, l.tail, n, cudaMemcpyDeviceToHost, st));
                at += n;
            }
        }
        CK(cudaStreamSynchronize(st));
        snaps.push_back(std::move(s));
    }
    /// The snapshot at exactly `p`, or null
    const Snap* snap_at(int64_t p) const {
        for (const Snap& s : snaps)
            if (s.pos == p) return &s;
        return nullptr;
    }
    /// The state back to (at most) its first `keep` tokens: the cells left whole, a snapshot's, or none.  Returns the
    /// tokens kept.
    int64_t rewind(int64_t keep) {
        keep = std::min<int64_t>(keep, pos());
        if (pos() - keep <= kMaxT) {
            seq.resize((size_t) keep);
        } else {
            const Snap* best = nullptr;
            for (const Snap& s : snaps)
                if (s.pos <= keep && (!best || s.pos > best->pos)) best = &s;
            if (best == nullptr) {
                seq.clear();
            } else {
                size_t at = 0;
                for (const Layer& l : m->layers) {
                    if (l.swa) {
                        const size_t n = (size_t) l.kv_cells * hy4k::kKvRow * 2;
                        CK(cudaMemcpyAsync(l.kv, best->bytes.data() + at, n, cudaMemcpyHostToDevice, st));
                        at += n;
                    }
                    if (l.has_idx) {
                        const size_t n = (size_t) kTail * 2 * (size_t) m->g.idx_dim * 4;
                        CK(cudaMemcpyAsync(l.tail, best->bytes.data() + at, n, cudaMemcpyHostToDevice, st));
                        at += n;
                    }
                }
                CK(cudaStreamSynchronize(st));
                seq.resize((size_t) best->pos);
            }
        }
        snaps.erase(std::remove_if(snaps.begin(), snaps.end(), [&](const Snap& s) { return s.pos > pos(); }), snaps.end());
        return pos();
    }

    void quant(const float* x, int64_t rows, int64_t n, uint8_t* out) { K::quantize_q8_1_rows(x, rows, n, out, st); }
    void mmvq(const Mat& w, const uint8_t* xq_, float* y_, int T) {
        K::native_mmvq(w.type, w.d, xq_, y_, w.n_in, w.n_out, T, st);
    }
    unsigned blocks(int64_t n) const { return (unsigned) ((n + hy4k::kThreads - 1) / hy4k::kThreads); }

    // ---- a forward.  Its GPU work runs in segments between the CPU's parts: segment 0 is the embedding and the layers
    // up to the first MoE layer's routing; segment s > 0 the s-th MoE layer's end (the CPU's rows merged, the weighted
    // sum, routed_up, the FFN's norm and iHC post) and the layers after it up to the next MoE layer's routing, the last
    // one the head.  Each (segment, rows) pair is captured once as a CUDA graph: what changes from one forward to the
    // next is in device memory (the tokens, the positions, the Engram rows, the slots' table) or in mapped host memory
    // (the CPU's rows), and every decision a row's position makes (its cells, its group's pooling, the indexer's pick)
    // is the GPU's.
    std::vector<int> moe_layers;
    std::map<int, cudaGraphExec_t> graphs;
    bool use_graphs = true;

    /// T rows (tokens toks, at positions pos() ..) through the model; logits of every row into `logits`.
    void forward(const int32_t* toks, int T) {
        const auto& g = m->g;
        const int64_t p0 = pos();
        for (int t = 0; t < T; ++t) seq.push_back(toks[t]);
        // Engram's rows: hashed now, read from the SSD while this forward's inputs go to the GPU
        const auto tt = Clock::now();
        const size_t n_mod = g.eng_layers.size(), eh = (size_t) g.eng_heads(), ed = (size_t) g.eng_dim;
        for (int t = 0; t < T; ++t) {
            std::vector<uint32_t> r(n_mod * eh);
            engram_ids(g, seq, p0 + t, r.data(), h_valid[t]);
            for (size_t e = 0; e < n_mod; ++e)
                std::memcpy(h_rows + (e * (size_t) T + (size_t) t) * eh, r.data() + e * eh, eh * 4);
            h_pos[t] = (int32_t) (p0 + t);
            h_tok[t] = toks[t];
        }
        const lamina::ngram::PleReader::Ticket ticket = reader.issue(h_rows, n_mod * (size_t) T * eh, h_raw);
        CK(cudaMemcpyAsync(tok_d, h_tok, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos_d, h_pos, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        std::string err;
        if (!reader.collect(ticket, err)) {
            std::fprintf(stderr, "lamina: the Engram table: %s\n", err.c_str());
            throw Fatal{1};
        }
        table_ms += ms_since(tt);
        CK(cudaMemcpyAsync(eraw, h_raw, n_mod * (size_t) T * eh * ed, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(evalid, h_valid, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        const int n_seg = (int) moe_layers.size() + 1;
        for (int s = 0; s < n_seg; ++s) {
            if (s > 0) cpu_part(m->layers[(size_t) moe_layers[(size_t) s - 1]], T);
            run_segment(s, T);
        }
        ++forwards;
    }

    void run_segment(int s, int T) {
        if (!use_graphs) {
            enqueue_segment(s, T);
            return;
        }
        const int key = s * 16 + T;
        auto it = graphs.find(key);
        if (it == graphs.end()) {
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            enqueue_segment(s, T);
            cudaGraph_t gr = nullptr;
            CK(cudaStreamEndCapture(st, &gr));
            cudaGraphExec_t ex = nullptr;
            CK(cudaGraphInstantiate(&ex, gr, 0));
            cudaGraphDestroy(gr);
            own(&Owned::graphs, ex);
            it = graphs.emplace(key, ex).first;
        }
        CK(cudaGraphLaunch(it->second, st));
    }

    void enqueue_segment(int s, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, hc = (int) g.hc;
        size_t li = 0;
        if (s == 0) {
            K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, g.n_vocab, tok_d, T, N, emb, st);
            hy4k::k_streams_init<<<T, hy4k::kThreads, 0, st>>>(emb, X, hc, N);
            hy4k::k_swa_cells<<<T, 256, 0, st>>>(pos_d, (int) g.n_swa, (int) ring, cells_swa, n_swa);
        } else {
            const int lm = moe_layers[(size_t) s - 1];
            moe_end(m->layers[(size_t) lm], T);
            li = (size_t) lm + 1;
        }
        for (; li < m->layers.size(); ++li) {
            Layer& l = m->layers[li];
            front(l, T);
            if (!l.dense) {
                moe_front(l, T);
                return;
            }
            dense_ffn(l, T);
        }
        // the head: the streams reduced, output_norm, the BF16 head
        hy4k::k_hc_pre<<<T, 256, 0, st>>>(X, m->ohc_fn, m->ohc_scale, m->ohc_base, hc, N, hc, g.eps, g.hc_eps, 0.f,
                                          nullptr, xh, m->out_norm, xn, g.eps);
        K::bf16_gemv_fp32_mmvf_multi(xn, m->head, logits, N, g.n_vocab, T, st);
    }

    /// Engram, the attention, and the FFN's input (xn, its q8_1 rows in xq; iHC's post weights in `post`)
    void front(Layer& l, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, hc = (int) g.hc, H = l.H;
        if (l.eng >= 0) {
            const int eh = (int) g.eng_heads(), ed = (int) g.eng_dim;
            hy4k::k_engram_rows<<<dim3(eh, T), 128, 0, st>>>(eraw + (size_t) l.eng * T * eh * ed, l.e_lnorm, evalid,
                                                             erows, eh, ed, kEngramEps);
            quant(erows, T, (int64_t) eh * ed, erq);
            mmvq(l.e_kv, erq, ekv, T);
            hy4k::k_engram_mix<<<T, 256, 0, st>>>(X, ekv, l.e_kv_b, l.e_gate, l.e_gamma, l.e_hcs, evalid, hc, N,
                                                  kEngramEps);
        }
        hy4k::k_hc_pre<<<T, 256, 0, st>>>(X, l.hca_fn, l.hca_scale, l.hca_base, hc, N, 2 * hc, g.eps, g.hc_eps,
                                          g.hc_mag, post, xh, l.attn_norm, xn, g.eps);
        quant(xn, T, N, xq);
        mmvq(l.q_a, xq, qa, T);
        hy4k::k_rms<<<T, hy4k::kThreads, 0, st>>>(qa, (int) g.q_lora, l.q_a_norm, qa, (int) g.q_lora, (int) g.q_lora, g.eps);
        quant(qa, T, g.q_lora, qrq);
        mmvq(l.q_b, qrq, q, T);
        mmvq(l.kv_a, xq, kva, T);
        mmvq(l.gate, xq, gt, T);
        hy4k::k_q_split<<<dim3(H, T), 256, 0, st>>>(q, qnope, qabs, H, (int) g.k_head, (int) g.nope());
        hy4k::k_rope_norm<<<dim3(H, T), 32, 0, st>>>(qabs, H, hy4k::kKvRow, hy4k::kLatent, (int) g.n_rot, pos_d, 0,
                                                     l.theta);
        quant(qnope, (int64_t) T * H, g.nope(), qnq);
        hy4k::k_head_q8_0<<<dim3((unsigned) ((H * hy4k::kLatent + 127) / 128), T), 128, 0, st>>>(
            l.k_b.d, qnq, qabs, H, (int) g.nope(), hy4k::kLatent, H * hy4k::kKvRow, hy4k::kKvRow);
        hy4k::k_kv_store<<<T, hy4k::kThreads, 0, st>>>(kva, l.kv_a_norm, l.kv, pos_d, l.swa ? (int) ring : 0, l.theta,
                                                       g.eps);
        const int32_t* cells = cells_swa;
        const int32_t* counts = n_swa;
        int ld = (int) g.n_swa;
        if (!l.swa) {
            indexer(l, T);
            cells = cells_sel;
            counts = n_sel;
            ld = (int) g.idx_top_k;
        }
        const int n_chunks = (ld + hy4k::kAttnChunk - 1) / hy4k::kAttnChunk;
        const size_t smem = (size_t) hy4k::kAttnChunk * hy4k::kKvRowPad * sizeof(__half);
        hy4k::k_mla_chunk<<<dim3(n_chunks, T), 256, smem, st>>>(qabs, l.kv, cells, counts, ld, H,
                                                                1.0f / sqrtf((float) g.k_head), part, n_chunks);
        hy4k::k_mla_merge<<<dim3(H, T), 128, 0, st>>>(part, l.sinks, att, H, n_chunks);
        quant(att, (int64_t) T * H, hy4k::kLatent, attq);
        hy4k::k_head_q8_0<<<dim3((unsigned) ((H * g.v_head + 127) / 128), T), 128, 0, st>>>(
            l.v_b.d, attq, o, H, hy4k::kLatent, (int) g.v_head, H * (int) g.v_head, (int) g.v_head);
        hy4k::k_gate_mul<<<blocks((int64_t) T * H * g.v_head), hy4k::kThreads, 0, st>>>(o, gt, (int64_t) T * H * g.v_head);
        quant(o, T, (int64_t) H * g.v_head, oq);
        mmvq(l.wo, oq, y, T);
        hy4k::k_rms_inplace<<<T, hy4k::kThreads, 0, st>>>(y, l.post_attn, N, g.eps);
        hy4k::k_hc_post<<<T, hy4k::kThreads, 0, st>>>(X, y, post, hc, N);
        hy4k::k_hc_pre<<<T, 256, 0, st>>>(X, l.hcf_fn, l.hcf_scale, l.hcf_base, hc, N, 2 * hc, g.eps, g.hc_eps,
                                          g.hc_mag, post, xh, l.ffn_norm, xn, g.eps);
        quant(xn, T, N, xq);
    }

    /// The sparse layer's indexer: the rows' index keys into the tail, the groups they complete pooled, and each row's
    /// cells: every cell while its visible groups (those before its own) are within the reach, else the top groups'.
    void indexer(Layer& l, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, ih = (int) g.idx_heads, idd = (int) g.idx_dim, pool_n = (int) g.idx_pool;
        mmvq(l.iq_b, qrq, iq, T);
        hy4k::k_rope_norm<<<dim3(ih, T), 32, 0, st>>>(iq, ih, idd, idd - (int) g.n_rot, (int) g.n_rot, pos_d, 0, l.theta);
        mmvq(l.ik, xq, ik, T);
        hy4k::k_rms<<<T, 128, 0, st>>>(ik, idd, l.ik_norm, ik, idd, idd, g.eps);
        quant(ik, T, idd, ikq);
        mmvq(l.icv, ikq, ivg, T);
        mmvq(l.icg, ikq, ivg + (size_t) kMaxT * idd, T);
        hy4k::k_tail_store<<<T, 128, 0, st>>>(ivg, ivg + (size_t) kMaxT * idd, l.tail, pos_d, kTail, idd);
        hy4k::k_pool_rows<<<T, 128, 0, st>>>(l.tail, kTail, idd, pool_n, l.ape, l.ipool_norm, g.eps, l.theta,
                                             (int) g.n_rot, pos_d, l.pooled);
        const int reach = (int) (g.idx_top_k / pool_n - (g.idx_tail ? 1 : 0));
        hy4k::k_gemv_f32<<<dim3((unsigned) ((ih + 7) / 8), T), 256, 0, st>>>(l.iproj, xn, iw, N, ih,
                                                                           1.0f / sqrtf((float) (idd * ih)));
        const size_t smem = ((size_t) ih * idd + ih) * sizeof(float);
        hy4k::k_index_scores_pos<<<dim3(kScoreBlocks, T), 256, smem, st>>>(iq, iw, l.pooled, pos_d, pool_n, reach, ih,
                                                                            idd, scores, (int) (ctx / pool_n));
        hy4k::k_select_cells<<<T, 1024, 0, st>>>(scores, (int) (ctx / pool_n), pos_d, pool_n, reach, cells_sel,
                                                 (int) g.idx_top_k, n_sel);
    }

    void dense_ffn(Layer& l, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, hc = (int) g.hc;
        mmvq(l.f_gate, xq, fg, T);
        mmvq(l.f_up, xq, fu, T);
        hy4k::k_swiglu<<<blocks((int64_t) T * g.n_ff), hy4k::kThreads, 0, st>>>(fg, fu, fh, (int64_t) T * g.n_ff);
        quant(fh, T, g.n_ff, hq);
        mmvq(l.f_down, hq, y, T);
        hy4k::k_rms_inplace<<<T, hy4k::kThreads, 0, st>>>(y, l.post_ffw, N, g.eps);
        hy4k::k_hc_post<<<T, hy4k::kThreads, 0, st>>>(X, y, post, hc, N);
    }

    /// The MoE layer's GPU work before the CPU's: the routing, the routed input to the host (an event when it is there),
    /// the experts with a VRAM slot, the shared expert.
    void moe_front(Layer& l, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, E = (int) g.n_expert, Kk = (int) g.top_k, Ld = (int) g.n_latent,
                  F = (int) g.n_ff_exp;
        hy4k::k_gemv_f32<<<dim3((unsigned) ((E + 7) / 8), T), 256, 0, st>>>(l.router, xn, logits_r, N, E, 1.f);
        hy4k::k_route<<<T, std::max(32, ((E + 31) / 32) * 32), 0, st>>>(logits_r, l.bias, E, Kk, g.w_norm, g.w_scale,
                                                                       ids, wts);
        mmvq(l.r_down, xq, rin, T);
        CK(cudaMemcpyAsync(h_rin, rin, (size_t) T * Ld * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(h_ids, ids, (size_t) T * Kk * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaEventRecordWithFlags(ev, st, cudaEventRecordExternal));   // a node of its own in a graph: the host waits on it
        if (l.n_slots > 0) {
            quant(rin, T, Ld, rinq);
            hy4k::k_group_slots<<<1, 512, 0, st>>>(ids, T * Kk, Kk, E, l.slot_of_d, l.slots, l.stride, grp_ptr, grp_start,
                                                  n_groups, ent_dst, ent_tok);
            K::native_expert_grouped(l.gl, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                     std::min<int64_t>((int64_t) T * Kk, E), (int64_t) T * Kk, rinq,
                                     (int64_t) (Ld / 32) * 36, escr, parts, st);
        }
        mmvq(l.s_gate, xq, sg, T);
        mmvq(l.s_up, xq, su, T);
        hy4k::k_swiglu<<<blocks((int64_t) T * F), hy4k::kThreads, 0, st>>>(sg, su, sh, (int64_t) T * F);
        quant(sh, T, F, hq);
        mmvq(l.s_down, hq, sy, T);
    }

    /// The CPU's part of an MoE layer, between two segments: the experts without a slot on the pool, their rows into
    /// mapped memory (h_out, h_miss), which the next segment merges.
    void cpu_part(Layer& l, int T) {
        const auto& g = m->g;
        const int Kk = (int) g.top_k, Ld = (int) g.n_latent;
        CK(cudaEventSynchronize(ev));
        const auto t0 = Clock::now();
        jobs.clear();
        std::map<int, size_t> open;
        int n_miss = 0;
        std::vector<char> quantized((size_t) T, 0);
        for (int e = 0; e < T * Kk; ++e) {
            const int ex = h_ids[e], t = e / Kk;
            l.use[(size_t) ex] += 1.f;
            h_miss[e] = l.slot_of[(size_t) ex] < 0;
            if (!h_miss[e]) continue;
            ++n_miss;
            if (!quantized[(size_t) t]) {
                C::native_quant_act(l.fmt, h_rin + (size_t) t * Ld, acts + (size_t) t * C::kNativeActBytes);
                quantized[(size_t) t] = 1;
            }
            auto it = open.find(ex);
            if (it == open.end() || jobs[it->second].nt == C::MAXT) {
                C::ExpertJobMulti j;
                j.blob = l.arena + (size_t) ex * l.stride;
                jobs.push_back(j);
                it = open.insert_or_assign(ex, jobs.size() - 1).first;
            }
            C::ExpertJobMulti& j = jobs[it->second];
            j.nact[j.nt] = acts + (size_t) t * C::kNativeActBytes;
            j.out[j.nt] = h_out + (size_t) e * Ld;
            ++j.nt;
        }
        lookups += T * Kk;
        hits += T * Kk - n_miss;
        if (n_miss > 0) pool->run_split_multi_native(l.fmt, jobs.data(), (int) jobs.size());
        cpu_ms += ms_since(t0);
    }

    /// The MoE layer's end: the CPU's rows merged (read through the mapping), the weighted sum, routed_up, the shared
    /// expert added, the FFN's norm, iHC's post.
    void moe_end(Layer& l, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, hc = (int) g.hc, Kk = (int) g.top_k, Ld = (int) g.n_latent;
        hy4k::k_merge_cpu<<<T * Kk, hy4k::kThreads, 0, st>>>(parts, h_out_d, h_miss_d, Ld);
        hy4k::k_moe_sum<<<T, hy4k::kThreads, 0, st>>>(parts, wts, moe, Kk, Ld);
        quant(moe, T, Ld, moeq);
        mmvq(l.r_up, moeq, y, T);
        hy4k::k_add<<<blocks((int64_t) T * N), hy4k::kThreads, 0, st>>>(y, sy, (int64_t) T * N);
        hy4k::k_rms_inplace<<<T, hy4k::kThreads, 0, st>>>(y, l.post_ffw, N, g.eps);
        hy4k::k_hc_post<<<T, hy4k::kThreads, 0, st>>>(X, y, post, hc, N);
    }

    // ---- the routed experts in VRAM
    /// Slots on the main GPU for every MoE layer, as many a layer as `gib` GiB hold (negative: the free VRAM less a
    /// reserve), filled from the profile's counts (else experts 0..); the GPU's view of each layer's blobs.
    void place_slots(double gib) {
        const auto& g = m->g;
        size_t row = 0;
        for (const Layer& l : m->layers)
            if (!l.dense) row += l.stride;
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const size_t reserve = (size_t) 1536 << 20;
        const size_t budget = gib >= 0 ? (size_t) (gib * (double) (1ull << 30)) : (fr > reserve ? fr - reserve : 0);
        const int S = (int) std::min<size_t>((size_t) g.n_expert, row ? budget / row : 0);
        load_profile();
        for (Layer& l : m->layers) {
            if (l.dense) continue;
            l.gl = K::native_expert_layout(l.fmt.gu_type, l.fmt.d_type, g.n_latent, g.n_ff_exp);
            l.gl.swiglu_limit = l.fmt.swiglu_limit;
            if (l.use.empty()) l.use.assign((size_t) g.n_expert, 0.f);
            l.slot_of.assign((size_t) g.n_expert, -1);
            l.slot_of_d = dalloc<int32_t>((size_t) g.n_expert);
            if (S <= 0 || !K::native_expert_supported(l.fmt.gu_type, l.fmt.d_type, g.n_latent, g.n_ff_exp)) continue;
            l.n_slots = S;
            l.slots = dalloc<uint8_t>((size_t) S * l.stride);
            l.slot_expert.assign((size_t) S, -1);
            std::vector<int32_t> order((size_t) g.n_expert);
            for (int e = 0; e < (int) g.n_expert; ++e) order[(size_t) e] = e;
            std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return l.use[(size_t) a] > l.use[(size_t) b]; });
            for (int k = 0; k < S; ++k) put_slot(l, k, order[(size_t) k]);
            CK(cudaMemcpy(l.slot_of_d, l.slot_of.data(), (size_t) g.n_expert * 4, cudaMemcpyHostToDevice));
        }
        CK(cudaDeviceSynchronize());
        std::fprintf(stderr, "lamina: %d of %lld routed experts a layer in VRAM (%.2f GiB)%s\n", std::max(S, 0),
                     (long long) g.n_expert, (double) std::max(S, 0) * (double) row / (double) (1ull << 30),
                     profile_loaded ? ", the profile's" : "");
    }
    void put_slot(Layer& l, int k, int32_t e) {
        if (l.slot_expert[(size_t) k] >= 0) l.slot_of[(size_t) l.slot_expert[(size_t) k]] = -1;
        CK(cudaMemcpyAsync(l.slots + (size_t) k * l.stride, l.arena + (size_t) e * l.stride, l.fmt.bytes,
                           cudaMemcpyHostToDevice, st));
        l.slot_expert[(size_t) k] = e;
        l.slot_of[(size_t) e] = k;
    }
    /// The slots to the experts the routing used most: in each layer the most used experts without a slot take the slots
    /// of the least used ones that are used less, at most `max_swaps` copies in all; then the counts are halved.
    int rebalance(int max_swaps) {
        const auto& g = m->g;
        int swaps = 0;
        for (Layer& l : m->layers) {
            if (l.n_slots == 0) continue;
            std::vector<int32_t> out, in;
            for (int e = 0; e < (int) g.n_expert; ++e) (l.slot_of[(size_t) e] < 0 ? out : in).push_back(e);
            std::sort(out.begin(), out.end(), [&](int32_t a, int32_t b) { return l.use[(size_t) a] > l.use[(size_t) b]; });
            std::sort(in.begin(), in.end(), [&](int32_t a, int32_t b) { return l.use[(size_t) a] < l.use[(size_t) b]; });
            bool moved = false;
            for (size_t i = 0; i < out.size() && i < in.size() && swaps < max_swaps; ++i) {
                if (l.use[(size_t) out[i]] <= 1.25f * l.use[(size_t) in[i]] + 0.5f) break;
                put_slot(l, l.slot_of[(size_t) in[i]], out[i]);
                ++swaps;
                moved = true;
            }
            if (moved) CK(cudaMemcpyAsync(l.slot_of_d, l.slot_of.data(), (size_t) g.n_expert * 4, cudaMemcpyHostToDevice, st));
            for (float& u : l.use) u *= 0.5f;
        }
        CK(cudaStreamSynchronize(st));
        return swaps;
    }
    // the profile: 'HY4P', n_layers, n_expert, then each MoE layer's counts (float)
    bool profile_loaded = false;
    void load_profile() {
        if (profile_path.empty()) return;
        FILE* f = std::fopen(profile_path.c_str(), "rb");
        if (!f) return;
        int32_t hdr[3] = {};
        const auto& g = m->g;
        if (std::fread(hdr, 4, 3, f) == 3 && hdr[0] == 0x50345948 && hdr[1] == (int32_t) g.n_layers &&
            hdr[2] == (int32_t) g.n_expert) {
            bool ok = true;
            for (Layer& l : m->layers) {
                if (l.dense) continue;
                l.use.assign((size_t) g.n_expert, 0.f);
                ok = ok && std::fread(l.use.data(), 4, l.use.size(), f) == l.use.size();
            }
            profile_loaded = ok;
            if (!ok)
                for (Layer& l : m->layers) l.use.clear();
        }
        std::fclose(f);
    }
    void save_profile() {
        if (profile_path.empty()) return;
        const std::string tmp = profile_path + ".tmp";
        FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) return;
        const auto& g = m->g;
        const int32_t hdr[3] = {0x50345948, (int32_t) g.n_layers, (int32_t) g.n_expert};
        bool ok = std::fwrite(hdr, 4, 3, f) == 3;
        for (const Layer& l : m->layers)
            if (!l.dense) ok = ok && std::fwrite(l.use.data(), 4, l.use.size(), f) == l.use.size();
        ok = std::fclose(f) == 0 && ok;
        if (ok) {
            std::remove(profile_path.c_str());
            std::rename(tmp.c_str(), profile_path.c_str());
        }
    }

    /// row r's logits on the host
    void host_logits(int r, std::vector<float>& out) {
        const auto& g = m->g;
        out.resize((size_t) g.n_vocab);
        CK(cudaMemcpyAsync(out.data(), logits + (size_t) r * g.n_vocab, (size_t) g.n_vocab * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }
};

int argmax(const std::vector<float>& v) { return (int) (std::max_element(v.begin(), v.end()) - v.begin()); }

// ------------------------------------------------------------------------------------------------ serve
/// lamina-server's engine protocol, one request at a time: GEN <max_new> [key=value ...] <ids> -> RESUME (the tokens
/// the state already held), PP lines while the prompt is read, a T line a token, DONE; STOP ends a request at its next
/// step, QUIT the loop.
struct Serve {
    Engine& E;
    int32_t eos = -1;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};

    explicit Serve(Engine& e, int32_t eos_id) : E(e), eos(eos_id) {}

    void request(const std::string& line) {
        const auto& g = E.m->g;
        const char* p = line.c_str() + 4;
        char* e = nullptr;
        const long long max_new = std::strtoll(p, &e, 10);
        p = e;
        lamina::program::Sampling smp;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t) (p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = const_cast<char*>(st0); break; }
            smp.set(tk.substr(0, eq), tk.c_str() + eq + 1);   // keys of other paths (img_keys, score) are left
        }
        const std::vector<int32_t> ids = lamina::program::parse_ids(p);
        if (max_new < 1 || ids.empty()) { serve_print("ERR bad request: max_new or the ids\n"); return; }
        for (const int32_t t : ids)
            if (t < 0 || t >= g.n_vocab) { serve_print("ERR a token id is outside the vocabulary\n"); return; }
        if ((int64_t) ids.size() + max_new > E.ctx - kMaxT) {
            serve_print("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%lld)\n", ids.size(), max_new,
                        (long long) (E.ctx - kMaxT));
            return;
        }
        stop.store(false);
        // what the state holds of this prompt: its common start, less one token when that is all of it (the last
        // prompt token's logits are the first pick)
        size_t common = 0;
        while (common < E.seq.size() && common < ids.size() && E.seq[common] == ids[common]) ++common;
        std::vector<float> lg;
        const Engine::Snap* whole = common == ids.size() ? E.snap_at((int64_t) common) : nullptr;
        if (whole) lg = whole->logits;   // the held prompt again: its snapshot has the first pick's logits
        else if (common == ids.size()) --common;
        const size_t reuse = (size_t) E.rewind((int64_t) common);
        if (reuse > 0) serve_print("RESUME %zu\n", reuse);
        const auto t0 = Clock::now();
        const int64_t fresh = (int64_t) (ids.size() - reuse);
        int64_t done = 0;
        auto last_pp = Clock::now();
        bool cancelled = false;
        while (done < fresh) {
            const int T = (int) std::min<int64_t>(kMaxT, fresh - done);
            E.forward(ids.data() + reuse + done, T);
            done += T;
            if (done == fresh || ms_since(last_pp) > 250) {
                serve_print("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) fresh, ms_since(t0),
                            (double) done / std::max(1e-9, ms_since(t0) / 1000.0));
                last_pp = Clock::now();
            }
            if (stop.load() && done < fresh) { cancelled = true; break; }
        }
        const double prompt_ms = ms_since(t0);
        if (cancelled) {
            serve_print("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0.0 %lld 0\n", ids.size(), prompt_ms, reuse,
                        (long long) done);
            return;
        }
        if (fresh > 0) {
            E.host_logits((int) ((fresh - 1) % kMaxT), lg);
            E.snapshot(lg);
        }
        std::mt19937_64 rng(smp.seed);
        std::vector<int32_t> hist = ids;
        const long long hits0 = E.hits, lookups0 = E.lookups;
        const auto d0 = Clock::now();
        long long produced = 0;
        const char* finish = "length";
        while (true) {
            const int nx = lamina::program::sample(lg, smp, hist, rng);
            if (nx < 0) { finish = "nonfinite"; break; }
            serve_print("T %d\n", nx);
            ++produced;
            hist.push_back(nx);
            if (nx == eos) { finish = "stop"; break; }
            if (stop.load()) { finish = "cancel"; break; }
            if (produced >= max_new || E.pos() + 1 >= E.ctx - kMaxT) { finish = "length"; break; }
            const int32_t one = nx;
            E.forward(&one, 1);
            E.host_logits(0, lg);
            if (produced % 128 == 0) E.rebalance(32);
        }
        const double decode_ms = ms_since(d0);
        const long long hits = E.hits - hits0, lookups = E.lookups - lookups0;
        serve_print("DONE %lld %zu %.1f %.1f %s 0 0 %zu %lld %lld 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms,
                    decode_ms, finish, reuse, hits, lookups, (long long) fresh);
        const int swaps = E.rebalance(512);
        E.save_profile();
        if (swaps > 0) std::fprintf(stderr, "lamina: %d experts moved into VRAM slots\n", swaps);
        std::fprintf(stderr, "lamina: %zu prompt tokens (%zu reused) in %.1f ms, %lld generated in %.1f ms (%.2f tok/s), "
                     "%s\n", ids.size(), reuse, prompt_ms, produced, decode_ms,
                     decode_ms > 0 ? 1000.0 * (double) produced / decode_ms : 0.0, finish);
    }

    int run() {
        serve_print("READY %lld stop\n", (long long) (E.ctx - kMaxT));
        const lamina::program::ServeIo& io = lamina::program::serve_io();
        std::thread reader([&] {
            std::string l;
            while (io.read_line(l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") { stop.store(true); continue; }
                std::lock_guard<std::mutex> lk(mu);
                lines.push_back(l);
                cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        });
        lamina::program::ReaderEnd reader_end{reader, io};
        for (;;) {
            std::string line;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return eof || !lines.empty(); });
                if (lines.empty()) return 0;
                line = lines.front();
                lines.pop_front();
            }
            if (line.empty()) continue;
            if (line.rfind("GEN ", 0) == 0) {
                request(line);
            } else if (line == "QUIT") {
                std::lock_guard<std::mutex> lk(mu);
                eof = true;
                lines.clear();
            } else {
                serve_print("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
            }
        }
    }
};

int run_engine(int argc, char** argv, bool check_only) {
    std::string model, tokens, force, dump_full;
    int64_t max_ctx = 8192;
    int n_gen = 64, threads = 0;
    bool serve = false;
    double expert_vram = -1;
    std::string profile;
    bool graphs = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "lamina: %s needs a value\n", a.c_str());
                throw Fatal{2};
            }
            return argv[++i];
        };
        if (a == "--native" || a == "--model") model = next();
        else if (a == "--max-context") max_ctx = std::atoll(next());
        else if (a == "--tokens") tokens = next();
        else if (a == "--tokens-file") {
            const char* f = next();
            FILE* fp = std::fopen(f, "rb");
            if (!fp) { std::fprintf(stderr, "lamina: cannot open %s\n", f); return 2; }
            char buf[1 << 16];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof buf, fp)) > 0) tokens.append(buf, n);
            std::fclose(fp);
        } else if (a == "--n") n_gen = std::atoi(next());
        else if (a == "--force") force = next();
        else if (a == "--dump-full-logits") dump_full = next();
        else if (a == "--cpu-threads" || a == "--pool-workers") threads = std::atoi(next());
        else if (a == "--serve") serve = true;
        else if (a == "--expert-vram") expert_vram = std::atof(next());
        else if (a == "--expert-profile") profile = next();
        else if (a == "--no-graph") graphs = false;
        else if (a == "--tier-gpus") (void) next();   // the expert tiers' GPUs: not used yet (open item ey)
        else {
            std::fprintf(stderr, "lamina: Hy4-flash's path does not take %s\n", a.c_str());
            return 2;
        }
    }
    if (model.empty()) { std::fprintf(stderr, "lamina: --native MODEL is required\n"); return 2; }
    if (check_only) return 0;
    Model m;
    std::string err;
    if (!load(model, m, 8, err)) {
        std::fprintf(stderr, "lamina: %s\n", err.c_str());
        return 1;
    }
    Engine E;
    E.profile_path = profile;
    E.use_graphs = graphs;
    E.init(m, max_ctx, threads);
    E.place_slots(expert_vram);
    if (serve) {
        Serve sv(E, m.eos);
        return sv.run();
    }
    const std::vector<int32_t> prompt = lamina::program::parse_ids(tokens.c_str()),
                               forced = lamina::program::parse_ids(force.c_str());
    if (prompt.empty()) { std::fprintf(stderr, "lamina: --tokens is required\n"); return 2; }
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    for (const std::vector<int32_t>* v : {&prompt, &forced})
        for (const int32_t t : *v)
            if (t < 0 || t >= m.g.n_vocab) {
                std::fprintf(stderr, "lamina: the token %d is outside the vocabulary (%lld)\n", (int) t, (long long) m.g.n_vocab);
                return 2;
            }
    if ((int64_t) prompt.size() + n_gen + kMaxT > E.ctx) {
        std::fprintf(stderr, "lamina: the prompt (%zu tokens) and %d generated exceed the context (%lld)\n", prompt.size(),
                     n_gen, (long long) E.ctx);
        return 2;
    }
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    const auto t0 = Clock::now();
    int last = 0;
    for (size_t i = 0; i < prompt.size(); i += kMaxT) {
        const int T = (int) std::min<size_t>(kMaxT, prompt.size() - i);
        E.forward(prompt.data() + i, T);
        last = T - 1;
    }
    std::vector<float> lg;
    E.host_logits(last, lg);
    const double pp_ms = ms_since(t0);
    const double cpu0 = E.cpu_ms, tab0 = E.table_ms;
    const auto t1 = Clock::now();
    std::vector<int32_t> gen;
    for (int i = 0; i < n_gen; ++i) {
        const int a = argmax(lg);
        gen.push_back(a);
        if (ff) std::fwrite(lg.data(), 4, lg.size(), ff);
        if (i + 1 == n_gen) break;
        const int32_t nx = i < (int) forced.size() ? forced[(size_t) i] : a;
        E.forward(&nx, 1);
        E.host_logits(0, lg);
        if ((i + 1) % 128 == 0) E.rebalance(32);
    }
    const double tg_ms = ms_since(t1);
    if (ff) std::fclose(ff);
    serve_print("gen:");
    for (int t : gen) serve_print(" %d", t);
    serve_print("\n");
    std::fprintf(stderr, "lamina: prompt %zu tokens in %.1f ms (%.1f tok/s), %zu generated in %.1f ms (%.2f tok/s); "
                 "the CPU's experts %.1f ms, the table %.1f ms while generating; VRAM hits %.1f%%\n", prompt.size(), pp_ms,
                 prompt.size() / (pp_ms / 1000.0), gen.size(), tg_ms, gen.size() / (tg_ms / 1000.0), E.cpu_ms - cpu0,
                 E.table_ms - tab0, E.lookups ? 100.0 * (double) E.hits / (double) E.lookups : 0.0);
    E.save_profile();
    return 0;
}

}  // namespace

int lamina::hy4::run(int argc, char** argv) {
    Owned owned;
    g_owned = &owned;
    struct Unset {
        ~Unset() { g_owned = nullptr; }
    } unset;
    try {
        return run_engine(argc, argv, false);
    } catch (const Fatal& f) {
        return f.code;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "lamina: %s\n", e.what());
        return 1;
    }
}

int lamina::hy4::check_args(int argc, char** argv) {
    try {
        return run_engine(argc, argv, true);
    } catch (const Fatal& f) {
        return f.code;
    }
}
