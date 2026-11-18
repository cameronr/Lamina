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
#include "hy4_prompt.cuh"

#include "lamina/artifact/gguf_reader.hpp"
#include "lamina/artifact/gguf_split.hpp"
#include "lamina/core/cuda_host.hpp"
#include "lamina/core/gpu_select.hpp"
#include "lamina/hy4/engine.hpp"
#include "lamina/hy4/geometry.hpp"
#include "lamina/kernels/bf16_gemv.hpp"
#include "lamina/kernels/cpu/native_expert.hpp"
#include "lamina/kernels/cpu/pool.hpp"
#include "lamina/kernels/iq_kernels.hpp"
#include "lamina/kernels/native_mmvq.hpp"
#include "lamina/ngram/ple_reader.hpp"
#include "lamina/platform/console_quit.hpp"
#include "lamina/platform/memory.hpp"
#include "lamina/prefill/moe_mmq.hpp"
#include "lamina/program/host_sampling.hpp"
#include "lamina/program/serve_io.hpp"

#include <cublas_v2.h>
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
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
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
    CK(cudaHostAlloc(&p, std::max<size_t>(n * sizeof(T), 256), cudaHostAllocPortable));   // the tiers copy from it
    own(&Owned::pinned, p);
    return (T*) p;
}
/// pinned host memory the GPU reads through the mapping: the host pointer, the device's in *dev
template <class T> T* hmapped(size_t n, T** dev) {
    void* p = nullptr;
    CK(cudaHostAlloc(&p, std::max<size_t>(n * sizeof(T), 256), cudaHostAllocMapped | cudaHostAllocPortable));
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
constexpr size_t kWeightPad = 4096;   // zero bytes after each dense weight

// ------------------------------------------------------------------------------------------------ the model
struct Mat {
    const uint8_t* d = nullptr;   // device, GGUF blocks
    int type = -1;
    int n_in = 0, n_out = 0;
};

/// A GPU's slots of one MoE layer
struct SlotSet {
    uint8_t* slots = nullptr;          // n blobs, the layer's stride apart (device)
    int n = 0;
    std::vector<int32_t> slot_of;      // [n_expert]: its slot or -1 (host)
    std::vector<int32_t> slot_expert;  // [n]: its expert or -1
    int32_t* slot_of_d = nullptr;      // slot_of on that GPU
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
    // the experts in VRAM: each GPU's slots (Engine::gpus: [0] the main GPU, then the expert tiers)
    K::NativeExpertLayout gl;          // the GPUs' view of a blob
    std::vector<SlotSet> held;
    std::vector<float> use;            // routing counts, halved at each rebalance
    std::vector<float> total;          // the run's routing counts (the learned profile's)
    std::vector<float> start;          // the start profile's shares (empty: none)
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
    size_t pinned = 0;   // the arena's bytes in pinned memory
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
        uint8_t* d = dalloc<uint8_t>(n + kWeightPad);   // MMQ's k loop reads past a short last row (zero activations)
        CK(cudaMemcpy(d, src, n, cudaMemcpyHostToDevice));
        CK(cudaMemset(d + n, 0, kWeightPad));
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

bool load(const std::string& path, Model& m, int threads, double pin_gib, std::string& err) {
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
        {   // whole blocks of both types (MMQ reads expert e at e * stride), 64-byte aligned
            int be = 0, bg = 0, bd = 0;
            lamina::block_geometry((uint32_t) tg->type, be, bg);
            lamina::block_geometry((uint32_t) td->type, be, bd);
            const size_t unit = std::lcm(std::lcm((size_t) 64, (size_t) bg), (size_t) bd);
            y.stride = (y.fmt.bytes + unit - 1) / unit * unit;
        }
        // pinned while the limit allows (the GPUs copy from it at their links' full rate, the host not waiting), else
        // pageable
        const size_t bytes = y.stride * (size_t) g.n_expert;
        if ((double) (m.pinned + bytes) <= pin_gib * (double) (1ull << 30)) {
            void* h = nullptr;
            if (cudaHostAlloc(&h, bytes, cudaHostAllocPortable) == cudaSuccess) {
                own(&Owned::pinned, h);
                y.arena = (uint8_t*) h;
                m.pinned += bytes;
            } else {
                (void) cudaGetLastError();
            }
        }
        if (!y.arena) y.arena = (uint8_t*) heap_new(bytes);
        if (!y.arena) { err = "out of host memory for the routed experts"; return false; }
        total += bytes;
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
                 "experts (%.1f GiB, %.1f of it pinned) in host memory in %.1f s\n", (long long) g.n_layers,
                 (long long) g.n_mtp, dense_s, m.arena_gib, (double) m.pinned / (double) (1ull << 30),
                 ms_since(t0) / 1000.0 - dense_s);
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
    std::string profile_path, save_path;   // the profile started from; where the learned one goes
    double save_every_min = 1.0;
    Clock::time_point last_save = Clock::now();
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

    // ---- a prompt part: chunks through the prompt path (Prompt, when it is set up and the part has chunk_min tokens
    // or more: pieces as even as chunk_cap allows), else forwards of kMaxT rows
    std::function<void(const int32_t*, int)> chunk_fn;
    int64_t chunk_cap = 0, chunk_min = 0;
    /// n tokens into the state; `on(done)` after each piece (false: stop there).  Returns the logits row of the last
    /// token (host_logits), -1 when stopped first.
    int feed(const int32_t* toks, int64_t n, const std::function<bool(int64_t)>& on) {
        int64_t done = 0;
        int last = 0;
        while (done < n) {
            // a segment up to the next multiple of kSnapEvery (a snapshot there) or the end, in even pieces
            const int64_t p = pos(), seg = std::min<int64_t>(n - done, (p / kSnapEvery + 1) * kSnapEvery - p);
            const bool chunks = chunk_fn && seg >= chunk_min;
            const int64_t step = chunks ? chunk_cap : kMaxT, pieces = (seg + step - 1) / step,
                          size = (seg + pieces - 1) / pieces;
            for (int64_t sd = 0; sd < seg;) {
                const int T = (int) std::min<int64_t>(size, seg - sd);
                if (chunks) chunk_fn(toks + done + sd, T);
                else forward(toks + done + sd, T);
                sd += T;
                last = chunks ? 0 : T - 1;
                if (on && !on(done + sd) && done + sd < n) return -1;
            }
            done += seg;
            if (pos() % kSnapEvery == 0 && done < n) {
                std::vector<float> lg;
                host_logits(last, lg);
                snapshot(lg, true);
            }
        }
        return last;
    }


    // ---- going back.  The sparse layers' cells and the pooled keys are by position: what a shorter state keeps is
    // already right.  The window layers' rings and the indexer's tail hold the last cells only: going back up to
    // kMaxT tokens leaves them whole (the ring has kMaxT cells to spare, the tail 16); further back takes a snapshot of
    // both taken at or before the point (snapshot(): at each prompt's end), else the state starts over.
    struct Snap {
        int64_t pos = 0;
        std::vector<uint8_t> bytes;   // every window layer's ring, then every indexer layer's tail
        std::vector<float> logits;    // the last row's (the next token's distribution there)
        bool mark = false;            // taken at a multiple of kSnapEvery during a read (kept), else at a prompt's end
    };
    std::vector<Snap> snaps;
    static constexpr size_t kSnaps = 8;              // prompt ends kept
    static constexpr int64_t kSnapEvery = 8192;      // a read keeps a snapshot at each multiple of this it passes
    size_t snap_bytes() const {
        size_t n = 0;
        for (const Layer& l : m->layers) {
            if (l.swa) n += (size_t) l.kv_cells * hy4k::kKvRow * 2;
            if (l.has_idx) n += (size_t) kTail * 2 * (size_t) m->g.idx_dim * 4;
        }
        return n;
    }
    void snapshot(const std::vector<float>& logits_now, bool mark = false) {
        if (snap_at(pos()) != nullptr) return;
        if (!mark && std::count_if(snaps.begin(), snaps.end(), [](const Snap& x) { return !x.mark; }) >= (long) kSnaps)
            snaps.erase(std::find_if(snaps.begin(), snaps.end(), [](const Snap& x) { return !x.mark; }));
        Snap s;
        s.pos = pos();
        s.mark = mark;
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
    int64_t rewind(int64_t keep, bool restore = false) {
        keep = std::min<int64_t>(keep, pos());
        if (pos() - keep <= kMaxT && !restore) {
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

    // ---- parked conversations: one a request leaves goes to RAM (its sparse layers' cells and pooled keys up to its
    // end, its snapshots and one at its end), for a later request that continues it; the least recently used leave
    // when the park passes its budget or the RAM available to the system falls under the reserve
    struct Parked {
        std::vector<int32_t> seq;
        std::vector<uint8_t> cells;   // per sparse layer: K/V rows [0, pos), then pooled keys [0, pos / pool)
        std::vector<Snap> snaps;
        uint64_t used = 0;
        size_t bytes() const {
            size_t n = cells.size();
            for (const Snap& x : snaps) n += x.bytes.size() + x.logits.size() * 4;
            return n;
        }
    };
    std::vector<Parked> park;
    uint64_t park_clock = 0;
    size_t park_budget = (size_t) 16384 << 20, ram_reserve = (size_t) 6144 << 20;
    static constexpr int64_t kParkMin = 1024;   // shorter conversations are not parked
    size_t parked_bytes() const {
        size_t n = 0;
        for (const Parked& c : park) n += c.bytes();
        return n;
    }
    /// The live conversation into the park, with a snapshot at its end (the logits of its last token unknown: empty);
    /// the live state keeps copies of its snapshots up to `keep` (the start the next request shares with it)
    void park_live(int64_t keep) {
        const auto& g = m->g;
        const int64_t p = pos(), groups = p / g.idx_pool;
        if (p < kParkMin) return;
        snapshot({}, false);
        Parked c;
        size_t n = 0;
        for (const Layer& l : m->layers)
            if (!l.swa) n += (size_t) p * hy4k::kKvRow * 2 + (size_t) groups * (size_t) g.idx_dim * 4;
        const size_t need = n + [&] { size_t k = 0; for (const Snap& x : snaps) k += x.bytes.size() + x.logits.size() * 4; return k; }();
        // the budget counts the snapshots too; the RAM, the cells alone (the snapshots are in RAM already)
        uint64_t avail = 0;
        const bool known = lamina::platform::available_ram(avail);
        while (!park.empty() && (parked_bytes() + need > park_budget || (known && avail < n + ram_reserve))) {
            auto lru = std::min_element(park.begin(), park.end(), [](const Parked& a, const Parked& b) { return a.used < b.used; });
            std::fprintf(stderr, "lamina: a parked conversation of %zu tokens leaves the park (%.0f MiB)\n", lru->seq.size(),
                         (double) lru->bytes() / (double) (1 << 20));
            park.erase(lru);
            if (known) lamina::platform::available_ram(avail);
        }
        if (need > park_budget || (known && avail < n + ram_reserve)) {
            std::fprintf(stderr, "lamina: a conversation of %lld tokens is not parked: %.0f MiB, over the park's budget "
                         "or the RAM reserve\n", (long long) p, (double) need / (double) (1 << 20));
            return;
        }
        c.cells.resize(n);
        size_t at = 0;
        for (const Layer& l : m->layers) {
            if (l.swa) continue;
            const size_t kb = (size_t) p * hy4k::kKvRow * 2, pb = (size_t) groups * (size_t) g.idx_dim * 4;
            CK(cudaMemcpyAsync(c.cells.data() + at, l.kv, kb, cudaMemcpyDeviceToHost, st));
            at += kb;
            if (pb) CK(cudaMemcpyAsync(c.cells.data() + at, l.pooled, pb, cudaMemcpyDeviceToHost, st));
            at += pb;
        }
        CK(cudaStreamSynchronize(st));
        c.seq = seq;
        c.snaps = std::move(snaps);
        snaps.clear();
        for (const Snap& x : c.snaps)
            if (x.pos <= keep) snaps.push_back(x);
        c.used = ++park_clock;
        std::fprintf(stderr, "lamina: a conversation of %lld tokens parked (%.0f MiB; the park %zu, %.0f MiB)\n",
                     (long long) p, (double) c.bytes() / (double) (1 << 20), park.size() + 1,
                     (double) (parked_bytes() + c.bytes()) / (double) (1 << 20));
        park.push_back(std::move(c));
    }
    /// A parked conversation (taken out of the park) as the live state: its cells and snapshots back, the state at its
    /// last snapshot (its end)
    void unpark(Parked c) {
        const auto& g = m->g;
        const int64_t p = (int64_t) c.seq.size(), groups = p / g.idx_pool;
        size_t at = 0;
        for (const Layer& l : m->layers) {
            if (l.swa) continue;
            const size_t kb = (size_t) p * hy4k::kKvRow * 2, pb = (size_t) groups * (size_t) g.idx_dim * 4;
            CK(cudaMemcpyAsync(l.kv, c.cells.data() + at, kb, cudaMemcpyHostToDevice, st));
            at += kb;
            if (pb) CK(cudaMemcpyAsync(l.pooled, c.cells.data() + at, pb, cudaMemcpyHostToDevice, st));
            at += pb;
        }
        CK(cudaStreamSynchronize(st));
        seq = std::move(c.seq);
        snaps = std::move(c.snaps);
        rewind(pos(), true);   // the rings and tails from its end's snapshot
        std::fprintf(stderr, "lamina: a parked conversation of %lld tokens resumed\n", (long long) p);
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
        hy4k::k_kv_store<<<T, hy4k::kThreads, 0, st>>>(kva, l.kv_a_norm, l.kv, pos_d, l.swa ? (int) ring : 0, 0, l.theta,
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
        if (!l.held.empty() && l.held[0].n > 0) {
            quant(rin, T, Ld, rinq);
            hy4k::k_group_slots<<<1, 512, 0, st>>>(ids, T * Kk, Kk, E, l.held[0].slot_of_d, l.held[0].slots, l.stride,
                                                  grp_ptr, grp_start, n_groups, ent_dst, ent_tok);
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

    /// The host's part of an MoE layer, between two segments: each entry to its holder (the main GPU's slots, which the
    /// segment before ran; a tier's, whose share runs now on that GPU; else the CPU pool), the tiers' and the CPU's rows
    /// into mapped memory (h_out, with h_miss), which the next segment merges.
    void cpu_part(Layer& l, int T) {
        const auto& g = m->g;
        const int Kk = (int) g.top_k, Ld = (int) g.n_latent;
        CK(cudaEventSynchronize(ev));
        const auto t0 = Clock::now();
        std::fill(tier_used.begin(), tier_used.end(), (char) 0);
        int n_miss = 0;
        for (int e = 0; e < T * Kk; ++e) {
            const int ex = h_ids[e];
            l.use[(size_t) ex] += 1.f;
            l.total[(size_t) ex] += 1.f;
            int holder = -1;
            for (size_t gi = 0; gi < l.held.size() && holder < 0; ++gi)
                if (l.held[gi].n > 0 && l.held[gi].slot_of[(size_t) ex] >= 0) holder = (int) gi;
            h_miss[e] = holder != 0;
            if (holder >= 0) ++gpus[(size_t) holder].hits;
            if (holder > 0) tier_used[(size_t) holder] = 1;
            if (holder < 0) ++n_miss;
        }
        for (size_t gi = 1; gi < tier_used.size(); ++gi)
            if (tier_used[gi]) tier_launch((int) gi, l, T);
        jobs.clear();
        std::map<int, size_t> open;
        std::vector<char> quantized((size_t) T, 0);
        for (int e = 0; e < T * Kk; ++e) {
            const int ex = h_ids[e], t = e / Kk;
            bool held = false;
            for (const SlotSet& s : l.held) held = held || (s.n > 0 && s.slot_of[(size_t) ex] >= 0);
            if (held) continue;
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
        for (size_t gi = 1; gi < tier_used.size(); ++gi)
            if (tier_used[gi]) {
                lamina::core::DeviceScope ds(gpus[gi].dev, main_dev);
                CK(cudaEventSynchronize(gpus[gi].ev));
            }
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

    // ---- the routed experts in VRAM: the main GPU's slots (holder 0) and the expert tiers' (the other GPUs, holders
    // 1..), every MoE layer the same count a GPU.  One ranking of a layer's experts by its routing counts places them:
    // the most used on the main GPU, the next on the first tier and so on, the rest on the CPU; an expert is held by one
    // GPU at most.
    struct Gpu {
        int dev = 0;
        std::string name;
        cudaStream_t st = nullptr;
        cudaEvent_t ev = nullptr;
        int S = 0;   // slots a layer
        // a tier's own buffers on its GPU: the routing and the routed input, its entries grouped, their rows
        int32_t* ids_d = nullptr;
        float* rin_d = nullptr;
        uint8_t *rinq_d = nullptr, *escr = nullptr;
        unsigned long long* grp_ptr = nullptr;
        int32_t *grp_start = nullptr, *n_groups = nullptr, *ent_dst = nullptr, *ent_tok = nullptr;
        float* rows_d = nullptr;
        float** out_d = nullptr;       // h_out's address (the mapping's, the same on every GPU)
        uint32_t** flag_d = nullptr;   // null: native_expert_rows_out raises no flag
        unsigned* counter_d = nullptr;
        std::map<int, cudaGraphExec_t> graphs;   // layer * 16 + rows
        int64_t hits = 0;
    };
    std::vector<Gpu> gpus;   // [0]: the main GPU
    int main_dev = 0;
    std::vector<char> tier_used;   // a forward's layer: the tiers given entries

    /// The main GPU's slots (as many a layer as `main_gib` GiB hold; negative: its free VRAM less a reserve) and each of
    /// `tier_devs`' (its free VRAM less a reserve), filled by the profile's ranking (else experts in order).
    void place_slots(double main_gib, const std::vector<int>& tier_devs) {
        const auto& g = m->g;
        CK(cudaGetDevice(&main_dev));
        size_t row = 0;
        for (const Layer& l : m->layers)
            if (!l.dense) row += l.stride;
        auto slots_for = [&](double gib, size_t reserve) {
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            const size_t budget = gib >= 0 ? (size_t) (gib * (double) (1ull << 30)) : (fr > reserve ? fr - reserve : 0);
            return (int) std::min<size_t>((size_t) g.n_expert, row ? budget / row : 0);
        };
        gpus.clear();
        Gpu g0;
        g0.dev = main_dev;
        g0.st = st;
        g0.ev = ev;
        g0.S = slots_for(main_gib, (size_t) 1536 << 20);
        cudaDeviceProp pr{};
        CK(cudaGetDeviceProperties(&pr, main_dev));
        g0.name = pr.name;
        gpus.push_back(g0);
        for (const int d : tier_devs) {
            lamina::core::DeviceScope ds(d, main_dev);
            Gpu t;
            t.dev = d;
            CK(cudaGetDeviceProperties(&pr, d));
            t.name = pr.name;
            CK(cudaStreamCreateWithFlags(&t.st, cudaStreamNonBlocking));
            own(&Owned::streams, t.st);
            CK(cudaEventCreateWithFlags(&t.ev, cudaEventDisableTiming));
            own(&Owned::events, t.ev);
            const size_t n_ent = (size_t) kMaxT * (size_t) g.top_k;
            t.ids_d = dalloc<int32_t>(n_ent);
            t.rin_d = dalloc<float>((size_t) kMaxT * (size_t) g.n_latent);
            t.rinq_d = dalloc<uint8_t>((size_t) kMaxT * (size_t) g.n_latent / 32 * 36);
            t.grp_ptr = dalloc<unsigned long long>(n_ent + 1);
            t.grp_start = dalloc<int32_t>(n_ent + 2);
            t.n_groups = dalloc<int32_t>(1);
            t.ent_dst = dalloc<int32_t>(n_ent);
            t.ent_tok = dalloc<int32_t>(n_ent);
            t.escr = dalloc<uint8_t>(K::native_expert_scratch_bytes((int64_t) n_ent, g.n_ff_exp));
            t.rows_d = dalloc<float>(n_ent * (size_t) g.n_latent);
            t.out_d = dalloc<float*>(1);
            t.flag_d = dalloc<uint32_t*>(1);
            t.counter_d = dalloc<unsigned>(1);
            float* out = h_out_d;
            uint32_t* none = nullptr;
            CK(cudaMemcpy(t.out_d, &out, sizeof out, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(t.flag_d, &none, sizeof none, cudaMemcpyHostToDevice));
            CK(cudaMemset(t.counter_d, 0, sizeof(unsigned)));
            t.S = slots_for(-1, (size_t) 1280 << 20);
            gpus.push_back(t);
        }
        load_profile();
        for (Layer& l : m->layers) {
            if (l.dense) continue;
            l.gl = K::native_expert_layout(l.fmt.gu_type, l.fmt.d_type, g.n_latent, g.n_ff_exp);
            l.gl.swiglu_limit = l.fmt.swiglu_limit;
            if (l.use.empty()) l.use.assign((size_t) g.n_expert, 0.f);
            l.total.assign((size_t) g.n_expert, 0.f);
            const bool ok = K::native_expert_supported(l.fmt.gu_type, l.fmt.d_type, g.n_latent, g.n_ff_exp);
            l.held.assign(gpus.size(), SlotSet{});
            for (size_t gi = 0; gi < gpus.size(); ++gi) {
                lamina::core::DeviceScope ds(gpus[gi].dev, main_dev, gi > 0);
                SlotSet& s = l.held[gi];
                s.slot_of.assign((size_t) g.n_expert, -1);
                s.slot_of_d = dalloc<int32_t>((size_t) g.n_expert);
                if (!ok || gpus[gi].S <= 0) continue;
                s.n = gpus[gi].S;
                s.slots = dalloc<uint8_t>((size_t) s.n * l.stride);
                s.slot_expert.assign((size_t) s.n, -1);
            }
            const std::vector<int32_t> order = ranking(l);
            size_t at = 0;
            for (size_t gi = 0; gi < gpus.size(); ++gi)
                for (int k = 0; k < l.held[gi].n && at < order.size(); ++k) put_slot(l, (int) gi, k, order[at++]);
            for (size_t gi = 0; gi < gpus.size(); ++gi) upload_table(l, (int) gi);
        }
        for (const Gpu& gp : gpus) {
            lamina::core::DeviceScope ds(gp.dev, main_dev, gp.dev != main_dev);
            CK(cudaDeviceSynchronize());
        }
        tier_used.assign(gpus.size(), 0);
        for (size_t gi = 0; gi < gpus.size(); ++gi)
            std::fprintf(stderr, "lamina: GPU %d, %s: %d of %lld routed experts a layer (%.2f GiB)%s\n", gpus[gi].dev,
                         gpus[gi].name.c_str(), std::max(gpus[gi].S, 0), (long long) g.n_expert,
                         (double) std::max(gpus[gi].S, 0) * (double) row / (double) (1ull << 30),
                         gi == 0 ? (profile_loaded ? ", ranked by " + profile_from : std::string(", in order (no profile)")).c_str() : "");
    }
    /// A layer's experts, the most used first (ties by id)
    std::vector<int32_t> ranking(const Layer& l) const {
        std::vector<int32_t> order(l.use.size());
        for (size_t e = 0; e < order.size(); ++e) order[e] = (int32_t) e;
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return l.use[(size_t) a] > l.use[(size_t) b]; });
        return order;
    }
    void upload_table(Layer& l, int gi) {
        lamina::core::DeviceScope ds(gpus[(size_t) gi].dev, main_dev, gi > 0);
        SlotSet& s = l.held[(size_t) gi];
        CK(cudaMemcpyAsync(s.slot_of_d, s.slot_of.data(), s.slot_of.size() * 4, cudaMemcpyHostToDevice, gpus[(size_t) gi].st));
    }
    /// Expert e into GPU gi's slot k (its blob from the RAM arena), out of any other GPU's slot
    void put_slot(Layer& l, int gi, int k, int32_t e) {
        SlotSet& s = l.held[(size_t) gi];
        for (size_t oi = 0; oi < l.held.size(); ++oi) {
            SlotSet& so = l.held[oi];
            if (so.slot_of.empty() || so.slot_of[(size_t) e] < 0) continue;
            so.slot_expert[(size_t) so.slot_of[(size_t) e]] = -1;
            so.slot_of[(size_t) e] = -1;
        }
        if (s.slot_expert[(size_t) k] >= 0) s.slot_of[(size_t) s.slot_expert[(size_t) k]] = -1;
        {
            lamina::core::DeviceScope ds(gpus[(size_t) gi].dev, main_dev, gi > 0);
            CK(cudaMemcpyAsync(s.slots + (size_t) k * l.stride, l.arena + (size_t) e * l.stride, l.fmt.bytes,
                               cudaMemcpyHostToDevice, gpus[(size_t) gi].st));
        }
        s.slot_expert[(size_t) k] = e;
        s.slot_of[(size_t) e] = k;
    }
    /// The GPUs' slots toward the ranking: in each layer, GPU by GPU (the main first), the experts its share of the
    /// ranking holds that it lacks take its empty slots, then the slots of its experts outside that share (the least
    /// used first) while they are used less (by 1.25x); at most `max_swaps` copies in all; then the counts are halved.
    /// `waiting` (if set) is asked before each copy: true ends the copies there (the counts are halved all the same).
    int rebalance(int max_swaps, const std::function<bool()>& waiting = nullptr) {
        int swaps = 0;
        for (Layer& l : m->layers) {
            if (l.dense || l.held.empty()) continue;
            const std::vector<int32_t> order = ranking(l);
            std::vector<int> want(order.size(), -1);   // expert -> the GPU its rank gives it
            size_t at = 0;
            for (size_t gi = 0; gi < l.held.size(); ++gi)
                for (int k = 0; k < l.held[gi].n && at < order.size(); ++k) want[(size_t) order[at++]] = (int) gi;
            std::vector<char> changed(l.held.size(), 0);
            for (size_t gi = 0; gi < l.held.size() && swaps < max_swaps; ++gi) {
                SlotSet& s = l.held[gi];
                if (s.n == 0) continue;
                std::vector<int32_t> in, out;
                for (const int32_t e : order)
                    if (want[(size_t) e] == (int) gi && s.slot_of[(size_t) e] < 0) in.push_back(e);   // most used first
                for (int k = 0; k < s.n; ++k) {
                    const int32_t e = s.slot_expert[(size_t) k];
                    if (e >= 0 && want[(size_t) e] != (int) gi) out.push_back(e);
                }
                std::sort(out.begin(), out.end(), [&](int32_t a, int32_t b) { return l.use[(size_t) a] < l.use[(size_t) b]; });
                size_t oi = 0;
                for (const int32_t e : in) {
                    if (swaps < max_swaps && waiting && waiting()) max_swaps = swaps;
                    if (swaps >= max_swaps) break;
                    int k = -1;
                    for (int j = 0; j < s.n && k < 0; ++j)
                        if (s.slot_expert[(size_t) j] < 0) k = j;
                    if (k < 0) {
                        if (oi >= out.size() || l.use[(size_t) e] <= 1.25f * l.use[(size_t) out[oi]] + 0.5f) break;
                        k = s.slot_of[(size_t) out[oi++]];
                    }
                    for (size_t hi = 0; hi < l.held.size(); ++hi)   // the GPU it leaves
                        if (!l.held[hi].slot_of.empty() && l.held[hi].slot_of[(size_t) e] >= 0) changed[hi] = 1;
                    put_slot(l, (int) gi, k, e);
                    changed[gi] = 1;
                    ++swaps;
                }
            }
            for (size_t gi = 0; gi < l.held.size(); ++gi)
                if (changed[gi]) upload_table(l, (int) gi);
            for (float& u : l.use) u *= 0.5f;
        }
        for (const Gpu& gp : gpus) {
            lamina::core::DeviceScope ds(gp.dev, main_dev, gp.dev != main_dev);
            CK(cudaStreamSynchronize(gp.st));
        }
        return swaps;
    }
    /// A tier's share of an MoE layer, on its GPU (current): the routing and the routed input from the host, its entries
    /// grouped and run, their rows into the mapped h_out (the next segment merges them as the CPU's)
    void tier_enqueue(Gpu& gp, int gi, Layer& l, int T) {
        const auto& g = m->g;
        const int E = (int) g.n_expert, Kk = (int) g.top_k, Ld = (int) g.n_latent;
        SlotSet& s = l.held[(size_t) gi];
        CK(cudaMemcpyAsync(gp.ids_d, h_ids, (size_t) T * Kk * 4, cudaMemcpyHostToDevice, gp.st));
        CK(cudaMemcpyAsync(gp.rin_d, h_rin, (size_t) T * Ld * 4, cudaMemcpyHostToDevice, gp.st));
        K::quantize_q8_1_rows(gp.rin_d, T, Ld, gp.rinq_d, gp.st);
        hy4k::k_group_slots<<<1, 512, 0, gp.st>>>(gp.ids_d, T * Kk, Kk, E, s.slot_of_d, s.slots, l.stride, gp.grp_ptr,
                                                 gp.grp_start, gp.n_groups, gp.ent_dst, gp.ent_tok);
        K::native_expert_grouped(l.gl, gp.grp_ptr, gp.grp_start, gp.n_groups, gp.ent_dst, gp.ent_tok,
                                 std::min<int64_t>((int64_t) T * Kk, E), (int64_t) T * Kk, gp.rinq_d,
                                 (int64_t) (Ld / 32) * 36, gp.escr, gp.rows_d, gp.st);
        K::native_expert_rows_out(gp.rows_d, gp.n_groups, gp.grp_start, gp.ent_dst, Ld, (int64_t) T * Kk, gp.out_d,
                                  gp.flag_d, nullptr, gp.counter_d, gp.st);
        CK(cudaEventRecordWithFlags(gp.ev, gp.st, cudaEventRecordExternal));
    }
    void tier_launch(int gi, Layer& l, int T) {
        Gpu& gp = gpus[(size_t) gi];
        lamina::core::DeviceScope ds(gp.dev, main_dev);
        if (!use_graphs) {
            tier_enqueue(gp, gi, l, T);
            return;
        }
        const int key = l.il * 16 + T;
        auto it = gp.graphs.find(key);
        if (it == gp.graphs.end()) {
            CK(cudaStreamBeginCapture(gp.st, cudaStreamCaptureModeThreadLocal));
            tier_enqueue(gp, gi, l, T);
            cudaGraph_t gr = nullptr;
            CK(cudaStreamEndCapture(gp.st, &gr));
            cudaGraphExec_t ex = nullptr;
            CK(cudaGraphInstantiate(&ex, gr, 0));
            cudaGraphDestroy(gr);
            own(&Owned::graphs, ex);
            it = gp.graphs.emplace(key, ex).first;
        }
        CK(cudaGraphLaunch(it->second, gp.st));
    }
    // a profile: 'HY4P', n_layers, n_expert, then each MoE layer's shares of its routing (float, summing to 1)
    bool profile_loaded = false;
    std::string profile_from;
    /// The start: the learned profile when it holds this model's shape, else --expert-profile's.  Its shares rank the
    /// experts and count as 1,000 entries a layer of routing until the rebalances halve them.
    void load_profile() {
        const auto& g = m->g;
        auto read = [&](const std::string& path) {
            if (path.empty()) return false;
            FILE* f = std::fopen(path.c_str(), "rb");
            if (!f) return false;
            int32_t hdr[3] = {};
            bool ok = std::fread(hdr, 4, 3, f) == 3 && hdr[0] == 0x50345948 && hdr[1] == (int32_t) g.n_layers &&
                      hdr[2] == (int32_t) g.n_expert;
            for (Layer& l : m->layers) {
                if (l.dense || !ok) continue;
                l.start.assign((size_t) g.n_expert, 0.f);
                ok = std::fread(l.start.data(), 4, l.start.size(), f) == l.start.size();
                double sum = 0;
                for (const float v : l.start) sum += v > 0 ? v : 0;
                for (float& v : l.start) v = sum > 0 && v > 0 ? (float) (v / sum) : 0.f;
            }
            std::fclose(f);
            if (!ok)
                for (Layer& l : m->layers) l.start.clear();
            return ok;
        };
        if (read(save_path)) profile_from = save_path;
        else if (read(profile_path)) profile_from = profile_path;
        profile_loaded = !profile_from.empty();
        if (!profile_loaded) return;
        for (Layer& l : m->layers) {
            if (l.dense) continue;
            l.use.resize(l.start.size());
            for (size_t e = 0; e < l.start.size(); ++e) l.use[e] = 1000.f * l.start[e];
        }
    }
    /// The learned profile: each layer's share of the run's routing and the start's share at equal weight (the run's
    /// alone without a start); written once `save_every_min` minutes passed since the last (`force`: now)
    void save_profile(bool force) {
        if (save_path.empty()) return;
        if (!force && (save_every_min <= 0 || ms_since(last_save) < save_every_min * 60000.0)) return;
        last_save = Clock::now();
        const std::string tmp = save_path + ".tmp";
        FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) return;
        const auto& g = m->g;
        const int32_t hdr[3] = {0x50345948, (int32_t) g.n_layers, (int32_t) g.n_expert};
        bool ok = std::fwrite(hdr, 4, 3, f) == 3;
        std::vector<float> share((size_t) g.n_expert);
        for (const Layer& l : m->layers) {
            if (l.dense) continue;
            double sum = 0;
            for (const float v : l.total) sum += v;
            const bool run = sum > 0, base = !l.start.empty();
            for (size_t e = 0; e < share.size(); ++e) {
                const double r = run ? l.total[e] / sum : 0.0, b = base ? l.start[e] : 0.0;
                share[e] = (float) (run && base ? 0.5 * (r + b) : run ? r : b);
            }
            ok = ok && std::fwrite(share.data(), 4, share.size(), f) == share.size();
        }
        ok = std::fclose(f) == 0 && ok;
        if (ok) {
            std::remove(save_path.c_str());
            std::rename(tmp.c_str(), save_path.c_str());
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

// ------------------------------------------------------------------------------------------------ the prompt path
namespace MMQ = lamina::prefill::mmq;
constexpr size_t kQ8Block = 144;   // bytes of a q8_1 block of MMQ's layout (128 values)

/// A GPU's share of a prompt chunk's routed experts: the entries of the experts in its slots and, on the streaming GPU,
/// of the experts no GPU holds (copied from the arena into two staging buffers in turn); their rows summed per token
/// into `acc`, which a tier hands to the main GPU through pinned memory.
struct MoeDev {
    int gi = 0, dev = 0;
    cudaStream_t st = nullptr, cst = nullptr;
    cudaEvent_t done = nullptr, copied[2] = {}, used[2] = {};
    std::unique_ptr<MMQ::Context> mmq;
    uint8_t *tq = nullptr, *eq = nullptr, *hq = nullptr;
    float *gu = nullptr, *h = nullptr, *rows = nullptr, *acc = nullptr;
    int32_t *ident = nullptr, *plan = nullptr, *plan_h = nullptr;
    int64_t ident_rows = 0;
    float* acc_h = nullptr;
    uint8_t* stage[2] = {};
};

/// The prompt path: a chunk of up to `cap` rows a forward.  The main GPU runs the projections on llama.cpp's MMQ (the
/// Q8_0 weights read once a chunk), the attention a head group of 32 at a time on the tensor cores (the window layers
/// over their cells in position order, the sparse ones over the indexer's pick, its scores on the tensor cores), the
/// router and the shared expert; the routed experts run on every GPU that holds some (MMQ over its slots), the ones no
/// GPU holds on the GPU with the fastest link, copied there from the arena.  The state it leaves is the decode's.
struct Prompt {
    Engine& E;
    Model* m = nullptr;
    int cap = 0;              // rows a chunk
    int min_rows = 32;        // a shorter part goes through the decode's forwards
    static constexpr int kG = hy4p::kQRows;   // heads a pass
    static constexpr int kEmax = 4096;        // a group's entries at most
    static constexpr int kStage = 32;         // experts a staging buffer
    static constexpr int kSub = 512;          // rows a pass of Engram's and the dense FFN's products
    int R = 0;                // the indexer's rows a pass
    int streamer = 0;         // the devs index that runs the experts no GPU holds
    cublasHandle_t blas = nullptr;
    cudaStream_t st = nullptr;
    cudaEvent_t routed = nullptr;
    // the main GPU's rows
    float *X = nullptr, *xh = nullptr, *xn = nullptr, *y = nullptr, *sy = nullptr, *emb = nullptr, *post = nullptr;
    uint8_t *xq = nullptr, *qrq = nullptr, *hdq = nullptr, *oq = nullptr, *ikq = nullptr, *subq = nullptr, *fhq = nullptr,
            *shq = nullptr, *mq = nullptr, *eraw = nullptr;
    float *qa = nullptr, *kva = nullptr, *qg = nullptr, *ql = nullptr, *gt = nullptr;
    float *iq = nullptr, *ik = nullptr, *iv = nullptr, *ig = nullptr, *iw = nullptr, *scores = nullptr;
    float *erows = nullptr, *ekv = nullptr, *evalid = nullptr, *fg = nullptr, *fu = nullptr, *fh = nullptr;
    float *logits_r = nullptr, *wts = nullptr, *rin = nullptr, *sg = nullptr, *su = nullptr, *sh = nullptr, *tmp = nullptr;
    int32_t *ids = nullptr, *cells = nullptr, *ncells = nullptr, *hids = nullptr, *hb = nullptr, *tok_d = nullptr,
            *pos_d = nullptr;
    int64_t hids_rows = 0;
    __half* lin = nullptr;
    // the host's (pinned)
    int32_t *h_ids = nullptr, *h_tok = nullptr, *h_pos = nullptr;
    float *h_wts = nullptr, *h_valid = nullptr;
    uint8_t *tq_h = nullptr, *h_raw = nullptr;
    uint32_t* h_rows = nullptr;
    std::vector<MoeDev> devs;   // [0] the main GPU, then the tiers in Engine::gpus' order
    size_t stage_bytes = 0;
    bool ok = false;
    double ms_moe = 0, ms_copy = 0;
    // LAMINA_HY4_PP_TRACE=1: every GPU waited for after each phase, the phases' sums printed after a chunk
    const bool trace = std::getenv("LAMINA_HY4_PP_TRACE") != nullptr;
    double ph[6] = {};
    Clock::time_point ph_t;
    void mark(int k) {
        if (!trace) return;
        for (const MoeDev& d : devs) {
            lamina::core::DeviceScope ds(d.dev, main_dev, d.gi > 0);
            CK(cudaDeviceSynchronize());
        }
        ph[k] += ms_since(ph_t);
        ph_t = Clock::now();
    }

    explicit Prompt(Engine& e) : E(e), m(e.m) {}

    /// The buffers of a chunk of `rows` on the main GPU and on each of `tier_devs` (before their slots take the rest);
    /// false (with a note) where MMQ cannot run one of the model's products.
    bool init(int rows, const std::vector<int>& tier_devs) {
        const auto& g = m->g;
        cap = std::max(kMaxT + 1, rows);
        for (const Layer& l : m->layers) {
            // every product's (type, rows) as the chunk runs it: the head-group slices, the per-head matrices
            std::vector<std::pair<int, int64_t>> need{{l.q_a.type, l.q_a.n_out}, {l.q_b.type, kG * g.k_head},
                                                      {l.kv_a.type, l.kv_a.n_out}, {l.wo.type, l.wo.n_out},
                                                      {l.gate.type, kG * g.v_head}, {l.k_b.type, hy4k::kLatent},
                                                      {l.v_b.type, g.v_head}};
            for (const Mat* w : {&l.iq_b, &l.ik, &l.icv, &l.icg, &l.e_kv, &l.f_gate, &l.f_up, &l.f_down, &l.r_down,
                                 &l.r_up, &l.s_gate, &l.s_up, &l.s_down})
                if (w->d) need.emplace_back(w->type, w->n_out);
            for (const auto& [t, r] : need)
                if (!MMQ::fits(t, r)) {
                    std::fprintf(stderr, "lamina: no MMQ for a [%lld rows] product of type %d: prompts read %d rows a "
                                 "forward\n", (long long) r, t, kMaxT);
                    return false;
                }
            if (!l.dense && (!MMQ::fits(l.fmt.gu_type, 2 * g.n_ff_exp) || !MMQ::fits(l.fmt.d_type, g.n_latent))) {
                std::fprintf(stderr, "lamina: no MMQ for the routed experts' types (%d, %d): prompts read %d rows a "
                             "forward\n", l.fmt.gu_type, l.fmt.d_type, kMaxT);
                return false;
            }
        }
        CK(cudaGetDevice(&main_dev));
        st = E.st;
        CK(cudaEventCreateWithFlags(&routed, cudaEventDisableTiming));
        own(&Owned::events, routed);
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(blas, st) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "lamina: cuBLAS did not start: prompts read %d rows a forward\n", kMaxT);
            return false;
        }
        const size_t T = (size_t) cap, N = (size_t) g.n_embd, hc = (size_t) g.hc, Ld = (size_t) g.n_latent,
                     F = (size_t) g.n_ff_exp, Ex = (size_t) g.n_expert, K = (size_t) g.top_k;
        size_t hmax = 0;
        for (const Layer& l : m->layers) hmax = std::max<size_t>(hmax, (size_t) l.H);
        X = dalloc<float>(T * hc * N);
        xh = dalloc<float>(T * N);
        xn = dalloc<float>(T * N);
        y = dalloc<float>(T * N);
        sy = dalloc<float>(T * N);
        emb = dalloc<float>(T * N);
        post = dalloc<float>(T * hc);
        xq = dalloc<uint8_t>(MMQ::q8_bytes((int64_t) T, (int64_t) N));
        qa = dalloc<float>(T * (size_t) g.q_lora);
        qrq = dalloc<uint8_t>(MMQ::q8_bytes((int64_t) T, g.q_lora));
        kva = dalloc<float>(T * (size_t) g.kv_row());
        qg = dalloc<float>(T * kG * (size_t) g.k_head);
        ql = dalloc<float>(T * kG * hy4k::kLatent);
        gt = dalloc<float>(T * kG * (size_t) g.v_head);
        hdq = dalloc<uint8_t>(MMQ::q8_bytes((int64_t) (kG * T), hy4k::kLatent));
        oq = dalloc<uint8_t>(MMQ::q8_bytes((int64_t) T, (int64_t) (hmax * g.v_head)));
        hids_rows = MMQ::ids_table_rows((int64_t) (kG * T), (int64_t) T);
        hids = dalloc<int32_t>((size_t) hids_rows);
        CK(cudaMemset(hids, 0, (size_t) hids_rows * 4));
        hb = dalloc<int32_t>(kG + 1);
        lin = dalloc<__half>(((size_t) g.n_swa + T) * hy4k::kKvRow);
        const size_t ih = (size_t) g.idx_heads, idd = (size_t) g.idx_dim, ld = (size_t) (E.ctx / g.idx_pool);
        iq = dalloc<float>(T * ih * idd);
        ik = dalloc<float>(T * idd);
        ikq = dalloc<uint8_t>(MMQ::q8_bytes((int64_t) T, (int64_t) idd));
        iv = dalloc<float>(T * idd);
        ig = dalloc<float>(T * idd);
        iw = dalloc<float>(T * ih);
        R = (int) std::max<size_t>(8, std::min<size_t>(T, ((size_t) 128 << 20) / (ld * 4)));
        scores = dalloc<float>((size_t) R * ld);
        cells = dalloc<int32_t>(T * (size_t) g.idx_top_k);
        ncells = dalloc<int32_t>(T);
        const size_t n_mod = g.eng_layers.size(), eh = (size_t) g.eng_heads(), ed = (size_t) g.eng_dim;
        eraw = dalloc<uint8_t>(n_mod * T * eh * ed);
        erows = dalloc<float>(T * eh * ed);
        evalid = dalloc<float>(T);
        const size_t sub = std::min<size_t>(T, kSub);
        size_t widest = std::max<size_t>(eh * ed, N);
        subq = dalloc<uint8_t>(MMQ::q8_bytes((int64_t) sub, (int64_t) std::max<size_t>(widest, (size_t) g.n_ff)));
        ekv = dalloc<float>(sub * (hc + 1) * N);
        fg = dalloc<float>(sub * (size_t) g.n_ff);
        fu = dalloc<float>(sub * (size_t) g.n_ff);
        fh = dalloc<float>(sub * (size_t) g.n_ff);
        logits_r = dalloc<float>(T * Ex);
        ids = dalloc<int32_t>(T * K);
        wts = dalloc<float>(T * K);
        rin = dalloc<float>(T * Ld);
        sg = dalloc<float>(T * F);
        su = dalloc<float>(T * F);
        sh = dalloc<float>(T * F);
        shq = dalloc<uint8_t>(MMQ::q8_bytes((int64_t) T, (int64_t) F));
        mq = dalloc<uint8_t>(MMQ::q8_bytes((int64_t) T, (int64_t) Ld));
        tmp = dalloc<float>(T * Ld);
        tok_d = dalloc<int32_t>(T);
        pos_d = dalloc<int32_t>(T);
        h_ids = halloc<int32_t>(T * K);
        h_wts = halloc<float>(T * K);
        h_tok = halloc<int32_t>(T);
        h_pos = halloc<int32_t>(T);
        h_valid = halloc<float>(T);
        h_rows = halloc<uint32_t>(n_mod * T * eh);
        h_raw = halloc<uint8_t>(n_mod * T * eh * ed);
        tq_h = halloc<uint8_t>(MMQ::q8_bytes((int64_t) T, (int64_t) Ld));
        for (const Layer& l : m->layers) stage_bytes = std::max(stage_bytes, l.stride);
        // the GPUs' shares: the main GPU, then the tiers
        std::vector<int> all{main_dev};
        all.insert(all.end(), tier_devs.begin(), tier_devs.end());
        const size_t plan_n = T * K * 6 + 16384;
        for (size_t i = 0; i < all.size(); ++i) {
            lamina::core::DeviceScope ds(all[i], main_dev, i > 0);
            MoeDev d;
            d.gi = (int) i;
            d.dev = all[i];
            if (i == 0) {
                d.st = st;
            } else {
                CK(cudaStreamCreateWithFlags(&d.st, cudaStreamNonBlocking));
                own(&Owned::streams, d.st);
            }
            CK(cudaStreamCreateWithFlags(&d.cst, cudaStreamNonBlocking));
            own(&Owned::streams, d.cst);
            for (cudaEvent_t* e : {&d.done, &d.copied[0], &d.copied[1], &d.used[0], &d.used[1]}) {
                CK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
                own(&Owned::events, *e);
            }
            CK(cudaEventRecord(d.used[0], d.st));
            CK(cudaEventRecord(d.used[1], d.st));
            d.mmq = std::make_unique<MMQ::Context>();
            d.tq = dalloc<uint8_t>(MMQ::q8_bytes((int64_t) T, (int64_t) Ld));
            d.eq = dalloc<uint8_t>(MMQ::q8_bytes(kEmax, (int64_t) Ld));
            d.gu = dalloc<float>((size_t) kEmax * 2 * F);
            d.h = dalloc<float>((size_t) kEmax * F);
            d.hq = dalloc<uint8_t>(MMQ::q8_bytes(kEmax, (int64_t) F));
            d.rows = dalloc<float>((size_t) kEmax * Ld);
            d.acc = dalloc<float>(T * Ld);
            d.ident_rows = MMQ::ids_table_rows(kEmax);
            d.ident = dalloc<int32_t>((size_t) d.ident_rows);
            MMQ::iota(d.ident, d.ident_rows, d.st);
            d.plan = dalloc<int32_t>(plan_n);
            d.plan_h = halloc<int32_t>(plan_n);
            if (i > 0) d.acc_h = halloc<float>(T * Ld);
            devs.push_back(std::move(d));
        }
        // the experts no GPU holds go to the GPU whose link copies fastest (core::probe_gpus: the links woken first)
        double best = 0;
        if (devs.size() > 1) {
            std::vector<int> ords;
            for (const MoeDev& d : devs) ords.push_back(d.dev);
            const std::vector<lamina::core::GpuProbe> pr = lamina::core::probe_gpus(ords, main_dev);
            for (size_t i = 0; i < pr.size(); ++i)
                if (pr[i].h2d_gbps > best) {
                    best = pr[i].h2d_gbps;
                    streamer = (int) i;
                }
        }
        {
            lamina::core::DeviceScope ds(devs[(size_t) streamer].dev, main_dev, streamer > 0);
            for (uint8_t*& s : devs[(size_t) streamer].stage) s = dalloc<uint8_t>((size_t) kStage * stage_bytes + 4096);
        }
        for (const MoeDev& d : devs) {
            lamina::core::DeviceScope ds(d.dev, main_dev, d.gi > 0);
            CK(cudaDeviceSynchronize());
        }
        CK(cudaFuncSetAttribute(hy4p::k_attn_prompt<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int) sizeof(hy4p::AttnSmem)));
        CK(cudaFuncSetAttribute(hy4p::k_attn_prompt<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int) sizeof(hy4p::AttnSmem)));
        std::fprintf(stderr, "lamina: prompts in chunks of up to %d rows; the experts no GPU holds on GPU %d%s\n", cap,
                     devs[(size_t) streamer].dev,
                     devs.size() > 1 ? (" (" + std::to_string((int) best) + " GB/s from pinned memory)").c_str() : "");
        ok = true;
        return true;
    }
    int main_dev = 0;

    unsigned blocks(int64_t n) const { return (unsigned) ((n + hy4k::kThreads - 1) / hy4k::kThreads); }
    void dense(const Mat& w, const uint8_t* x, int T, float* out, int64_t rows = 0, size_t row0 = 0, int64_t ld = 0) {
        const int64_t r = rows ? rows : w.n_out;
        const size_t row_bytes = (size_t) w.n_in / 32 * 34;
        devs[0].mmq->run_dense(w.d + row0 * row_bytes, w.type, r, w.n_in, x, T, out, ld ? ld : r, st);
    }
    void q8(const float* x, int64_t rows, int64_t cols, uint8_t* out, int type = 8, const int32_t* map = nullptr,
            int64_t ld = 0) {
        MMQ::quantize(x, map, out, type, cols, ld ? ld : cols, rows, st);
    }

    /// T rows (tokens toks at positions pos() ..) through the model; the last row's logits into the engine's logits row 0.
    void run(const int32_t* toks, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, hc = (int) g.hc;
        const int64_t p0 = E.pos();
        for (int t = 0; t < T; ++t) E.seq.push_back(toks[t]);
        const auto tt = Clock::now();
        const size_t n_mod = g.eng_layers.size(), eh = (size_t) g.eng_heads(), ed = (size_t) g.eng_dim;
        std::vector<uint32_t> r(n_mod * eh);
        for (int t = 0; t < T; ++t) {
            engram_ids(g, E.seq, p0 + t, r.data(), h_valid[t]);
            for (size_t e = 0; e < n_mod; ++e)
                std::memcpy(h_rows + (e * (size_t) T + (size_t) t) * eh, r.data() + e * eh, eh * 4);
            h_pos[t] = (int32_t) (p0 + t);
            h_tok[t] = toks[t];
        }
        const lamina::ngram::PleReader::Ticket ticket = E.reader.issue(h_rows, n_mod * (size_t) T * eh, h_raw);
        CK(cudaMemcpyAsync(tok_d, h_tok, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos_d, h_pos, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        hy4p::k_head_ids<<<(unsigned) ((kG * T + 255) / 256), 256, 0, st>>>(hids, kG, T);
        hy4p::k_head_bounds<<<1, 64, 0, st>>>(hb, kG, T);
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, g.n_vocab, tok_d, T, N, emb, st);
        hy4k::k_streams_init<<<T, hy4k::kThreads, 0, st>>>(emb, X, hc, N);
        std::string err;
        if (!E.reader.collect(ticket, err)) {
            std::fprintf(stderr, "lamina: the Engram table: %s\n", err.c_str());
            throw Fatal{1};
        }
        E.table_ms += ms_since(tt);
        CK(cudaMemcpyAsync(eraw, h_raw, n_mod * (size_t) T * eh * ed, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(evalid, h_valid, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        std::fill(std::begin(ph), std::end(ph), 0.0);
        ph_t = Clock::now();
        const auto c0 = Clock::now();
        for (Layer& l : m->layers) {
            front(l, T, p0);
            mark(0);
            if (l.dense) dense_ffn(l, T);
            else moe(l, T);
            mark(5);
        }
        // the head over the last row
        const float* Xl = X + (size_t) (T - 1) * hc * N;
        hy4k::k_hc_pre<<<1, 256, 0, st>>>(Xl, m->ohc_fn, m->ohc_scale, m->ohc_base, hc, N, hc, g.eps, g.hc_eps, 0.f,
                                          nullptr, xh, m->out_norm, xn, g.eps);
        K::bf16_gemv_fp32_mmvf_multi(xn, m->head, E.logits, N, g.n_vocab, 1, st);
        ++E.forwards;
        if (trace) {
            mark(5);
            std::fprintf(stderr, "lamina: chunk of %d rows at %lld: %.0f ms; attention and indexer %.0f, routing %.0f, plan "
                         "%.0f, experts %.0f, the rest %.0f\n", T, (long long) p0, ms_since(c0), ph[0], ph[1], ph[2], ph[3],
                         ph[5]);
        }
    }

    /// Engram, the attention, and the FFN's input (xn, its q8_1 rows in xq; iHC's post weights in `post`)
    void front(Layer& l, int T, int64_t p0) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, hc = (int) g.hc, H = l.H;
        if (l.eng >= 0) {
            const int ehn = (int) g.eng_heads(), edn = (int) g.eng_dim, w = ehn * edn;
            hy4k::k_engram_rows<<<dim3(ehn, T), 128, 0, st>>>(eraw + (size_t) l.eng * T * w, l.e_lnorm, evalid, erows, ehn,
                                                               edn, kEngramEps);
            for (int r0 = 0; r0 < T; r0 += kSub) {
                const int n = std::min(kSub, T - r0);
                q8(erows + (size_t) r0 * w, n, w, subq);
                dense(l.e_kv, subq, n, ekv);
                hy4k::k_engram_mix<<<n, 256, 0, st>>>(X + (size_t) r0 * hc * N, ekv, l.e_kv_b, l.e_gate, l.e_gamma,
                                                      l.e_hcs, evalid + r0, hc, N, kEngramEps);
            }
        }
        hy4k::k_hc_pre<<<T, 256, 0, st>>>(X, l.hca_fn, l.hca_scale, l.hca_base, hc, N, 2 * hc, g.eps, g.hc_eps,
                                          g.hc_mag, post, xh, l.attn_norm, xn, g.eps);
        q8(xn, T, N, xq);
        dense(l.q_a, xq, T, qa);
        hy4k::k_rms<<<T, hy4k::kThreads, 0, st>>>(qa, (int) g.q_lora, l.q_a_norm, qa, (int) g.q_lora, (int) g.q_lora, g.eps);
        q8(qa, T, g.q_lora, qrq);
        dense(l.kv_a, xq, T, kva);
        const int64_t p_end = p0 + T;
        const int ring = (int) E.ring, n_swa = (int) g.n_swa;
        int lin0 = 0;
        if (l.swa) {
            // the window's cells in position order: the ring's before the chunk, then the chunk's; the ring keeps the last
            lin0 = (int) std::max<int64_t>(0, p0 - n_swa + 1);
            if (p0 > lin0) hy4p::k_copy_cells<<<(unsigned) (p0 - lin0), 64, 0, st>>>(l.kv, ring, lin, 0, lin0, lin0);
            hy4k::k_kv_store<<<T, hy4k::kThreads, 0, st>>>(kva, l.kv_a_norm, lin, pos_d, 0, lin0, l.theta, g.eps);
            const int64_t pa = std::max<int64_t>(p0, p_end - ring);
            hy4p::k_copy_cells<<<(unsigned) (p_end - pa), 64, 0, st>>>(lin, 0, l.kv, ring, (int) pa, lin0);
        } else {
            hy4k::k_kv_store<<<T, hy4k::kThreads, 0, st>>>(kva, l.kv_a_norm, l.kv, pos_d, 0, 0, l.theta, g.eps);
            indexer(l, T, p0);
        }
        const float scale = 1.0f / sqrtf((float) g.k_head);
        const size_t kb_bytes = (size_t) hy4k::kLatent * ((size_t) g.nope() / 32 * 34);
        const size_t vb_bytes = (size_t) g.v_head * ((size_t) hy4k::kLatent / 32 * 34);
        for (int h0 = 0; h0 < H; h0 += kG) {
            dense(l.q_b, qrq, T, qg, kG * g.k_head, (size_t) h0 * g.k_head);
            hy4k::k_rope_norm<<<dim3(kG, T), 32, 0, st>>>(qg, kG, (int) g.k_head, (int) g.nope(), (int) g.n_rot, pos_d, 0,
                                                          l.theta);
            q8(qg, (int64_t) kG * T, g.nope(), hdq, 8, hids, g.k_head);
            MMQ::Product kb;
            kb.w = l.k_b.d + (size_t) h0 * kb_bytes;
            kb.type = l.k_b.type;
            kb.w_rows = hy4k::kLatent;
            kb.w_cols = g.nope();
            kb.expert_bytes = kb_bytes;
            kb.n = kG;
            kb.xq = hdq;
            kb.bounds = hb;
            kb.ids = hids;
            kb.ids_rows = hids_rows;
            kb.total_rows = (int64_t) kG * T;
            kb.max_rows = T;
            kb.dst = ql;
            kb.ld_dst = hy4k::kLatent;
            devs[0].mmq->run(kb, st);
            if (l.swa)
                hy4p::k_attn_prompt<false><<<T, 256, sizeof(hy4p::AttnSmem), st>>>(ql, qg, lin, nullptr, nullptr, 0, pos_d,
                                                                                  n_swa, lin0, l.sinks + h0, scale);
            else
                hy4p::k_attn_prompt<true><<<T, 256, sizeof(hy4p::AttnSmem), st>>>(ql, qg, l.kv, cells, ncells,
                                                                                 (int) g.idx_top_k, pos_d, 0, 0,
                                                                                 l.sinks + h0, scale);
            q8(ql, (int64_t) kG * T, hy4k::kLatent, hdq, 8, hids);
            MMQ::Product vb = kb;
            vb.w = l.v_b.d + (size_t) h0 * vb_bytes;
            vb.type = l.v_b.type;
            vb.w_rows = g.v_head;
            vb.w_cols = hy4k::kLatent;
            vb.expert_bytes = vb_bytes;
            vb.dst = qg;
            vb.ld_dst = g.v_head;
            devs[0].mmq->run(vb, st);
            dense(l.gate, xq, T, gt, kG * g.v_head, (size_t) h0 * g.v_head);
            const int64_t n_o = (int64_t) T * kG * g.v_head;
            hy4k::k_gate_mul<<<blocks(n_o), hy4k::kThreads, 0, st>>>(qg, gt, n_o);
            // into the wo input's column blocks of these heads
            q8(qg, T, (int64_t) kG * g.v_head, oq + (size_t) h0 * g.v_head / 128 * T * kQ8Block);
        }
        dense(l.wo, oq, T, y);
        hy4k::k_rms_inplace<<<T, hy4k::kThreads, 0, st>>>(y, l.post_attn, N, g.eps);
        hy4k::k_hc_post<<<T, hy4k::kThreads, 0, st>>>(X, y, post, hc, N);
        hy4k::k_hc_pre<<<T, 256, 0, st>>>(X, l.hcf_fn, l.hcf_scale, l.hcf_base, hc, N, 2 * hc, g.eps, g.hc_eps,
                                          g.hc_mag, post, xh, l.ffn_norm, xn, g.eps);
        q8(xn, T, N, xq);
    }

    /// The sparse layer's indexer over a chunk: its rows' keys pooled into the groups they complete (the tail's cells
    /// before the chunk), the tail's last cells, each row's cells (every position while its visible groups are within
    /// the reach, else the top groups' by the tensor-core scores, R rows a pass).
    void indexer(Layer& l, int T, int64_t p0) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, ih = (int) g.idx_heads, idd = (int) g.idx_dim, pool_n = (int) g.idx_pool;
        dense(l.iq_b, qrq, T, iq);
        hy4k::k_rope_norm<<<dim3(ih, T), 32, 0, st>>>(iq, ih, idd, idd - (int) g.n_rot, (int) g.n_rot, pos_d, 0, l.theta);
        dense(l.ik, xq, T, ik);
        hy4k::k_rms<<<T, 128, 0, st>>>(ik, idd, l.ik_norm, ik, idd, idd, g.eps);
        q8(ik, T, idd, ikq);
        dense(l.icv, ikq, T, iv);
        dense(l.icg, ikq, T, ig);
        const int64_t p_end = p0 + T;
        const int64_t g_a = (p0 + pool_n) / pool_n - 1, g_b = p_end / pool_n;   // the groups the chunk completes
        if (g_b > g_a)
            hy4p::k_pool_chunk<<<(unsigned) (g_b - g_a), idd, 0, st>>>(iv, ig, l.tail, kTail, (int) p0, (int) g_a, idd,
                                                                      pool_n, l.ape, l.ipool_norm, g.eps, l.theta,
                                                                      (int) g.n_rot, l.pooled);
        const int keep = std::min(T, kTail);
        hy4k::k_tail_store<<<keep, 128, 0, st>>>(iv + (size_t) (T - keep) * idd, ig + (size_t) (T - keep) * idd, l.tail,
                                                 pos_d + (T - keep), kTail, idd);
        hy4k::k_gemv_f32<<<dim3((unsigned) ((ih + 7) / 8), T), 256, 0, st>>>(l.iproj, xn, iw, N, ih,
                                                                           1.0f / sqrtf((float) (idd * ih)));
        const int reach = (int) (g.idx_top_k / pool_n - (g.idx_tail ? 1 : 0));
        const int ld = (int) (E.ctx / pool_n);
        for (int r0 = 0; r0 < T; r0 += R) {
            const int rr = std::min(R, T - r0);
            const int vis_max = (int) ((p0 + r0 + rr - 1) / pool_n);
            static const bool f32 = std::getenv("LAMINA_HY4_IDX_F32") != nullptr;   // the decode's FP32 scores (A/B)
            if (vis_max > reach && f32) {
                const size_t smem = ((size_t) ih * idd + ih) * sizeof(float);
                hy4k::k_index_scores_pos<<<dim3(kScoreBlocks, rr), 256, smem, st>>>(
                    iq + (size_t) r0 * ih * idd, iw + (size_t) r0 * ih, l.pooled, pos_d + r0, pool_n, reach, ih, idd,
                    scores, ld);
            } else if (vis_max > reach) {
                const int splits = std::max(1, std::min(64, (vis_max + 1023) / 1024));
                const int per = (vis_max + splits - 1) / splits / hy4p::kIdxTile * hy4p::kIdxTile + hy4p::kIdxTile;
                hy4p::k_index_scores_mma<<<dim3((unsigned) ((rr + 7) / 8), (unsigned) splits), 256, 0, st>>>(
                    iq, iw, l.pooled, pos_d, pool_n, reach, r0, rr, per, scores, ld);
            }
            hy4k::k_select_cells<<<rr, 1024, 0, st>>>(scores, ld, pos_d + r0, pool_n, reach,
                                                      cells + (size_t) r0 * g.idx_top_k, (int) g.idx_top_k, ncells + r0);
        }
    }

    void dense_ffn(Layer& l, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, hc = (int) g.hc, FF = (int) g.n_ff;
        for (int r0 = 0; r0 < T; r0 += kSub) {
            const int n = std::min(kSub, T - r0);
            q8(xn + (size_t) r0 * N, n, N, subq);
            dense(l.f_gate, subq, n, fg);
            dense(l.f_up, subq, n, fu);
            hy4k::k_swiglu<<<blocks((int64_t) n * FF), hy4k::kThreads, 0, st>>>(fg, fu, fh, (int64_t) n * FF);
            q8(fh, n, FF, subq);
            dense(l.f_down, subq, n, y + (size_t) r0 * N);
        }
        hy4k::k_rms_inplace<<<T, hy4k::kThreads, 0, st>>>(y, l.post_ffw, N, g.eps);
        hy4k::k_hc_post<<<T, hy4k::kThreads, 0, st>>>(X, y, post, hc, N);
    }

    // ---- the routed experts.  The host takes each layer's routing, hands every entry to a GPU (its expert's slot, else
    // the streamer, which copies the expert from the arena) and plans each GPU's groups: experts of consecutive blobs
    // whose entries (up to kEmax) run as one MMQ product, their rows summed into the tokens' sums in pick order.
    struct Group {
        const uint8_t* w = nullptr;   // the first expert's blob on its GPU
        int n = 0, n_ent = 0, max_rows = 0, n_runs = 0;
        int o_bounds = 0, o_tok = 0, o_rtok = 0, o_rstart = 0, o_rent = 0, o_rw = 0;   // into the plan (int32)
    };
    struct Batch {                    // the streamer's experts copied into one staging buffer, and its groups
        std::vector<int32_t> experts;
        int g0 = 0, g1 = 0;
    };
    struct DevPlan {
        std::vector<int32_t> P;
        std::vector<Group> groups;    // the batches' groups first, then the slots'
        std::vector<Batch> batches;
        int n_slot_groups = 0;
    };
    std::vector<DevPlan> plans;
    std::vector<std::vector<std::vector<int32_t>>> slot_ents;   // [dev][slot]: entries
    std::vector<std::vector<int32_t>> free_ents;                // [expert]: entries no GPU holds

    /// Groups over (blob index, entries) in ascending blob order from `base` (`stride` apart), split at kEmax entries
    void add_groups(DevPlan& dp, const uint8_t* base, size_t stride,
                    const std::vector<std::pair<int, const std::vector<int32_t>*>>& seq) {
        const int K = (int) m->g.top_k;
        std::vector<int32_t>& P = dp.P;
        size_t i = 0, off = 0;
        std::vector<int32_t> ents, cnt;
        std::vector<int> loc;
        while (i < seq.size()) {
            ents.clear();
            cnt.clear();
            const int ja = seq[i].first;
            while (i < seq.size() && (int) ents.size() < kEmax) {
                const int j = seq[i].first;
                const std::vector<int32_t>& v = *seq[i].second;
                const size_t take = std::min<size_t>((size_t) kEmax - ents.size(), v.size() - off);
                cnt.resize((size_t) (j - ja + 1), 0);
                cnt[(size_t) (j - ja)] += (int32_t) take;
                ents.insert(ents.end(), v.begin() + (ptrdiff_t) off, v.begin() + (ptrdiff_t) (off + take));
                off += take;
                if (off < v.size()) break;   // the rest of this expert's entries: the next group
                ++i;
                off = 0;
            }
            Group gr;
            gr.w = base + (size_t) ja * stride;
            gr.n = (int) cnt.size();
            gr.n_ent = (int) ents.size();
            gr.max_rows = *std::max_element(cnt.begin(), cnt.end());
            gr.o_bounds = (int) P.size();
            int32_t acc = 0;
            P.push_back(0);
            for (const int32_t c : cnt) P.push_back(acc += c);
            gr.o_tok = (int) P.size();
            for (const int32_t e : ents) P.push_back(e / K);
            // the runs: the group's entries by token, in pick order
            loc.resize(ents.size());
            for (size_t k = 0; k < loc.size(); ++k) loc[k] = (int) k;
            std::sort(loc.begin(), loc.end(), [&](int a, int b) { return ents[(size_t) a] < ents[(size_t) b]; });
            std::vector<int32_t> rtok, rstart;
            for (size_t k = 0; k < loc.size(); ++k) {
                const int32_t tk = ents[(size_t) loc[k]] / K;
                if (rtok.empty() || rtok.back() != tk) {
                    rtok.push_back(tk);
                    rstart.push_back((int32_t) k);
                }
            }
            rstart.push_back((int32_t) loc.size());
            gr.n_runs = (int) rtok.size();
            gr.o_rtok = (int) P.size();
            P.insert(P.end(), rtok.begin(), rtok.end());
            gr.o_rstart = (int) P.size();
            P.insert(P.end(), rstart.begin(), rstart.end());
            gr.o_rent = (int) P.size();
            for (const int k : loc) P.push_back(k);
            gr.o_rw = (int) P.size();
            for (const int k : loc) {
                int32_t bits;
                std::memcpy(&bits, &h_wts[ents[(size_t) k]], 4);
                P.push_back(bits);
            }
            dp.groups.push_back(gr);
        }
    }

    void plan_layer(Layer& l, int T) {
        const auto& g = m->g;
        const int K = (int) g.top_k, nE = (int) g.n_expert;
        const size_t nd = devs.size();
        plans.assign(nd, DevPlan{});
        slot_ents.resize(nd);
        for (size_t d = 0; d < nd; ++d) {
            slot_ents[d].resize((size_t) l.held[d].n);
            for (auto& v : slot_ents[d]) v.clear();
        }
        free_ents.resize((size_t) nE);
        for (auto& v : free_ents) v.clear();
        int64_t on_gpus = 0;
        for (int e = 0; e < T * K; ++e) {
            const int x = h_ids[e];
            l.use[(size_t) x] += 1.f;
            l.total[(size_t) x] += 1.f;
            int holder = -1;
            for (size_t d = 0; d < nd && holder < 0; ++d)
                if (l.held[d].n > 0 && l.held[d].slot_of[(size_t) x] >= 0) holder = (int) d;
            if (holder >= 0) {
                slot_ents[(size_t) holder][(size_t) l.held[(size_t) holder].slot_of[(size_t) x]].push_back(e);
                ++E.gpus[(size_t) holder].hits;
                ++on_gpus;
            } else {
                free_ents[(size_t) x].push_back(e);
            }
        }
        E.lookups += (int64_t) T * K;
        E.hits += on_gpus;
        std::vector<std::pair<int, const std::vector<int32_t>*>> seq;
        for (size_t d = 0; d < nd; ++d) {
            DevPlan& dp = plans[d];
            if ((int) d == streamer) {   // the experts no GPU holds, kStage a staging buffer
                std::vector<int32_t> used;
                for (int x = 0; x < nE; ++x)
                    if (!free_ents[(size_t) x].empty()) used.push_back(x);
                for (size_t b0 = 0; b0 < used.size(); b0 += kStage) {
                    Batch bt;
                    bt.experts.assign(used.begin() + (ptrdiff_t) b0,
                                      used.begin() + (ptrdiff_t) std::min(used.size(), b0 + kStage));
                    seq.clear();
                    for (size_t j = 0; j < bt.experts.size(); ++j)
                        seq.emplace_back((int) j, &free_ents[(size_t) bt.experts[j]]);
                    bt.g0 = (int) dp.groups.size();
                    add_groups(dp, devs[d].stage[dp.batches.size() % 2], l.stride, seq);
                    bt.g1 = (int) dp.groups.size();
                    dp.batches.push_back(std::move(bt));
                }
            }
            seq.clear();
            for (int s = 0; s < l.held[d].n; ++s)
                if (!slot_ents[d][(size_t) s].empty()) seq.emplace_back(s, &slot_ents[d][(size_t) s]);
            const size_t g_before = dp.groups.size();
            add_groups(dp, l.held[d].slots, l.stride, seq);
            dp.n_slot_groups = (int) (dp.groups.size() - g_before);
        }
    }

    void run_group(MoeDev& d, Layer& l, int T, const Group& gr) {
        const auto& g = m->g;
        const int Ld = (int) g.n_latent, F = (int) g.n_ff_exp;
        hy4p::k_gather_q8<<<gr.n_ent, 128, 0, d.st>>>(reinterpret_cast<const uint4*>(d.tq), reinterpret_cast<uint4*>(d.eq),
                                                      d.plan + gr.o_tok, gr.n_ent, T, Ld / 128);
        MMQ::Product gu;
        gu.w = gr.w;
        gu.type = l.fmt.gu_type;
        gu.w_rows = 2 * F;
        gu.w_cols = Ld;
        gu.expert_bytes = l.stride;
        gu.n = gr.n;
        gu.xq = d.eq;
        gu.bounds = d.plan + gr.o_bounds;
        gu.ids = d.ident;
        gu.ids_rows = d.ident_rows;
        gu.total_rows = gr.n_ent;
        gu.max_rows = gr.max_rows;
        gu.dst = d.gu;
        gu.ld_dst = 2 * F;
        d.mmq->run(gu, d.st);
        const int64_t nh = (int64_t) gr.n_ent * F;
        hy4p::k_swiglu_split<<<blocks(nh), hy4k::kThreads, 0, d.st>>>(d.gu, d.h, F, nh, l.fmt.swiglu_limit);
        MMQ::quantize(d.h, nullptr, d.hq, l.fmt.d_type, F, F, gr.n_ent, d.st);
        MMQ::Product dn = gu;
        dn.w = gr.w + l.fmt.down_off;
        dn.type = l.fmt.d_type;
        dn.w_rows = Ld;
        dn.w_cols = F;
        dn.xq = d.hq;
        dn.dst = d.rows;
        dn.ld_dst = Ld;
        d.mmq->run(dn, d.st);
        hy4p::k_accum_runs<<<gr.n_runs, 256, 0, d.st>>>(d.rows, d.plan + gr.o_rtok, d.plan + gr.o_rstart, d.plan + gr.o_rent,
                                                        reinterpret_cast<const float*>(d.plan + gr.o_rw), d.acc, Ld);
    }

    void enqueue(MoeDev& d, Layer& l, int T) {
        const auto& g = m->g;
        const int Ld = (int) g.n_latent;
        DevPlan& dp = plans[(size_t) d.gi];
        lamina::core::DeviceScope ds(d.dev, main_dev, d.gi > 0);
        std::memcpy(d.plan_h, dp.P.data(), dp.P.size() * 4);
        CK(cudaMemcpyAsync(d.plan, d.plan_h, dp.P.size() * 4, cudaMemcpyHostToDevice, d.st));
        CK(cudaMemsetAsync(d.acc, 0, (size_t) T * Ld * 4, d.st));
        if (d.gi > 0) CK(cudaMemcpyAsync(d.tq, tq_h, MMQ::q8_bytes(T, Ld), cudaMemcpyHostToDevice, d.st));
        const size_t n_batch_groups = dp.groups.size() - (size_t) dp.n_slot_groups;
        for (size_t gi = n_batch_groups; gi < dp.groups.size(); ++gi) run_group(d, l, T, dp.groups[gi]);
        for (size_t b = 0; b < dp.batches.size(); ++b) {
            const Batch& bt = dp.batches[b];
            const int buf = (int) (b % 2);
            CK(cudaStreamWaitEvent(d.cst, d.used[buf], 0));
            for (size_t j = 0; j < bt.experts.size(); ++j)
                CK(cudaMemcpyAsync(d.stage[buf] + j * l.stride, l.arena + (size_t) bt.experts[j] * l.stride, l.fmt.bytes,
                                   cudaMemcpyHostToDevice, d.cst));
            CK(cudaEventRecord(d.copied[buf], d.cst));
            CK(cudaStreamWaitEvent(d.st, d.copied[buf], 0));
            for (int gi = bt.g0; gi < bt.g1; ++gi) run_group(d, l, T, dp.groups[(size_t) gi]);
            CK(cudaEventRecord(d.used[buf], d.st));
        }
        if (d.gi > 0) {
            CK(cudaMemcpyAsync(d.acc_h, d.acc, (size_t) T * Ld * 4, cudaMemcpyDeviceToHost, d.st));
            CK(cudaEventRecord(d.done, d.st));
        }
    }

    void moe(Layer& l, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, hc = (int) g.hc, nE = (int) g.n_expert, K = (int) g.top_k, Ld = (int) g.n_latent,
                  F = (int) g.n_ff_exp;
        const float one = 1.f, zero = 0.f;
        if (cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, nE, T, N, &one, l.router, N, xn, N, &zero, logits_r, nE) !=
            CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "lamina: cublasSgemm (the router) failed\n");
            throw Fatal{1};
        }
        hy4k::k_route<<<T, std::max(32, ((nE + 31) / 32) * 32), 0, st>>>(logits_r, l.bias, nE, K, g.w_norm, g.w_scale, ids,
                                                                          wts);
        dense(l.r_down, xq, T, rin);
        MoeDev& d0 = devs[0];
        q8(rin, T, Ld, d0.tq, l.fmt.gu_type);
        CK(cudaMemcpyAsync(h_ids, ids, (size_t) T * K * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(h_wts, wts, (size_t) T * K * 4, cudaMemcpyDeviceToHost, st));
        if (devs.size() > 1) CK(cudaMemcpyAsync(tq_h, d0.tq, MMQ::q8_bytes(T, Ld), cudaMemcpyDeviceToHost, st));
        CK(cudaEventRecord(routed, st));
        CK(cudaEventSynchronize(routed));
        mark(1);
        const auto t0 = Clock::now();
        plan_layer(l, T);
        mark(2);
        if (streamer != 0) enqueue(d0, l, T);
        for (size_t d = 1; d < devs.size(); ++d)
            if ((int) d != streamer) enqueue(devs[d], l, T);
        enqueue(devs[(size_t) streamer], l, T);
        // the shared expert
        dense(l.s_gate, xq, T, sg);
        dense(l.s_up, xq, T, su);
        hy4k::k_swiglu<<<blocks((int64_t) T * F), hy4k::kThreads, 0, st>>>(sg, su, sh, (int64_t) T * F);
        q8(sh, T, F, shq);
        dense(l.s_down, shq, T, sy);
        // the tiers' sums, in their order
        for (size_t d = 1; d < devs.size(); ++d) {
            CK(cudaStreamWaitEvent(st, devs[d].done, 0));
            CK(cudaMemcpyAsync(tmp, devs[d].acc_h, (size_t) T * Ld * 4, cudaMemcpyHostToDevice, st));
            hy4k::k_add<<<blocks((int64_t) T * Ld), hy4k::kThreads, 0, st>>>(d0.acc, tmp, (int64_t) T * Ld);
        }
        ms_moe += ms_since(t0);
        mark(3);
        q8(d0.acc, T, Ld, mq);
        dense(l.r_up, mq, T, y);
        hy4k::k_add<<<blocks((int64_t) T * N), hy4k::kThreads, 0, st>>>(y, sy, (int64_t) T * N);
        hy4k::k_rms_inplace<<<T, hy4k::kThreads, 0, st>>>(y, l.post_ffw, N, g.eps);
        hy4k::k_hc_post<<<T, hy4k::kThreads, 0, st>>>(X, y, post, hc, N);
    }
};

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
        auto common_of = [&](const std::vector<int32_t>& s) {
            size_t c = 0;
            while (c < s.size() && c < ids.size() && s[c] == ids[c]) ++c;
            return c;
        };
        size_t common = common_of(E.seq);
        // a parked conversation the prompt continues further than the live one: it comes back, and the live one is
        // parked when the request leaves it (it drops kParkMin or more of its cells and more than it keeps)
        size_t best = 0, best_c = common;
        for (size_t i = 0; i < E.park.size(); ++i)
            if (const size_t c = common_of(E.park[i].seq); c > best_c) {
                best = i + 1;
                best_c = c;
            }
        const int64_t drop = E.pos() - (int64_t) common;
        if (best > 0) {
            Engine::Parked back = std::move(E.park[best - 1]);
            E.park.erase(E.park.begin() + (ptrdiff_t) (best - 1));
            E.park_live(0);
            E.unpark(std::move(back));
            common = common_of(E.seq);
        } else if (drop >= Engine::kParkMin && drop > (int64_t) common) {
            E.park_live((int64_t) common);
        }
        std::vector<float> lg;
        const Engine::Snap* whole = common == ids.size() ? E.snap_at((int64_t) common) : nullptr;
        if (whole != nullptr && whole->logits.empty()) whole = nullptr;
        if (whole) lg = whole->logits;   // the held prompt again: its snapshot has the first pick's logits
        else if (common == ids.size()) --common;
        const size_t reuse = (size_t) E.rewind((int64_t) common);
        if (reuse > 0) serve_print("RESUME %zu\n", reuse);
        const auto t0 = Clock::now();
        const int64_t fresh = (int64_t) (ids.size() - reuse);
        int64_t done = 0;
        auto last_pp = Clock::now();
        const int last = E.feed(ids.data() + reuse, fresh, [&](int64_t d) {
            done = d;
            if (done == fresh || ms_since(last_pp) > 250) {
                serve_print("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) fresh, ms_since(t0),
                            (double) done / std::max(1e-9, ms_since(t0) / 1000.0));
                last_pp = Clock::now();
            }
            return !stop.load();
        });
        const bool cancelled = last < 0;
        const double prompt_ms = ms_since(t0);
        if (cancelled) {
            serve_print("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0.0 %lld 0\n", ids.size(), prompt_ms, reuse,
                        (long long) done);
            return;
        }
        if (fresh > 0) {
            E.host_logits(last, lg);
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
        const auto r0 = Clock::now();
        const int swaps = E.rebalance(512, [&] {
            std::lock_guard<std::mutex> lk(mu);
            return eof || !lines.empty();
        });
        E.save_profile(false);
        if (swaps > 0) std::fprintf(stderr, "lamina: %d experts moved into VRAM slots in %.0f ms\n", swaps, ms_since(r0));
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
        // Ctrl+C in the console ends the engine as QUIT does, before any request queued (the running one stops at its
        // next token), then the exit work runs (the learned profile).  In-process the host handles it.
        std::optional<lamina::platform::ConsoleQuit> console_quit;
        if (!io.in_process)
            console_quit.emplace([&] {
                stop.store(true);
                std::lock_guard<std::mutex> lk(mu);
                lines.push_front("QUIT");
                cv.notify_one();
            });
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
    std::string profile, profile_save;
    double save_every = 1.0;
    bool graphs = true;
    int prefill = 4096, prefill_min = 24;
    double arena_pin = 72;
    double park_mib = 16384, reserve_mib = 6144;
    std::vector<int> tier_devs;
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
        else if (a == "--expert-profile-save") profile_save = next();
        else if (a == "--expert-profile-save-every") save_every = std::atof(next());
        else if (a == "--no-graph") graphs = false;
        else if (a == "--prefill") prefill = std::atoi(next());            // rows a prompt chunk (0: the decode's forwards)
        else if (a == "--prefill-min") prefill_min = std::atoi(next());    // a shorter prompt part: the decode's forwards
        else if (a == "--arena-pin") arena_pin = std::atof(next());        // GiB of the arena in pinned memory at most
        else if (a == "--park-mib" || a == "--conversation-cache-mib") park_mib = std::atof(next());
        else if (a == "--ram-reserve-mib") reserve_mib = std::atof(next());
        else if (a == "--tier-gpus") {   // the expert tiers' GPUs (CUDA ordinals, the family switch's)
            for (const int32_t d : lamina::program::parse_ids(next())) tier_devs.push_back((int) d);
        }
        else {
            std::fprintf(stderr, "lamina: Hy4-flash's path does not take %s\n", a.c_str());
            return 2;
        }
    }
    if (model.empty()) { std::fprintf(stderr, "lamina: --native MODEL is required\n"); return 2; }
    if (check_only) return 0;
    Model m;
    std::string err;
    if (!load(model, m, 8, arena_pin, err)) {
        std::fprintf(stderr, "lamina: %s\n", err.c_str());
        return 1;
    }
    Engine E;
    E.profile_path = profile;
    E.save_path = profile_save;
    E.save_every_min = save_every;
    E.use_graphs = graphs;
    E.init(m, max_ctx, threads);
    E.park_budget = (size_t) (std::max(0.0, park_mib) * (double) (1 << 20));
    E.ram_reserve = (size_t) (std::max(0.0, reserve_mib) * (double) (1 << 20));
    Prompt P(E);   // its buffers before the slots take the GPUs' free memory
    if (prefill > 0 && P.init(prefill, tier_devs)) {
        E.chunk_fn = [&P](const int32_t* t, int n) { P.run(t, n); };
        E.chunk_cap = P.cap;
        E.chunk_min = std::max(1, prefill_min);
    }
    E.place_slots(expert_vram, tier_devs);
    if (serve) {
        Serve sv(E, m.eos);
        const int rc = sv.run();
        E.save_profile(true);
        return rc;
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
    const int last = E.feed(prompt.data(), (int64_t) prompt.size(), nullptr);
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
    for (const Engine::Gpu& gp : E.gpus)
        std::fprintf(stderr, "lamina: GPU %d, %s: %.1f%% of the routed entries\n", gp.dev, gp.name.c_str(),
                     E.lookups ? 100.0 * (double) gp.hits / (double) E.lookups : 0.0);
    E.save_profile(true);
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
