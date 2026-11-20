// src/vision/ggml_encoder.cpp - the image encoder as a ggml graph (include/lamina/vision/encoder.hpp).
//
// What llama.cpp's mtmd does for these two projector types at the pinned commit, kept operation for operation and in
// its order (tools/mtmd/clip.cpp, models/qwen3vl.cpp, models/hunyuanvl.cpp, MIT; Hy4's adapter and limits from pull
// request 32517's models/hy-v4-flash.cpp): the weights in their GGUF types on the backend, a graph built per picture
// for its size, ggml's scheduler placing each operation on the GPU or the CPU.  The graph's nodes are added in mtmd's
// order, so the backend's kernels see the same work and the rows come out the same.
#include "lamina/vision/encoder.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace lamina::vision {
namespace {

void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

// ggml's clock, its FP16 tables and its backends, once (what llama_backend_init does); its messages from warnings up
void backends_init() {
    static const bool once = [] {
        ggml_log_set(quiet_log, nullptr);
        ggml_time_init();
        ggml_init_params ip = {0, nullptr, false};
        ggml_free(ggml_init(ip));
        if (ggml_backend_reg_count() == 0) ggml_backend_load_all();
        return true;
    }();
    (void) once;
}

enum class Kind { qwen3vl, hy4 };

struct Layer {
    ggml_tensor* ln_1_w = nullptr;
    ggml_tensor* ln_1_b = nullptr;
    ggml_tensor* ln_2_w = nullptr;
    ggml_tensor* ln_2_b = nullptr;
    ggml_tensor* qkv_w = nullptr;   // the three projections as one matrix (Qwen's files) ...
    ggml_tensor* qkv_b = nullptr;
    ggml_tensor* q_w = nullptr;     // ... or each its own (Hy4's)
    ggml_tensor* q_b = nullptr;
    ggml_tensor* k_w = nullptr;
    ggml_tensor* k_b = nullptr;
    ggml_tensor* v_w = nullptr;
    ggml_tensor* v_b = nullptr;
    ggml_tensor* o_w = nullptr;
    ggml_tensor* o_b = nullptr;
    ggml_tensor* ff_up_w = nullptr;
    ggml_tensor* ff_up_b = nullptr;
    ggml_tensor* ff_down_w = nullptr;
    ggml_tensor* ff_down_b = nullptr;
};

struct Model {
    Kind kind = Kind::qwen3vl;
    int n_embd = 0, n_head = 0, n_embd_head = 0, n_head_kv = 0, n_layer = 0, projection_dim = 0;
    int patch_size = 0, n_merge = 2;
    float eps = 0.0f;
    float mean[3] = {}, std[3] = {};
    // the picture's sizing (image.hpp)
    SizeRule size;
    Filter filter = Filter::bicubic;
    bool letterbox = true;
    std::array<uint8_t, 3> pad = {0, 0, 0};

    ggml_tensor* patch_embd_0 = nullptr;
    ggml_tensor* patch_embd_1 = nullptr;
    ggml_tensor* patch_bias = nullptr;
    ggml_tensor* position_embd = nullptr;
    ggml_tensor* pre_ln_w = nullptr;
    ggml_tensor* pre_ln_b = nullptr;
    ggml_tensor* post_ln_w = nullptr;
    ggml_tensor* post_ln_b = nullptr;
    std::vector<Layer> layers;
    ggml_tensor* mm_0_w = nullptr;
    ggml_tensor* mm_0_b = nullptr;
    ggml_tensor* mm_1_w = nullptr;
    ggml_tensor* mm_1_b = nullptr;
    ggml_tensor* mm_pre_norm_w = nullptr;   // Hy4's adapter
    ggml_tensor* image_newline = nullptr;
};

constexpr int kMaxNodes = 8192;

// A picture's graph: the builders are mtmd's clip_graph's, by name.
struct Graph {
    const Model& m;
    ggml_context* ctx0;
    ggml_cgraph* gf;
    const int img_w, img_h;
    const int patch_size, n_patches_x, n_patches_y, n_patches;
    const int n_embd, n_head, n_head_kv, d_head, n_layer;
    const float eps, kq_scale;
    const bool flash;
    const int head_pad;      // the flash attention's heads widened by this many zeros (GgmlEncoder::open)
    const int n_batch = 1;   // a still picture
    // the inputs the encoder fills
    ggml_tensor* inp_raw = nullptr;
    ggml_tensor* positions = nullptr;   // Qwen's rotary positions
    ggml_tensor* pos_embd = nullptr;    // Hy4's resampled position table

    Graph(const Model& model, ggml_context* ctx, int w, int h, bool flash_attention, int pad)
        : m(model), ctx0(ctx), gf(ggml_new_graph_custom(ctx, kMaxNodes, false)), img_w(w), img_h(h),
          patch_size(model.patch_size), n_patches_x(w / model.patch_size), n_patches_y(h / model.patch_size),
          n_patches(n_patches_x * n_patches_y), n_embd(model.n_embd), n_head(model.n_head),
          n_head_kv(model.n_head_kv), d_head(model.n_embd_head > 0 ? model.n_embd_head : model.n_embd / model.n_head),
          n_layer(model.n_layer), eps(model.eps), kq_scale(1.0f / sqrtf((float) d_head)), flash(flash_attention),
          head_pad(pad) {}

    ggml_tensor* build_inp_raw() {
        inp_raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, img_w, img_h, 3, n_batch);
        ggml_set_name(inp_raw, "inp_raw");
        ggml_set_input(inp_raw);
        return inp_raw;
    }

    ggml_tensor* build_norm(ggml_tensor* cur, ggml_tensor* mw, ggml_tensor* mb, bool rms, float norm_eps) const {
        cur = rms ? ggml_rms_norm(ctx0, cur, norm_eps) : ggml_norm(ctx0, cur, norm_eps);
        if (mw) cur = ggml_mul(ctx0, cur, mw);
        if (mb) cur = ggml_add(ctx0, cur, mb);
        return cur;
    }

    // up, GELU (the tanh form, ggml_gelu), down
    ggml_tensor* build_ffn(ggml_tensor* cur, ggml_tensor* up, ggml_tensor* up_b, ggml_tensor* down,
                           ggml_tensor* down_b) const {
        ggml_tensor* tmp = ggml_mul_mat(ctx0, up, cur);
        if (up_b) tmp = ggml_add(ctx0, tmp, up_b);
        cur = ggml_gelu(ctx0, tmp);
        cur = ggml_mul_mat(ctx0, down, cur);
        if (down_b) cur = ggml_add(ctx0, cur, down_b);
        return cur;
    }

    // q, k, v [d_head, n_head, n_pos, B] -> the block's attention output [n_embd, n_pos * B]
    ggml_tensor* build_attn(ggml_tensor* wo, ggml_tensor* wo_b, ggml_tensor* q_cur, ggml_tensor* k_cur,
                            ggml_tensor* v_cur) const {
        // into the graph together, in this order
        ggml_build_forward_order(gf, q_cur);
        ggml_build_forward_order(gf, k_cur);
        ggml_build_forward_order(gf, v_cur);

        ggml_tensor* q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
        ggml_tensor* k = ggml_permute(ctx0, k_cur, 0, 2, 1, 3);
        ggml_tensor* cur;
        if (flash) {
            // Padded heads: zeros after each head's values in q, k and v leave every score as it was and add zero
            // outputs, which are dropped again; the scale stays the true head size's.
            if (head_pad > 0) {
                q = ggml_permute(ctx0, ggml_pad(ctx0, q_cur, head_pad, 0, 0, 0), 0, 2, 1, 3);
                k = ggml_permute(ctx0, ggml_pad(ctx0, k_cur, head_pad, 0, 0, 0), 0, 2, 1, 3);
                v_cur = ggml_pad(ctx0, v_cur, head_pad, 0, 0, 0);
            }
            ggml_tensor* v = ggml_permute(ctx0, v_cur, 0, 2, 1, 3);
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
            cur = ggml_flash_attn_ext(ctx0, q, k, v, nullptr, kq_scale, 0.0f, 0.0f);
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            if (head_pad > 0)
                cur = ggml_cont(ctx0, ggml_view_4d(ctx0, cur, d_head, cur->ne[1], cur->ne[2], cur->ne[3], cur->nb[1],
                                                   cur->nb[2], cur->nb[3], 0));
            cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
        } else {
            ggml_tensor* v = ggml_permute(ctx0, v_cur, 1, 2, 0, 3);
            v = ggml_cont(ctx0, v);
            ggml_tensor* kq = ggml_mul_mat(ctx0, k, q);
            kq = ggml_soft_max_ext(ctx0, kq, nullptr, kq_scale, 0.0f);
            ggml_tensor* kqv = ggml_mul_mat(ctx0, v, kq);
            cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
            cur = ggml_cont_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
        }
        if (wo) cur = ggml_mul_mat(ctx0, wo, cur);
        if (wo_b) cur = ggml_add(ctx0, cur, wo_b);
        return cur;
    }

    // models/qwen3vl.cpp
    ggml_cgraph* build_qwen3vl() {
        const int batch_size = 1;
        const int n_pos = n_patches;
        const int num_position_ids = n_pos * 4;   // 4 values a position
        int mrope_sections[4] = {d_head / 4, d_head / 4, d_head / 4, d_head / 4};

        // a still picture: the two patch kernels over the same frame, summed
        build_inp_raw();
        ggml_tensor* inp =
            ggml_add(ctx0, ggml_conv_2d(ctx0, m.patch_embd_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1),
                     ggml_conv_2d(ctx0, m.patch_embd_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1));

        // the patches in 2x2 groups from here on (the merger concatenates each group's four at the end)
        {
            inp = ggml_permute(ctx0, inp, 1, 2, 0, 3);   // [w, h, c, b] -> [c, w, h, b]
            inp = ggml_cont_4d(ctx0, inp, n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
            inp = ggml_reshape_4d(ctx0, inp, n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
            inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
            inp = ggml_cont_3d(ctx0, inp, n_embd, n_patches_x * n_patches_y, batch_size);
        }

        inp = ggml_add(ctx0, inp, m.patch_bias);

        // the learned position table at the picture's grid, in the same grouping
        ggml_tensor* learned_pos_embd = resize_position_embeddings(GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
        learned_pos_embd = ggml_cont_4d(ctx0, learned_pos_embd, n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        learned_pos_embd =
            ggml_reshape_4d(ctx0, learned_pos_embd, n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        learned_pos_embd = ggml_permute(ctx0, learned_pos_embd, 0, 2, 1, 3);
        learned_pos_embd = ggml_cont_3d(ctx0, learned_pos_embd, n_embd, n_patches_x * n_patches_y, batch_size);
        inp = ggml_add(ctx0, inp, learned_pos_embd);

        ggml_tensor* inpL = inp;

        positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
        ggml_set_name(positions, "positions");
        ggml_set_input(positions);

        if (m.pre_ln_w) inpL = build_norm(inpL, m.pre_ln_w, m.pre_ln_b, false, eps);

        for (int il = 0; il < n_layer; il++) {
            const Layer& layer = m.layers[(size_t) il];
            ggml_tensor* cur = inpL;

            cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, false, eps);

            {
                cur = ggml_mul_mat(ctx0, layer.qkv_w, cur);
                cur = ggml_add(ctx0, cur, layer.qkv_b);

                ggml_tensor* Qcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos, ggml_row_size(cur->type, d_head),
                                                 cur->nb[1], 0);
                ggml_tensor* Kcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos, ggml_row_size(cur->type, d_head),
                                                 cur->nb[1], ggml_row_size(cur->type, n_embd));
                ggml_tensor* Vcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos, ggml_row_size(cur->type, d_head),
                                                 cur->nb[1], ggml_row_size(cur->type, 2 * n_embd));

                Qcur = ggml_rope_multi(ctx0, Qcur, positions, nullptr, d_head / 2, mrope_sections,
                                       GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
                Kcur = ggml_rope_multi(ctx0, Kcur, positions, nullptr, d_head / 2, mrope_sections,
                                       GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);

                cur = build_attn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur);
            }

            cur = ggml_add(ctx0, cur, inpL);
            inpL = cur;

            cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, false, eps);
            cur = build_ffn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_down_w, layer.ff_down_b);

            cur = ggml_add(ctx0, inpL, cur);
            inpL = cur;
        }

        if (m.post_ln_w) inpL = build_norm(inpL, m.post_ln_w, m.post_ln_b, false, eps);

        // the merger: a group's four patches as one row, two linear layers
        ggml_tensor* embeddings = inpL;
        embeddings = ggml_reshape_3d(ctx0, embeddings, n_embd * 4, n_pos / 4, batch_size);
        embeddings = build_ffn(embeddings, m.mm_0_w, m.mm_0_b, m.mm_1_w, m.mm_1_b);

        ggml_build_forward_expand(gf, embeddings);
        return gf;
    }

    // clip_graph::resize_position_embeddings
    ggml_tensor* resize_position_embeddings(uint32_t mode) const {
        ggml_tensor* pos = m.position_embd;
        const int height = img_h / patch_size;
        const int width = img_w / patch_size;
        const int n_per_side = (int) std::sqrt(pos->ne[1]);
        if (height == n_per_side && width == n_per_side) return pos;

        pos = ggml_reshape_3d(ctx0, pos, n_embd, n_per_side, n_per_side);   // -> (n_embd, n_per_side, n_per_side)
        pos = ggml_permute(ctx0, pos, 2, 0, 1, 3);                         // -> (n_per_side, n_per_side, n_embd)
        pos = ggml_interpolate(ctx0, pos, width, height, n_embd, 1, mode);  // -> (width, height, n_embd)
        pos = ggml_permute(ctx0, pos, 1, 2, 0, 3);                         // -> (n_embd, width, height)
        pos = ggml_cont_2d(ctx0, pos, n_embd, width * height);             // -> (n_embd, width * height)
        return pos;
    }

    // clip_graph::build_inp
    ggml_tensor* build_inp() {
        build_inp_raw();
        ggml_tensor* inp = ggml_conv_2d(ctx0, m.patch_embd_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
        inp = ggml_reshape_3d(ctx0, inp, n_patches, n_embd, n_batch);
        inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
        if (m.patch_bias) inp = ggml_add(ctx0, inp, m.patch_bias);
        return inp;
    }

    // clip_graph::build_vit for a tower of separate q, k and v projections, with a learned position table
    ggml_tensor* build_vit(ggml_tensor* inp, int64_t n_pos, ggml_tensor* learned_pos_embd) {
        const int64_t B = inp->ne[2];

        if (learned_pos_embd) inp = ggml_add(ctx0, inp, learned_pos_embd);

        inp = ggml_reshape_2d(ctx0, inp, n_embd, n_pos * B);
        ggml_tensor* inpL = inp;

        if (m.pre_ln_w) inpL = build_norm(inpL, m.pre_ln_w, m.pre_ln_b, false, eps);

        for (int il = 0; il < n_layer; il++) {
            const Layer& layer = m.layers[(size_t) il];
            ggml_tensor* cur = inpL;

            cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, false, eps);

            {
                ggml_tensor* Qcur = ggml_mul_mat(ctx0, layer.q_w, cur);
                if (layer.q_b) Qcur = ggml_add(ctx0, Qcur, layer.q_b);

                ggml_tensor* Kcur = ggml_mul_mat(ctx0, layer.k_w, cur);
                if (layer.k_b) Kcur = ggml_add(ctx0, Kcur, layer.k_b);

                ggml_tensor* Vcur = ggml_mul_mat(ctx0, layer.v_w, cur);
                if (layer.v_b) Vcur = ggml_add(ctx0, Vcur, layer.v_b);

                Qcur = ggml_reshape_4d(ctx0, Qcur, d_head, n_head, n_pos, B);
                Kcur = ggml_reshape_4d(ctx0, Kcur, d_head, n_head_kv, n_pos, B);
                Vcur = ggml_reshape_4d(ctx0, Vcur, d_head, n_head_kv, n_pos, B);

                cur = build_attn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur);
            }

            cur = ggml_add(ctx0, cur, inpL);
            inpL = cur;

            cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, false, eps);
            cur = build_ffn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_down_w, layer.ff_down_b);

            cur = ggml_add(ctx0, inpL, cur);
            inpL = cur;
        }

        if (m.post_ln_w) inpL = build_norm(inpL, m.post_ln_w, m.post_ln_b, false, eps);

        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, inpL->ne[1] / B, B);
        return inpL;
    }

    // llama.cpp pull request 32517's models/hy-v4-flash.cpp
    ggml_cgraph* build_hy4() {
        const int merge = m.n_merge;
        const int pw = n_patches_x;
        const int ph = n_patches_y;

        // the position table at the picture's grid: an input, resampled on the host (hy4_positions)
        pos_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, ph * pw);
        ggml_set_name(pos_embd, "hunyuanvl_pos_embd");
        ggml_set_input(pos_embd);

        ggml_tensor* inp = build_inp();
        ggml_tensor* cur = build_vit(inp, n_patches, pos_embd);

        cur = build_norm(cur, m.mm_pre_norm_w, nullptr, true, eps);

        // [C, W*H] -> [W, H, C] for the convolutions
        cur = ggml_reshape_3d(ctx0, cur, n_embd, pw, ph);
        cur = ggml_permute(ctx0, cur, 2, 0, 1, 3);
        cur = ggml_cont(ctx0, cur);

        // a 2x2 convolution of stride 2, GELU (erf), a 1x1 convolution
        cur = ggml_conv_2d(ctx0, m.mm_0_w, cur, merge, merge, 0, 0, 1, 1);
        cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, m.mm_0_b, 1, 1, m.mm_0_b->ne[0]));
        cur = ggml_gelu_erf(ctx0, cur);
        cur = ggml_conv_2d(ctx0, m.mm_1_w, cur, 1, 1, 0, 0, 1, 1);
        cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, m.mm_1_b, 1, 1, m.mm_1_b->ne[0]));

        const int ow = pw / merge;
        const int oh = ph / merge;
        const int idim = (int) cur->ne[2];

        // the newline row after each row of cells
        ggml_tensor* nl = ggml_reshape_4d(ctx0, m.image_newline, 1, 1, idim, 1);
        nl = ggml_repeat_4d(ctx0, nl, 1, oh, idim, 1);
        cur = ggml_concat(ctx0, cur, nl, 0);

        // [OW+1, OH, OC] -> [OC, (OW+1)*OH]
        cur = ggml_permute(ctx0, cur, 1, 2, 0, 3);
        cur = ggml_cont_2d(ctx0, cur, idim, (int64_t) (ow + 1) * oh);

        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    ggml_cgraph* build() { return m.kind == Kind::qwen3vl ? build_qwen3vl() : build_hy4(); }
};

// Qwen's rotary positions (mtmd's encode): a patch's row and column, the patches in the 2x2 groups' order
std::vector<int32_t> qwen_positions(int pw, int ph, int merge) {
    const int num_patches = pw * ph;
    std::vector<int32_t> positions((size_t) num_patches * 4);
    int ptr = 0;
    for (int y = 0; y < ph; y += merge) {
        for (int x = 0; x < pw; x += merge) {
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    positions[(size_t) ptr] = y + dy;
                    positions[(size_t) num_patches + ptr] = x + dx;
                    positions[(size_t) 2 * num_patches + ptr] = y + dy;
                    positions[(size_t) 3 * num_patches + ptr] = x + dx;
                    ptr++;
                }
            }
        }
    }
    return positions;
}

// Hy4's position table at out_w x out_h (mtmd's encode, HunyuanVL's): bilinear between the table's cells, a sample's
// place by the scale (target + 0.1) / n_grid, pixel centres (align_corners off).  dst[(y * out_w + x) * n_embd + c].
std::vector<float> hy4_positions(const std::vector<float>& src, int64_t n_embd, int n_grid, int out_w, int out_h) {
    std::vector<float> dst((size_t) n_embd * out_h * out_w);

    const float sx = (float) (out_w + 0.1f) / (float) n_grid;
    const float sy = (float) (out_h + 0.1f) / (float) n_grid;

    for (int y = 0; y < out_h; ++y) {
        const float fy = ((float) y + 0.5f) / sy - 0.5f;
        int y0 = (int) std::floor(fy);
        int y1 = y0 + 1;
        y0 = std::clamp(y0, 0, n_grid - 1);
        y1 = std::clamp(y1, 0, n_grid - 1);
        float wy1 = std::clamp(fy - (float) y0, 0.0f, 1.0f);
        const float wy0 = 1.0f - wy1;
        for (int x = 0; x < out_w; ++x) {
            const float fx = ((float) x + 0.5f) / sx - 0.5f;
            int x0 = (int) std::floor(fx);
            int x1 = x0 + 1;
            x0 = std::clamp(x0, 0, n_grid - 1);
            x1 = std::clamp(x1, 0, n_grid - 1);
            float wx1 = std::clamp(fx - (float) x0, 0.0f, 1.0f);
            const float wx0 = 1.0f - wx1;

            const float w00 = wy0 * wx0;
            const float w01 = wy0 * wx1;
            const float w10 = wy1 * wx0;
            const float w11 = wy1 * wx1;

            const float* s00 = &src[((size_t) y0 * n_grid + x0) * n_embd];
            const float* s01 = &src[((size_t) y0 * n_grid + x1) * n_embd];
            const float* s10 = &src[((size_t) y1 * n_grid + x0) * n_embd];
            const float* s11 = &src[((size_t) y1 * n_grid + x1) * n_embd];
            float* d = &dst[((size_t) y * out_w + x) * n_embd];
            for (int c = 0; c < n_embd; ++c) {
                d[c] = w00 * s00[c] + w01 * s01[c] + w10 * s10[c] + w11 * s11[c];
            }
        }
    }
    return dst;
}

int seek64(std::FILE* f, uint64_t at) {
#ifdef _WIN32
    return _fseeki64(f, (long long) at, SEEK_SET);
#else
    return fseeko(f, (off_t) at, SEEK_SET);
#endif
}

class GgmlEncoder final : public Encoder {
public:
    GgmlEncoder() = default;
    ~GgmlEncoder() override {
        sched_.reset();
        buf_.reset();
        if (backend_ != nullptr) ggml_backend_free(backend_);
        if (backend_cpu_ != nullptr && backend_cpu_ != backend_) ggml_backend_free(backend_cpu_);
    }
    GgmlEncoder(const GgmlEncoder&) = delete;
    GgmlEncoder& operator=(const GgmlEncoder&) = delete;

    /// throws std::runtime_error
    void open(const EncoderParams& p) {
        flash_ = p.flash_attention;
        threads_ = p.threads > 0 ? p.threads : 4;

        // the backends: the GPU first when asked for, the CPU for what it lacks
        backend_cpu_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu_ == nullptr) throw std::runtime_error("the CPU backend did not start");
        if (p.gpu) {
            backend_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
            if (backend_ == nullptr) backend_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
            if (backend_ == nullptr) throw std::runtime_error("no GPU backend started");
        }
        if (backend_ != nullptr) {
            backends_.push_back(backend_);
            bufts_.push_back(ggml_backend_get_default_buffer_type(backend_));
        } else {
            backend_ = backend_cpu_;
        }
        backends_.push_back(backend_cpu_);
        bufts_.push_back(ggml_backend_get_default_buffer_type(backend_cpu_));
        sched_.reset(ggml_backend_sched_new(backends_.data(), bufts_.data(), (int) backends_.size(), kMaxNodes, false, true));

        load(p);
        // ggml's CUDA flash attention has tensor-core kernels for heads of 64, 80, 96, 112 and 128 values; a head
        // of another size (this tower's 72) goes to its tile kernel, which at the pinned commit is slower and, the
        // more patches a picture has, further from the FP32 attention: Hy4's 3,645-row screenshot 3.05 s and rows
        // 27% from the CPU's, padded to 80 1.19 s and 6%; Flash-Next's pictures of ~1,000 tokens 0.33-0.35 s and
        // 5-13% from the CPU's FP32 attention, padded 0.19-0.21 s and 2.4-5.5%.  So on a GPU the heads are padded
        // to the next of those sizes (mtmd leaves them: Qwen's rows on a GPU are no longer its bytes; on the CPU
        // they are).  LAMINA_VISION_HEAD_PAD=N pads them to N values (0: none), for comparisons.
        if (backend_ != backend_cpu_) {
            const int d_head = m_.n_embd_head > 0 ? m_.n_embd_head : m_.n_embd / m_.n_head;
            int to = d_head;
            for (const int size : {64, 80, 96, 112, 128})
                if (to == d_head && size >= d_head) to = size;
            if (const char* s = std::getenv("LAMINA_VISION_HEAD_PAD")) to = std::max(d_head, std::atoi(s));
            head_pad_ = to - d_head;
        }
        compute_meta_.resize((size_t) kMaxNodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(kMaxNodes, false));
    }

    int n_embd() const override { return m_.projection_dim; }

    bool encode(const Rgb& picture, Rows& out, std::string& err) override {
        try {
            encode_or_throw(picture, out);
            return true;
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
    }

private:
    // ---- the projector file ----------------------------------------------------------------------------------------
    int find(const char* key) const { return (int) gguf_find_key(gguf_.get(), key); }
    /// an integer key's value; `dflt` when the key is not there and not `required`
    int get_int(const char* key, bool required, int dflt = 0) const {
        const int i = find(key);
        if (i < 0) {
            if (required) throw std::runtime_error(std::string("the projector has no ") + key);
            return dflt;
        }
        int64_t v = 0;
        switch (gguf_get_kv_type(gguf_.get(), i)) {
            case GGUF_TYPE_UINT8: v = gguf_get_val_u8(gguf_.get(), i); break;
            case GGUF_TYPE_INT8: v = gguf_get_val_i8(gguf_.get(), i); break;
            case GGUF_TYPE_UINT16: v = gguf_get_val_u16(gguf_.get(), i); break;
            case GGUF_TYPE_INT16: v = gguf_get_val_i16(gguf_.get(), i); break;
            case GGUF_TYPE_UINT32: v = gguf_get_val_u32(gguf_.get(), i); break;
            case GGUF_TYPE_INT32: v = gguf_get_val_i32(gguf_.get(), i); break;
            case GGUF_TYPE_UINT64: v = (int64_t) gguf_get_val_u64(gguf_.get(), i); break;
            case GGUF_TYPE_INT64: v = gguf_get_val_i64(gguf_.get(), i); break;
            default: throw std::runtime_error(std::string("the projector's ") + key + " is not an integer");
        }
        if (v < 0 || v > std::numeric_limits<int32_t>::max())
            throw std::runtime_error(std::string("the projector's ") + key + " is out of range");
        return (int) v;
    }
    float get_f32(const char* key) const {
        const int i = find(key);
        if (i < 0) throw std::runtime_error(std::string("the projector has no ") + key);
        if (gguf_get_kv_type(gguf_.get(), i) != GGUF_TYPE_FLOAT32)
            throw std::runtime_error(std::string("the projector's ") + key + " is not a float");
        return gguf_get_val_f32(gguf_.get(), i);
    }
    bool get_bool(const char* key) const {
        const int i = find(key);
        return i >= 0 && gguf_get_kv_type(gguf_.get(), i) == GGUF_TYPE_BOOL && gguf_get_val_bool(gguf_.get(), i);
    }
    std::string get_str(const char* key) const {
        const int i = find(key);
        if (i < 0 || gguf_get_kv_type(gguf_.get(), i) != GGUF_TYPE_STRING) return "";
        return gguf_get_val_str(gguf_.get(), i);
    }
    void get_rgb(const char* key, float out[3]) const {
        const int i = find(key);
        if (i < 0 || gguf_get_kv_type(gguf_.get(), i) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(gguf_.get(), i) != GGUF_TYPE_FLOAT32 || gguf_get_arr_n(gguf_.get(), i) < 3)
            throw std::runtime_error(std::string("the projector's ") + key + " is not three floats");
        const float* v = (const float*) gguf_get_arr_data(gguf_.get(), i);
        for (int c = 0; c < 3; ++c) out[c] = v[c];
    }

    void load(const EncoderParams& p) {
        ggml_context* meta = nullptr;
        gguf_init_params gp = {/*.no_alloc =*/ true, /*.ctx =*/ &meta};
        gguf_.reset(gguf_init_from_file(p.mmproj.c_str(), gp));
        if (!gguf_) throw std::runtime_error("not a GGUF file that can be read");
        meta_.reset(meta);

        // the tower's shape and the picture's sizing
        if (!get_bool("clip.has_vision_encoder")) throw std::runtime_error("the file has no vision encoder");
        std::string type = get_str("clip.projector_type");
        if (type.empty()) type = get_str("clip.vision.projector_type");
        if (type == "qwen3vl_merger") m_.kind = Kind::qwen3vl;
        else if (type == "hy_v4_flash") m_.kind = Kind::hy4;
        else
            throw std::runtime_error("its projector type is " + (type.empty() ? std::string("not named") : type) +
                                     ": this encoder reads qwen3vl_merger and hy_v4_flash");
        m_.n_embd = get_int("clip.vision.embedding_length", true);
        m_.n_head = get_int("clip.vision.attention.head_count", true);
        m_.n_embd_head = get_int("clip.vision.attention.head_dim", false);
        m_.n_head_kv = get_int("clip.vision.attention.head_count_kv", false, m_.n_head);
        m_.n_layer = get_int("clip.vision.block_count", true);
        m_.projection_dim = get_int("clip.vision.projection_dim", true);
        m_.eps = get_f32("clip.vision.attention.layer_norm_epsilon");
        m_.patch_size = get_int("clip.vision.patch_size", true);
        get_rgb("clip.vision.image_mean", m_.mean);
        get_rgb("clip.vision.image_std", m_.std);
        if (!get_bool("clip.use_gelu")) throw std::runtime_error("its feed-forward activation is not GELU");
        if (m_.n_embd <= 0 || m_.n_head <= 0 || m_.n_layer <= 0 || m_.patch_size <= 0 || m_.n_embd % m_.n_head != 0 ||
            m_.n_head_kv != m_.n_head)
            throw std::runtime_error("its tower's sizes are not ones this encoder reads");
        m_.n_merge = get_int("clip.vision.spatial_merge_size", false, 2);
        if (m_.n_merge != 2) throw std::runtime_error("its spatial merge is not 2x2");
        const int patch_area = m_.patch_size * m_.patch_size * m_.n_merge * m_.n_merge;
        m_.size.align = m_.patch_size * m_.n_merge;
        if (m_.kind == Kind::qwen3vl) {
            // 8 to 4,096 tokens a picture unless the caller says otherwise; letterboxed into its size, bicubic
            m_.size.min_pixels = (p.min_tokens > 0 ? p.min_tokens : 8) * patch_area;
            m_.size.max_pixels = (p.max_tokens > 0 ? p.max_tokens : 4096) * patch_area;
            m_.filter = Filter::bicubic;
            m_.letterbox = true;
        } else {
            // the reference processor's 4 to 4,096 tokens, then the file's own pixel limits over them (the caller's
            // too, as the pull request has it); resized straight to its size, Lanczos
            m_.size.min_pixels = (p.min_tokens > 0 ? p.min_tokens : 4) * patch_area;
            m_.size.max_pixels = (p.max_tokens > 0 ? p.max_tokens : 4096) * patch_area;
            m_.size.min_pixels = get_int("clip.vision.image_min_pixels", false, m_.size.min_pixels);
            m_.size.max_pixels = get_int("clip.vision.image_max_pixels", false, m_.size.max_pixels);
            m_.filter = Filter::lanczos;
            m_.letterbox = false;
        }

        // the weights: each tensor's description copied into a context of its own, then allocated on the backend
        const int64_t n_tensors = gguf_get_n_tensors(gguf_.get());
        ggml_init_params ip = {/*.mem_size =*/ (size_t) (n_tensors + 1) * ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr,
                               /*.no_alloc =*/ true};
        data_.reset(ggml_init(ip));
        if (!data_) throw std::runtime_error("no memory for the projector's tensors");
        std::vector<ggml_tensor*> to_load;
        auto get = [&](const std::string& name, bool required = true) -> ggml_tensor* {
            ggml_tensor* cur = ggml_get_tensor(meta_.get(), name.c_str());
            if (cur == nullptr) {
                if (required) throw std::runtime_error("the projector has no tensor " + name);
                return nullptr;
            }
            ggml_tensor* t = ggml_dup_tensor(data_.get(), cur);
            ggml_set_name(t, cur->name);
            to_load.push_back(t);
            return t;
        };
        auto blk = [](int il, const char* what) { return "v.blk." + std::to_string(il) + "." + what; };

        if (ggml_get_tensor(meta_.get(), "v.class_embd") != nullptr)
            throw std::runtime_error("its tower has a class embedding, which this encoder does not read");
        m_.pre_ln_w = get("v.pre_ln.weight", false);
        m_.pre_ln_b = get("v.pre_ln.bias", false);
        m_.post_ln_w = get("v.post_ln.weight", false);
        m_.post_ln_b = get("v.post_ln.bias", false);
        m_.patch_bias = get("v.patch_embd.bias", m_.kind == Kind::qwen3vl);
        m_.patch_embd_0 = get("v.patch_embd.weight");
        m_.patch_embd_1 = get("v.patch_embd.weight.1", m_.kind == Kind::qwen3vl);
        m_.position_embd = get("v.position_embd.weight");
        m_.layers.resize((size_t) m_.n_layer);
        for (int il = 0; il < m_.n_layer; ++il) {
            Layer& l = m_.layers[(size_t) il];
            if (m_.kind == Kind::qwen3vl) {
                l.qkv_w = get(blk(il, "attn_qkv.weight"));
                l.qkv_b = get(blk(il, "attn_qkv.bias"));
            } else {
                l.q_w = get(blk(il, "attn_q.weight"));
                l.q_b = get(blk(il, "attn_q.bias"), false);
                l.k_w = get(blk(il, "attn_k.weight"));
                l.k_b = get(blk(il, "attn_k.bias"), false);
                l.v_w = get(blk(il, "attn_v.weight"));
                l.v_b = get(blk(il, "attn_v.bias"), false);
            }
            l.o_w = get(blk(il, "attn_out.weight"));
            l.o_b = get(blk(il, "attn_out.bias"), false);
            l.ln_1_w = get(blk(il, "ln1.weight"), false);
            l.ln_1_b = get(blk(il, "ln1.bias"), false);
            l.ln_2_w = get(blk(il, "ln2.weight"), false);
            l.ln_2_b = get(blk(il, "ln2.bias"), false);
            l.ff_up_w = get(blk(il, "ffn_up.weight"));
            l.ff_up_b = get(blk(il, "ffn_up.bias"), false);
            l.ff_down_w = get(blk(il, "ffn_down.weight"));
            l.ff_down_b = get(blk(il, "ffn_down.bias"), false);
            // what mtmd's tower builder would also apply, and these projectors do not have
            for (const char* other : {"attn_q_norm.weight", "attn_k_norm.weight", "ls1.weight", "ls2.weight",
                                      "ffn_gate.weight", "attn_post_norm.weight", "ffn_post_norm.weight"})
                if (ggml_get_tensor(meta_.get(), blk(il, other).c_str()) != nullptr)
                    throw std::runtime_error("its tower has " + blk(il, other) + ", which this encoder does not read");
            if (ggml_get_tensor(meta_.get(), ("v.deepstack." + std::to_string(il) + ".fc1.weight").c_str()) != nullptr)
                throw std::runtime_error("its tower has deepstack layers, which the engine's models do not take");
        }
        m_.mm_0_w = get("mm.0.weight");
        m_.mm_0_b = get("mm.0.bias");
        m_.mm_1_w = get("mm.2.weight");
        m_.mm_1_b = get("mm.2.bias");
        if (m_.kind == Kind::hy4) {
            m_.mm_pre_norm_w = get("mm.pre_norm.weight");
            m_.image_newline = get("v.image_newline");
            if (m_.position_embd->type != GGML_TYPE_F32)   // resampled on the host
                throw std::runtime_error("its position table is not F32");
            const int n_grid = (int) std::lround(std::sqrt((double) m_.position_embd->ne[1]));
            if ((int64_t) n_grid * n_grid != m_.position_embd->ne[1] || m_.position_embd->ne[0] != m_.n_embd)
                throw std::runtime_error("its position table is not a square grid of the tower's width");
        }
        // the rows' width is the projection's
        const int64_t width = m_.kind == Kind::qwen3vl ? m_.mm_1_b->ne[0] : m_.mm_1_w->ne[3];
        if (width != m_.projection_dim) throw std::runtime_error("its adapter's output is not its projection_dim wide");

        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend_);
        buf_.reset(ggml_backend_alloc_ctx_tensors_from_buft(data_.get(), buft));
        if (!buf_) throw std::runtime_error("the projector's weights do not fit the device's memory");
        ggml_backend_buffer_set_usage(buf_.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        std::FILE* f = ggml_fopen(p.mmproj.c_str(), "rb");
        if (f == nullptr) throw std::runtime_error("the file cannot be opened");
        struct Close {
            std::FILE* f;
            ~Close() { std::fclose(f); }
        } close{f};
        const bool host = ggml_backend_buft_is_host(buft);
        std::vector<uint8_t> read_buf;
        for (ggml_tensor* t : to_load) {
            const int64_t idx = gguf_find_tensor(gguf_.get(), t->name);
            if (idx < 0) throw std::runtime_error(std::string("the projector has no tensor ") + t->name);
            const uint64_t at = gguf_get_data_offset(gguf_.get()) + gguf_get_tensor_offset(gguf_.get(), idx);
            const size_t n = ggml_nbytes(t);
            bool ok = seek64(f, at) == 0;
            if (ok && host) {
                ok = std::fread(t->data, 1, n, f) == n;   // the CPU's buffer: read in place
            } else if (ok) {
                read_buf.resize(n);
                ok = std::fread(read_buf.data(), 1, n, f) == n;
                if (ok) ggml_backend_tensor_set(t, read_buf.data(), 0, n);
            }
            if (!ok) throw std::runtime_error(std::string("the file ends inside the tensor ") + t->name);
        }
        if (m_.kind == Kind::hy4) {
            pos_host_.resize((size_t) ggml_nelements(m_.position_embd));
            ggml_backend_tensor_get(m_.position_embd, pos_host_.data(), 0, ggml_nbytes(m_.position_embd));
        }
    }

    // ---- a picture -------------------------------------------------------------------------------------------------
    ggml_cgraph* build(ggml_context_ptr& ctx0, std::unique_ptr<Graph>& g, int w, int h, bool flash) {
        ggml_init_params ip = {/*.mem_size =*/ compute_meta_.size(), /*.mem_buffer =*/ compute_meta_.data(),
                               /*.no_alloc =*/ true};
        ctx0.reset(ggml_init(ip));
        g = std::make_unique<Graph>(m_, ctx0.get(), w, h, flash, head_pad_);
        return g->build();
    }

    /// the buffers for a w x h picture's graph reserved; false where the backend lacks its flash attention
    bool reserve(int w, int h, bool flash) {
        ggml_context_ptr ctx0;
        std::unique_ptr<Graph> g;
        ggml_cgraph* gf = build(ctx0, g, w, h, flash);
        if (!ggml_backend_sched_reserve(sched_.get(), gf))
            throw std::runtime_error("the encoder's buffers do not fit the device's memory");
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor* node = ggml_graph_node(gf, i);
            if (node->op == GGML_OP_FLASH_ATTN_EXT && !ggml_backend_supports_op(backend_, node)) return false;
        }
        return true;
    }

    void encode_or_throw(const Rgb& picture, Rows& out) {
        if (picture.w <= 0 || picture.h <= 0 || picture.px.size() != (size_t) picture.w * (size_t) picture.h * 3)
            throw std::runtime_error("an empty picture");
        int tw = 0, th = 0;
        target_size(picture.w, picture.h, m_.size, tw, th);
        if (tw <= 0 || th <= 0 || tw % m_.size.align != 0 || th % m_.size.align != 0)
            throw std::runtime_error("the picture's size cannot be brought to the projector's grid");
        const Rgb sized = fit(picture, tw, th, m_.filter, m_.letterbox, m_.pad);
        const std::vector<float> pixels = planes(sized, m_.mean, m_.std);

        // The first picture settles the flash attention (automatic: on where the backend has it for this tower) and
        // reserves the buffers, as mtmd's warm-up does.
        if (!allocated_) {
            if (flash_ == FlashAttention::automatic) {
                flash_ = FlashAttention::on;
                if (!reserve(tw, th, true)) {
                    flash_ = FlashAttention::off;
                    reserve(tw, th, false);
                }
            } else {
                reserve(tw, th, flash_ == FlashAttention::on);
            }
            allocated_ = true;
        }

        ggml_backend_sched_reset(sched_.get());
        ggml_context_ptr ctx0;
        std::unique_ptr<Graph> g;
        ggml_cgraph* gf = build(ctx0, g, tw, th, flash_ == FlashAttention::on);
        if (!ggml_backend_sched_alloc_graph(sched_.get(), gf))
            throw std::runtime_error("the picture's buffers do not fit the device's memory");

        ggml_backend_tensor_set(g->inp_raw, pixels.data(), 0, ggml_nbytes(g->inp_raw));
        const int pw = tw / m_.patch_size, ph = th / m_.patch_size;
        if (m_.kind == Kind::qwen3vl) {
            const std::vector<int32_t> positions = qwen_positions(pw, ph, m_.n_merge);
            ggml_backend_tensor_set(g->positions, positions.data(), 0, ggml_nbytes(g->positions));
        } else {
            const int n_grid = (int) std::lround(std::sqrt((double) m_.position_embd->ne[1]));
            const std::vector<float> table = hy4_positions(pos_host_, m_.n_embd, n_grid, pw, ph);
            ggml_backend_tensor_set(g->pos_embd, table.data(), 0, ggml_nbytes(g->pos_embd));
        }

        ggml_backend_cpu_set_n_threads(backend_cpu_, threads_);
        const ggml_status status = ggml_backend_sched_graph_compute(sched_.get(), gf);
        if (status != GGML_STATUS_SUCCESS)
            throw std::runtime_error("the encoder's graph failed (ggml status " + std::to_string((int) status) + ")");

        ggml_tensor* rows = ggml_graph_node(gf, -1);
        const int n = (int) rows->ne[1];
        const int merged_x = pw / m_.n_merge, merged_y = ph / m_.n_merge;
        const int expected = m_.kind == Kind::qwen3vl ? merged_x * merged_y : (merged_x + 1) * merged_y;
        if (rows->ne[0] != m_.projection_dim || n != expected || ggml_nelements(rows) != (int64_t) n * m_.projection_dim)
            throw std::runtime_error("the encoder gave " + std::to_string(n) + " rows of " +
                                     std::to_string(rows->ne[0]) + " where " + std::to_string(expected) + " of " +
                                     std::to_string(m_.projection_dim) + " were due");
        out.n = n;
        out.n_embd = m_.projection_dim;
        if (m_.kind == Kind::qwen3vl) {
            out.nx = merged_x;
            out.ny = merged_y;
        } else {   // read at text positions: one row of n
            out.nx = n;
            out.ny = 1;
        }
        out.data.resize((size_t) n * (size_t) m_.projection_dim);
        ggml_backend_tensor_get(rows, out.data.data(), 0, ggml_nbytes(rows));
    }

    Model m_;
    gguf_context_ptr gguf_;
    ggml_context_ptr meta_, data_;
    ggml_backend_t backend_ = nullptr, backend_cpu_ = nullptr;
    std::vector<ggml_backend_t> backends_;
    std::vector<ggml_backend_buffer_type_t> bufts_;
    ggml_backend_buffer_ptr buf_;
    ggml_backend_sched_ptr sched_;
    std::vector<uint8_t> compute_meta_;
    std::vector<float> pos_host_;   // Hy4's position table
    FlashAttention flash_ = FlashAttention::automatic;
    int threads_ = 4;
    int head_pad_ = 0;
    bool allocated_ = false;
};

}  // namespace

bool gpu_visible() {
    backends_init();
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
           ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU) != nullptr;
}

bool cpu_has_avx512() { return ggml_cpu_has_avx512() != 0; }

std::unique_ptr<Encoder> open_encoder(const EncoderParams& params, std::string& err) {
    backends_init();
    auto enc = std::make_unique<GgmlEncoder>();
    try {
        enc->open(params);
    } catch (const std::exception& e) {
        err = e.what();
        return nullptr;
    }
    return enc;
}

}  // namespace lamina::vision
