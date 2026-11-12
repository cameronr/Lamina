// src/kernels/head_screen_parity.cpp - the M1 screen kernels against host references: the k-th largest by
// radix select (exact bits, ties included), the q8_1 input norms, the row norms through the reference dequant
// (Q8_0 bytes), and the histogram's bucket math. `--selftest` for ctest; exit 77 without a CUDA device.

#include "lamina/kernels/native_head_screen.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

void* dmalloc(size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        std::printf("skip: cudaMalloc\n");
        std::exit(77);
    }
    return p;
}

void test_kth(std::mt19937& rnd) {
    std::uniform_real_distribution<float> uni(-50.0f, 50.0f);
    std::uniform_int_distribution<int> dup(0, 9);   // thousands of rows over 10 distinct values: dense ties
    for (const size_t n : {65u, 1000u, 200000u}) {
        std::vector<float> row(n);
        for (size_t i = 0; i < n; ++i) row[i] = n > 1000 ? (float) dup(rnd) : uni(rnd);
        std::vector<float> sorted = row;
        std::sort(sorted.begin(), sorted.end(), std::greater<float>());

        float* d_row = (float*) dmalloc(n * sizeof(float));
        float* d_tau = (float*) dmalloc(sizeof(float));
        cudaMemcpy(d_row, row.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        float tau = 0.0f;
        lamina::kernels::head_screen_kth(d_row, 1, (int64_t) n, 1, d_tau, nullptr);
        cudaMemcpy(&tau, d_tau, sizeof(float), cudaMemcpyDeviceToHost);
        check(tau == sorted[0], "kth k=1 is the maximum");
        lamina::kernels::head_screen_kth(d_row, 1, (int64_t) n, 64, d_tau, nullptr);
        cudaMemcpy(&tau, d_tau, sizeof(float), cudaMemcpyDeviceToHost);
        check(tau == sorted[63], "kth k=64 is the 64th largest, exact bits");
        cudaFree(d_row);
        cudaFree(d_tau);
    }
    // two columns at once, and a negative-heavy row where every candidate answer is below zero
    std::vector<float> row(1000);
    std::uniform_real_distribution<float> neg(-1.0f, -0.5f);
    for (auto& v : row) v = neg(rnd);
    float* d_two = (float*) dmalloc(2 * row.size() * sizeof(float));
    float* d_tau2 = (float*) dmalloc(2 * sizeof(float));
    cudaMemcpy(d_two, row.data(), row.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_two + row.size(), row.data(), row.size() * sizeof(float), cudaMemcpyHostToDevice);
    std::vector<float> sorted = row;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    float tau[2] = {};
    lamina::kernels::head_screen_kth(d_two, 2, (int64_t) row.size(), 64, d_tau2, nullptr);
    cudaMemcpy(tau, d_tau2, sizeof tau, cudaMemcpyDeviceToHost);
    check(tau[0] == sorted[63] && tau[1] == sorted[63], "kth over two all-negative columns");
    cudaFree(d_two);
    cudaFree(d_tau2);
}

void write_q81(std::vector<uint8_t>& host, int slot, std::mt19937& rnd, std::uniform_real_distribution<float>& uni,
               std::uniform_int_distribution<int>& q, double& acc) {
    uint8_t* p = host.data() + (size_t) slot * 36;   // half2 d (the scalar in the low half) + 32 int8
    const half d = __float2half_rn(uni(rnd));
    const uint16_t du = __half_as_ushort(d);
    p[0] = (uint8_t) du;
    p[1] = (uint8_t) (du >> 8);
    p[2] = 0;
    p[3] = 0;
    const double dv = __half2float(d);
    for (int i = 0; i < 32; ++i) {
        const int v = q(rnd);
        p[4 + i] = (uint8_t) (int8_t) v;
        acc += (double) v * dv * ((double) v * dv);
    }
}

void test_input_norms(std::mt19937& rnd) {
    const int n_in = 512, ncols = 3, nb = n_in / 32;
    std::vector<uint8_t> host((size_t) ncols * nb * 36);
    std::uniform_real_distribution<float> uni(-4.0f, 4.0f);
    std::uniform_int_distribution<int> q(-127, 127);
    std::vector<double> expect(ncols, 0.0);
    for (int c = 0; c < ncols; ++c)
        for (int b = 0; b < nb; ++b) write_q81(host, c * nb + b, rnd, uni, q, expect[c]);
    void* d_x = dmalloc(host.size());
    float* d_norms = (float*) dmalloc(ncols * sizeof(float));
    cudaMemcpy(d_x, host.data(), host.size(), cudaMemcpyHostToDevice);
    lamina::kernels::head_screen_input_norms(d_x, d_norms, n_in, ncols, nullptr);
    std::vector<float> got(ncols);
    cudaMemcpy(got.data(), d_norms, ncols * sizeof(float), cudaMemcpyDeviceToHost);
    for (int c = 0; c < ncols; ++c) {
        const double want = std::sqrt(expect[c]);
        check(std::fabs(got[c] - want) <= 1e-3 * want + 1e-4, "input norms match the host sum");
    }
    cudaFree(d_x);
    cudaFree(d_norms);
}

void test_row_norms(std::mt19937& rnd) {
    // Q8_0 (ggml type 8): half d + 32 int8, 34 bytes per block. The oracle is the same dequant in doubles.
    const int64_t rows = 130, cols = 64;   // two chunks at 128, and a tail
    const int nb = (int) (cols / 32);
    std::vector<uint8_t> host((size_t) rows * nb * 34);
    std::uniform_real_distribution<float> uni(-2.0f, 2.0f);
    std::uniform_int_distribution<int> q(-127, 127);
    std::vector<double> expect(rows, 0.0);
    for (int64_t r = 0; r < rows; ++r) {
        for (int b = 0; b < nb; ++b) {
            uint8_t* p = host.data() + (size_t) (r * nb + b) * 34;
            const half d = __float2half_rn(uni(rnd));
            const uint16_t du = __half_as_ushort(d);
            p[0] = (uint8_t) du;
            p[1] = (uint8_t) (du >> 8);
            const double dv = __half2float(d);
            for (int i = 0; i < 32; ++i) {
                const int v = q(rnd);
                p[2 + i] = (uint8_t) (int8_t) v;
                expect[r] += ((double) v * dv) * ((double) v * dv);
            }
        }
    }
    void* d_w = dmalloc(host.size());
    float* d_norms = (float*) dmalloc(rows * sizeof(float));
    float* d_scratch = (float*) dmalloc((size_t) rows * cols * sizeof(float));
    cudaMemcpy(d_w, host.data(), host.size(), cudaMemcpyHostToDevice);
    lamina::kernels::head_screen_row_norms(8, d_w, rows, cols, 128, d_norms, d_scratch, nullptr);
    std::vector<float> got(rows);
    cudaMemcpy(got.data(), d_norms, rows * sizeof(float), cudaMemcpyDeviceToHost);
    for (int64_t r = 0; r < rows; ++r) {
        const double want = std::sqrt(expect[r]);
        check(std::fabs(got[r] - want) <= 1e-3 * want + 1e-4, "row norms match the host dequant");
    }
    cudaFree(d_w);
    cudaFree(d_norms);
    cudaFree(d_scratch);
}

void test_count() {
    // norms {1,2,3,4}; column 0: xnorms 1, tau 2.5 keeps the rows whose bound 1.0001*n + 1e-4 clears it:
    // {3,4} = 2. Column 1: xnorms 2, tau 3.5: {2,3,4} = 3. Both under 32: bucket 0, one sample per column.
    float norms[4] = {1.0f, 2.0f, 3.0f, 4.0f}, xn[2] = {1.0f, 2.0f}, tau[2] = {2.5f, 3.5f};
    void* d_n = dmalloc(sizeof norms);
    void* d_x = dmalloc(sizeof xn);
    void* d_t = dmalloc(sizeof tau);
    unsigned long long* d_cur = (unsigned long long*) dmalloc(2 * sizeof(unsigned long long));
    unsigned* d_done = (unsigned*) dmalloc(sizeof(unsigned));
    const size_t hb = 2 * lamina::kernels::kHeadScreenBuckets;
    unsigned long long* d_hist = (unsigned long long*) dmalloc(hb * sizeof(unsigned long long));
    cudaMemcpy(d_n, norms, sizeof norms, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, xn, sizeof xn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_t, tau, sizeof tau, cudaMemcpyHostToDevice);
    cudaMemset(d_cur, 0, 2 * sizeof(unsigned long long));
    cudaMemset(d_done, 0, sizeof(unsigned));
    cudaMemset(d_hist, 0, hb * sizeof(unsigned long long));
    lamina::kernels::head_screen_count((const float*) d_n, 4, (const float*) d_x, 2, (const float*) d_t, d_cur,
                                       d_done, d_hist, nullptr);
    std::vector<unsigned long long> hist(hb);
    cudaMemcpy(hist.data(), d_hist, hb * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    check(hist[0] == 1 && hist[1] == 1, "count writes one bucket sample per column");
    for (size_t i = 2; i < hist.size(); ++i) check(hist[i] == 0, "count leaves the other buckets empty");
    cudaFree(d_n);
    cudaFree(d_x);
    cudaFree(d_t);
    cudaFree(d_cur);
    cudaFree(d_done);
    cudaFree(d_hist);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::strcmp(argv[1], "--selftest") != 0) {
        std::printf("usage: head_screen_parity --selftest\n");
        return 2;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("skip: no CUDA device\n");
        return 77;
    }
    std::mt19937 rnd(20261112);
    test_kth(rnd);
    test_input_norms(rnd);
    test_row_norms(rnd);
    test_count();
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("head_screen_parity: ok\n");
    return 0;
}
