// lamina-server -m's config pieces that need no model and no GPU (src/server/model_config.hpp): the preset by the
// header's architecture and quantizer, the main GPU and the image encoder's, a GPU's name in a config, the cards a
// config's --main-gpu and --second-gpu name (none for a --main-gpu the engine would not find), the command
// line's engine options in place of the config's, image projectors told apart by their width, and a config written
// over an earlier one.  The projectors are small GGUF headers written here.  (CPU only.)
#include "config.hpp"
#include "model_config.hpp"
#include "process.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using lamina::json;
namespace fs = std::filesystem;

static int fails = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        if (!(c)) {                                                               \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);   \
            ++fails;                                                              \
        }                                                                         \
    } while (0)

namespace {

/// a GGUF v3 header without tensors: string, bool and uint32 keys
struct Header {
    std::vector<std::pair<std::string, std::string>> strs;
    std::vector<std::pair<std::string, bool>> bools;
    std::vector<std::pair<std::string, uint32_t>> u32s;
    void write(const fs::path& path, size_t pad_to = 0) const {
        std::string b = "GGUF";
        auto u32 = [&](uint32_t v) { b.append((const char*) &v, 4); };
        auto u64 = [&](uint64_t v) { b.append((const char*) &v, 8); };
        auto str = [&](const std::string& s) {
            u64(s.size());
            b += s;
        };
        u32(3);
        u64(0);
        u64(strs.size() + bools.size() + u32s.size());
        for (const auto& [k, v] : strs) {
            str(k);
            u32(8);
            str(v);
        }
        for (const auto& [k, v] : bools) {
            str(k);
            u32(7);
            b += (char) (v ? 1 : 0);
        }
        for (const auto& [k, v] : u32s) {
            str(k);
            u32(4);
            u32(v);
        }
        while (b.size() % 32 != 0 || b.size() < pad_to) b += '\0';
        std::ofstream(path, std::ios::binary) << b;
    }
};

void projector(const fs::path& path, uint32_t dim, size_t size = 0) {
    Header h;
    h.strs = {{"general.architecture", "clip"}};
    h.bools = {{"clip.has_vision_encoder", true}};
    h.u32s = {{"clip.vision.projection_dim", dim}};
    h.write(path, size);
}

lamina::GpuInfo gpu(int index, const char* name, const char* uuid, int64_t mib, int cc, bool display) {
    lamina::GpuInfo g{index, name, "", uuid};
    g.memory_mib = mib;
    g.cc = cc;
    g.display = display;
    return g;
}

std::vector<std::string> args(const json& cfg) {
    std::vector<std::string> out;
    for (const auto& a : cfg["args"]) out.push_back(a.get<std::string>());
    return out;
}

}  // namespace

int main() {
    // the presets: by architecture, and for Flash-Next by quantizer
    CHECK(lamina::preset_name("qwen4exp", "ISTA DASLab") == "iq3_s");
    CHECK(lamina::preset_name("qwen4exp", "Unsloth") == "ud-q4_k_xl");
    CHECK(lamina::preset_name("qwen4exp", "Agention") == "ud-q4_k_xl");
    CHECK(lamina::preset_name("qwen4exp", "") == "ud-q4_k_xl");
    CHECK(lamina::preset_name("qwen35moe", "Unsloth") == "qwen36-ud-q4_k_xl");
    CHECK(lamina::preset_name("qwen35", "") == "qwen38-27b-ud-q6_k_m");
    CHECK(lamina::preset_name("llama", "Unsloth").empty());
    for (const char* name : {"iq3_s", "ud-q4_k_xl", "qwen36-ud-q4_k_xl", "qwen38-27b-ud-q6_k_m"}) {
        const json p = lamina::preset(name);   // built in from examples/
        const std::vector<std::string> a = args(p);
        CHECK(std::find(a.begin(), a.end(), "--native") != a.end());
        CHECK(p["port"].is_number_integer());
    }

    // the main GPU: no display first, then the most VRAM, then the lower number; compute capability 8.0 or newer
    const std::vector<lamina::GpuInfo> pc = {gpu(0, "NVIDIA GeForce RTX 5070 Ti", "GPU-aaa", 16303, 120, true),
                                             gpu(1, "NVIDIA GeForce RTX 3090", "GPU-bbb", 24576, 86, false),
                                             gpu(2, "NVIDIA GeForce RTX 3090", "GPU-ccc", 24576, 86, false)};
    CHECK(lamina::pick_main_gpu(pc, {}) == 1);
    CHECK(lamina::pick_main_gpu(pc, {0, 2}) == 2);
    CHECK(lamina::pick_main_gpu(pc, {0}) == 0);   // only the display's card is left: it runs the model
    CHECK(lamina::pick_main_gpu({}, {}) == -1);
    const std::vector<lamina::GpuInfo> old = {gpu(0, "Tesla V100", "GPU-v", 32768, 70, false),
                                              gpu(1, "NVIDIA GeForce RTX 4080", "GPU-x", 16376, 89, true)};
    CHECK(lamina::pick_main_gpu(old, {}) == 1);
    CHECK(lamina::pick_main_gpu({gpu(0, "Tesla V100", "GPU-v", 32768, 70, false)}, {}) == -1);
    CHECK(lamina::gpu_spec(pc[0], pc) == "NVIDIA GeForce RTX 5070 Ti");
    CHECK(lamina::gpu_spec(pc[1], pc) == "GPU-bbb");   // two cards answer to that name

    // the image encoder's GPU: besides the main one, no display first, then the least VRAM, then the higher number
    CHECK(lamina::pick_vision_gpu(pc, 1) == 2);   // this development PC's: the other 3090, not the display's 5070 Ti
    CHECK(lamina::pick_vision_gpu(pc, 2) == 1);
    const std::vector<lamina::GpuInfo> three = {gpu(0, "NVIDIA GeForce RTX 3090", "GPU-a", 24576, 86, false),
                                                gpu(1, "NVIDIA GeForce RTX 4060 Ti", "GPU-b", 16380, 89, false),
                                                gpu(2, "NVIDIA GeForce RTX 4060 Ti", "GPU-c", 16380, 89, false)};
    CHECK(lamina::pick_vision_gpu(three, 0) == 2);   // the least VRAM; of two alike the higher number
    CHECK(lamina::pick_vision_gpu({pc[0], pc[1]}, 1) == 0);   // only a display's card is left besides the main one
    CHECK(lamina::pick_vision_gpu({pc[1]}, 1) == 1);   // one card: the main one
    CHECK(lamina::pick_vision_gpu(old, 1) == 1);   // the V100 is too old: the main card
    CHECK(lamina::pick_vision_gpu(pc, -1) == 2);   // the main card unknown: the least VRAM without a display
    CHECK(lamina::pick_vision_gpu({}, -1) == -1);

    // the cards the engine works on, as it reads --main-gpu: one it cannot find gives none and why, not a guess
    auto on = [&](std::vector<std::string> a, const std::vector<lamina::GpuInfo>& gpus, std::string* why = nullptr) {
        a.insert(a.begin(), {"--native", "m.gguf"});
        return lamina::engine_gpus(json{{"args", a}}, gpus, why);
    };
    std::string why;
    CHECK((on({"--main-gpu", "GPU-bbb", "--second-gpu", "auto"}, pc) == std::vector<int>{1, 0, 2}));
    CHECK((on({"--main-gpu", "RTX 5070", "--second-gpu", "GPU-ccc"}, pc) == std::vector<int>{0, 2}));
    CHECK((on({"--second-gpu", "auto"}, pc) == std::vector<int>{0, 1, 2}));   // the engine's default, GPU 0
    CHECK((on({"--main-gpu", "2", "--second-gpu", "GPU-dead"}, pc, &why) == std::vector<int>{2}) && why.empty());
    CHECK(on({"--main-gpu", "RTX 3090", "--second-gpu", "auto"}, pc, &why).empty());
    CHECK(why == "--main-gpu RTX 3090: several GPUs match RTX 3090 (name one by its UUID): 1 NVIDIA GeForce RTX 3090 "
                 "GPU-bbb; 2 NVIDIA GeForce RTX 3090 GPU-ccc");
    CHECK(on({"--main-gpu", "GPU-dead"}, pc, &why).empty() && why == "--main-gpu GPU-dead: no GPU is GPU-dead");
    CHECK(on({"--main-gpu", "5"}, pc, &why).empty() && why == "--main-gpu 5: no GPU 5 (nvidia-smi lists 3) or GPU named so");
    CHECK((on({"--main-gpu", "1"}, {}) == std::vector<int>{1}));   // nvidia-smi lists nothing: the number as given
    CHECK((lamina::engine_gpus(json{{"gpu", "0,2"}, {"args", {"--main-gpu", "GPU-dead"}}}, pc) == std::vector<int>{0, 2}));

    // the command line's engine options in place of the config's, with their values and the other form's
    json cfg = {{"args", {"--native", "m.gguf", "--kv", "int8", "--kv-grow", "--second-gpu", "-1",
                          "--control-vector-layer-range", "4", "44", "--hc-q8"}}};
    lamina::apply_engine_options(cfg, {{"--kv", "q4_0"}, {"--no-kv-grow"}, {"--control-vector-layer-range", "1", "2"},
                                       {"--second-gpu", "auto"}, {"--max-context", "65536"}});
    CHECK((args(cfg) == std::vector<std::string>{"--native", "m.gguf", "--hc-q8", "--kv", "q4_0", "--no-kv-grow",
                                                  "--control-vector-layer-range", "1", "2", "--second-gpu", "auto",
                                                  "--max-context", "65536"}));
    json flip = {{"args", {"--no-kv-grow", "--spec", "4"}}};
    lamina::apply_engine_options(flip, {{"--kv-grow"}});
    CHECK((args(flip) == std::vector<std::string>{"--spec", "4", "--kv-grow"}));

    // projectors: for the model's width only, the smallest first; not one: another architecture, not a GGUF
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path() / ("lamina-model-config-test-" + std::to_string(std::rand()));
    fs::create_directories(dir);
    lamina::ModelFacts m;
    m.path = lamina::u8_str(dir / "model.gguf");
    m.n_embd = 2560;
    Header model;
    model.strs = {{"general.architecture", "qwen4exp"}};
    model.write(dir / "model.gguf");
    projector(dir / "mmproj-BF16.gguf", 2560, 4096);
    projector(dir / "mmproj-Q8_0.gguf", 2560, 2048);
    projector(dir / "MMPROJ-other.gguf", 2048);
    projector(dir / "vision.gguf", 2560);   // not named as a projector: not looked at
    std::ofstream(dir / "mmproj-broken.gguf") << "not a GGUF";
    CHECK(lamina::projector_problem(lamina::u8_str(dir / "mmproj-Q8_0.gguf"), m).empty());
    CHECK(!lamina::projector_problem(lamina::u8_str(dir / "MMPROJ-other.gguf"), m).empty());
    CHECK(!lamina::projector_problem(lamina::u8_str(dir / "model.gguf"), m).empty());
    CHECK(!lamina::projector_problem(lamina::u8_str(dir / "mmproj-broken.gguf"), m).empty());
    const std::vector<std::string> found = lamina::projectors_beside(m);
    CHECK(found.size() == 2);
    CHECK(found.size() == 2 && fs::path(lamina::u8_path(found[0])).filename() == "mmproj-Q8_0.gguf");
    CHECK(found.size() == 2 && fs::path(lamina::u8_path(found[1])).filename() == "mmproj-BF16.gguf");

    // a config written, then written again over it: the earlier file's own keys stay, the earlier file as .bak
    const std::string path = lamina::u8_str(dir / "lamina-test.json");
    CHECK(lamina::write_config(path, {{"args", {"--native", "a.gguf"}}, {"port", 8081}}).empty());
    json first;
    {
        std::ifstream f(lamina::u8_path(path), std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(f)), {});
        CHECK(lamina::py_loads(text, first) && first["port"] == 8081);
        CHECK(text.find('\r') == std::string::npos && !text.empty() && text.back() == '\n');
    }
    first["api_key"] = "secret";
    first["port"] = 9000;
    std::ofstream(lamina::u8_path(path), std::ios::binary) << lamina::py_dumps(first, true, false, 2) << "\n";
    const std::vector<std::string> kept = lamina::write_config(path, {{"args", {"--native", "b.gguf"}}, {"port", 8082}});
    CHECK((kept == std::vector<std::string>{"api_key"}));
    json second;
    {
        std::ifstream f(lamina::u8_path(path), std::ios::binary);
        CHECK(lamina::py_loads(std::string((std::istreambuf_iterator<char>(f)), {}), second));
    }
    CHECK(second["port"] == 8082 && second["api_key"] == "secret" && second["args"][1] == "b.gguf");
    CHECK(fs::is_regular_file(dir / "lamina-test.json.bak"));
    // a key the preset has, left out this time (pictures off), is not taken back from the earlier file
    second["vision"] = {{"mmproj", "p.gguf"}};
    std::ofstream(lamina::u8_path(path), std::ios::binary) << lamina::py_dumps(second, true, false, 2) << "\n";
    CHECK((lamina::write_config(path, {{"args", {"--native", "c.gguf"}}}, {"args", "port", "vision"}) ==
           std::vector<std::string>{"api_key"}));
    json third;
    {
        std::ifstream f(lamina::u8_path(path), std::ios::binary);
        CHECK(lamina::py_loads(std::string((std::istreambuf_iterator<char>(f)), {}), third));
    }
    CHECK(!third.contains("vision") && !third.contains("port") && third["api_key"] == "secret");

    fs::remove_all(dir, ec);
    if (fails == 0) std::printf("model_config_test: ok\n");
    return fails == 0 ? 0 : 1;
}
