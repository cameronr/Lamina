// src/hy4/hy4_dfly_file.cpp - DFly2's file: the safetensors shard and the config.json beside it
#include "hy4_dfly_file.hpp"

#include <nlohmann/json.hpp>

#include <cstring>
#include <exception>
#include <filesystem>

namespace lamina::hy4 {

bool SafeTensors::open(const std::string& path, std::string& err) {
    in_.open(std::filesystem::path(path), std::ios::binary);
    uint64_t n = 0;
    if (!in_ || !in_.read(reinterpret_cast<char*>(&n), 8) || n < 2 || n > (uint64_t{64} << 20)) {
        err = path + ": not a safetensors file";
        return false;
    }
    std::string text((size_t) n, '\0');
    if (!in_.read(text.data(), (std::streamsize) n)) {
        err = path + ": its header is cut short";
        return false;
    }
    const nlohmann::json head = nlohmann::json::parse(text, nullptr, false);
    if (head.is_discarded() || !head.is_object()) {
        err = path + ": its header is not a JSON object";
        return false;
    }
    try {
        for (const auto& [name, t] : head.items()) {
            if (name == "__metadata__") continue;
            Entry e;
            e.dtype = t.at("dtype").get<std::string>();
            e.shape = t.at("shape").get<std::vector<int64_t>>();
            e.begin = t.at("data_offsets").at(0).get<uint64_t>();
            e.end = t.at("data_offsets").at(1).get<uint64_t>();
            entries_.emplace(name, std::move(e));
        }
    } catch (const std::exception& e) {
        err = path + ": " + e.what();
        return false;
    }
    data_ = 8 + n;
    return true;
}

bool SafeTensors::read(const std::string& name, std::vector<float>& out, std::vector<int64_t>& shape) {
    const auto it = entries_.find(name);
    if (it == entries_.end()) return false;
    const Entry& e = it->second;
    const uint64_t width = e.dtype == "BF16" ? 2 : e.dtype == "F32" ? 4 : 0;
    uint64_t count = 1;
    for (const int64_t d : e.shape) count *= (uint64_t) d;
    if (width == 0 || e.end - e.begin != count * width) return false;
    std::vector<uint8_t> raw((size_t) (e.end - e.begin));
    in_.clear();
    if (!in_.seekg((std::streamoff) (data_ + e.begin)) ||
        !in_.read(reinterpret_cast<char*>(raw.data()), (std::streamsize) raw.size()))
        return false;
    out.resize((size_t) count);
    if (width == 4) {
        std::memcpy(out.data(), raw.data(), raw.size());
    } else {
        for (size_t i = 0; i < out.size(); ++i) {
            const uint32_t u = (uint32_t) (raw[2 * i] | (raw[2 * i + 1] << 8)) << 16;
            std::memcpy(&out[i], &u, 4);
        }
    }
    shape = e.shape;
    return true;
}

bool read_dfly_config(const std::string& dir, DFlyConfig& c, std::string& err) {
    const std::filesystem::path p = std::filesystem::path(dir) / "config.json";
    std::ifstream in(p);
    if (!in) {
        err = p.string() + " is missing: the checkpoint's config.json goes beside its DFly shard";
        return false;
    }
    const nlohmann::json j = nlohmann::json::parse(in, nullptr, false);
    if (j.is_discarded() || !j.is_object() || !j.contains("dfly_target_kv_layers")) {
        err = p.string() + ": no dfly_target_kv_layers (not Hy4-flash's config with DFly)";
        return false;
    }
    try {
        c.kv_layers = j.at("dfly_target_kv_layers").get<std::vector<int>>();
        std::vector<int> index = c.kv_layers;
        if (const auto& v = j.value("dfly_target_index_layer", nlohmann::json()); v.is_number())
            index.assign(c.kv_layers.size(), v.get<int>());
        else if (v.is_array())
            index = v.get<std::vector<int>>();
        c.mask = j.at("dfly_mask_token_id").get<int32_t>();
        c.width = j.at("speculative_config").at("num_speculative_tokens").get<int>() + 1;
        c.top_k = j.at("dfly_target_index_topk").get<int>();
        const nlohmann::json window = j.value("dfly_context_local_window", nlohmann::json());
        if (index != c.kv_layers) {
            err = p.string() + ": DFly's index layers differ from its K/V layers, which this path does not run";
            return false;
        }
        if (!window.is_null() && window.get<int>() != 0) {
            err = p.string() + ": DFly's local window (dfly_context_local_window) is not run on this path";
            return false;
        }
        if (j.value("dfly_num_hidden_layers", 0) != (int) c.kv_layers.size()) {
            err = p.string() + ": dfly_num_hidden_layers is not one a K/V layer";
            return false;
        }
    } catch (const std::exception& e) {
        err = p.string() + ": " + e.what();
        return false;
    }
    return true;
}

}  // namespace lamina::hy4
