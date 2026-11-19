// src/hy4/hy4_dfly_file.hpp - DFly2's file: the Hy4-flash checkpoint's last safetensors shard (model.dfly_layers.*)
// and the config.json beside it.  Host code apart from the engine's CUDA file (the JSON library stays out of nvcc).
#pragma once

#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace lamina::hy4 {

/// A safetensors file's tensors: the header read at open, a tensor's data on request
class SafeTensors {
public:
    bool open(const std::string& path, std::string& err);
    /// tensor `name` widened to floats (BF16 or F32), its shape outermost first; false when missing or of another type
    bool read(const std::string& name, std::vector<float>& out, std::vector<int64_t>& shape);

private:
    struct Entry {
        std::string dtype;
        std::vector<int64_t> shape;
        uint64_t begin = 0, end = 0;
    };
    std::map<std::string, Entry> entries_;
    std::ifstream in_;
    uint64_t data_ = 0;
};

/// DFly's settings from the checkpoint's config.json
struct DFlyConfig {
    std::vector<int> kv_layers;   // dfly_target_kv_layers: the trunk layer whose K/V and index picks each layer reads
    int32_t mask = -1;            // dfly_mask_token_id
    int width = 0;                // the query block's rows: speculative_config.num_speculative_tokens + 1
    int top_k = 0;                // dfly_target_index_topk
};
/// The config.json in `dir`; false (and `err`) without its DFly keys or with settings this path does not run
bool read_dfly_config(const std::string& dir, DFlyConfig& c, std::string& err);

}  // namespace lamina::hy4
