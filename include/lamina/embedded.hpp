// include/lamina/embedded.hpp - files built into the programs (cmake/embed_files.cmake): the web app, the server's
// presets, the engine's expert profile.
#pragma once

#include <string_view>

namespace lamina {

struct EmbeddedFile {
    std::string_view path;   ///< below the folder it was taken from ("index.html", "iq3_s.json")
    std::string_view data;
};

/// src/server/web's files (lamina-server)
const EmbeddedFile* web_asset(std::string_view path);
/// examples/*.json, the presets lamina-server builds a config from (src/server/model_config.hpp)
const EmbeddedFile* preset_file(std::string_view path);
/// data/expert-profile.bin, Qwen3.8-Flash-Next's built-in expert profile (lamina/core/expert_cache.hpp)
const EmbeddedFile* profile_file(std::string_view path);

}  // namespace lamina
