// include/lamina/program/engine.hpp - the engine as a function: the `lamina` binary's main, and the engine that
// lamina-server runs on a thread of its own (with in-process channels: serve_io.hpp).
#pragma once

namespace lamina::program {

/// The engine's command line (argv[0] is the program's name). Returns its exit code.
int run(int argc, char** argv);

/// Where Qwen3.8-Flash-Next's draft layer comes from, for the warnings of a start without it (the engine's log, the
/// server's window): this platform's download script and the layer's repository.
#ifdef _WIN32
inline constexpr const char* kDraftLayerScript = "download-draft-layer.bat";
#else
inline constexpr const char* kDraftLayerScript = "download-draft-layer.sh";
#endif
inline constexpr const char* kDraftLayerUrl = "https://huggingface.co/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP";

}  // namespace lamina::program
