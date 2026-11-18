// lamina-server's settings table (src/server/params.hpp): every option names one key and the other way round, the
// command line's values by their kinds (switches and their --no- forms, numbers, lists given again, NAME=VALUE pairs,
// JSON), the engine's options apart, the errors, a section merged key by key, the config's keys the table lacks, and
// --help naming every option.  (CPU only.)
#include "params.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

using lamina::json;

static int fails = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        if (!(c)) {                                                               \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);   \
            ++fails;                                                              \
        }                                                                         \
    } while (0)

namespace {

lamina::CommandLine parse(std::vector<std::string> words) {
    words.insert(words.begin(), "lamina-server");
    std::vector<char*> argv;
    for (auto& w : words) argv.push_back(w.data());
    return lamina::parse_command_line((int) argv.size(), argv.data());
}

bool refused(std::vector<std::string> words) {
    try {
        parse(std::move(words));
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    // one key a flag, one flag a key; none is a command's
    std::set<std::string> flags, keys;
    for (const lamina::Param& p : lamina::kParams) {
        if (p.flag.empty()) continue;
        CHECK(keys.insert(std::string(p.key)).second);
        CHECK(flags.insert(std::string(p.flag)).second);
        std::string a(p.aliases);
        for (size_t at = 0; at < a.size();) {
            const size_t sp = std::min(a.find(' ', at), a.size());
            CHECK(flags.insert(a.substr(at, sp - at)).second);
            at = sp + 1;
        }
    }
    for (const char* c : {"--config", "--write-config", "--check", "--help", "-h"}) CHECK(flags.count(c) == 0);

    bool neg = false;
    CHECK(lamina::find_param("-m")->key == "model");
    CHECK(lamina::find_param("--temp")->key == "sampling.temperature");
    CHECK(lamina::find_param("--no-lazy", &neg)->key == "lazy_load" && neg);
    CHECK(lamina::find_param("--no-mmproj", &neg)->key == "vision" && !neg);
    CHECK(lamina::find_param("--no-port") == nullptr);
    CHECK(lamina::find_param("--max-context") == nullptr);   // the engine's

    const lamina::CommandLine cl = parse(
        {"-m", "x.gguf", "--port", "8090", "--temp", "0.6", "--top-k", "20", "--lazy", "--no-answer-room",
         "--aliases", "a,b", "--alias", "c", "--env", "A=1", "--env", "B=x=y", "--gpu", "0,2", "--replicas", "2",
         "--before-load", "[\"a\", \"b\"]", "--script", "x, y", "--script", "z", "--max-context", "65536", "--kv-grow",
         "--second-gpu", "-1", "--no-mmproj", "--mmproj", "p.gguf", "--open", "--idle-unload", "600",
         "--queue-timeout", "1.5", "--fit-max-tokens=false", "--reasoning-loop-recovery", "false", "--config", "c.json",
         "--write-config", "w.json", "--check"});
    const json& s = cl.settings;
    CHECK(s["model"] == lamina::u8_str(std::filesystem::absolute("x.gguf").lexically_normal()));   // a path
    CHECK(s["port"] == 8090 && s["port"].is_number_integer());
    CHECK(s["sampling"]["temperature"] == 0.6 && s["sampling"]["top_k"] == 20);
    CHECK(s["lazy_load"] == true && s["answer_room"] == false && s["open_browser"] == true);
    CHECK((s["aliases"] == json{"a", "b", "c"}));
    CHECK((s["env"] == json{{"A", "1"}, {"B", "x=y"}}));
    CHECK(s["gpu"] == "0,2" && s["replicas"] == 2);
    CHECK((s["before_load"] == json{"a", "b"}));
    CHECK((s["scripts"] == json{"x, y", "z"}));
    CHECK((s["vision"] == json{{"mmproj", lamina::u8_str(std::filesystem::absolute("p.gguf").lexically_normal())}}));
    // (--no-mmproj, then --mmproj: the last)
    CHECK(s["idle_unload_s"] == 600 && s["idle_unload_s"].is_number_integer());
    CHECK(s["queue_timeout"] == 1.5 && s["fit_max_tokens"] == false && s["reasoning_loop_recovery"] == false);
    CHECK(cl.engine.size() == 3 && cl.engine[0] == (std::vector<std::string>{"--max-context", "65536"}) &&
          cl.engine[1] == (std::vector<std::string>{"--kv-grow"}) &&
          cl.engine[2] == (std::vector<std::string>{"--second-gpu", "-1"}));
    CHECK(cl.config == "c.json" && cl.write_config == "w.json" && cl.check && !cl.help);
    CHECK(parse({"--mmproj", "p.gguf", "--no-mmproj"}).settings["vision"] == false);
    CHECK(parse({"--port=9000"}).settings["port"] == 9000);

    CHECK(refused({"--port", "x"}));
    CHECK(refused({"--port"}));
    CHECK(refused({"--lazy=maybe"}));
    CHECK(refused({"--no-lazy=1"}));
    CHECK(refused({"--env", "NOEQ"}));
    CHECK(refused({"--aliases", "[1"}));
    CHECK(refused({"stray"}));
    CHECK(refused({"-x"}));

    // a section merged key by key, anything else replaced
    json base = {{"sampling", {{"temperature", 0.7}, {"top_p", 0.95}}}, {"port", 8081}, {"args", {"a"}},
                 {"vision", false}};
    lamina::merge_settings(base, {{"sampling", {{"temperature", 0.6}}}, {"args", {"b"}}, {"vision", {{"gpu", true}}}});
    CHECK(base["sampling"]["temperature"] == 0.6 && base["sampling"]["top_p"] == 0.95 && base["port"] == 8081);
    CHECK((base["args"] == json{"b"}) && (base["vision"] == json{{"gpu", true}}));

    CHECK((lamina::unknown_keys({{"port", 1}, {"args", json::array()}, {"mcpServers", json::object()},
                                 {"sampling", json::object()}, {"images", json::object()}, {"portt", 2}, {"backend", "x"}})
           == std::vector<std::string>{"portt", "backend"}));

    // the benchmark's options: commands, not keys; the engine keeps its own --bench-* (the qwen35 path's --bench-pp)
    {
        const lamina::CommandLine b = parse({"--bench", "--bench-runs", "2", "--bench-clients=4", "--bench-long", "--bench-think",
                                             "--bench-out", "out", "--bench-compare", "old", "--bench-pp", "512"});
        CHECK(b.bench && b.bench_runs == 2 && b.bench_clients == 4 && b.bench_long && b.bench_think && b.bench_options);
        CHECK(std::filesystem::path(b.bench_out).is_absolute() && b.bench_compare.size() == 1);
        CHECK(b.engine.size() == 1 && b.engine[0] == (lamina::EngineOption{"--bench-pp", "512"}) && b.settings.empty());
        const lamina::CommandLine two = parse({"--bench-compare", "a", "b"});
        CHECK(two.bench_compare.size() == 2 && !two.bench && !two.bench_options);
        const lamina::CommandLine url = parse({"--bench-url", "http://127.0.0.1:8081"});
        CHECK(url.bench_url == "http://127.0.0.1:8081" && !url.bench_options);
        CHECK(refused({"--bench-runs", "0"}));
        CHECK(refused({"--bench-runs", "two"}));
        CHECK(refused({"--bench=1"}));
        CHECK(refused({"--bench-compare", "a", "b", "--bench-compare", "c"}));
    }

    const std::string help = lamina::params_help();
    for (const std::string& f : flags) CHECK(help.find(f) != std::string::npos);
    for (const std::string& k : keys) CHECK(help.find("[" + k + "]") != std::string::npos);
    for (const char* f : {"--bench ", "--bench-url", "--bench-runs", "--bench-clients", "--bench-long", "--bench-think",
                          "--bench-out", "--bench-compare"})
        CHECK(help.find(f) != std::string::npos);

    if (fails == 0) std::printf("params_test: ok (%zu options)\n", keys.size());
    return fails == 0 ? 0 : 1;
}
