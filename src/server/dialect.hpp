// src/server/dialect.hpp - the chat forms of a template that are not Qwen's: Hy4-flash's.  Its tags carry the
// template's suffix (HYTK, ":6124c78e"): reasoning in <think:..>; calls as <tool_calls:..><tool_call:..>NAME
// <arg_key:..>K</arg_key:..><arg_value:..>V</arg_value:..></tool_call:..></tool_calls:..>; reasoning_effort no_think /
// low / high / max.  They are translated to Qwen's forms (<think>, <tool_call>\n<function=NAME>\n<parameter=K>\nV\n
// </parameter>\n</function>\n</tool_call>), which the output parser, the template's probes and the request flow read,
// and the server's own Qwen-form text (a forced call's opening, the close of thinking) back to Hy4's for a prompt.
#pragma once

#include "json.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace lamina {

struct Hy4Tags {
    std::string think_open, think_close, calls_open, calls_close, call_open, call_close, key_open, key_close,
        val_open, val_close;
    /// a Hy4-flash template's tags (from its HYTK), or nullopt for another template
    static std::optional<Hy4Tags> from_template(std::string_view source);
};

/// Hy4's tags -> Qwen's.  `piece`: a token's text, translated when it is a whole tag (the tags are special tokens);
/// `text`: every tag in a rendered text.  A call's name is closed by the tag after it.
class TagTranslator {
public:
    explicit TagTranslator(Hy4Tags t) : t_(std::move(t)) {}
    std::string piece(std::string_view s);
    std::string text(std::string_view s);

private:
    std::optional<std::string> tag(std::string_view s);
    Hy4Tags t_;
    bool name_ = false;   // a call's name opened and not closed yet
};

/// the server's Qwen-form text for a prompt (the close of thinking, a forced call's opening) in Hy4's tags
std::string qwen_to_hy4(std::string_view s, const Hy4Tags& t);
/// a request's template kwargs (Qwen's: enable_thinking false, reasoning_effort low / medium / xhigh) as Hy4's
/// reasoning_effort (no_think / low / high; Hy4's own values pass)
json hy4_kwargs(const json& kwargs);

}  // namespace lamina
