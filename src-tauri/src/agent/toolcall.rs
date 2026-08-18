//! Tolerant tool-call extraction.
//!
//! Strong models return tool calls in Ollama's native `tool_calls` field and
//! nothing here runs. Weak models — the ones this project exists for — emit the
//! call as prose instead: wrapped in `<tool_call>` tags, inside a fenced code
//! block, as a bare object mid-sentence, or as Python-ish call syntax. A strict
//! parser sees no tool call, the agent loop stalls, and the session dies before
//! any work happens. That failure is the single most common reason local agents
//! appear "broken".
//!
//! So we try progressively looser strategies and stop at the first that yields a
//! call. Each strategy records how it matched, which the UI surfaces so users
//! can see when a model needed rescuing.

use serde_json::{Map, Value};
use std::sync::LazyLock;

use crate::ollama::types::{ToolCall, ToolCallFunction};

/// How a call was recovered. Ordered loosest-last; also shown in the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Recovery {
    /// Model used the API correctly.
    Native,
    /// `<tool_call>` / `<function_call>` / `<invoke>` tags.
    XmlTag,
    /// ```json fenced block.
    CodeFence,
    /// Bare JSON object located in free text.
    LooseJson,
    /// `<function=name>{...}</function>`.
    FunctionTag,
    /// `name(arg="value")`.
    PythonCall,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Extracted {
    pub calls: Vec<ToolCall>,
    pub recovery: Recovery,
    /// Content with the call syntax stripped out, so the UI does not display
    /// raw JSON to the user as if it were prose.
    pub residual_text: String,
}

static XML_TAG: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::RegexBuilder::new(
        r"<(?:tool_call|tool-call|function_call|invoke)>\s*(.*?)\s*</(?:tool_call|tool-call|function_call|invoke)>",
    )
    .dot_matches_new_line(true)
    .build()
    .expect("static regex")
});

static FUNCTION_TAG: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::RegexBuilder::new(r"<function\s*=\s*([A-Za-z0-9_.-]+)\s*>\s*(.*?)\s*</function>")
        .dot_matches_new_line(true)
        .build()
        .expect("static regex")
});

static CODE_FENCE: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::RegexBuilder::new(r"```(?:json|tool_code|tool|python)?\s*\n(.*?)\n?\s*```")
        .dot_matches_new_line(true)
        .build()
        .expect("static regex")
});

static PYTHON_CALL: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*([^)]*)\s*\)\s*$")
        .expect("static regex")
});

static PY_KWARG: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r#"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|[^,]+)"#)
        .expect("static regex")
});

/// Extract tool calls from a model turn.
///
/// `native` is whatever arrived in the response's `tool_calls` field; when it is
/// non-empty we trust it and skip text scanning entirely.
pub fn extract(content: &str, native: Option<&[ToolCall]>, known_tools: &[String]) -> Option<Extracted> {
    if let Some(calls) = native {
        if !calls.is_empty() {
            return Some(Extracted {
                calls: calls.to_vec(),
                recovery: Recovery::Native,
                residual_text: content.trim().to_string(),
            });
        }
    }

    for strategy in [
        try_xml_tag,
        try_function_tag,
        try_code_fence,
        try_loose_json,
        try_python_call,
    ] {
        if let Some(found) = strategy(content, known_tools) {
            if !found.calls.is_empty() {
                return Some(found);
            }
        }
    }
    None
}

fn try_xml_tag(content: &str, known: &[String]) -> Option<Extracted> {
    collect_spans(content, &XML_TAG, Recovery::XmlTag, known, |caps| {
        parse_call_object(caps.get(1)?.as_str(), known)
    })
}

fn try_function_tag(content: &str, known: &[String]) -> Option<Extracted> {
    collect_spans(content, &FUNCTION_TAG, Recovery::FunctionTag, known, |caps| {
        let name = caps.get(1)?.as_str().to_string();
        let args = repair_json(caps.get(2)?.as_str())
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default();
        Some(ToolCall {
            function: ToolCallFunction { name, arguments: Value::Object(args) },
        })
    })
}

fn try_code_fence(content: &str, known: &[String]) -> Option<Extracted> {
    collect_spans(content, &CODE_FENCE, Recovery::CodeFence, known, |caps| {
        parse_call_object(caps.get(1)?.as_str(), known)
    })
}

fn try_python_call(content: &str, known: &[String]) -> Option<Extracted> {
    collect_spans(content, &PYTHON_CALL, Recovery::PythonCall, known, |caps| {
        let name = caps.get(1)?.as_str().to_string();
        if !known.iter().any(|k| k == &name) {
            return None; // Avoid mistaking ordinary prose for a call.
        }
        let mut args = Map::new();
        for kw in PY_KWARG.captures_iter(caps.get(2)?.as_str()) {
            let key = kw.get(1)?.as_str().to_string();
            args.insert(key, scalar_from_literal(kw.get(2)?.as_str().trim()));
        }
        Some(ToolCall {
            function: ToolCallFunction { name, arguments: Value::Object(args) },
        })
    })
}

/// Run `build` over every regex match, accumulating calls and removing the
/// matched spans from the residual text.
fn collect_spans<F>(
    content: &str,
    re: &regex::Regex,
    recovery: Recovery,
    _known: &[String],
    mut build: F,
) -> Option<Extracted>
where
    F: FnMut(&regex::Captures) -> Option<ToolCall>,
{
    let mut calls = Vec::new();
    let mut residual = String::new();
    let mut last = 0usize;

    for caps in re.captures_iter(content) {
        let whole = caps.get(0)?;
        if let Some(call) = build(&caps) {
            residual.push_str(&content[last..whole.start()]);
            last = whole.end();
            calls.push(call);
        }
    }
    if calls.is_empty() {
        return None;
    }
    residual.push_str(&content[last..]);
    Some(Extracted { calls, recovery, residual_text: residual.trim().to_string() })
}

/// Scan free text for a balanced `{...}` span that parses as a call.
fn try_loose_json(content: &str, known: &[String]) -> Option<Extracted> {
    let bytes = content.as_bytes();
    let mut calls = Vec::new();
    let mut residual = String::new();
    let mut last = 0usize;
    let mut i = 0usize;

    while i < bytes.len() {
        if bytes[i] != b'{' {
            i += 1;
            continue;
        }
        match balanced_end(content, i) {
            Some(end) => {
                if let Some(call) = parse_call_object(&content[i..end], known) {
                    residual.push_str(&content[last..i]);
                    last = end;
                    calls.push(call);
                    i = end;
                } else {
                    i += 1;
                }
            }
            // Never closed — generation almost certainly hit the token limit
            // mid-object. `repair_json` can close the open braces, so give the
            // tail one chance rather than discarding a recoverable call.
            None => {
                if let Some(call) = parse_call_object(&content[i..], known) {
                    residual.push_str(&content[last..i]);
                    last = content.len();
                    calls.push(call);
                }
                break;
            }
        }
    }

    if calls.is_empty() {
        return None;
    }
    residual.push_str(&content[last..]);
    Some(Extracted {
        calls,
        recovery: Recovery::LooseJson,
        residual_text: residual.trim().to_string(),
    })
}

/// Index just past the `}` that closes the object starting at `start`,
/// respecting string literals and escapes.
fn balanced_end(s: &str, start: usize) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escaped = false;
    let mut quote = b'"';

    for (offset, &b) in bytes[start..].iter().enumerate() {
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == quote {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' | b'\'' => {
                in_str = true;
                quote = b;
            }
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(start + offset + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// Interpret one JSON-ish object as a tool call.
///
/// Accepts the several key spellings models use for the same two concepts:
/// the tool's name and its arguments.
fn parse_call_object(raw: &str, known: &[String]) -> Option<ToolCall> {
    let value = repair_json(raw)?;

    // Some models wrap the call: {"tool_call": {...}} or {"function": {...}}.
    let obj = value.as_object()?;
    for wrapper in ["tool_call", "function_call", "invoke"] {
        if let Some(inner) = obj.get(wrapper) {
            if let Some(call) = call_from_map(inner.as_object()?, known) {
                return Some(call);
            }
        }
    }
    call_from_map(obj, known)
}

fn call_from_map(obj: &Map<String, Value>, known: &[String]) -> Option<ToolCall> {
    let name = ["name", "tool", "tool_name", "function", "action", "recipient_name"]
        .iter()
        .find_map(|k| match obj.get(*k) {
            // `{"function": {"name": ...}}` — descend one level.
            Some(Value::Object(inner)) => {
                inner.get("name").and_then(Value::as_str).map(str::to_string)
            }
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })?;

    // Models rename tools freely, so compare loosely and adopt the canonical
    // spelling. A name that still matches nothing is far more likely to be
    // prose than a call, and is rejected.
    let name = resolve_tool_name(&name, known)?;

    let args = ["arguments", "args", "parameters", "params", "input", "tool_input"]
        .iter()
        .find_map(|k| obj.get(*k))
        .cloned()
        .or_else(|| {
            obj.get("function")
                .and_then(Value::as_object)
                .and_then(|f| f.get("arguments").cloned())
        })
        .unwrap_or(Value::Object(Map::new()));

    // Arguments occasionally arrive as a JSON string rather than an object.
    let args = match args {
        Value::String(s) => repair_json(&s).unwrap_or(Value::Object(Map::new())),
        other => other,
    };

    Some(ToolCall {
        function: ToolCallFunction { name, arguments: args },
    })
}

/// Match a model's spelling of a tool name against the canonical one.
///
/// Models re-case and re-punctuate names constantly: `qwen2.5-coder:0.5b`
/// answers a `read_file` schema with `"ReadFile"`, and Llama variants emit
/// `read-file`. Comparing verbatim throws away a perfectly good call, so
/// comparison ignores case and separators — and the *canonical* spelling is
/// returned, because the executor looks tools up by their registered name.
fn resolve_tool_name(name: &str, known: &[String]) -> Option<String> {
    if known.is_empty() {
        return Some(name.to_string());
    }
    if let Some(exact) = known.iter().find(|k| k.as_str() == name) {
        return Some(exact.clone());
    }
    let fold = |s: &str| {
        s.chars()
            .filter(char::is_ascii_alphanumeric)
            .flat_map(char::to_lowercase)
            .collect::<String>()
    };
    let target = fold(name);
    if target.is_empty() {
        return None;
    }
    known.iter().find(|k| fold(k) == target).cloned()
}

/// Parse JSON, falling back to repairs for the malformations weak models make.
pub fn repair_json(raw: &str) -> Option<Value> {
    let trimmed = raw.trim().trim_start_matches("json").trim();
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return Some(v);
    }

    let mut s = trimmed.to_string();
    s = strip_trailing_commas(&s);
    s = quote_bare_keys(&s);
    s = single_to_double_quotes(&s);
    s = s
        .replace("True", "true")
        .replace("False", "false")
        .replace("None", "null")
        .replace("NaN", "null");

    if let Ok(v) = serde_json::from_str::<Value>(&s) {
        return Some(v);
    }
    // Truncated output: close whatever braces are still open.
    close_open_braces(&s).and_then(|c| serde_json::from_str::<Value>(&c).ok())
}

/// Apply `f` to each byte outside of string literals.
fn outside_strings(s: &str, mut f: impl FnMut(&mut String, usize, u8, &[u8])) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut in_str = false;
    let mut escaped = false;
    let mut quote = b'"';

    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            out.push(b as char);
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == quote {
                in_str = false;
            }
            continue;
        }
        if b == b'"' || b == b'\'' {
            in_str = true;
            quote = b;
            out.push(b as char);
            continue;
        }
        f(&mut out, i, b, bytes);
    }
    out
}

fn strip_trailing_commas(s: &str) -> String {
    outside_strings(s, |out, i, b, bytes| {
        if b == b',' {
            let next = bytes[i + 1..].iter().find(|c| !c.is_ascii_whitespace());
            if matches!(next, Some(b'}') | Some(b']')) {
                return; // drop it
            }
        }
        out.push(b as char);
    })
}

fn quote_bare_keys(s: &str) -> String {
    static BARE_KEY: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)").expect("static regex")
    });
    BARE_KEY.replace_all(s, r#"$1"$2"$3"#).into_owned()
}

fn single_to_double_quotes(s: &str) -> String {
    // Only rewrite when there is no double-quoted string to confuse things.
    if s.contains('"') {
        return s.to_string();
    }
    s.replace('\'', "\"")
}

fn close_open_braces(s: &str) -> Option<String> {
    let mut stack = Vec::new();
    let mut in_str = false;
    let mut escaped = false;

    for &b in s.as_bytes() {
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => stack.push(b'}'),
            b'[' => stack.push(b']'),
            b'}' | b']' => {
                stack.pop()?;
            }
            _ => {}
        }
    }
    if stack.is_empty() && !in_str {
        return None; // Nothing to repair; the failure was something else.
    }

    let mut out = s.trim_end().trim_end_matches(',').to_string();
    if in_str {
        out.push('"');
    }
    while let Some(closer) = stack.pop() {
        out.push(closer as char);
    }
    Some(out)
}

fn scalar_from_literal(lit: &str) -> Value {
    let unquoted = lit
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .or_else(|| lit.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')));
    if let Some(s) = unquoted {
        return Value::String(s.to_string());
    }
    match lit {
        "true" | "True" => Value::Bool(true),
        "false" | "False" => Value::Bool(false),
        "None" | "null" => Value::Null,
        _ => lit
            .parse::<i64>()
            .map(Value::from)
            .or_else(|_| lit.parse::<f64>().map(Value::from))
            .unwrap_or_else(|_| Value::String(lit.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Vec<String> {
        vec!["read_file".into(), "run_command".into(), "list_dir".into()]
    }

    fn assert_read_file(ex: &Extracted, expected_path: &str, expected: Recovery) {
        assert_eq!(ex.recovery, expected);
        assert_eq!(ex.calls.len(), 1);
        assert_eq!(ex.calls[0].function.name, "read_file");
        assert_eq!(
            ex.calls[0].function.arguments_object().get("path").and_then(Value::as_str),
            Some(expected_path)
        );
    }

    #[test]
    fn native_calls_win() {
        let native = vec![ToolCall {
            function: ToolCallFunction {
                name: "read_file".into(),
                arguments: serde_json::json!({"path": "a.txt"}),
            },
        }];
        let ex = extract("ignored", Some(&native), &tools()).unwrap();
        assert_read_file(&ex, "a.txt", Recovery::Native);
    }

    #[test]
    fn qwen_xml_tags() {
        let text = r#"Let me look.
<tool_call>
{"name": "read_file", "arguments": {"path": "src/main.rs"}}
</tool_call>"#;
        let ex = extract(text, None, &tools()).unwrap();
        assert_read_file(&ex, "src/main.rs", Recovery::XmlTag);
        assert_eq!(ex.residual_text, "Let me look.");
    }

    #[test]
    fn fenced_json_block() {
        let text = "I'll read it:\n```json\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.rs\"}}\n```";
        let ex = extract(text, None, &tools()).unwrap();
        assert_read_file(&ex, "a.rs", Recovery::CodeFence);
    }

    #[test]
    fn llama_function_tag() {
        let text = r#"<function=read_file>{"path": "x.md"}</function>"#;
        let ex = extract(text, None, &tools()).unwrap();
        assert_read_file(&ex, "x.md", Recovery::FunctionTag);
    }

    #[test]
    fn bare_json_in_prose() {
        let text = r#"Sure. {"name": "read_file", "arguments": {"path": "z.py"}} Done."#;
        let ex = extract(text, None, &tools()).unwrap();
        assert_read_file(&ex, "z.py", Recovery::LooseJson);
    }

    #[test]
    fn python_style_call() {
        let text = "read_file(path=\"conf.toml\")";
        let ex = extract(text, None, &tools()).unwrap();
        assert_read_file(&ex, "conf.toml", Recovery::PythonCall);
    }

    #[test]
    fn repairs_single_quotes_and_trailing_comma() {
        let text = "<tool_call>{'name': 'read_file', 'arguments': {'path': 'a.txt',},}</tool_call>";
        let ex = extract(text, None, &tools()).unwrap();
        assert_read_file(&ex, "a.txt", Recovery::XmlTag);
    }

    #[test]
    fn repairs_unquoted_keys() {
        let text = r#"<tool_call>{name: "read_file", arguments: {path: "a.txt"}}</tool_call>"#;
        let ex = extract(text, None, &tools()).unwrap();
        assert_read_file(&ex, "a.txt", Recovery::XmlTag);
    }

    #[test]
    fn repairs_truncated_output() {
        // Generation hit the token limit mid-object.
        let text = r#"<tool_call>{"name": "read_file", "arguments": {"path": "a.txt""#;
        let ex = extract(text, None, &tools()).unwrap();
        assert_read_file(&ex, "a.txt", Recovery::LooseJson);
    }

    #[test]
    fn accepts_alternate_key_spellings() {
        let text = r#"{"tool": "read_file", "parameters": {"path": "a.txt"}}"#;
        let ex = extract(text, None, &tools()).unwrap();
        assert_read_file(&ex, "a.txt", Recovery::LooseJson);
    }

    #[test]
    fn accepts_nested_function_shape() {
        let text = r#"{"function": {"name": "read_file", "arguments": {"path": "a.txt"}}}"#;
        let ex = extract(text, None, &tools()).unwrap();
        assert_read_file(&ex, "a.txt", Recovery::LooseJson);
    }

    #[test]
    fn arguments_as_json_string() {
        let text = r#"<tool_call>{"name": "read_file", "arguments": "{\"path\": \"a.txt\"}"}</tool_call>"#;
        let ex = extract(text, None, &tools()).unwrap();
        assert_read_file(&ex, "a.txt", Recovery::XmlTag);
    }

    #[test]
    fn multiple_calls_in_one_turn() {
        let text = r#"<tool_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call>
<tool_call>{"name":"list_dir","arguments":{"path":"."}}</tool_call>"#;
        let ex = extract(text, None, &tools()).unwrap();
        assert_eq!(ex.calls.len(), 2);
        assert_eq!(ex.calls[1].function.name, "list_dir");
    }

    #[test]
    fn plain_prose_is_not_a_call() {
        assert!(extract("I think we should read the file first.", None, &tools()).is_none());
    }

    #[test]
    fn pascal_case_name_resolves_to_the_snake_case_tool() {
        // Observed from qwen2.5-coder:0.5b answering a `read_file` schema:
        // it emits "ReadFile" in a fenced block. Rejecting that loses a
        // perfectly good call.
        let text = "```json
{
  \"name\": \"ReadFile\",
  \"arguments\": {
    \"path\": \"src/main.rs\"
  }
}
```";
        let ex = extract(text, None, &tools()).unwrap();
        assert_eq!(ex.recovery, Recovery::CodeFence);
        // The canonical spelling is what the executor looks up.
        assert_eq!(ex.calls[0].function.name, "read_file");
    }

    #[test]
    fn separator_and_case_variants_all_resolve() {
        for spelling in ["read_file", "ReadFile", "readFile", "read-file", "READ_FILE"] {
            let text = format!(r#"{{"name": "{spelling}", "arguments": {{"path": "a"}}}}"#);
            let ex = extract(&text, None, &tools())
                .unwrap_or_else(|| panic!("{spelling} should resolve"));
            assert_eq!(ex.calls[0].function.name, "read_file", "from {spelling}");
        }
    }

    #[test]
    fn bare_json_from_a_tool_capable_model_is_recovered() {
        // qwen2.5-coder:7b returns this in `content` with an empty native
        // `tool_calls` field — a strict parser stalls the agent here.
        let text = r#"{"name": "read_file", "arguments": {"path": "src/main.rs"}}"#;
        let ex = extract(text, None, &tools()).unwrap();
        assert_eq!(ex.recovery, Recovery::LooseJson);
        assert_eq!(ex.calls[0].function.name, "read_file");
    }

    #[test]
    fn unknown_tool_name_is_rejected() {
        let text = r#"{"name": "definitely_not_a_tool", "arguments": {}}"#;
        assert!(extract(text, None, &tools()).is_none());
    }

    #[test]
    fn json_example_in_prose_is_not_a_call() {
        // A config snippet must not be mistaken for an invocation.
        let text = r#"Your config should look like {"host": "localhost", "port": 8080}."#;
        assert!(extract(text, None, &tools()).is_none());
    }
}
