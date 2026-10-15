//! Tool discovery for models that cannot see the whole tool surface.
//!
//! When the context window is too small for every schema, the runner shows a
//! relevance-ranked subset. Three things keep the hidden rest reachable:
//!
//! - [`rank_by_relevance`] orders tools by how well their name and description
//!   match the user's instruction, so the cut keeps what the task is about
//!   rather than whatever a server happened to list first.
//! - [`FIND_TOOLS`] is a built-in tool that searches the full registry and
//!   returns compact signatures; the runner then exposes the matches.
//! - [`nearest`] turns "unknown tool" into a list of real candidates, so a
//!   model that guessed a name has something to correct it with.
//!
//! Everything here is generic: it only looks at names, descriptions and
//! parameter schemas, never at a particular server or tool.

use crate::ollama::types::Tool;
use serde_json::json;
use std::collections::HashSet;

/// Name of the built-in discovery tool.
pub const FIND_TOOLS: &str = "find_tools";

/// Schema of the built-in discovery tool.
pub fn find_tools_schema() -> Tool {
    Tool::function(
        FIND_TOOLS,
        "Search ALL available tools, including ones not listed here. Use it before \
         guessing a tool name. Matches are added to your tools.",
        json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords, e.g. what you want to change or read."}
            },
            "required": ["query"]
        }),
    )
}

/// Lowercased word tokens, split on `_`, `-`, digits boundaries and any
/// non-letter. Works for Turkish and English alike (Unicode letters).
pub fn tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut prev_lower = false;
    for c in text.chars() {
        if c.is_alphanumeric() {
            // camelCase boundary.
            if c.is_uppercase() && prev_lower && !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            prev_lower = c.is_lowercase();
            cur.extend(c.to_lowercase());
        } else {
            prev_lower = false;
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out.retain(|t| t.chars().count() >= 2 && !STOP.contains(&t.as_str()));
    out
}

/// Words that match everything and so rank nothing.
const STOP: &[&str] = &[
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "it", "this", "that",
    "with", "by", "be", "as", "at", "from", "mcp", "tool", "tools", "please", "can", "you",
    "bir", "ve", "ile", "bu", "şu", "da", "de", "mi", "mı", "mu", "mü", "için", "lütfen",
];

/// Prefix-tolerant token match: "efektlerde" matches "effects"? No, but
/// "effect" matches "effects" and "config" matches "configuration".
fn token_match(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    let (short, long) = if a.chars().count() <= b.chars().count() { (a, b) } else { (b, a) };
    short.chars().count() >= 4 && long.starts_with(short)
}

/// Relevance of `tool` to `query`: name hits weigh more than description hits.
pub fn score(tool: &Tool, query: &[String]) -> usize {
    if query.is_empty() {
        return 0;
    }
    let name = tokens(&tool.function.name);
    let mut desc = tokens(&tool.function.description);
    if let Some(props) = tool.function.parameters.get("properties").and_then(|p| p.as_object()) {
        for (k, v) in props {
            desc.extend(tokens(k));
            if let Some(d) = v.get("description").and_then(|d| d.as_str()) {
                desc.extend(tokens(d));
            }
        }
    }
    let mut s = 0;
    for q in query.iter().collect::<HashSet<_>>() {
        if name.iter().any(|n| token_match(n, q)) {
            s += 3;
        } else if desc.iter().any(|d| token_match(d, q)) {
            s += 1;
        }
    }
    s
}

/// Tools ordered best match first. Ties keep their original order, so with no
/// signal at all this is the server order the old code used.
pub fn rank_by_relevance<'a>(tools: &[&'a Tool], query: &str) -> Vec<&'a Tool> {
    let q = tokens(query);
    let mut scored: Vec<(usize, usize, &Tool)> =
        tools.iter().enumerate().map(|(i, t)| (score(t, &q), i, *t)).collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.into_iter().map(|(_, _, t)| t).collect()
}

/// Up to `limit` tools matching `query`, best first, zero-score ones dropped.
pub fn search<'a>(tools: &'a [Tool], query: &str, limit: usize) -> Vec<&'a Tool> {
    let q = tokens(query);
    let mut scored: Vec<(usize, usize, &Tool)> = tools
        .iter()
        .enumerate()
        .map(|(i, t)| (score(t, &q), i, t))
        .filter(|(s, _, _)| *s > 0)
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.into_iter().take(limit).map(|(_, _, t)| t).collect()
}

/// `name(params) - description` on one line, the form every recovery message uses.
pub fn one_line(tool: &Tool) -> String {
    let sig = crate::mcp::schema::signature(&tool.function.parameters);
    let desc: String = tool.function.description.chars().take(90).collect();
    format!("- {}({sig}) - {}", tool.function.name, desc.trim())
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// The `limit` real tools closest to an invented `name`: shared tokens first,
/// then edit distance.
pub fn nearest<'a>(tools: &'a [Tool], name: &str, limit: usize) -> Vec<&'a Tool> {
    let want = tokens(name);
    let mut scored: Vec<(usize, usize, &Tool)> = tools
        .iter()
        .map(|t| {
            let have = tokens(&t.function.name);
            let shared = want.iter().filter(|w| have.iter().any(|h| token_match(h, w))).count();
            (shared, levenshtein(&t.function.name.to_lowercase(), &name.to_lowercase()), t)
        })
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.into_iter().take(limit).map(|(_, _, t)| t).collect()
}

/// Message for a call to a tool that does not exist anywhere.
pub fn unknown_tool_message(tools: &[Tool], name: &str) -> String {
    let close: Vec<String> = nearest(tools, name, 5).into_iter().map(one_line).collect();
    format!(
        "Error: there is no tool named `{name}`. Do not retry this name. The closest real \
         tools are:\n{}\nCall one of these, use `{FIND_TOOLS}` to search, or answer without a tool.",
        close.join("\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(name: &str, desc: &str, params: serde_json::Value) -> Tool {
        Tool::function(name, desc, params)
    }

    fn registry() -> Vec<Tool> {
        let p = json!({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]});
        let mut v: Vec<Tool> = (0..20)
            .map(|i| t(&format!("mcp__srv__misc_{i}"), "Unrelated helper.", json!({"type": "object", "properties": {}})))
            .collect();
        v.push(t("mcp__srv__get_config", "Read the config or a dotted path.", p.clone()));
        v.push(t(
            "mcp__srv__set_effect_param",
            "Change a parameter of a post effect such as chroma or bloom.",
            json!({"type": "object", "properties": {"effect": {"type": "string"}, "value": {"type": "number"}}, "required": ["effect", "value"]}),
        ));
        v
    }

    #[test]
    fn tokens_split_snake_camel_and_turkish() {
        assert_eq!(tokens("mcp__srv__setLayerSettings"), vec!["srv", "set", "layer", "settings"]);
        assert!(tokens("Renk sapmasını yükselt").contains(&"sapmasını".to_string()));
    }

    #[test]
    fn ranking_beats_server_order() {
        let reg = registry();
        let refs: Vec<&Tool> = reg.iter().collect();
        let ranked = rank_by_relevance(&refs, "increase the chroma effect");
        assert_eq!(ranked[0].function.name, "mcp__srv__set_effect_param");
    }

    #[test]
    fn search_finds_hidden_tools_and_skips_noise() {
        let reg = registry();
        let hits = search(&reg, "effect chroma", 5);
        assert_eq!(hits[0].function.name, "mcp__srv__set_effect_param");
        assert!(hits.iter().all(|h| !h.function.name.contains("misc")));
    }

    #[test]
    fn unknown_name_lists_real_neighbours_with_signatures() {
        let reg = registry();
        let msg = unknown_tool_message(&reg, "mcp__srv__set_config");
        assert!(msg.contains("Do not retry this name"), "{msg}");
        assert!(msg.contains("mcp__srv__get_config(path: string*)"), "{msg}");
        assert!(msg.contains("mcp__srv__set_effect_param(effect: string*, value: number*)"), "{msg}");
    }

    #[test]
    fn one_line_keeps_each_tools_own_params() {
        let reg = registry();
        let a = one_line(&reg[20]);
        let b = one_line(&reg[21]);
        assert!(a.contains("get_config(path") && !a.contains("value"));
        assert!(b.contains("set_effect_param(effect") && !b.contains("path"));
    }
}
