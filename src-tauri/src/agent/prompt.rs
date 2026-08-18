//! System prompt construction.
//!
//! Two constraints shape everything here, and they pull in the same direction.
//!
//! **Weak models have a much smaller effective instruction window than their
//! advertised context.** A 7B model given a 4,000-token system prompt with
//! twenty tool definitions will follow roughly the first few instructions and
//! ignore the rest. Trimming the prompt is not a token-cost optimisation; it is
//! what makes the model behave at all.
//!
//! **A byte-identical prefix is reused by the daemon's KV cache.** Re-prefilling
//! 3,000 tokens every turn at ~50 tok/s costs a minute of wall time across a
//! session. Anything that varies per turn — timestamps, step counters, the
//! current file list — must live in the *user* message, never in the system
//! prompt, or the prefix changes and the cache is thrown away every time.
//!
//! So: short, and stable. Volatile context goes downstream.

use crate::ollama::types::Tool;

/// How much instruction a model can absorb before it starts ignoring things.
///
/// These are budgets for the system prompt, not the context window, and they
/// are deliberately far below what the model advertises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptTier {
    /// <= 2B. Barely follows multi-step instruction; give it one job.
    Minimal,
    /// 3B–9B. The common case for local coding work.
    Compact,
    /// 10B+. Can hold a fuller brief.
    Full,
}

impl PromptTier {
    /// Pick a tier from the model's parameter count in billions.
    pub fn for_params(billions: Option<f64>) -> Self {
        match billions {
            Some(b) if b < 3.0 => Self::Minimal,
            Some(b) if b < 10.0 => Self::Compact,
            Some(_) => Self::Full,
            // Unknown size: assume small. Over-prompting a weak model breaks it;
            // under-prompting a strong one merely makes it terser.
            None => Self::Compact,
        }
    }

    /// Rough ceiling for the system prompt, in tokens.
    pub fn budget_tokens(&self) -> usize {
        match self {
            Self::Minimal => 200,
            Self::Compact => 600,
            Self::Full => 1400,
        }
    }

    /// How many tool schemas this tier can hold before choosing badly.
    ///
    /// A model does not fail loudly when given too many tools — it starts
    /// picking the wrong one, or inventing names. The full surface is 23
    /// schemas (files, GitHub, documents, images, devices, web), far past what
    /// a 0.5B can choose between.
    ///
    /// These numbers are judgement, not measurement: 12 was chosen to hold the
    /// core tools plus one capability from each family. The live tests confirm
    /// a 7B still finishes a task at 12 and a 0.5B still produces usable calls
    /// at 5; where either actually degrades has not been measured.
    pub fn tool_budget(&self) -> usize {
        match self {
            Self::Minimal => 5,
            Self::Compact => 12,
            Self::Full => usize::MAX,
        }
    }
}

/// Tools every tier keeps.
///
/// A coding task cannot proceed without these, so they are never dropped to
/// make room for anything else.
const CORE_TOOLS: &[&str] = &[
    "read_file",
    "edit_file",
    "write_file",
    "list_dir",
    "run_command",
];

/// What fills the remaining budget, best first.
///
/// This list exists because the obvious implementation — keep the core tools,
/// then take whatever comes next — silently deleted a whole capability. The
/// surface is assembled files → GitHub → media → web, and GitHub alone is
/// twelve schemas, so "whatever comes next" was twelve GitHub tools and a 7B
/// model never saw `web_search`, `read_document`, or even `gh_open_pr`.
///
/// Ordering by usefulness in code keeps that decision visible, instead of
/// leaving it to the order the surface happens to be built in.
const PRIORITY_TOOLS: &[&str] = &[
    // Research, and the most common reason to reach outside the project.
    "web_search",
    "web_read",
    // Attachments the user drops into a task.
    "read_document",
    // The GitHub tools that carry the most work per call.
    "gh_status",
    "gh_diff",
    "gh_open_pr",
    "gh_commit",
    "view_image",
];

/// Trim the tool list to what the model can actually choose between.
///
/// Core tools first, then [`PRIORITY_TOOLS`], then the remainder in the order
/// given. Dropping a tool is better than keeping it and having the model pick
/// it by mistake — and a tool the model never needed still costs prefill on
/// every single step.
pub fn select_tools(tools: &[Tool], tier: PromptTier) -> Vec<Tool> {
    let budget = tier.tool_budget();
    if tools.len() <= budget {
        return tools.to_vec();
    }

    let mut chosen: Vec<Tool> = Vec::with_capacity(budget);
    let take = |name: &str, chosen: &mut Vec<Tool>| {
        if chosen.len() < budget && !chosen.iter().any(|c| c.function.name == name) {
            if let Some(tool) = tools.iter().find(|t| t.function.name == name) {
                chosen.push(tool.clone());
            }
        }
    };

    for name in CORE_TOOLS {
        take(name, &mut chosen);
    }
    for name in PRIORITY_TOOLS {
        take(name, &mut chosen);
    }
    for tool in tools {
        if chosen.len() >= budget {
            break;
        }
        if !chosen.iter().any(|c| c.function.name == tool.function.name) {
            chosen.push(tool.clone());
        }
    }
    chosen
}

/// Build the system prompt.
///
/// `workspace` is included because it is stable for the whole session. Nothing
/// else that varies may be added here — see the module note on prefix reuse.
pub fn system_prompt(tier: PromptTier, workspace: &str, tools: &[Tool]) -> String {
    let mut out = String::with_capacity(1024);

    match tier {
        PromptTier::Minimal => {
            out.push_str(
                "You are a coding assistant with tools.\n\
                 Call one tool at a time. Wait for its result before the next.\n\
                 Answer directly when no tool is needed.\n",
            );
        }
        PromptTier::Compact => {
            out.push_str(
                "You are a coding assistant working in a local project.\n\n\
                 Rules:\n\
                 - Use tools to read files before editing them. Never guess a file's contents.\n\
                 - Call one tool at a time and wait for the result.\n\
                 - Keep replies short. No preamble, no summary of what you are about to do.\n\
                 - When the task is done, say so in one sentence.\n",
            );
        }
        PromptTier::Full => {
            out.push_str(
                "You are a coding assistant working in a local project.\n\n\
                 Approach:\n\
                 - Read before you write. Inspect the file, then make the edit.\n\
                 - Prefer the smallest change that solves the problem.\n\
                 - Match the surrounding code's style, naming, and comment density.\n\
                 - Call one tool at a time and wait for its result before deciding the next step.\n\
                 - If a tool fails, read the error and adapt. Do not retry the same call unchanged.\n\n\
                 Output:\n\
                 - Keep replies short and concrete. No preamble.\n\
                 - Do not narrate what you are about to do; do it.\n\
                 - When the task is complete, state that in one sentence.\n",
            );
        }
    }

    out.push_str("\nWorkspace: ");
    out.push_str(workspace);
    out.push('\n');

    // Native tool calling passes schemas out of band, so listing them again in
    // prose would duplicate hundreds of tokens for no benefit.
    if !tools.is_empty() && tier == PromptTier::Minimal {
        out.push_str("\nTools: ");
        let names: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();
        out.push_str(&names.join(", "));
        out.push('\n');
    }

    out
}

/// Approximate token count.
///
/// Deliberately a heuristic: exact counting needs the model's tokenizer, which
/// costs a round-trip per call. ~3.6 chars/token is close enough for budgeting
/// on code-heavy English text, and every use here is a threshold check.
pub fn estimate_tokens(text: &str) -> usize {
    (text.len() as f64 / 3.6).ceil() as usize
}

/// Strip tool schemas down to what a small model can actually use.
///
/// Long parameter descriptions and deep nesting are where tool definitions get
/// expensive. For a `Minimal`-tier model, twenty verbose schemas can outweigh
/// the entire system prompt.
pub fn compact_tools(tools: &[Tool], tier: PromptTier) -> Vec<Tool> {
    if tier == PromptTier::Full {
        return tools.to_vec();
    }

    let description_limit = if tier == PromptTier::Minimal { 60 } else { 140 };

    tools
        .iter()
        .map(|tool| {
            let mut t = tool.clone();
            t.function.description = truncate_on_boundary(&t.function.description, description_limit);

            if let Some(props) = t
                .function
                .parameters
                .get_mut("properties")
                .and_then(|p| p.as_object_mut())
            {
                for (_, schema) in props.iter_mut() {
                    if let Some(obj) = schema.as_object_mut() {
                        if let Some(d) = obj.get("description").and_then(|d| d.as_str()) {
                            let short = truncate_on_boundary(d, 60);
                            obj.insert("description".into(), serde_json::Value::String(short));
                        }
                        // Examples and defaults are inferable; drop the tokens.
                        obj.remove("examples");
                        obj.remove("default");
                    }
                }
            }
            t
        })
        .collect()
}

/// Truncate at a word boundary, without a trailing ellipsis.
fn truncate_on_boundary(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let truncated: String = text.chars().take(limit).collect();
    match truncated.rfind(|c: char| c.is_whitespace()) {
        Some(idx) if idx > limit / 2 => truncated[..idx].trim_end().to_string(),
        _ => truncated.trim_end().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_tool() -> Tool {
        Tool::function(
            "read_file",
            "Read a file from disk and return its contents as UTF-8 text, \
             with line numbers prefixed to each line for easy reference in edits.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or workspace-relative path to the file to read, \
                                        which must exist and be readable by the current user.",
                        "examples": ["src/main.rs"],
                        "default": "."
                    }
                },
                "required": ["path"]
            }),
        )
    }

    #[test]
    fn tier_tracks_model_size() {
        assert_eq!(PromptTier::for_params(Some(0.5)), PromptTier::Minimal);
        assert_eq!(PromptTier::for_params(Some(7.0)), PromptTier::Compact);
        assert_eq!(PromptTier::for_params(Some(27.3)), PromptTier::Full);
        assert_eq!(PromptTier::for_params(None), PromptTier::Compact);
    }

    #[test]
    fn every_tier_stays_within_its_budget() {
        for tier in [PromptTier::Minimal, PromptTier::Compact, PromptTier::Full] {
            let p = system_prompt(tier, "/work/project", &[sample_tool()]);
            assert!(
                estimate_tokens(&p) <= tier.budget_tokens(),
                "{tier:?} prompt used {} tokens, budget {}",
                estimate_tokens(&p),
                tier.budget_tokens()
            );
        }
    }

    #[test]
    fn prompt_is_byte_identical_across_calls() {
        // The KV-cache reuse guarantee. If this fails, every turn re-prefills.
        let a = system_prompt(PromptTier::Compact, "/work/project", &[sample_tool()]);
        let b = system_prompt(PromptTier::Compact, "/work/project", &[sample_tool()]);
        assert_eq!(a, b);
    }

    #[test]
    fn prompt_carries_no_volatile_content() {
        let p = system_prompt(PromptTier::Full, "/work/project", &[sample_tool()]);
        for marker in ["2026", "step ", "timestamp", ":00"] {
            assert!(!p.to_lowercase().contains(marker), "found volatile marker {marker:?}");
        }
    }

    fn named(name: &str) -> Tool {
        Tool::function(name, "does a thing", json!({"type": "object", "properties": {}}))
    }

    /// The surface `run_agent` actually builds: files + GitHub + media.
    fn full_surface() -> Vec<Tool> {
        let mut tools: Vec<Tool> = CORE_TOOLS.iter().map(|n| named(n)).collect();
        for i in 0..12 {
            let name = format!("gh_tool_{i}");
            tools.push(named(&name));
        }
        for n in ["read_document", "view_image", "screenshot", "adb"] {
            tools.push(named(n));
        }
        tools
    }

    #[test]
    fn a_small_model_is_not_handed_the_whole_surface() {
        let all = full_surface();
        assert_eq!(all.len(), 21, "the surface this guards against");

        let minimal = select_tools(&all, PromptTier::Minimal);
        assert_eq!(minimal.len(), 5);

        let compact = select_tools(&all, PromptTier::Compact);
        assert_eq!(compact.len(), 12);

        // A capable model still gets everything.
        assert_eq!(select_tools(&all, PromptTier::Full).len(), 21);
    }

    #[test]
    fn file_tools_survive_every_trim() {
        // Dropping `read_file` would leave a coding agent unable to work at all.
        let all = full_surface();
        for tier in [PromptTier::Minimal, PromptTier::Compact, PromptTier::Full] {
            let selected = select_tools(&all, tier);
            let names: Vec<&str> = selected.iter().map(|t| t.function.name.as_str()).collect();
            for core in CORE_TOOLS {
                assert!(names.contains(core), "{tier:?} dropped {core}");
            }
        }
    }

    #[test]
    fn a_short_list_is_returned_unchanged() {
        let few: Vec<Tool> = CORE_TOOLS.iter().take(3).map(|n| named(n)).collect();
        assert_eq!(select_tools(&few, PromptTier::Minimal).len(), 3);
    }

    #[test]
    fn compaction_shrinks_schemas_for_small_models() {
        let original = sample_tool();
        let compact = compact_tools(&[original.clone()], PromptTier::Minimal);

        assert!(compact[0].function.description.len() < original.function.description.len());

        let props = compact[0].function.parameters.get("properties").unwrap();
        let path = props.get("path").unwrap();
        assert!(path.get("examples").is_none(), "examples should be dropped");
        assert!(path.get("default").is_none(), "defaults should be dropped");
    }

    #[test]
    fn compaction_preserves_the_contract() {
        // Trimming prose must never change names or required fields.
        let compact = compact_tools(&[sample_tool()], PromptTier::Minimal);
        assert_eq!(compact[0].function.name, "read_file");
        let required = compact[0].function.parameters.get("required").unwrap();
        assert_eq!(required, &json!(["path"]));
        assert!(compact[0]
            .function
            .parameters
            .get("properties")
            .and_then(|p| p.get("path"))
            .is_some());
    }

    #[test]
    fn full_tier_leaves_schemas_untouched() {
        let original = sample_tool();
        let compact = compact_tools(&[original.clone()], PromptTier::Full);
        assert_eq!(compact[0].function.description, original.function.description);
    }

    #[test]
    fn truncation_does_not_split_a_word() {
        let s = truncate_on_boundary("the quick brown fox jumps", 12);
        assert!(!s.ends_with("bro"));
        assert_eq!(s, "the quick");
    }
}
