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
    /// The Compact number is measured: `scripts/drift.py` offers exactly this
    /// surface to qwen2.5-coder:7b and it still makes zero false tool calls on
    /// eight conversational inputs while missing none of eight tasks. It holds
    /// the core tools, both web tools, both attachment tools, the screen, and
    /// four GitHub tools — one capability from every family the app has.
    ///
    /// The Minimal number is judgement: a 0.5B still produces usable calls at
    /// five, and where it actually degrades has not been measured.
    pub fn tool_budget(&self) -> usize {
        match self {
            Self::Minimal => 6,
            Self::Compact => 16,
            Self::Full => usize::MAX,
        }
    }

    /// Extra room for MCP tools, on top of [`Self::tool_budget`].
    ///
    /// The user attached these on purpose, so they are not left to compete
    /// with the built-in surface for the same slots; but a weak model still
    /// cannot choose between forty of them, so they are capped too. Not
    /// measured: a judgement in line with the budgets above.
    pub fn mcp_budget(&self) -> usize {
        match self {
            Self::Minimal => 4,
            Self::Compact => 10,
            Self::Full => 40,
        }
    }
}

fn is_mcp(tool: &Tool) -> bool {
    tool.function.name.starts_with(crate::mcp::TOOL_PREFIX)
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
    "search",
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
    // Asking beats guessing, and a model that cannot ask guesses. Cheap enough
    // to keep at every tier that has room.
    "ask_user",
    // Research, and the most common reason to reach outside the project.
    "web_search",
    "web_read",
    // Attachments the user drops into a task. Images before documents is
    // deliberate: dragging a screenshot into the box is the common case.
    "view_image",
    "read_document",
    // Looking at the user's screen. Without this a trimmed surface leaves the
    // model answering "I can't see your screen" to a question it has a tool for.
    "screenshot",
    // The GitHub tools that carry the most work per call.
    "gh_status",
    "gh_diff",
    "gh_open_pr",
    "gh_commit",
];

/// Trim the tool list to what the model can actually choose between.
///
/// Core tools first, then [`PRIORITY_TOOLS`], then the remainder in the order
/// given. Dropping a tool is better than keeping it and having the model pick
/// it by mistake — and a tool the model never needed still costs prefill on
/// every single step.
pub fn select_tools(tools: &[Tool], tier: PromptTier) -> Vec<Tool> {
    let mcp_reserved = tools.iter().filter(|t| is_mcp(t)).count().min(tier.mcp_budget());
    let budget = tier.tool_budget().saturating_add(mcp_reserved);
    if tools.len() <= budget && tools.iter().filter(|t| is_mcp(t)).count() <= tier.mcp_budget() {
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
    // MCP tools next, in server order, up to their own reservation.
    for tool in tools.iter().filter(|t| is_mcp(t)).take(mcp_reserved) {
        take(&tool.function.name, &mut chosen);
    }
    for name in PRIORITY_TOOLS {
        take(name, &mut chosen);
    }
    for tool in tools {
        if chosen.len() >= budget {
            break;
        }
        // MCP tools past the reservation stay out, so they cannot crowd out
        // the built-in surface either.
        if is_mcp(tool) {
            continue;
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
                 Greetings and general questions get a plain answer, no tool.\n\
                 Call one tool at a time. Wait for its result before the next.\n\
                 Use only the tool names you were given. Never invent one.\n",
            );
        }
        // Both of these are measured with `scripts/drift.py`, not guessed.
        //
        // "Tools are for acting on the project. Most messages are not that."
        // took false tool calls on greetings from six out of eight to zero on
        // qwen2.5-coder:7b — every one of them a `web_search` for the word
        // "hello".
        //
        // But "specific in this project" was too narrow, and models quoted it
        // back: asked "bilgisayarımın ekranında şuanda ne açık?" with
        // `screenshot` in its tool list, gpt-oss:20b reasoned that this was a
        // general question and answered that it has no access to the screen.
        // Naming what else counts as specific — the screen, a page, a phone —
        // fixes that case and still measures zero false calls on the 7B.
        PromptTier::Compact => {
            out.push_str(
                "You are a coding assistant working in a local project.\n\n\
                 Tools are for acting on the project. Most messages are not that.\n\
                 - If the user is greeting you, thanking you, or asking a general \
                 question, just reply. Do not call a tool.\n\
                 - Call a tool when asked to look at or change something specific: \
                 a file here, the screen, a web page, a connected phone.\n\
                 - Never say you cannot see something you have a tool for.\n\
                 - Read a file before editing it. Never guess its contents.\n\
                 - One tool at a time. Wait for the result.\n\
                 - Keep replies short. No preamble.\n",
            );
        }
        PromptTier::Full => {
            out.push_str(
                "You are a coding assistant working in a local project.\n\n\
                 Tools are for acting on the project. Most messages are not that.\n\
                 - If the user is greeting you, thanking you, or asking a general \
                 question, just reply. Do not call a tool.\n\
                 - Call a tool when asked to look at or change something specific: \
                 a file here, the screen, a web page, a connected phone.\n\
                 - Never say you cannot see something you have a tool for.\n\n\
                 Approach:\n\
                 - Read before you write. Inspect the file, then make the edit.\n\
                 - Prefer the smallest change that solves the problem.\n\
                 - Match the surrounding code's style, naming, and comment density.\n\
                 - There is no tool for making an image, a sound or a video, but you \
                 can write a script and run it. Say you cannot only after that fails.\n\
                 - One tool at a time. Wait for its result before deciding the next step.\n\
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

/// Tool instructions for a model that cannot call tools natively.
///
/// Appended to the system prompt instead of sending schemas out of band. One
/// strict format, stated once with an example, because a weak model copies
/// examples far better than it follows descriptions. The parser accepts much
/// more than this (fences, tags, trailing prose), so a near miss still works.
pub fn text_tool_instructions(tools: &[Tool]) -> String {
    let mut out = String::from(
        "\nTo use a tool, reply with ONLY this JSON object and nothing else:\n\
         {\"name\": \"<tool name>\", \"arguments\": {<arguments>}}\n\
         Then wait for the result. When you can answer, reply in plain text with no JSON.\n\
         Arguments marked * are required.\nTools:\n",
    );
    for t in tools {
        let sig = crate::mcp::schema::signature(&t.function.parameters);
        let desc = truncate_on_boundary(&t.function.description, 100);
        out.push_str(&format!("- {}({sig}): {desc}\n", t.function.name));
    }
    if let Some(first) = tools.first() {
        out.push_str(&format!(
            "Example: {{\"name\": \"{}\", \"arguments\": {{}}}}\n",
            first.function.name
        ));
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
        for n in ["read_document", "view_image", "screenshot", "adb", "web_search", "web_read"] {
            tools.push(named(n));
        }
        tools
    }

    #[test]
    fn a_small_model_is_not_handed_the_whole_surface() {
        let all = full_surface();
        assert_eq!(all.len(), 24, "the surface this guards against");

        assert_eq!(
            select_tools(&all, PromptTier::Minimal).len(),
            PromptTier::Minimal.tool_budget()
        );
        assert_eq!(
            select_tools(&all, PromptTier::Compact).len(),
            PromptTier::Compact.tool_budget()
        );
        // A capable model still gets everything.
        assert_eq!(select_tools(&all, PromptTier::Full).len(), all.len());
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

    fn mcp_named(name: &str) -> Tool {
        Tool::function(name, "A tool.", json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}))
    }

    #[test]
    fn mcp_tools_get_their_own_room_and_are_capped() {
        let mut all: Vec<Tool> = CORE_TOOLS.iter().map(|n| mcp_named(n)).collect();
        all.extend((0..30).map(|i| mcp_named(&format!("gh_tool_{i}"))));
        all.extend((0..20).map(|i| mcp_named(&format!("mcp__srv__t{i}"))));

        for tier in [PromptTier::Minimal, PromptTier::Compact] {
            let chosen = select_tools(&all, tier);
            let mcp = chosen.iter().filter(|t| t.function.name.starts_with("mcp__")).count();
            assert_eq!(mcp, tier.mcp_budget(), "{tier:?}");
            assert_eq!(chosen.len(), tier.tool_budget() + tier.mcp_budget(), "{tier:?}");
            for core in CORE_TOOLS {
                assert!(chosen.iter().any(|t| t.function.name == *core), "{tier:?} dropped {core}");
            }
        }
    }

    #[test]
    fn a_surface_without_mcp_tools_is_trimmed_as_before() {
        let all: Vec<Tool> = (0..30).map(|i| mcp_named(&format!("tool_{i}"))).collect();
        assert_eq!(select_tools(&all, PromptTier::Compact).len(), PromptTier::Compact.tool_budget());
    }

    #[test]
    fn text_mode_instructions_show_the_format_and_signatures() {
        let text = text_tool_instructions(&[mcp_named("mcp__srv__find")]);
        assert!(text.contains("{\"name\": \"<tool name>\", \"arguments\""), "{text}");
        assert!(text.contains("- mcp__srv__find(q: string*)"), "{text}");
    }
}
