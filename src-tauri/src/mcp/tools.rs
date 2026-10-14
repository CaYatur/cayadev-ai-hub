//! Offers MCP tools to the agent loop next to the built-in tools.

use serde_json::{Map, Value};
use std::sync::Arc;
use tokio::sync::mpsc;

use super::{McpManager, McpTool};
use crate::agent::runner::ToolExecutor;
use crate::ollama::types::Tool;
use crate::perms::Action;

/// A tool surface plus the MCP tools attached to the agent.
pub struct WithMcp<T: ToolExecutor> {
    inner: T,
    manager: Arc<McpManager>,
    tools: Vec<McpTool>,
    auto_approve: Vec<String>,
    /// Characters of one result handed back to the model.
    result_cap: usize,
}

impl<T: ToolExecutor> WithMcp<T> {
    /// `auto_approve` lists server ids whose tools run without asking.
    pub fn new(inner: T, manager: Arc<McpManager>, tools: Vec<McpTool>, auto_approve: Vec<String>) -> Self {
        Self {
            inner,
            manager,
            tools,
            auto_approve,
            result_cap: super::sanitize::cap_for_context(None),
        }
    }

    /// Size results for a model with `num_ctx` tokens of context.
    pub fn with_context(mut self, num_ctx: Option<u32>) -> Self {
        self.result_cap = super::sanitize::cap_for_context(num_ctx);
        self
    }

    fn find(&self, name: &str) -> Option<&McpTool> {
        self.tools.iter().find(|t| t.exposed == name)
    }
}

/// Schema offered to the model for one MCP tool.
pub fn schema_for(tool: &McpTool) -> Tool {
    // Saying which server a tool comes from helps a model pick between two
    // servers that both offer `search`.
    let description = if tool.description.is_empty() {
        format!("[{}] {}", tool.server_name, tool.name)
    } else {
        format!("[{}] {}", tool.server_name, tool.description)
    };
    Tool::function(&tool.exposed, &description, tool.input_schema.clone())
}

#[async_trait::async_trait]
impl<T: ToolExecutor> ToolExecutor for WithMcp<T> {
    fn schemas(&self) -> Vec<Tool> {
        let mut all = self.inner.schemas();
        all.extend(self.tools.iter().map(schema_for));
        all
    }

    fn action_for(&self, name: &str, args: &Map<String, Value>) -> Option<Action> {
        match self.find(name) {
            Some(t) => Some(Action::McpCall {
                server: t.server_name.clone(),
                tool: t.name.clone(),
                read_only: t.read_only,
                trusted: self.auto_approve.contains(&t.server_id),
            }),
            None => self.inner.action_for(name, args),
        }
    }

    fn set_progress(&mut self, tx: mpsc::Sender<String>) {
        self.inner.set_progress(tx);
    }

    async fn execute(&self, name: &str, args: &Map<String, Value>) -> Result<String, String> {
        match self.find(name) {
            // Errors are server text too, and can carry the same payloads.
            Some(t) => self
                .manager
                .call(&t.server_id, &t.name, args, self.result_cap)
                .await
                .map_err(|e| super::sanitize::cap(&super::sanitize::clean_text_or_json(&e), self.result_cap)),
            None => self.inner.execute(name, args).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::toolcall::{extract, Recovery};
    use serde_json::json;

    fn known() -> Vec<String> {
        vec!["mcp__fs__read_text_file".into(), "read_file".into()]
    }

    /// The replies a weak model actually sends in text mode. Each must come
    /// back as one call to the MCP tool with its argument intact.
    #[test]
    fn text_mode_replies_parse_despite_fences_trailing_text_and_bad_json() {
        let replies = [
            r#"{"name": "mcp__fs__read_text_file", "arguments": {"path": "a.txt"}}"#,
            "```json\n{\"name\": \"mcp__fs__read_text_file\", \"arguments\": {\"path\": \"a.txt\"}}\n```\nI will read it now.",
            r#"Sure! {'name': 'mcp__fs__read_text_file', 'arguments': {'path': 'a.txt'},} hope that helps"#,
            r#"{"tool": "MCP__FS__READ_TEXT_FILE", "args": "{\"path\": \"a.txt\"}"}"#,
            r#"{"name": "mcp__fs__read_text_file", "arguments": {"path": "a.txt""#,
        ];
        for reply in replies {
            let found = extract(reply, None, &known()).unwrap_or_else(|| panic!("no call in: {reply}"));
            assert_ne!(found.recovery, Recovery::Native);
            let call = &found.calls[0].function;
            assert_eq!(call.name, "mcp__fs__read_text_file", "{reply}");
            assert_eq!(call.arguments_object().get("path"), Some(&json!("a.txt")), "{reply}");
        }
    }

    #[test]
    fn the_schema_names_the_server() {
        let t = McpTool {
            server_id: "fs".into(),
            server_name: "Files".into(),
            name: "read".into(),
            exposed: "mcp__fs__read".into(),
            description: "Read a file.".into(),
            input_schema: json!({"type": "object", "properties": {}}),
            read_only: true,
            enabled: true,
        };
        let s = schema_for(&t);
        assert_eq!(s.function.name, "mcp__fs__read");
        assert_eq!(s.function.description, "[Files] Read a file.");
    }
}
