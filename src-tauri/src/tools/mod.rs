//! The tool surface the agent acts through.
//!
//! Every tool resolves its paths against the workspace root and reports the
//! [`Action`] it intends before running, so the permission engine judges the
//! *resolved* target rather than whatever string the model produced. A model
//! that asks for `../../etc/passwd` gets that path normalised and then refused
//! — the tool never sees a decision it can influence.
//!
//! Outputs are capped. A 0.5B model handed a 200 KB file will spend its whole
//! context on it and then have nothing left to reason with, so reads truncate
//! with an explicit marker rather than silently filling the window.

use serde_json::{Map, Value};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use crate::agent::runner::ToolExecutor;
use crate::ollama::types::Tool;
use crate::perms::Action;

/// Largest slice of a file returned to the model, in bytes.
const MAX_READ_BYTES: usize = 60_000;
/// Largest command output returned, in bytes.
const MAX_OUTPUT_BYTES: usize = 20_000;
/// Entries listed before a directory listing is cut short.
const MAX_DIR_ENTRIES: usize = 300;
/// Wall-clock ceiling for a single command.
const COMMAND_TIMEOUT: Duration = Duration::from_secs(120);

pub struct WorkspaceTools {
    workspace: PathBuf,
    /// When set, commands are classified and reported but never executed.
    /// Used by tests that must exercise destructive paths without running them.
    dry_run: bool,
}

impl WorkspaceTools {
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        Self { workspace: workspace.into(), dry_run: false }
    }

    /// Build a tool surface that never actually executes commands.
    pub fn dry_run(workspace: impl Into<PathBuf>) -> Self {
        Self { workspace: workspace.into(), dry_run: true }
    }

    /// Resolve a model-supplied path against the workspace.
    ///
    /// Relative paths join the root; absolute paths are taken as given. Either
    /// way the result is normalised, so containment is judged on a path with no
    /// `.` or `..` left in it.
    pub fn resolve(&self, raw: &str) -> PathBuf {
        let candidate = Path::new(raw);
        let joined = if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            self.workspace.join(candidate)
        };
        crate::perms::normalize(&joined)
    }

    fn arg_str(args: &Map<String, Value>, key: &str) -> Option<String> {
        args.get(key).and_then(Value::as_str).map(str::to_string)
    }
}

/// Split a command line into program and arguments, respecting quotes.
///
/// Used both to classify risk and to execute, so the string the permission
/// engine judged is the string that runs.
pub fn split_command(line: &str) -> Option<(String, Vec<String>)> {
    let mut parts: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;

    for c in line.chars() {
        match c {
            '"' | '\'' if quote.is_none() => quote = Some(c),
            c if Some(c) == quote => quote = None,
            c if c.is_whitespace() && quote.is_none() => {
                if !current.is_empty() {
                    parts.push(std::mem::take(&mut current));
                }
            }
            c => current.push(c),
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    if parts.is_empty() {
        return None;
    }
    let program = parts.remove(0);
    Some((program, parts))
}

/// Cut a string to `limit` bytes on a char boundary, noting what was dropped.
fn truncate(text: &str, limit: usize, unit: &str) -> String {
    if text.len() <= limit {
        return text.to_string();
    }
    let mut end = limit;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n\n[truncated — {} of {} {unit} shown]",
        &text[..end],
        end,
        text.len()
    )
}

#[async_trait::async_trait]
impl ToolExecutor for WorkspaceTools {
    fn schemas(&self) -> Vec<Tool> {
        use serde_json::json;
        vec![
            Tool::function(
                "read_file",
                "Read a file's contents.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path, relative to the workspace"}
                    },
                    "required": ["path"]
                }),
            ),
            Tool::function(
                "write_file",
                "Create a file or replace its entire contents.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path, relative to the workspace"},
                        "content": {"type": "string", "description": "Full new contents"}
                    },
                    "required": ["path", "content"]
                }),
            ),
            Tool::function(
                "edit_file",
                "Replace one exact string in a file.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path, relative to the workspace"},
                        "old": {"type": "string", "description": "Exact text to replace; must be unique"},
                        "new": {"type": "string", "description": "Replacement text"}
                    },
                    "required": ["path", "old", "new"]
                }),
            ),
            Tool::function(
                "list_dir",
                "List the entries of a directory.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory, relative to the workspace"}
                    },
                    "required": ["path"]
                }),
            ),
            Tool::function(
                "run_command",
                "Run a shell command in the workspace.",
                json!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Full command line"}
                    },
                    "required": ["command"]
                }),
            ),
        ]
    }

    fn action_for(&self, name: &str, args: &Map<String, Value>) -> Option<Action> {
        let path = || Self::arg_str(args, "path").map(|p| self.resolve(&p));

        match name {
            "read_file" => Some(Action::ReadFile { path: path()? }),
            "write_file" | "edit_file" => Some(Action::WriteFile { path: path()? }),
            "list_dir" => Some(Action::ListDir { path: path()? }),
            "run_command" => {
                let line = Self::arg_str(args, "command")?;
                let (program, args) = split_command(&line)?;
                Some(Action::RunCommand {
                    program,
                    args,
                    cwd: Some(self.workspace.clone()),
                })
            }
            _ => None,
        }
    }

    async fn execute(&self, name: &str, args: &Map<String, Value>) -> Result<String, String> {
        match name {
            "read_file" => {
                let raw = Self::arg_str(args, "path").ok_or("missing `path`")?;
                let path = self.resolve(&raw);
                let content = tokio::fs::read_to_string(&path)
                    .await
                    .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
                Ok(truncate(&content, MAX_READ_BYTES, "bytes"))
            }

            "write_file" => {
                let raw = Self::arg_str(args, "path").ok_or("missing `path`")?;
                let content = Self::arg_str(args, "content").ok_or("missing `content`")?;
                let path = self.resolve(&raw);
                if let Some(parent) = path.parent() {
                    tokio::fs::create_dir_all(parent)
                        .await
                        .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
                }
                tokio::fs::write(&path, &content)
                    .await
                    .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
                Ok(format!("Wrote {} bytes to {}", content.len(), path.display()))
            }

            "edit_file" => {
                let raw = Self::arg_str(args, "path").ok_or("missing `path`")?;
                let old = Self::arg_str(args, "old").ok_or("missing `old`")?;
                let new = Self::arg_str(args, "new").ok_or("missing `new`")?;
                let path = self.resolve(&raw);

                let content = tokio::fs::read_to_string(&path)
                    .await
                    .map_err(|e| format!("cannot read {}: {e}", path.display()))?;

                // Ambiguity is reported rather than guessed at: replacing the
                // first of several matches is how agents silently corrupt files.
                match content.matches(&old).count() {
                    0 => Err(format!("`old` text not found in {}", path.display())),
                    1 => {
                        let updated = content.replace(&old, &new);
                        tokio::fs::write(&path, &updated)
                            .await
                            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
                        Ok(format!("Edited {}", path.display()))
                    }
                    n => Err(format!(
                        "`old` text appears {n} times in {}. Include more surrounding \
                         context so it matches exactly once.",
                        path.display()
                    )),
                }
            }

            "list_dir" => {
                let raw = Self::arg_str(args, "path").ok_or("missing `path`")?;
                let path = self.resolve(&raw);
                let mut entries = tokio::fs::read_dir(&path)
                    .await
                    .map_err(|e| format!("cannot list {}: {e}", path.display()))?;

                let mut names = Vec::new();
                while let Ok(Some(entry)) = entries.next_entry().await {
                    let is_dir = entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false);
                    let name = entry.file_name().to_string_lossy().to_string();
                    names.push(if is_dir { format!("{name}/") } else { name });
                    if names.len() >= MAX_DIR_ENTRIES {
                        names.push(format!("[truncated at {MAX_DIR_ENTRIES} entries]"));
                        break;
                    }
                }
                names.sort();
                Ok(if names.is_empty() { "(empty)".into() } else { names.join("\n") })
            }

            "run_command" => {
                let line = Self::arg_str(args, "command").ok_or("missing `command`")?;
                let (program, argv) =
                    split_command(&line).ok_or("command is empty")?;

                if self.dry_run {
                    return Ok(format!("[dry-run] would execute: {program} {}", argv.join(" ")));
                }

                let mut cmd = tokio::process::Command::new(&program);
                cmd.args(&argv)
                    .current_dir(&self.workspace)
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped());

                let output = tokio::time::timeout(COMMAND_TIMEOUT, cmd.output())
                    .await
                    .map_err(|_| {
                        format!("command timed out after {}s", COMMAND_TIMEOUT.as_secs())
                    })?
                    .map_err(|e| format!("cannot run `{program}`: {e}"))?;

                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let code = output.status.code().unwrap_or(-1);

                let mut result = String::new();
                if !stdout.trim().is_empty() {
                    result.push_str(&truncate(&stdout, MAX_OUTPUT_BYTES, "bytes"));
                }
                if !stderr.trim().is_empty() {
                    if !result.is_empty() {
                        result.push('\n');
                    }
                    result.push_str("stderr:\n");
                    result.push_str(&truncate(&stderr, MAX_OUTPUT_BYTES, "bytes"));
                }
                if result.is_empty() {
                    result.push_str("(no output)");
                }
                // The model needs the exit code to know whether to keep going.
                Ok(format!("exit {code}\n{result}"))
            }

            other => Err(format!("unknown tool `{other}`")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perms::{Decision, PermissionEngine, PermissionMode};
    use serde_json::json;

    fn scratch() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cayahub-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        // Resolve symlinks (macOS /tmp, Windows short names) so the workspace
        // root matches what `resolve` produces.
        dir.canonicalize().map(|p| crate::perms::normalize(&p)).unwrap_or(dir)
    }

    fn args(v: Value) -> Map<String, Value> {
        serde_json::from_value(v).expect("object")
    }

    #[test]
    fn split_command_handles_quotes() {
        let (p, a) = split_command(r#"git commit -m "hello world""#).unwrap();
        assert_eq!(p, "git");
        assert_eq!(a, vec!["commit", "-m", "hello world"]);
    }

    #[test]
    fn split_command_rejects_an_empty_line() {
        assert!(split_command("   ").is_none());
    }

    #[test]
    fn relative_paths_resolve_inside_the_workspace() {
        let tools = WorkspaceTools::new("/work/project");
        assert_eq!(tools.resolve("src/main.rs"), crate::perms::normalize(Path::new("/work/project/src/main.rs")));
    }

    #[test]
    fn traversal_is_normalised_before_the_engine_sees_it() {
        // The engine must judge the resolved path, not the raw string.
        let tools = WorkspaceTools::new("/work/project");
        let action = tools
            .action_for("write_file", &args(json!({"path": "../../etc/hosts"})))
            .unwrap();
        let Action::WriteFile { path } = &action else { panic!("expected a write") };
        assert!(!path.to_string_lossy().contains(".."), "path still contains ..");

        let engine = PermissionEngine::new("/work/project", PermissionMode::ProjectAuto);
        assert_ne!(engine.evaluate(&action).decision, Decision::Allow);
    }

    #[tokio::test]
    async fn write_then_read_round_trips() {
        let dir = scratch();
        let tools = WorkspaceTools::new(&dir);

        tools
            .execute("write_file", &args(json!({"path": "a.txt", "content": "hello"})))
            .await
            .expect("write should succeed");

        let out = tools
            .execute("read_file", &args(json!({"path": "a.txt"})))
            .await
            .expect("read should succeed");
        assert_eq!(out, "hello");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn edit_refuses_an_ambiguous_match() {
        let dir = scratch();
        let tools = WorkspaceTools::new(&dir);
        tools
            .execute("write_file", &args(json!({"path": "b.txt", "content": "x\nx\n"})))
            .await
            .unwrap();

        let err = tools
            .execute("edit_file", &args(json!({"path": "b.txt", "old": "x", "new": "y"})))
            .await
            .expect_err("two matches must not be edited silently");
        assert!(err.contains("2 times"), "got: {err}");

        // The file must be untouched after a refused edit.
        let content = std::fs::read_to_string(dir.join("b.txt")).unwrap();
        assert_eq!(content, "x\nx\n");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn edit_reports_a_missing_target() {
        let dir = scratch();
        let tools = WorkspaceTools::new(&dir);
        tools
            .execute("write_file", &args(json!({"path": "c.txt", "content": "abc"})))
            .await
            .unwrap();

        let err = tools
            .execute("edit_file", &args(json!({"path": "c.txt", "old": "zzz", "new": "y"})))
            .await
            .expect_err("missing text should be an error");
        assert!(err.contains("not found"), "got: {err}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn read_truncates_oversized_files() {
        let dir = scratch();
        let tools = WorkspaceTools::new(&dir);
        let big = "a".repeat(MAX_READ_BYTES + 5_000);
        std::fs::write(dir.join("big.txt"), &big).unwrap();

        let out = tools
            .execute("read_file", &args(json!({"path": "big.txt"})))
            .await
            .unwrap();
        assert!(out.contains("[truncated"), "oversized read was not capped");
        assert!(out.len() < big.len());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn commands_run_and_report_their_exit_code() {
        let dir = scratch();
        let tools = WorkspaceTools::new(&dir);
        // `cargo` is guaranteed present in this repo's test environment.
        let out = tools
            .execute("run_command", &args(json!({"command": "cargo --version"})))
            .await
            .expect("cargo --version should run");
        assert!(out.starts_with("exit 0"), "got: {out}");
        assert!(out.contains("cargo"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn destructive_commands_are_classified_without_being_executed() {
        // Exercises the Unrestricted path without touching the real filesystem:
        // the engine's verdict and the resolved argv are asserted, the command
        // itself never runs.
        let dir = scratch();
        let tools = WorkspaceTools::dry_run(&dir);
        let engine = PermissionEngine::new(&dir, PermissionMode::Unrestricted);

        let call = args(json!({"command": "rm -rf /"}));
        let action = tools.action_for("run_command", &call).unwrap();

        assert_eq!(
            engine.evaluate(&action).decision,
            Decision::Deny,
            "`rm -rf /` must be refused even in Unrestricted mode"
        );

        let out = tools.execute("run_command", &call).await.unwrap();
        assert!(out.starts_with("[dry-run]"), "dry run must not execute");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn a_recoverable_destructive_command_is_allowed_only_in_unrestricted() {
        let dir = scratch();
        let tools = WorkspaceTools::dry_run(&dir);
        let call = args(json!({"command": "git reset --hard"}));
        let action = tools.action_for("run_command", &call).unwrap();

        let strict = PermissionEngine::new(&dir, PermissionMode::Balanced);
        assert_eq!(strict.evaluate(&action).decision, Decision::Ask);

        let loose = PermissionEngine::new(&dir, PermissionMode::Unrestricted);
        assert_eq!(loose.evaluate(&action).decision, Decision::Allow);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn missing_arguments_are_reported_not_panicked_on() {
        let dir = scratch();
        let tools = WorkspaceTools::new(&dir);
        let err = tools.execute("read_file", &Map::new()).await.expect_err("should error");
        assert!(err.contains("path"), "got: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
