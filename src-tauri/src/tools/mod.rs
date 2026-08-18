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

pub mod github;
pub mod media;
pub mod web;

use crate::agent::runner::ToolExecutor;
use crate::checkpoint::{ChangeKind, CheckpointStore};
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
    /// Prior contents are captured here before a file is changed. `None`
    /// disables undo, which is what tests that only exercise reads want.
    checkpoints: Option<CheckpointStore>,
    /// GitHub tools, offered only when the workspace is actually a repository.
    github: Option<github::GitHubTools>,
    /// Document, image, and device tools. `None` until `with_media` is called,
    /// because they need an Ollama client to delegate vision to.
    media: Option<media::MediaTools>,
    /// Search and page reading. Always available — they need nothing installed.
    web: web::WebTools,
}

impl WorkspaceTools {
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        let workspace = workspace.into();
        Self {
            checkpoints: Some(CheckpointStore::new(&workspace)),
            github: workspace
                .join(".git")
                .exists()
                .then(|| github::GitHubTools::new(&workspace)),
            workspace,
            dry_run: false,
            media: None,
            web: web::WebTools::new(),
        }
    }

    /// Enable document, image, and device tools.
    ///
    /// `vision_model` is whichever model routing picked for images; without one
    /// the image tools are not advertised at all.
    pub fn with_media(
        mut self,
        client: crate::ollama::OllamaClient,
        vision_model: Option<String>,
    ) -> Self {
        self.media = Some(media::MediaTools::new(&self.workspace, client, vision_model));
        self
    }

    /// Build a tool surface that never actually executes commands.
    pub fn dry_run(workspace: impl Into<PathBuf>) -> Self {
        let workspace = workspace.into();
        Self {
            workspace,
            dry_run: true,
            checkpoints: None,
            github: None,
            media: None,
            web: web::WebTools::new(),
        }
    }

    /// Build a tool surface with no undo history.
    pub fn without_checkpoints(workspace: impl Into<PathBuf>) -> Self {
        let workspace = workspace.into();
        Self {
            workspace,
            dry_run: false,
            checkpoints: None,
            github: None,
            media: None,
            web: web::WebTools::new(),
        }
    }

    /// Capture a file's prior state, labelled with the call that will change it.
    ///
    /// A capture failure is logged and ignored rather than blocking the edit:
    /// losing undo is bad, refusing to work is worse.
    fn checkpoint(&self, path: &Path, kind: ChangeKind, label: &str) {
        let Some(store) = &self.checkpoints else { return };
        match store.capture(path, kind).and_then(|snap| store.record(label, vec![snap])) {
            Ok(_) => {}
            Err(e) => tracing::warn!("checkpoint for {} failed: {e}", path.display()),
        }
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
        let mut tools = vec![
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
        ];
        // Only advertise GitHub tools in a repository. Offering a model tools
        // that cannot work is a reliable way to waste its steps.
        if self.github.is_some() {
            tools.extend(github::schemas());
        }
        if let Some(media) = &self.media {
            tools.extend(media.schemas());
        }
        tools.extend(web::WebTools::schemas());
        tools
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
            other if other.starts_with("gh_") => {
                github::action_for(other, args, &self.workspace)
            }
            other if web::WebTools::handles(other) => web::WebTools::action_for(other, args),
            other => self.media.as_ref().and_then(|m| m.action_for(other, args)),
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
                self.checkpoint(&path, ChangeKind::Modified, &format!("write_file {raw}"));
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
                        self.checkpoint(&path, ChangeKind::Modified, &format!("edit_file {raw}"));
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

            other if other.starts_with("gh_") => match &self.github {
                Some(gh) => gh.execute(other, args).await,
                None => Err("this folder is not a git repository".into()),
            },

            other if web::WebTools::handles(other) => self.web.execute(other, args).await,

            other => match &self.media {
                Some(media) if media.handles(other) => media.execute(other, args).await,
                _ => Err(format!("unknown tool `{other}`")),
            },
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
    async fn a_write_can_be_undone() {
        let dir = scratch();
        let tools = WorkspaceTools::new(&dir);
        std::fs::write(dir.join("a.txt"), "before").unwrap();

        tools
            .execute("write_file", &args(json!({"path": "a.txt", "content": "after"})))
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(dir.join("a.txt")).unwrap(), "after");

        let store = crate::checkpoint::CheckpointStore::new(&dir);
        let latest = store.list().unwrap().pop().expect("a checkpoint");
        store.restore(&latest.id).unwrap();

        assert_eq!(std::fs::read_to_string(dir.join("a.txt")).unwrap(), "before");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn an_edit_can_be_undone() {
        let dir = scratch();
        let tools = WorkspaceTools::new(&dir);
        std::fs::write(dir.join("b.txt"), "hello world").unwrap();

        tools
            .execute("edit_file", &args(json!({"path": "b.txt", "old": "world", "new": "there"})))
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(dir.join("b.txt")).unwrap(), "hello there");

        let store = crate::checkpoint::CheckpointStore::new(&dir);
        let latest = store.list().unwrap().pop().unwrap();
        store.restore(&latest.id).unwrap();

        assert_eq!(std::fs::read_to_string(dir.join("b.txt")).unwrap(), "hello world");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn a_created_file_is_removed_by_undo() {
        let dir = scratch();
        let tools = WorkspaceTools::new(&dir);

        tools
            .execute("write_file", &args(json!({"path": "fresh.txt", "content": "new"})))
            .await
            .unwrap();
        assert!(dir.join("fresh.txt").exists());

        let store = crate::checkpoint::CheckpointStore::new(&dir);
        let latest = store.list().unwrap().pop().unwrap();
        store.restore(&latest.id).unwrap();

        assert!(!dir.join("fresh.txt").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The surface `run_agent` builds, not the hand-made one the other tests use.
    fn real_surface(dir: &Path) -> WorkspaceTools {
        // GitHub tools only appear inside a repository.
        std::fs::create_dir_all(dir.join(".git")).unwrap();
        WorkspaceTools::new(dir).with_media(
            crate::ollama::OllamaClient::new(
                crate::ollama::DEFAULT_HOST,
                crate::ollama::Timeouts::default(),
            ),
            Some("qwen3-vl:8b".into()),
        )
    }

    #[test]
    fn trimming_keeps_a_capability_from_each_family() {
        // The bug this guards: the surface is assembled files -> GitHub -> media
        // -> web, and GitHub alone is twelve schemas. Filling the budget in
        // assembly order handed a 7B five file tools and seven GitHub tools,
        // silently deleting web search, documents, images, and gh_open_pr.
        use crate::agent::prompt::{select_tools, PromptTier};

        let dir = scratch();
        let tools = real_surface(&dir);
        let all = tools.schemas();
        assert!(all.len() > 20, "expected the full surface, got {}", all.len());

        let compact = select_tools(&all, PromptTier::Compact);
        let names: Vec<&str> = compact.iter().map(|t| t.function.name.as_str()).collect();

        for expected in ["read_file", "run_command", "web_search", "read_document", "gh_open_pr"] {
            assert!(
                names.contains(&expected),
                "Compact tier dropped {expected}; got {names:?}"
            );
        }

        // The smallest tier keeps only what a task cannot proceed without.
        let minimal = select_tools(&all, PromptTier::Minimal);
        let minimal_names: Vec<&str> =
            minimal.iter().map(|t| t.function.name.as_str()).collect();
        assert!(minimal_names.contains(&"read_file"));
        assert!(!minimal_names.iter().any(|n| n.starts_with("gh_")));

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
