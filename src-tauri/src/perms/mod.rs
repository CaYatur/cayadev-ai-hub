//! Permission engine.
//!
//! Every side-effecting tool call passes through [`PermissionEngine::evaluate`]
//! before it runs. The engine is deliberately conservative and deliberately
//! boring: it makes decisions from explicit rules, never from a model's opinion.
//!
//! A model *may* be consulted for the ambiguous middle ground (see
//! `agent::approver`), but only to turn "ask the user" into "allow" for calls
//! the rules already classified as reversible. A model can never upgrade a
//! denied call, and never authorises anything destructive. That asymmetry is
//! the point: a misled classifier costs a confirmation prompt, not a wiped
//! directory.

pub mod command;
pub mod path;

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub use command::{classify_command, CommandRisk};
pub use path::{contains, normalize};

/// How much the agent may do without asking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionMode {
    /// Reads inside the workspace are automatic. Everything that writes or
    /// executes is confirmed. The safe default.
    ReadOnly,
    /// Reads and writes inside the workspace are automatic. Commands are
    /// confirmed. This is the mode most work happens in.
    ProjectAuto,
    /// As `ProjectAuto`, plus commands the rules classify as read-only
    /// (`git status`, `ls`, `cargo check`) run automatically.
    Balanced,
    /// Everything runs unattended, including destructive commands and writes
    /// outside the workspace.
    ///
    /// Genuinely dangerous, and named to say so. The UI requires typed
    /// confirmation to enter it and shows a persistent banner while it is on.
    Unrestricted,
}

impl Default for PermissionMode {
    fn default() -> Self {
        Self::ProjectAuto
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    /// Run without interrupting the user.
    Allow,
    /// Stop and ask.
    Ask,
    /// Refuse outright. Not offered as a prompt.
    Deny,
}

/// The outcome of evaluating one call, including why — the reason is shown in
/// the UI and written to the audit log so auto-approvals stay reviewable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evaluation {
    pub decision: Decision,
    pub reason: String,
    /// Set when the call would touch something outside the workspace.
    pub escapes_workspace: bool,
    pub risk: CommandRisk,
}

impl Evaluation {
    fn allow(reason: impl Into<String>) -> Self {
        Self {
            decision: Decision::Allow,
            reason: reason.into(),
            escapes_workspace: false,
            risk: CommandRisk::ReadOnly,
        }
    }

    fn ask(reason: impl Into<String>, risk: CommandRisk, escapes: bool) -> Self {
        Self {
            decision: Decision::Ask,
            reason: reason.into(),
            escapes_workspace: escapes,
            risk,
        }
    }

    fn deny(reason: impl Into<String>, risk: CommandRisk) -> Self {
        Self {
            decision: Decision::Deny,
            reason: reason.into(),
            escapes_workspace: true,
            risk,
        }
    }
}

/// The action a tool wants to perform, reduced to what the engine must judge.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Action {
    ReadFile { path: PathBuf },
    WriteFile { path: PathBuf },
    DeleteFile { path: PathBuf },
    ListDir { path: PathBuf },
    RunCommand { program: String, args: Vec<String>, cwd: Option<PathBuf> },
    /// Any request that leaves the machine.
    Network { url: String },
}

#[derive(Debug, Clone)]
pub struct PermissionEngine {
    pub mode: PermissionMode,
    /// Canonical workspace root. All containment checks are relative to this.
    workspace: PathBuf,
    /// Extra roots the user has explicitly opened, e.g. a sibling repo.
    extra_roots: Vec<PathBuf>,
}

impl PermissionEngine {
    pub fn new(workspace: impl AsRef<Path>, mode: PermissionMode) -> Self {
        Self {
            mode,
            workspace: normalize(workspace.as_ref()),
            extra_roots: Vec::new(),
        }
    }

    pub fn workspace(&self) -> &Path {
        &self.workspace
    }

    pub fn allow_root(&mut self, root: impl AsRef<Path>) {
        self.extra_roots.push(normalize(root.as_ref()));
    }

    fn in_workspace(&self, path: &Path) -> bool {
        let target = normalize(path);
        contains(&self.workspace, &target)
            || self.extra_roots.iter().any(|r| contains(r, &target))
    }

    pub fn evaluate(&self, action: &Action) -> Evaluation {
        match action {
            Action::ReadFile { path } | Action::ListDir { path } => self.eval_read(path),
            Action::WriteFile { path } => self.eval_write(path, false),
            Action::DeleteFile { path } => self.eval_write(path, true),
            Action::RunCommand { program, args, cwd } => self.eval_command(program, args, cwd.as_deref()),
            Action::Network { url } => self.eval_network(url),
        }
    }

    fn eval_read(&self, path: &Path) -> Evaluation {
        if path::is_sensitive(path) {
            return Evaluation::deny(
                "Path holds credentials or private keys. Reading it is never automatic.",
                CommandRisk::Destructive,
            );
        }
        if self.in_workspace(path) {
            return Evaluation::allow("Read inside the workspace.");
        }
        if self.mode == PermissionMode::Unrestricted {
            return Evaluation::allow("Unrestricted mode: read outside the workspace.");
        }
        Evaluation::ask("Read outside the workspace.", CommandRisk::ReadOnly, true)
    }

    fn eval_write(&self, path: &Path, destructive: bool) -> Evaluation {
        if path::is_sensitive(path) {
            return Evaluation::deny(
                "Path holds credentials or private keys. Writing to it is refused.",
                CommandRisk::Destructive,
            );
        }
        if path::is_system_path(path) {
            return Evaluation::deny(
                "Path is inside a protected system directory.",
                CommandRisk::Destructive,
            );
        }

        let inside = self.in_workspace(path);
        let verb = if destructive { "Delete" } else { "Write" };

        match self.mode {
            PermissionMode::Unrestricted => {
                Evaluation::allow(format!("Unrestricted mode: {} allowed.", verb.to_lowercase()))
            }
            _ if !inside => Evaluation::ask(
                format!("{verb} outside the workspace."),
                CommandRisk::Destructive,
                true,
            ),
            // Deletion stays a confirmation even inside the workspace: the
            // checkpoint system can restore a file's contents, but a user who
            // did not expect the deletion should see it first.
            _ if destructive => {
                Evaluation::ask("Delete inside the workspace.", CommandRisk::Destructive, false)
            }
            PermissionMode::ReadOnly => Evaluation::ask(
                "Read-only mode: writes are always confirmed.",
                CommandRisk::Mutating,
                false,
            ),
            PermissionMode::ProjectAuto | PermissionMode::Balanced => {
                Evaluation::allow("Write inside the workspace.")
            }
        }
    }

    fn eval_command(&self, program: &str, args: &[String], cwd: Option<&Path>) -> Evaluation {
        let risk = classify_command(program, args);

        if let Some(dir) = cwd {
            if !self.in_workspace(dir) && self.mode != PermissionMode::Unrestricted {
                return Evaluation::ask(
                    "Command runs outside the workspace.",
                    risk,
                    true,
                );
            }
        }

        match risk {
            // Refused in every mode, including Unrestricted. These are the
            // commands with no recoverable failure mode — a disk format or a
            // recursive delete of a drive root is not a permission question.
            CommandRisk::Catastrophic => Evaluation::deny(
                "Command is irreversible at system scope and is never run automatically. \
                 Run it yourself if you intend it.",
                risk,
            ),
            CommandRisk::Destructive => match self.mode {
                PermissionMode::Unrestricted => {
                    Evaluation::allow("Unrestricted mode: destructive command allowed.")
                }
                _ => Evaluation::ask("Command can destroy data.", risk, false),
            },
            CommandRisk::Mutating => match self.mode {
                PermissionMode::Unrestricted => {
                    Evaluation::allow("Unrestricted mode: command allowed.")
                }
                _ => Evaluation::ask("Command changes state.", risk, false),
            },
            CommandRisk::ReadOnly => match self.mode {
                PermissionMode::Balanced | PermissionMode::Unrestricted => {
                    Evaluation::allow("Read-only command.")
                }
                _ => Evaluation::ask("Command execution is confirmed in this mode.", risk, false),
            },
        }
    }

    fn eval_network(&self, url: &str) -> Evaluation {
        let local = url.starts_with("http://localhost")
            || url.starts_with("http://127.0.0.1")
            || url.starts_with("http://[::1]");
        if local {
            return Evaluation::allow("Request to the local machine.");
        }
        match self.mode {
            PermissionMode::Unrestricted => Evaluation::allow("Unrestricted mode: network allowed."),
            _ => Evaluation::ask("Request leaves this machine.", CommandRisk::Mutating, true),
        }
    }
}

/// One decided call, retained for the activity log the user reviews.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: String,
    pub timestamp: String,
    pub tool: String,
    pub summary: String,
    pub evaluation: Evaluation,
    /// What actually happened, once the user (or the rules) settled it.
    pub outcome: Outcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    AutoApproved,
    UserApproved,
    UserRejected,
    Denied,
    Failed,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine(mode: PermissionMode) -> PermissionEngine {
        PermissionEngine::new(workspace_root(), mode)
    }

    fn workspace_root() -> PathBuf {
        if cfg!(windows) {
            PathBuf::from(r"C:\work\project")
        } else {
            PathBuf::from("/work/project")
        }
    }

    fn inside(rel: &str) -> PathBuf {
        workspace_root().join(rel)
    }

    fn outside(rel: &str) -> PathBuf {
        if cfg!(windows) {
            PathBuf::from(r"C:\elsewhere").join(rel)
        } else {
            PathBuf::from("/elsewhere").join(rel)
        }
    }

    #[test]
    fn workspace_reads_are_automatic_in_every_mode() {
        for mode in [
            PermissionMode::ReadOnly,
            PermissionMode::ProjectAuto,
            PermissionMode::Balanced,
        ] {
            let e = engine(mode).evaluate(&Action::ReadFile { path: inside("src/main.rs") });
            assert_eq!(e.decision, Decision::Allow, "mode {mode:?}");
        }
    }

    #[test]
    fn read_only_mode_confirms_writes() {
        let e = engine(PermissionMode::ReadOnly)
            .evaluate(&Action::WriteFile { path: inside("src/main.rs") });
        assert_eq!(e.decision, Decision::Ask);
    }

    #[test]
    fn project_auto_allows_workspace_writes() {
        let e = engine(PermissionMode::ProjectAuto)
            .evaluate(&Action::WriteFile { path: inside("src/main.rs") });
        assert_eq!(e.decision, Decision::Allow);
    }

    #[test]
    fn writes_outside_the_workspace_are_confirmed() {
        let e = engine(PermissionMode::ProjectAuto)
            .evaluate(&Action::WriteFile { path: outside("notes.txt") });
        assert_eq!(e.decision, Decision::Ask);
        assert!(e.escapes_workspace);
    }

    #[test]
    fn traversal_out_of_the_workspace_is_not_treated_as_inside() {
        let escape = inside("../../Windows/System32/drivers/etc/hosts");
        let e = engine(PermissionMode::ProjectAuto).evaluate(&Action::WriteFile { path: escape });
        assert_ne!(e.decision, Decision::Allow, "`..` must not grant workspace trust");
    }

    #[test]
    fn deletion_inside_the_workspace_still_asks() {
        let e = engine(PermissionMode::ProjectAuto)
            .evaluate(&Action::DeleteFile { path: inside("src/main.rs") });
        assert_eq!(e.decision, Decision::Ask);
    }

    #[test]
    fn balanced_mode_runs_read_only_commands() {
        let e = engine(PermissionMode::Balanced).evaluate(&Action::RunCommand {
            program: "git".into(),
            args: vec!["status".into()],
            cwd: Some(workspace_root()),
        });
        assert_eq!(e.decision, Decision::Allow);
    }

    #[test]
    fn balanced_mode_still_confirms_mutating_commands() {
        let e = engine(PermissionMode::Balanced).evaluate(&Action::RunCommand {
            program: "git".into(),
            args: vec!["push".into(), "--force".into()],
            cwd: Some(workspace_root()),
        });
        assert_eq!(e.decision, Decision::Ask);
    }

    #[test]
    fn catastrophic_commands_are_denied_even_when_unrestricted() {
        let engine = engine(PermissionMode::Unrestricted);
        for (program, args) in [
            ("rm", vec!["-rf".to_string(), "/".to_string()]),
            ("format", vec!["C:".to_string()]),
            ("mkfs.ext4", vec!["/dev/sda".to_string()]),
        ] {
            let e = engine.evaluate(&Action::RunCommand {
                program: program.into(),
                args,
                cwd: Some(workspace_root()),
            });
            assert_eq!(e.decision, Decision::Deny, "{program} must never auto-run");
        }
    }

    #[test]
    fn credential_paths_are_denied_in_every_mode() {
        let engine = engine(PermissionMode::Unrestricted);
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/home/u"));
        for p in [home.join(".ssh").join("id_rsa"), home.join(".aws").join("credentials")] {
            let e = engine.evaluate(&Action::ReadFile { path: p.clone() });
            assert_eq!(e.decision, Decision::Deny, "{} must be refused", p.display());
        }
    }

    #[test]
    fn unrestricted_mode_allows_ordinary_destructive_commands() {
        let e = engine(PermissionMode::Unrestricted).evaluate(&Action::RunCommand {
            program: "git".into(),
            args: vec!["reset".into(), "--hard".into()],
            cwd: Some(workspace_root()),
        });
        assert_eq!(e.decision, Decision::Allow);
    }

    #[test]
    fn localhost_requests_do_not_prompt() {
        let e = engine(PermissionMode::ProjectAuto)
            .evaluate(&Action::Network { url: "http://localhost:11434/api/tags".into() });
        assert_eq!(e.decision, Decision::Allow);
    }

    #[test]
    fn external_requests_prompt() {
        let e = engine(PermissionMode::ProjectAuto)
            .evaluate(&Action::Network { url: "https://example.com".into() });
        assert_eq!(e.decision, Decision::Ask);
    }
}
