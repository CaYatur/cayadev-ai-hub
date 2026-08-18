//! GitHub tools, over the `gh` CLI.
//!
//! ## Why these are coarse rather than thin wrappers
//!
//! Opening a pull request from a dirty working tree is six commands: create a
//! branch, stage, commit, push with upstream, open the PR, report the URL. A 7B
//! model asked to do that as six tool calls will drop one, mis-order two, and
//! run out of step budget. Each call is also a full model turn — at 42 tok/s
//! that is most of a minute of pure overhead.
//!
//! So `gh_open_pr` does all six and returns the URL. The model spends one step
//! and cannot get the order wrong. That is the same reasoning as the routing
//! and prompt work: fewer, larger steps is how a small model finishes a task.
//!
//! ## Publishing always asks
//!
//! Anything other people can see — a pull request, a release, a comment, a
//! merge — is [`Action::Publish`], which the permission engine confirms in
//! *every* mode, `Unrestricted` included. Unrestricted means "do not interrupt
//! me about my own machine"; it does not mean "post as me without asking".

use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use crate::ollama::types::Tool;
use crate::perms::Action;

const GH_TIMEOUT: Duration = Duration::from_secs(90);
const MAX_OUTPUT: usize = 20_000;

pub struct GitHubTools {
    workspace: PathBuf,
    /// Report the command instead of running it. Used by tests so no test can
    /// open a real pull request.
    dry_run: bool,
}

impl GitHubTools {
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        Self { workspace: workspace.into(), dry_run: false }
    }

    pub fn dry_run(workspace: impl Into<PathBuf>) -> Self {
        Self { workspace: workspace.into(), dry_run: true }
    }

    fn arg(args: &Map<String, Value>, key: &str) -> Option<String> {
        args.get(key).and_then(Value::as_str).map(str::to_string)
    }

    fn required(args: &Map<String, Value>, key: &str) -> Result<String, String> {
        Self::arg(args, key).ok_or_else(|| format!("missing `{key}`"))
    }

    /// Run a program in the workspace and return its combined output.
    async fn exec(&self, program: &str, argv: &[String]) -> Result<String, String> {
        if self.dry_run {
            return Ok(format!("[dry-run] {program} {}", argv.join(" ")));
        }

        let output = tokio::time::timeout(
            GH_TIMEOUT,
            tokio::process::Command::new(program)
                .args(argv)
                .current_dir(&self.workspace)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output(),
        )
        .await
        .map_err(|_| format!("`{program}` timed out after {}s", GH_TIMEOUT.as_secs()))?
        .map_err(|e| format!("cannot run `{program}`: {e}. Is the GitHub CLI installed?"))?;

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();

        if output.status.success() {
            let text = if stdout.is_empty() { stderr } else { stdout };
            Ok(text.chars().take(MAX_OUTPUT).collect())
        } else {
            // `gh` puts the useful diagnosis on stderr, so surface it verbatim —
            // the model needs the real reason to adapt.
            Err(if stderr.is_empty() { stdout } else { stderr })
        }
    }

    async fn gh(&self, argv: &[&str]) -> Result<String, String> {
        let owned: Vec<String> = argv.iter().map(|s| s.to_string()).collect();
        self.exec("gh", &owned).await
    }

    async fn git(&self, argv: &[&str]) -> Result<String, String> {
        let owned: Vec<String> = argv.iter().map(|s| s.to_string()).collect();
        self.exec("git", &owned).await
    }

    /// A branch name derived from a title, safe for git refs.
    fn slugify(title: &str) -> String {
        let slug: String = title
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
            .collect();
        let trimmed = slug.trim_matches('-').to_string();
        let collapsed = trimmed
            .split('-')
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("-");
        collapsed.chars().take(40).collect::<String>().trim_matches('-').to_string()
    }
}

/// Tool names that publish something other people can see.
pub const PUBLISHING_TOOLS: &[&str] = &[
    "gh_open_pr",
    "gh_create_issue",
    "gh_comment",
    "gh_merge_pr",
    "gh_create_release",
    "gh_push",
];

pub fn schemas() -> Vec<Tool> {
    vec![
        Tool::function(
            "gh_status",
            "Repository, branch, uncommitted files, and how far ahead or behind the remote.",
            json!({"type": "object", "properties": {}}),
        ),
        Tool::function(
            "gh_list",
            "List issues, pull requests, releases, or Actions runs.",
            json!({
                "type": "object",
                "properties": {
                    "what": {"type": "string", "description": "issues, prs, releases, or runs"},
                    "state": {"type": "string", "description": "open, closed, merged, or all"},
                    "limit": {"type": "integer", "description": "How many to return"}
                },
                "required": ["what"]
            }),
        ),
        Tool::function(
            "gh_view",
            "Read one issue, pull request, release, or Actions run in full.",
            json!({
                "type": "object",
                "properties": {
                    "what": {"type": "string", "description": "issue, pr, release, or run"},
                    "id": {"type": "string", "description": "Number, tag, or run id"}
                },
                "required": ["what", "id"]
            }),
        ),
        Tool::function(
            "gh_pr_checks",
            "CI check results for a pull request, or the current branch's.",
            json!({
                "type": "object",
                "properties": {"pr": {"type": "string", "description": "PR number; omit for the current branch"}}
            }),
        ),
        Tool::function(
            "gh_diff",
            "Uncommitted changes, or the diff of a pull request.",
            json!({
                "type": "object",
                "properties": {"pr": {"type": "string", "description": "PR number; omit for the working tree"}}
            }),
        ),
        Tool::function(
            "gh_commit",
            "Stage everything and commit locally. Does not push.",
            json!({
                "type": "object",
                "properties": {"message": {"type": "string", "description": "Commit message"}},
                "required": ["message"]
            }),
        ),
        Tool::function(
            "gh_push",
            "Push the current branch to the remote, setting upstream if needed.",
            json!({"type": "object", "properties": {}}),
        ),
        Tool::function(
            "gh_open_pr",
            "Branch, commit, push, and open a pull request in one step.",
            json!({
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Pull request title"},
                    "body": {"type": "string", "description": "Pull request description"},
                    "branch": {"type": "string", "description": "Branch name; derived from the title if omitted"},
                    "draft": {"type": "boolean", "description": "Open as a draft"}
                },
                "required": ["title", "body"]
            }),
        ),
        Tool::function(
            "gh_create_issue",
            "Open an issue.",
            json!({
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "labels": {"type": "string", "description": "Comma-separated labels"}
                },
                "required": ["title", "body"]
            }),
        ),
        Tool::function(
            "gh_comment",
            "Comment on an issue or pull request.",
            json!({
                "type": "object",
                "properties": {
                    "what": {"type": "string", "description": "issue or pr"},
                    "id": {"type": "string", "description": "Number"},
                    "body": {"type": "string"}
                },
                "required": ["what", "id", "body"]
            }),
        ),
        Tool::function(
            "gh_merge_pr",
            "Merge a pull request.",
            json!({
                "type": "object",
                "properties": {
                    "pr": {"type": "string", "description": "PR number"},
                    "method": {"type": "string", "description": "merge, squash, or rebase"}
                },
                "required": ["pr"]
            }),
        ),
        Tool::function(
            "gh_create_release",
            "Tag the current commit and publish a release in one step.",
            json!({
                "type": "object",
                "properties": {
                    "tag": {"type": "string", "description": "Version tag, e.g. v0.2.0"},
                    "title": {"type": "string"},
                    "notes": {"type": "string", "description": "Release notes"},
                    "prerelease": {"type": "boolean"},
                    "files": {"type": "string", "description": "Comma-separated paths to attach"}
                },
                "required": ["tag", "title", "notes"]
            }),
        ),
    ]
}

/// What a GitHub tool would do, for the permission engine.
pub fn action_for(name: &str, args: &Map<String, Value>, workspace: &Path) -> Option<Action> {
    if PUBLISHING_TOOLS.contains(&name) {
        return Some(Action::Publish { summary: describe(name, args) });
    }
    match name {
        "gh_status" | "gh_list" | "gh_view" | "gh_pr_checks" | "gh_diff" => {
            Some(Action::RunCommand {
                program: "gh".into(),
                args: vec!["--read-only".into()],
                cwd: Some(workspace.to_path_buf()),
            })
        }
        // Local commit: changes the repository but nothing anyone else sees.
        "gh_commit" => Some(Action::RunCommand {
            program: "git".into(),
            args: vec!["commit".into()],
            cwd: Some(workspace.to_path_buf()),
        }),
        _ => None,
    }
}

/// One line describing a publishing action, shown in the approval prompt.
fn describe(name: &str, args: &Map<String, Value>) -> String {
    let get = |k: &str| GitHubTools::arg(args, k).unwrap_or_default();
    match name {
        "gh_open_pr" => format!("Open a pull request titled \"{}\"", get("title")),
        "gh_create_issue" => format!("Open an issue titled \"{}\"", get("title")),
        "gh_comment" => format!("Comment on {} #{}", get("what"), get("id")),
        "gh_merge_pr" => format!("Merge pull request #{}", get("pr")),
        "gh_create_release" => format!("Publish release {}", get("tag")),
        "gh_push" => "Push the current branch to the remote".to_string(),
        other => other.to_string(),
    }
}

impl GitHubTools {
    pub async fn execute(&self, name: &str, args: &Map<String, Value>) -> Result<String, String> {
        match name {
            "gh_status" => {
                let repo = self
                    .gh(&["repo", "view", "--json", "nameWithOwner,defaultBranchRef", "-q",
                          ".nameWithOwner + \" (default: \" + .defaultBranchRef.name + \")\""])
                    .await
                    .unwrap_or_else(|e| format!("no GitHub remote: {e}"));
                let branch = self.git(&["rev-parse", "--abbrev-ref", "HEAD"]).await?;
                let dirty = self.git(&["status", "--porcelain"]).await?;
                let tracking = self
                    .git(&["status", "-sb", "--porcelain=v1"])
                    .await
                    .unwrap_or_default()
                    .lines()
                    .next()
                    .unwrap_or("")
                    .to_string();

                Ok(format!(
                    "repo: {repo}\nbranch: {branch}\ntracking: {tracking}\nuncommitted:\n{}",
                    if dirty.is_empty() { "(clean)".into() } else { dirty }
                ))
            }

            "gh_list" => {
                let what = Self::required(args, "what")?;
                let state = Self::arg(args, "state").unwrap_or_else(|| "open".into());
                let limit = args
                    .get("limit")
                    .and_then(Value::as_u64)
                    .unwrap_or(20)
                    .to_string();

                match what.as_str() {
                    "issues" | "issue" => {
                        self.gh(&["issue", "list", "--state", &state, "--limit", &limit]).await
                    }
                    "prs" | "pr" | "pulls" => {
                        self.gh(&["pr", "list", "--state", &state, "--limit", &limit]).await
                    }
                    "releases" | "release" => self.gh(&["release", "list", "--limit", &limit]).await,
                    "runs" | "actions" => self.gh(&["run", "list", "--limit", &limit]).await,
                    other => Err(format!(
                        "unknown `what`: {other}. Use issues, prs, releases, or runs."
                    )),
                }
            }

            "gh_view" => {
                let what = Self::required(args, "what")?;
                let id = Self::required(args, "id")?;
                match what.as_str() {
                    "issue" => self.gh(&["issue", "view", &id, "--comments"]).await,
                    "pr" => self.gh(&["pr", "view", &id, "--comments"]).await,
                    "release" => self.gh(&["release", "view", &id]).await,
                    "run" => self.gh(&["run", "view", &id, "--log-failed"]).await,
                    other => Err(format!("unknown `what`: {other}")),
                }
            }

            "gh_pr_checks" => match Self::arg(args, "pr") {
                Some(pr) => self.gh(&["pr", "checks", &pr]).await,
                None => self.gh(&["pr", "checks"]).await,
            },

            "gh_diff" => match Self::arg(args, "pr") {
                Some(pr) => self.gh(&["pr", "diff", &pr]).await,
                None => {
                    let staged = self.git(&["diff", "--staged"]).await.unwrap_or_default();
                    let unstaged = self.git(&["diff"]).await.unwrap_or_default();
                    let combined = format!("{staged}\n{unstaged}");
                    Ok(if combined.trim().is_empty() {
                        "(no uncommitted changes)".into()
                    } else {
                        combined.chars().take(MAX_OUTPUT).collect()
                    })
                }
            },

            "gh_commit" => {
                let message = Self::required(args, "message")?;
                self.git(&["add", "-A"]).await?;
                let status = self.git(&["status", "--porcelain"]).await?;
                if status.trim().is_empty() {
                    return Ok("Nothing to commit.".into());
                }
                self.git(&["commit", "-m", &message]).await?;
                self.git(&["log", "--oneline", "-1"]).await
            }

            "gh_push" => {
                let branch = self.git(&["rev-parse", "--abbrev-ref", "HEAD"]).await?;
                // `-u` is harmless when upstream already exists and necessary
                // when it does not, so it is always passed.
                self.git(&["push", "-u", "origin", &branch]).await
            }

            "gh_open_pr" => {
                let title = Self::required(args, "title")?;
                let body = Self::required(args, "body")?;
                let branch = Self::arg(args, "branch")
                    .unwrap_or_else(|| format!("agent/{}", Self::slugify(&title)));
                let draft = args.get("draft").and_then(Value::as_bool).unwrap_or(false);

                let current = self.git(&["rev-parse", "--abbrev-ref", "HEAD"]).await?;
                // Never open a PR from the default branch against itself.
                if current == branch {
                    tracing::info!("already on {branch}");
                } else {
                    self.git(&["checkout", "-b", &branch]).await?;
                }

                self.git(&["add", "-A"]).await?;
                let dirty = self.git(&["status", "--porcelain"]).await?;
                if !dirty.trim().is_empty() {
                    self.git(&["commit", "-m", &title]).await?;
                }
                self.git(&["push", "-u", "origin", &branch]).await?;

                let mut argv = vec![
                    "pr".to_string(),
                    "create".to_string(),
                    "--title".to_string(),
                    title,
                    "--body".to_string(),
                    body,
                ];
                if draft {
                    argv.push("--draft".to_string());
                }
                self.exec("gh", &argv).await
            }

            "gh_create_issue" => {
                let mut argv = vec![
                    "issue".to_string(),
                    "create".to_string(),
                    "--title".to_string(),
                    Self::required(args, "title")?,
                    "--body".to_string(),
                    Self::required(args, "body")?,
                ];
                if let Some(labels) = Self::arg(args, "labels") {
                    argv.push("--label".to_string());
                    argv.push(labels);
                }
                self.exec("gh", &argv).await
            }

            "gh_comment" => {
                let what = Self::required(args, "what")?;
                let id = Self::required(args, "id")?;
                let body = Self::required(args, "body")?;
                let kind = match what.as_str() {
                    "issue" => "issue",
                    "pr" => "pr",
                    other => return Err(format!("unknown `what`: {other}")),
                };
                self.exec(
                    "gh",
                    &[kind.into(), "comment".into(), id, "--body".into(), body],
                )
                .await
            }

            "gh_merge_pr" => {
                let pr = Self::required(args, "pr")?;
                let method = Self::arg(args, "method").unwrap_or_else(|| "squash".into());
                let flag = match method.as_str() {
                    "merge" => "--merge",
                    "rebase" => "--rebase",
                    _ => "--squash",
                };
                self.gh(&["pr", "merge", &pr, flag, "--delete-branch"]).await
            }

            "gh_create_release" => {
                let tag = Self::required(args, "tag")?;
                let title = Self::required(args, "title")?;
                let notes = Self::required(args, "notes")?;

                let mut argv = vec![
                    "release".to_string(),
                    "create".to_string(),
                    tag,
                    "--title".to_string(),
                    title,
                    "--notes".to_string(),
                    notes,
                ];
                if args.get("prerelease").and_then(Value::as_bool).unwrap_or(false) {
                    argv.push("--prerelease".to_string());
                }
                if let Some(files) = Self::arg(args, "files") {
                    for file in files.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                        argv.push(file.to_string());
                    }
                }
                self.exec("gh", &argv).await
            }

            other => Err(format!("unknown GitHub tool `{other}`")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perms::{Decision, PermissionEngine, PermissionMode};

    fn args(v: Value) -> Map<String, Value> {
        serde_json::from_value(v).expect("object")
    }

    #[test]
    fn slugify_produces_a_usable_branch_name() {
        assert_eq!(GitHubTools::slugify("Add rate limiting to /api/chat!"), "add-rate-limiting-to-api-chat");
        assert_eq!(GitHubTools::slugify("  Fix   bug  "), "fix-bug");
        assert!(GitHubTools::slugify(&"x".repeat(200)).len() <= 40);
    }

    #[test]
    fn publishing_is_confirmed_even_in_unrestricted_mode() {
        // Unrestricted means "stop interrupting me about my own machine", not
        // "post as me without asking".
        let workspace = std::env::temp_dir();
        let engine = PermissionEngine::new(&workspace, PermissionMode::Unrestricted);

        for (tool, payload) in [
            ("gh_open_pr", json!({"title": "T", "body": "B"})),
            ("gh_create_issue", json!({"title": "T", "body": "B"})),
            ("gh_comment", json!({"what": "pr", "id": "1", "body": "B"})),
            ("gh_merge_pr", json!({"pr": "1"})),
            ("gh_create_release", json!({"tag": "v1", "title": "T", "notes": "N"})),
            ("gh_push", json!({})),
        ] {
            let action = action_for(tool, &args(payload), &workspace)
                .unwrap_or_else(|| panic!("{tool} should map to an action"));
            assert_eq!(
                engine.evaluate(&action).decision,
                Decision::Ask,
                "{tool} must be confirmed before it reaches other people"
            );
        }
    }

    #[test]
    fn the_approval_prompt_says_what_will_be_published() {
        let summary = describe("gh_open_pr", &args(json!({"title": "Add rate limiting"})));
        assert!(summary.contains("Add rate limiting"), "got: {summary}");
        let release = describe("gh_create_release", &args(json!({"tag": "v0.2.0"})));
        assert!(release.contains("v0.2.0"), "got: {release}");
    }

    #[test]
    fn read_only_github_tools_are_not_publishing() {
        let workspace = std::env::temp_dir();
        for tool in ["gh_status", "gh_list", "gh_view", "gh_pr_checks", "gh_diff"] {
            let action = action_for(tool, &Map::new(), &workspace).unwrap();
            assert!(
                !matches!(action, Action::Publish { .. }),
                "{tool} should not be treated as publishing"
            );
        }
    }

    #[tokio::test]
    async fn a_dry_run_never_executes() {
        let tools = GitHubTools::dry_run(std::env::temp_dir());
        let out = tools
            .execute("gh_create_issue", &args(json!({"title": "T", "body": "B"})))
            .await
            .unwrap();
        assert!(out.starts_with("[dry-run]"), "got: {out}");
        assert!(out.contains("issue create"));
    }

    #[tokio::test]
    async fn missing_arguments_are_named() {
        let tools = GitHubTools::dry_run(std::env::temp_dir());
        let err = tools.execute("gh_comment", &args(json!({"what": "pr"}))).await.unwrap_err();
        assert!(err.contains("id"), "got: {err}");
    }

    #[test]
    fn every_schema_has_a_matching_action() {
        // A tool the engine cannot classify would run ungated.
        let workspace = std::env::temp_dir();
        for tool in schemas() {
            let name = tool.function.name;
            assert!(
                action_for(&name, &Map::new(), &workspace).is_some(),
                "{name} has no Action and would bypass the permission engine"
            );
        }
    }
}
