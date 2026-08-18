//! Workspace search.
//!
//! Without this the agent reaches for `run_command` with `grep` or
//! `Select-String`, which is one more command to get right, differs per
//! platform, and needs approval in most modes. Searching is a read; it should
//! be as cheap as reading a file.
//!
//! Results are capped hard. A match list is the least useful thing to spend a
//! small model's context on — it needs enough hits to choose a file, not every
//! hit in the repository.

use serde_json::{json, Map, Value};
use std::path::Path;

use crate::ollama::types::Tool;
use crate::perms::Action;

/// Matches returned before the search stops.
const MAX_MATCHES: usize = 60;
/// Files opened before the search gives up, so a huge tree cannot hang a turn.
const MAX_FILES_SCANNED: usize = 8_000;
/// Files larger than this are skipped — they are assets, not source.
const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;
/// Characters of each matching line that are shown.
const MAX_LINE_CHARS: usize = 200;

/// Directories never worth searching. Skipping them is the difference between
/// a search that takes 200 ms and one that reads a `node_modules` tree.
const SKIP_DIRS: &[&str] = &[
    ".git", "node_modules", "target", "dist", "build", ".next", ".venv", "venv",
    "__pycache__", ".cayahub", ".idea", ".vscode", "vendor", ".cargo",
];

/// Extensions that are never text.
const SKIP_EXTS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "webp", "bmp", "ico", "pdf", "zip", "gz", "tar",
    "7z", "rar", "exe", "dll", "so", "dylib", "bin", "wasm", "mp3", "mp4", "mov",
    "wav", "ogg", "ttf", "otf", "woff", "woff2", "lock",
];

pub fn schemas() -> Vec<Tool> {
    vec![Tool::function(
        "search",
        "Find text or file names in the workspace.",
        json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text to find"},
                "in_names": {
                    "type": "boolean",
                    "description": "Search file names instead of contents"
                },
                "extension": {"type": "string", "description": "Limit to one extension, e.g. rs"}
            },
            "required": ["query"]
        }),
    )]
}

pub fn handles(name: &str) -> bool {
    name == "search"
}

pub fn action_for(workspace: &Path) -> Action {
    // Searching only reads, and only inside the workspace.
    Action::ListDir { path: workspace.to_path_buf() }
}

fn skip_dir(entry: &walkdir::DirEntry) -> bool {
    entry.file_type().is_dir()
        && entry
            .file_name()
            .to_str()
            .map(|n| SKIP_DIRS.contains(&n))
            .unwrap_or(false)
}

fn is_probably_text(path: &Path) -> bool {
    match path.extension().map(|e| e.to_string_lossy().to_lowercase()) {
        Some(ext) => !SKIP_EXTS.contains(&ext.as_str()),
        // No extension: usually a script, a Makefile, or a LICENSE.
        None => true,
    }
}

/// Trim a matching line so one long minified line cannot fill the output.
fn clip(line: &str) -> String {
    let trimmed = line.trim();
    if trimmed.chars().count() <= MAX_LINE_CHARS {
        return trimmed.to_string();
    }
    let cut: String = trimmed.chars().take(MAX_LINE_CHARS).collect();
    format!("{cut}…")
}

pub fn execute(workspace: &Path, args: &Map<String, Value>) -> Result<String, String> {
    let query = args
        .get("query")
        .and_then(Value::as_str)
        .ok_or("missing `query`")?;
    if query.trim().is_empty() {
        return Err("`query` is empty".into());
    }

    let in_names = args.get("in_names").and_then(Value::as_bool).unwrap_or(false);
    let extension = args
        .get("extension")
        .and_then(Value::as_str)
        .map(|e| e.trim_start_matches('.').to_lowercase());

    let needle = query.to_lowercase();
    let mut matches: Vec<String> = Vec::new();
    let mut scanned = 0usize;
    let mut truncated = false;

    let walker = walkdir::WalkDir::new(workspace)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| !skip_dir(e));

    for entry in walker.filter_map(Result::ok) {
        if matches.len() >= MAX_MATCHES || scanned >= MAX_FILES_SCANNED {
            truncated = true;
            break;
        }
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();

        if let Some(ext) = &extension {
            let actual = path.extension().map(|e| e.to_string_lossy().to_lowercase());
            if actual.as_deref() != Some(ext.as_str()) {
                continue;
            }
        }

        let relative = path
            .strip_prefix(workspace)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");

        if in_names {
            if relative.to_lowercase().contains(&needle) {
                matches.push(relative);
            }
            continue;
        }

        if !is_probably_text(path) {
            continue;
        }
        match std::fs::metadata(path) {
            Ok(meta) if meta.len() > MAX_FILE_BYTES => continue,
            Err(_) => continue,
            _ => {}
        }
        // A binary file that slipped past the extension check reads as an error
        // here rather than producing garbage matches.
        let Ok(content) = std::fs::read_to_string(path) else { continue };
        scanned += 1;

        for (number, line) in content.lines().enumerate() {
            if matches.len() >= MAX_MATCHES {
                truncated = true;
                break;
            }
            if line.to_lowercase().contains(&needle) {
                matches.push(format!("{relative}:{}: {}", number + 1, clip(line)));
            }
        }
    }

    if matches.is_empty() {
        return Ok(format!("No matches for \"{query}\"."));
    }

    let mut out = matches.join("\n");
    if truncated {
        out.push_str(&format!(
            "\n\n[stopped at {MAX_MATCHES} matches — narrow the query or pass an extension]"
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("cayahub-search-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn args(v: Value) -> Map<String, Value> {
        serde_json::from_value(v).expect("object")
    }

    fn seed(dir: &Path) {
        std::fs::write(dir.join("main.rs"), "fn main() {\n    let timeout = 30;\n}\n").unwrap();
        std::fs::write(dir.join("config.toml"), "timeout = 30\n").unwrap();
        std::fs::create_dir_all(dir.join("node_modules/pkg")).unwrap();
        std::fs::write(dir.join("node_modules/pkg/index.js"), "timeout\n").unwrap();
        std::fs::create_dir_all(dir.join(".git")).unwrap();
        std::fs::write(dir.join(".git/COMMIT_EDITMSG"), "timeout\n").unwrap();
    }

    #[test]
    fn finds_text_with_file_and_line() {
        let dir = scratch();
        seed(&dir);
        let out = execute(&dir, &args(json!({"query": "timeout"}))).unwrap();
        assert!(out.contains("main.rs:2:"), "got: {out}");
        assert!(out.contains("config.toml:1:"), "got: {out}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn skips_dependency_and_vcs_directories() {
        // Searching node_modules is the difference between 200 ms and minutes.
        let dir = scratch();
        seed(&dir);
        let out = execute(&dir, &args(json!({"query": "timeout"}))).unwrap();
        assert!(!out.contains("node_modules"), "got: {out}");
        assert!(!out.contains(".git"), "got: {out}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn filters_by_extension() {
        let dir = scratch();
        seed(&dir);
        let out = execute(&dir, &args(json!({"query": "timeout", "extension": "rs"}))).unwrap();
        assert!(out.contains("main.rs"));
        assert!(!out.contains("config.toml"), "got: {out}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn searches_file_names() {
        let dir = scratch();
        seed(&dir);
        let out = execute(&dir, &args(json!({"query": "config", "in_names": true}))).unwrap();
        assert!(out.contains("config.toml"));
        // Contents must not be searched in name mode.
        assert!(!out.contains(":1:"), "got: {out}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn matching_is_case_insensitive() {
        let dir = scratch();
        seed(&dir);
        let out = execute(&dir, &args(json!({"query": "TIMEOUT"}))).unwrap();
        assert!(out.contains("main.rs"), "got: {out}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn no_matches_says_so_rather_than_returning_nothing() {
        let dir = scratch();
        seed(&dir);
        let out = execute(&dir, &args(json!({"query": "zzzznotpresent"}))).unwrap();
        assert!(out.starts_with("No matches"), "got: {out}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_long_line_is_clipped() {
        let dir = scratch();
        std::fs::write(dir.join("min.js"), format!("needle{}", "x".repeat(5_000))).unwrap();
        let out = execute(&dir, &args(json!({"query": "needle"}))).unwrap();
        assert!(out.len() < 500, "one minified line filled the output: {} chars", out.len());
        assert!(out.contains('…'));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn results_are_capped() {
        let dir = scratch();
        let body = "needle\n".repeat(500);
        std::fs::write(dir.join("many.txt"), body).unwrap();
        let out = execute(&dir, &args(json!({"query": "needle"}))).unwrap();
        assert_eq!(out.lines().filter(|l| l.contains("many.txt")).count(), MAX_MATCHES);
        assert!(out.contains("stopped at"), "truncation was not reported");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_empty_query_is_refused() {
        let dir = scratch();
        assert!(execute(&dir, &args(json!({"query": "   "}))).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
