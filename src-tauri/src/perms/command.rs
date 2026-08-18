//! Command risk classification.
//!
//! Decides how much damage a command could do, so the engine knows whether it
//! can run unattended. Classification is allow-list driven: a command earns
//! `ReadOnly` only by being recognised as read-only. Anything unrecognised is
//! `Mutating`, which means "ask", never "run".
//!
//! Two traps this module is written around:
//!
//! * **Shell wrappers.** `bash -c "git status && rm -rf /"` is not a `bash`
//!   invocation to be classified once; the payload has to be re-analysed, and
//!   the result is the *worst* risk found in any segment.
//! * **Subcommand sensitivity.** `git status` and `git push --force` share a
//!   program name and nothing else. Risk is decided by the full argument
//!   vector, not the executable.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandRisk {
    /// Observes without changing anything.
    ReadOnly,
    /// Changes state recoverably.
    Mutating,
    /// Can destroy work that is not easily recovered.
    Destructive,
    /// Irreversible at system scope. Never runs automatically in any mode.
    Catastrophic,
}

/// Programs whose *entire* purpose is to run another command line.
const SHELLS: &[&str] = &["sh", "bash", "zsh", "fish", "dash", "cmd", "powershell", "pwsh"];

/// `program subcommand` pairs that only read.
const READ_ONLY_SUBCOMMANDS: &[(&str, &[&str])] = &[
    ("git", &["status", "log", "diff", "show", "branch", "remote", "blame", "describe", "ls-files", "rev-parse", "config"]),
    ("cargo", &["check", "tree", "metadata", "search"]),
    ("npm", &["list", "ls", "view", "outdated", "why", "search"]),
    ("pnpm", &["list", "ls", "why", "outdated"]),
    ("yarn", &["list", "why", "info"]),
    ("pip", &["list", "show", "freeze"]),
    ("docker", &["ps", "images", "logs", "inspect", "version"]),
    ("kubectl", &["get", "describe", "logs", "version"]),
    ("gh", &["pr", "issue", "repo", "run", "release", "api", "auth"]),
    ("adb", &["devices", "shell", "logcat"]),
];

/// Whole programs that only read.
const READ_ONLY_PROGRAMS: &[&str] = &[
    "ls", "dir", "pwd", "cat", "type", "head", "tail", "wc", "file", "stat",
    "find", "grep", "rg", "ripgrep", "fd", "which", "where", "whoami", "hostname",
    "date", "env", "printenv", "echo", "uname", "df", "du", "ps", "top",
    "tree", "diff", "cmp", "md5sum", "sha256sum", "node", "python", "python3",
    "rustc", "tsc", "jq", "curl", "wget",
];

/// Destructive patterns, matched against the normalised command line.
const DESTRUCTIVE_PATTERNS: &[&str] = &[
    "git reset --hard",
    "git clean -fd",
    "git clean -df",
    "git push --force",
    "git push -f",
    "git branch -d",
    "git checkout --",
    "git restore",
    "npm publish",
    "cargo publish",
    "drop table",
    "drop database",
    "truncate table",
    "docker system prune",
    "docker volume rm",
    "kubectl delete",
];

/// Catastrophic patterns. Refused in every mode, Unrestricted included.
const CATASTROPHIC_PATTERNS: &[&str] = &[
    "mkfs",
    "diskpart",
    "fdisk",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "reg delete hklm",
    "vssadmin delete",
    "bcdedit",
    "cipher /w",
    "takeown /f c:\\",
    "icacls c:\\",
    ":(){:|:&};:",
];

/// True for Windows-style switches like `/s` or `/q`, but not for paths.
fn is_windows_switch(arg: &str) -> bool {
    arg.starts_with('/')
        && (2..=3).contains(&arg.len())
        && arg[1..].chars().all(|c| c.is_ascii_alphabetic())
}

/// Split a shell payload on operators that begin a new command.
///
/// Quoted spans are respected so a literal `;` inside a string does not create
/// a phantom segment.
fn split_shell_segments(line: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '\'' if !in_double => {
                in_single = !in_single;
                current.push(c);
            }
            '"' if !in_single => {
                in_double = !in_double;
                current.push(c);
            }
            _ if in_single || in_double => current.push(c),
            ';' | '\n' => {
                segments.push(std::mem::take(&mut current));
            }
            '&' | '|' => {
                // Consume a doubled operator so `&&` yields one break.
                if chars.peek() == Some(&c) {
                    chars.next();
                }
                segments.push(std::mem::take(&mut current));
            }
            '`' => {
                // Command substitution: treat the contents as their own segment.
                segments.push(std::mem::take(&mut current));
            }
            _ => current.push(c),
        }
    }
    segments.push(current);

    segments
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Expand `$(...)` substitutions into separate segments to be classified.
fn extract_substitutions(line: &str) -> Vec<String> {
    let mut found = Vec::new();
    let bytes = line.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'$' && bytes[i + 1] == b'(' {
            let mut depth = 1;
            let mut j = i + 2;
            while j < bytes.len() && depth > 0 {
                match bytes[j] {
                    b'(' => depth += 1,
                    b')' => depth -= 1,
                    _ => {}
                }
                j += 1;
            }
            if depth == 0 {
                found.push(line[i + 2..j - 1].to_string());
            }
            i = j;
        } else {
            i += 1;
        }
    }
    found
}

/// Classify a command by program and arguments.
pub fn classify_command(program: &str, args: &[String]) -> CommandRisk {
    let program_name = program
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(program)
        .trim_end_matches(".exe")
        .to_lowercase();

    // A shell invocation is only as safe as what it was asked to run.
    if SHELLS.contains(&program_name.as_str()) {
        let payload = args
            .iter()
            .skip_while(|a| a.starts_with('-') || a.eq_ignore_ascii_case("/c"))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ");

        if payload.is_empty() {
            return CommandRisk::Mutating; // Interactive shell; nothing to inspect.
        }

        let mut segments = split_shell_segments(&payload);
        segments.extend(extract_substitutions(&payload));

        return segments
            .iter()
            .map(|segment| {
                let mut parts = segment.split_whitespace();
                match parts.next() {
                    Some(prog) => {
                        let rest: Vec<String> = parts.map(String::from).collect();
                        classify_command(prog, &rest)
                    }
                    None => CommandRisk::ReadOnly,
                }
            })
            .max()
            .unwrap_or(CommandRisk::Mutating);
    }

    let full = format!("{program_name} {}", args.join(" ")).to_lowercase();
    let full = full.trim().to_string();

    if CATASTROPHIC_PATTERNS.iter().any(|p| full.contains(p)) {
        return CommandRisk::Catastrophic;
    }

    // Recursive force-delete: catastrophic at a filesystem root, destructive
    // otherwise. `rm -rf build/` is routine; `rm -rf /` is not.
    if matches!(program_name.as_str(), "rm" | "rmdir" | "del" | "erase" | "rd") {
        let recursive_force = args.iter().any(|a| {
            let a = a.to_lowercase();
            (a.starts_with('-') && a.contains('r')) || a == "/s" || a == "/q"
        });
        // `/` is both the Unix root and the prefix of Windows switches such as
        // `/s`. Excluding everything that starts with `/` would drop the single
        // most dangerous target there is, so only short alphabetic tokens count
        // as switches — `/s` and `/q` do, `/` and `/home` do not.
        let targets: Vec<String> = args
            .iter()
            .filter(|a| !a.starts_with('-') && !is_windows_switch(a))
            .map(|a| {
                let trimmed = a.trim_end_matches(['/', '\\']);
                // Trimming leaves "/" and "C:\" empty; keep the original so the
                // root check below still sees them.
                if trimmed.is_empty() { a.to_lowercase() } else { trimmed.to_lowercase() }
            })
            .collect();

        let hits_root = targets.iter().any(|t| {
            t.is_empty()
                || t == "/"
                || t == "*"
                || t == "~"
                || t == "."
                || t.len() == 2 && t.ends_with(':')
                || matches!(t.as_str(), "c:" | "c:\\" | "/home" | "/users" | "/etc" | "/usr" | "/var")
        });

        if hits_root {
            return CommandRisk::Catastrophic;
        }
        return if recursive_force { CommandRisk::Destructive } else { CommandRisk::Mutating };
    }

    // Raw device writes.
    if program_name == "dd" && args.iter().any(|a| a.starts_with("of=/dev/")) {
        return CommandRisk::Catastrophic;
    }
    if program_name == "format" {
        return CommandRisk::Catastrophic;
    }

    if DESTRUCTIVE_PATTERNS.iter().any(|p| full.contains(p)) {
        return CommandRisk::Destructive;
    }

    if let Some((_, subs)) = READ_ONLY_SUBCOMMANDS.iter().find(|(p, _)| *p == program_name) {
        let sub = args
            .iter()
            .find(|a| !a.starts_with('-'))
            .map(|s| s.to_lowercase())
            .unwrap_or_default();

        if subs.contains(&sub.as_str()) {
            // `gh` and `adb` subcommands split further into read and write
            // verbs; only the clearly-reading ones qualify.
            if matches!(program_name.as_str(), "gh" | "adb" | "kubectl" | "docker") {
                let writes = ["create", "delete", "edit", "merge", "close", "push",
                              "install", "uninstall", "reboot", "apply", "rm"];
                if args.iter().any(|a| writes.contains(&a.to_lowercase().as_str())) {
                    return CommandRisk::Mutating;
                }
            }
            return CommandRisk::ReadOnly;
        }
        return CommandRisk::Mutating;
    }

    if READ_ONLY_PROGRAMS.contains(&program_name.as_str()) {
        // Output redirection turns any reader into a writer.
        if args.iter().any(|a| a.contains('>')) {
            return CommandRisk::Mutating;
        }
        // `curl -o file` and `wget` write to disk; downloads are not read-only.
        if matches!(program_name.as_str(), "curl" | "wget")
            && args.iter().any(|a| matches!(a.as_str(), "-o" | "-O" | "--output"))
        {
            return CommandRisk::Mutating;
        }
        return CommandRisk::ReadOnly;
    }

    // Unrecognised: assume it changes something.
    CommandRisk::Mutating
}

#[cfg(test)]
mod tests {
    use super::*;

    fn risk(line: &str) -> CommandRisk {
        let mut parts = line.split_whitespace();
        let program = parts.next().unwrap();
        let args: Vec<String> = parts.map(String::from).collect();
        classify_command(program, &args)
    }

    #[test]
    fn inspection_commands_are_read_only() {
        for line in ["git status", "git log --oneline", "ls -la", "cargo check", "rg pattern"] {
            assert_eq!(risk(line), CommandRisk::ReadOnly, "{line}");
        }
    }

    #[test]
    fn ordinary_state_changes_are_mutating() {
        for line in ["git commit -m x", "npm install", "cargo build", "mkdir out"] {
            assert_eq!(risk(line), CommandRisk::Mutating, "{line}");
        }
    }

    #[test]
    fn history_rewrites_are_destructive() {
        for line in ["git reset --hard HEAD~3", "git push --force origin main", "git clean -fd"] {
            assert_eq!(risk(line), CommandRisk::Destructive, "{line}");
        }
    }

    #[test]
    fn scoped_recursive_delete_is_destructive_not_catastrophic() {
        assert_eq!(risk("rm -rf build/"), CommandRisk::Destructive);
        assert_eq!(risk("rm -rf node_modules"), CommandRisk::Destructive);
    }

    #[test]
    fn deleting_a_filesystem_root_is_catastrophic() {
        for line in ["rm -rf /", "rm -rf ~", "rm -rf *", "rm -rf C:\\", "rm -rf /home"] {
            assert_eq!(risk(line), CommandRisk::Catastrophic, "{line}");
        }
    }

    #[test]
    fn disk_and_power_operations_are_catastrophic() {
        for line in ["mkfs.ext4 /dev/sda", "format C:", "diskpart", "shutdown /s", "dd of=/dev/sda"] {
            assert_eq!(risk(line), CommandRisk::Catastrophic, "{line}");
        }
    }

    #[test]
    fn a_safe_prefix_cannot_hide_a_dangerous_suffix() {
        // The injection this module exists to catch: classifying `bash` once
        // would report ReadOnly and auto-run a disk wipe.
        let args = vec!["-c".into(), "git status && rm -rf /".into()];
        assert_eq!(classify_command("bash", &args), CommandRisk::Catastrophic);
    }

    #[test]
    fn every_chaining_operator_is_split() {
        for payload in [
            "ls; rm -rf /",
            "ls && rm -rf /",
            "ls || rm -rf /",
            "ls | rm -rf /",
            "ls\nrm -rf /",
        ] {
            let args = vec!["-c".into(), payload.into()];
            assert_eq!(
                classify_command("sh", &args),
                CommandRisk::Catastrophic,
                "payload: {payload}"
            );
        }
    }

    #[test]
    fn command_substitution_is_inspected() {
        let args = vec!["-c".into(), "echo $(rm -rf /)".into()];
        assert_eq!(classify_command("bash", &args), CommandRisk::Catastrophic);

        let backtick = vec!["-c".into(), "echo `shutdown /s`".into()];
        assert_eq!(classify_command("bash", &backtick), CommandRisk::Catastrophic);
    }

    #[test]
    fn windows_shells_are_unwrapped_too() {
        let args = vec!["/c".into(), "dir && format C:".into()];
        assert_eq!(classify_command("cmd.exe", &args), CommandRisk::Catastrophic);
    }

    #[test]
    fn a_semicolon_inside_quotes_is_not_a_new_command() {
        let args = vec!["-c".into(), r#"echo "a; b""#.into()];
        assert_eq!(classify_command("bash", &args), CommandRisk::ReadOnly);
    }

    #[test]
    fn redirection_makes_a_reader_mutating() {
        let args = vec!["hello".into(), ">".into(), "out.txt".into()];
        assert_eq!(classify_command("echo", &args), CommandRisk::Mutating);
    }

    #[test]
    fn downloads_are_not_read_only() {
        let args = vec!["-o".into(), "payload.sh".into(), "https://example.com".into()];
        assert_eq!(classify_command("curl", &args), CommandRisk::Mutating);
    }

    #[test]
    fn gh_read_verbs_and_write_verbs_are_distinguished() {
        let read = vec!["pr".into(), "list".into()];
        assert_eq!(classify_command("gh", &read), CommandRisk::ReadOnly);

        let write = vec!["pr".into(), "create".into(), "--title".into(), "x".into()];
        assert_eq!(classify_command("gh", &write), CommandRisk::Mutating);
    }

    #[test]
    fn full_paths_to_executables_are_classified_by_basename() {
        let args = vec!["status".into()];
        assert_eq!(classify_command(r"C:\Program Files\Git\bin\git.exe", &args), CommandRisk::ReadOnly);
    }

    #[test]
    fn unknown_programs_default_to_asking() {
        assert_eq!(risk("some-unknown-binary --flag"), CommandRisk::Mutating);
    }
}
