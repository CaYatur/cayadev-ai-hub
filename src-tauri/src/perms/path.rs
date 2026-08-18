//! Path containment.
//!
//! "Is this path inside the workspace?" is the question that decides whether a
//! write happens silently. Getting it wrong in the permissive direction lets an
//! agent edit files anywhere on the machine, so the checks here are lexical and
//! strict rather than convenient.
//!
//! Two mistakes are easy to make and both are guarded against:
//!
//! * **String prefixes.** `/work/project-backup` starts with `/work/project`
//!   but is a different directory. Comparison is component-wise.
//! * **Resolving against the filesystem.** `canonicalize` fails on paths that do
//!   not exist yet, which is exactly the case when creating a file — and on
//!   Windows it rewrites paths into `\\?\` form. Normalisation here is purely
//!   lexical, so it works on paths that have never existed.

use std::path::{Component, Path, PathBuf};

/// Lexically resolve `.` and `..` and strip Windows verbatim prefixes.
///
/// No filesystem access, so this is safe for paths that do not exist. `..` past
/// the root is clamped rather than escaping.
pub fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    let mut depth = 0usize;

    for component in path.components() {
        match component {
            // `\\?\C:\x` and `\\?\UNC\...` normalise to their plain form so a
            // verbatim path and a normal one compare equal.
            Component::Prefix(prefix) => {
                let raw = prefix.as_os_str().to_string_lossy();
                let cleaned = raw
                    .trim_start_matches(r"\\?\UNC\")
                    .trim_start_matches(r"\\?\")
                    .to_string();
                out.push(if cleaned.is_empty() { raw.to_string() } else { cleaned });
            }
            Component::RootDir => out.push(Component::RootDir.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if depth > 0 {
                    out.pop();
                    depth -= 1;
                }
                // At depth 0 the `..` is dropped: it cannot escape the root, and
                // silently discarding it keeps the result inside the tree.
            }
            Component::Normal(part) => {
                out.push(part);
                depth += 1;
            }
        }
    }
    out
}

/// Compare two path components for equality under platform rules.
fn components_equal(a: &std::ffi::OsStr, b: &std::ffi::OsStr) -> bool {
    if cfg!(windows) {
        // NTFS is case-insensitive in practice, so `SRC` and `src` are the
        // same directory and must not be treated as different.
        a.to_string_lossy().eq_ignore_ascii_case(&b.to_string_lossy())
    } else {
        a == b
    }
}

/// True when `target` is `root` itself or lies beneath it.
///
/// Both arguments are normalised first, so callers may pass raw user input.
pub fn contains(root: &Path, target: &Path) -> bool {
    let root = normalize(root);
    let target = normalize(target);

    let mut root_parts = root.components();
    let mut target_parts = target.components();

    loop {
        match (root_parts.next(), target_parts.next()) {
            // Root exhausted: every one of its components matched, so the
            // target is at or below it.
            (None, _) => return true,
            // Target exhausted first: it is an ancestor, not a descendant.
            (Some(_), None) => return false,
            (Some(r), Some(t)) => {
                if !components_equal(r.as_os_str(), t.as_os_str()) {
                    return false;
                }
            }
        }
    }
}

/// Paths holding credentials, keys, or tokens.
///
/// Refused for both reads and writes in every mode, including Unrestricted.
/// The agent has no legitimate reason to open these, and exfiltration through a
/// "just read this file" tool call is the cheapest attack on a local agent.
pub fn is_sensitive(path: &Path) -> bool {
    let normalized = normalize(path);
    let lower = normalized.to_string_lossy().to_lowercase().replace('\\', "/");

    const SECRET_DIRS: &[&str] = &[
        "/.ssh/",
        "/.aws/",
        "/.gnupg/",
        "/.docker/config.json",
        "/.kube/",
        "/.config/gh/",
        "/appdata/roaming/microsoft/credentials/",
        "/appdata/local/microsoft/credentials/",
    ];
    if SECRET_DIRS.iter().any(|d| lower.contains(d)) {
        return true;
    }

    let file_name = normalized
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    const SECRET_FILES: &[&str] = &[
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        "id_dsa",
        ".npmrc",
        ".pypirc",
        ".netrc",
        "credentials",
        "shadow",
        "sam",
    ];
    if SECRET_FILES.contains(&file_name.as_str()) {
        return true;
    }

    // Private keys and certificate bundles by extension.
    matches!(
        normalized.extension().map(|e| e.to_string_lossy().to_lowercase()).as_deref(),
        Some("pem" | "key" | "pfx" | "p12" | "keystore" | "jks")
    )
}

/// Directories the operating system owns.
///
/// Writing here breaks the machine rather than the project, so it is refused
/// regardless of mode. `.env` files are *not* on this list — they are ordinary
/// project files that agents legitimately edit.
pub fn is_system_path(path: &Path) -> bool {
    let lower = normalize(path).to_string_lossy().to_lowercase().replace('\\', "/");

    const SYSTEM_ROOTS: &[&str] = &[
        "c:/windows/",
        "c:/program files/",
        "c:/program files (x86)/",
        "c:/programdata/microsoft/",
        "/system/",
        "/usr/bin/",
        "/usr/sbin/",
        "/bin/",
        "/sbin/",
        "/boot/",
        "/etc/",
        "/dev/",
        "/proc/",
        "/sys/",
        "/library/",
    ];
    SYSTEM_ROOTS.iter().any(|root| lower.starts_with(root))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    #[test]
    fn normalize_resolves_dot_and_parent() {
        assert_eq!(normalize(&p("/a/b/../c/./d")), p("/a/c/d"));
    }

    #[test]
    fn normalize_clamps_parent_at_root() {
        // `..` must never climb above the root and produce an escape.
        assert_eq!(normalize(&p("/a/../../../etc")), p("/etc"));
    }

    #[test]
    fn contains_accepts_descendants_and_self() {
        assert!(contains(&p("/work/project"), &p("/work/project")));
        assert!(contains(&p("/work/project"), &p("/work/project/src/main.rs")));
    }

    #[test]
    fn contains_rejects_ancestors_and_siblings() {
        assert!(!contains(&p("/work/project"), &p("/work")));
        assert!(!contains(&p("/work/project"), &p("/work/other")));
    }

    #[test]
    fn contains_is_not_fooled_by_a_shared_string_prefix() {
        // The bug this whole module exists to prevent.
        assert!(!contains(&p("/work/project"), &p("/work/project-backup/secrets")));
        assert!(!contains(&p("/work/project"), &p("/work/projectX")));
    }

    #[test]
    fn contains_rejects_traversal_escapes() {
        assert!(!contains(&p("/work/project"), &p("/work/project/../other/file")));
        assert!(!contains(&p("/work/project"), &p("/work/project/../../etc/passwd")));
    }

    #[test]
    fn contains_allows_traversal_that_stays_inside() {
        assert!(contains(&p("/work/project"), &p("/work/project/src/../tests/a.rs")));
    }

    #[cfg(windows)]
    #[test]
    fn contains_is_case_insensitive_on_windows() {
        assert!(contains(&p(r"C:\Work\Project"), &p(r"c:\work\project\src\main.rs")));
    }

    #[cfg(windows)]
    #[test]
    fn verbatim_prefix_matches_plain_path() {
        assert!(contains(&p(r"C:\work\project"), &p(r"\\?\C:\work\project\a.txt")));
    }

    #[cfg(windows)]
    #[test]
    fn contains_rejects_a_different_drive() {
        assert!(!contains(&p(r"C:\work"), &p(r"D:\work\file.txt")));
    }

    #[test]
    fn ssh_and_cloud_credentials_are_sensitive() {
        assert!(is_sensitive(&p("/home/u/.ssh/id_rsa")));
        assert!(is_sensitive(&p("/home/u/.aws/credentials")));
        assert!(is_sensitive(&p("/home/u/certs/server.pem")));
        assert!(is_sensitive(&p(r"C:\Users\u\.ssh\id_ed25519")));
    }

    #[test]
    fn ordinary_project_files_are_not_sensitive() {
        assert!(!is_sensitive(&p("/work/project/src/main.rs")));
        assert!(!is_sensitive(&p("/work/project/README.md")));
        // A project .env is an ordinary file the agent may edit.
        assert!(!is_sensitive(&p("/work/project/.env")));
    }

    #[test]
    fn system_directories_are_recognised() {
        assert!(is_system_path(&p(r"C:\Windows\System32\drivers\etc\hosts")));
        assert!(is_system_path(&p("/etc/passwd")));
        assert!(is_system_path(&p("/usr/bin/ls")));
        assert!(!is_system_path(&p("/work/project/src/main.rs")));
    }
}
