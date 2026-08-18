//! Undo for agent file changes.
//!
//! Before a tool writes, deletes, or edits a file, its previous contents are
//! stored and journalled. Restoring puts them back.
//!
//! ## Why it stays small
//!
//! Content is addressed by SHA-256 and written once. An agent that reads and
//! rewrites the same file across twenty steps produces at most twenty *journal
//! lines* but only as many objects as there were distinct contents, and a file
//! restored to an earlier state costs nothing new. A file that never changes is
//! never stored twice.
//!
//! ## Why the store is refused to the agent
//!
//! The store lives under the workspace, which is exactly where `ProjectAuto`
//! auto-approves writes. Without a guard the agent could overwrite its own undo
//! history — so [`crate::perms::path::is_checkpoint_store`] refuses the path in
//! every mode, `Unrestricted` included.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Directory name inside the workspace.
pub const STORE_DIR: &str = ".cayahub";

/// Files larger than this are journalled as "not captured" rather than copied.
/// A 200 MB asset is not what undo is for, and copying it would make the store
/// bigger than the project.
const MAX_CAPTURE_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeKind {
    Created,
    Modified,
    Deleted,
}

/// One file's prior state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSnapshot {
    /// Path relative to the workspace, so a moved project still restores.
    pub path: String,
    pub kind: ChangeKind,
    /// Content hash of the *previous* contents. `None` when the file did not
    /// exist, or was too large to capture.
    pub before: Option<String>,
    /// Set when the file existed but was skipped for size.
    pub skipped_bytes: Option<u64>,
}

/// A restorable point, one per tool call that changed files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub id: String,
    pub timestamp: String,
    /// Which tool call produced it, for the UI.
    pub label: String,
    pub files: Vec<FileSnapshot>,
}

pub struct CheckpointStore {
    workspace: PathBuf,
}

impl CheckpointStore {
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        Self { workspace: workspace.into() }
    }

    pub fn store_root(&self) -> PathBuf {
        self.workspace.join(STORE_DIR)
    }

    fn objects_dir(&self) -> PathBuf {
        self.store_root().join("objects")
    }

    fn journal_path(&self) -> PathBuf {
        self.store_root().join("journal.jsonl")
    }

    /// Write content to the object store, returning its hash.
    ///
    /// A hash that already exists is not rewritten — that is the deduplication.
    fn put_object(&self, bytes: &[u8]) -> std::io::Result<String> {
        let hash = format!("{:x}", Sha256::digest(bytes));
        let dir = self.objects_dir().join(&hash[..2]);
        let path = dir.join(&hash);

        if path.exists() {
            return Ok(hash);
        }
        std::fs::create_dir_all(&dir)?;
        std::fs::write(&path, bytes)?;
        Ok(hash)
    }

    fn read_object(&self, hash: &str) -> std::io::Result<Vec<u8>> {
        std::fs::read(self.objects_dir().join(&hash[..2]).join(hash))
    }

    /// Capture a file's current contents before it is changed.
    pub fn capture(&self, absolute: &Path, kind: ChangeKind) -> std::io::Result<FileSnapshot> {
        let relative = absolute
            .strip_prefix(&self.workspace)
            .unwrap_or(absolute)
            .to_string_lossy()
            .replace('\\', "/");

        let metadata = std::fs::metadata(absolute).ok();
        let Some(metadata) = metadata else {
            // Nothing there yet: undo means deleting whatever gets created.
            return Ok(FileSnapshot {
                path: relative,
                kind: ChangeKind::Created,
                before: None,
                skipped_bytes: None,
            });
        };

        if metadata.len() > MAX_CAPTURE_BYTES {
            return Ok(FileSnapshot {
                path: relative,
                kind,
                before: None,
                skipped_bytes: Some(metadata.len()),
            });
        }

        let bytes = std::fs::read(absolute)?;
        let hash = self.put_object(&bytes)?;
        Ok(FileSnapshot { path: relative, kind, before: Some(hash), skipped_bytes: None })
    }

    /// Journal a checkpoint. Returns its id.
    pub fn record(&self, label: &str, files: Vec<FileSnapshot>) -> std::io::Result<String> {
        let checkpoint = Checkpoint {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            label: label.to_string(),
            files,
        };

        std::fs::create_dir_all(self.store_root())?;
        self.ensure_ignored()?;

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.journal_path())?;
        writeln!(file, "{}", serde_json::to_string(&checkpoint)?)?;
        Ok(checkpoint.id)
    }

    /// Keep the store out of the user's commits without them having to ask.
    fn ensure_ignored(&self) -> std::io::Result<()> {
        let marker = self.store_root().join(".gitignore");
        if !marker.exists() {
            std::fs::write(&marker, "*\n")?;
        }
        Ok(())
    }

    pub fn list(&self) -> std::io::Result<Vec<Checkpoint>> {
        let path = self.journal_path();
        if !path.exists() {
            return Ok(Vec::new());
        }
        let text = std::fs::read_to_string(path)?;
        // A truncated final line (power loss mid-write) should cost that entry,
        // not the whole history.
        Ok(text
            .lines()
            .filter_map(|line| serde_json::from_str::<Checkpoint>(line).ok())
            .collect())
    }

    /// Put every file in a checkpoint back to its captured state.
    ///
    /// Returns the paths restored. Files that were skipped for size are
    /// reported as errors rather than silently left alone.
    pub fn restore(&self, id: &str) -> Result<Vec<String>, String> {
        let checkpoints = self.list().map_err(|e| e.to_string())?;
        let checkpoint = checkpoints
            .iter()
            .find(|c| c.id == id)
            .ok_or_else(|| format!("no checkpoint {id}"))?;

        let mut restored = Vec::new();
        let mut problems = Vec::new();

        for file in &checkpoint.files {
            let target = self.workspace.join(&file.path);

            if let Some(size) = file.skipped_bytes {
                problems.push(format!("{} was too large to capture ({size} bytes)", file.path));
                continue;
            }

            let outcome = match &file.before {
                // It did not exist before, so undoing means removing it.
                None => match std::fs::remove_file(&target) {
                    Ok(()) => Ok(()),
                    // Already gone is the desired state.
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                    Err(e) => Err(e.to_string()),
                },
                Some(hash) => self
                    .read_object(hash)
                    .map_err(|e| e.to_string())
                    .and_then(|bytes| {
                        if let Some(parent) = target.parent() {
                            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                        }
                        std::fs::write(&target, bytes).map_err(|e| e.to_string())
                    }),
            };

            match outcome {
                Ok(()) => restored.push(file.path.clone()),
                Err(e) => problems.push(format!("{}: {e}", file.path)),
            }
        }

        if problems.is_empty() {
            Ok(restored)
        } else {
            Err(problems.join("; "))
        }
    }

    /// Bytes held by the object store, so the UI can show what undo costs.
    pub fn size_bytes(&self) -> u64 {
        walkdir::WalkDir::new(self.objects_dir())
            .into_iter()
            .filter_map(Result::ok)
            .filter_map(|e| e.metadata().ok())
            .filter(|m| m.is_file())
            .map(|m| m.len())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cayahub-ckpt-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn restores_previous_contents() {
        let dir = scratch();
        let store = CheckpointStore::new(&dir);
        let file = dir.join("a.txt");
        std::fs::write(&file, "original").unwrap();

        let snap = store.capture(&file, ChangeKind::Modified).unwrap();
        let id = store.record("write_file a.txt", vec![snap]).unwrap();

        std::fs::write(&file, "clobbered").unwrap();
        store.restore(&id).unwrap();

        assert_eq!(std::fs::read_to_string(&file).unwrap(), "original");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn undoing_a_creation_removes_the_file() {
        let dir = scratch();
        let store = CheckpointStore::new(&dir);
        let file = dir.join("new.txt");

        // Captured before it exists.
        let snap = store.capture(&file, ChangeKind::Created).unwrap();
        assert_eq!(snap.kind, ChangeKind::Created);
        let id = store.record("write_file new.txt", vec![snap]).unwrap();

        std::fs::write(&file, "created by the agent").unwrap();
        store.restore(&id).unwrap();

        assert!(!file.exists(), "undo of a creation should delete the file");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn identical_contents_are_stored_once() {
        let dir = scratch();
        let store = CheckpointStore::new(&dir);
        let a = dir.join("a.txt");
        let b = dir.join("b.txt");
        std::fs::write(&a, "same bytes").unwrap();
        std::fs::write(&b, "same bytes").unwrap();

        let sa = store.capture(&a, ChangeKind::Modified).unwrap();
        let sb = store.capture(&b, ChangeKind::Modified).unwrap();
        assert_eq!(sa.before, sb.before, "same content must hash the same");

        let objects: Vec<_> = walkdir::WalkDir::new(store.objects_dir())
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
            .collect();
        assert_eq!(objects.len(), 1, "deduplication failed");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn capturing_the_same_file_repeatedly_does_not_grow_the_store() {
        let dir = scratch();
        let store = CheckpointStore::new(&dir);
        let file = dir.join("a.txt");
        std::fs::write(&file, "steady").unwrap();

        for _ in 0..20 {
            let snap = store.capture(&file, ChangeKind::Modified).unwrap();
            store.record("edit_file a.txt", vec![snap]).unwrap();
        }
        assert_eq!(store.size_bytes(), "steady".len() as u64);
        assert_eq!(store.list().unwrap().len(), 20, "every step should be journalled");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn oversized_files_are_noted_rather_than_copied() {
        let dir = scratch();
        let store = CheckpointStore::new(&dir);
        let file = dir.join("big.bin");
        std::fs::write(&file, vec![0u8; (MAX_CAPTURE_BYTES + 1) as usize]).unwrap();

        let snap = store.capture(&file, ChangeKind::Modified).unwrap();
        assert!(snap.before.is_none());
        assert!(snap.skipped_bytes.is_some());
        assert_eq!(store.size_bytes(), 0, "an oversized file must not be copied");

        // Restoring must say so rather than silently doing nothing.
        let id = store.record("write_file big.bin", vec![snap]).unwrap();
        assert!(store.restore(&id).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_store_ignores_itself_in_git() {
        let dir = scratch();
        let store = CheckpointStore::new(&dir);
        store.record("noop", vec![]).unwrap();
        assert_eq!(
            std::fs::read_to_string(store.store_root().join(".gitignore")).unwrap(),
            "*\n"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_corrupt_journal_line_costs_only_that_entry() {
        let dir = scratch();
        let store = CheckpointStore::new(&dir);
        store.record("first", vec![]).unwrap();
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(store.journal_path())
            .unwrap();
        writeln!(f, "{{ truncated").unwrap();
        drop(f);
        store.record("third", vec![]).unwrap();

        let all = store.list().unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[1].label, "third");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn restoring_an_unknown_id_errors() {
        let dir = scratch();
        let store = CheckpointStore::new(&dir);
        assert!(store.restore("nope").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
