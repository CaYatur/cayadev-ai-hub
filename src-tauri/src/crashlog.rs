//! Leaves evidence when something goes wrong.
//!
//! A panic used to end the process without a trace: the window vanished and
//! nothing on disk said why. The hook installed here appends every panic, with
//! its thread, location and backtrace, to `crash.log` in the app's log
//! directory. Errors the frontend catches (a render that threw, an unhandled
//! promise rejection) are written to the same file, so a crash report can
//! start from one place.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// The log file is rotated once it grows past this.
const MAX_LOG_BYTES: u64 = 1024 * 1024;
const FILE_NAME: &str = "crash.log";

static LOG_DIR: OnceLock<Mutex<PathBuf>> = OnceLock::new();

fn dir_cell() -> &'static Mutex<PathBuf> {
    LOG_DIR.get_or_init(|| Mutex::new(std::env::temp_dir().join("com.cayadev.aihub")))
}

/// Where crash reports go. Until the app knows its log directory, the
/// system temp directory stands in.
pub fn set_dir(dir: PathBuf) {
    // A poisoned lock still holds a usable path.
    let mut guard = dir_cell().lock().unwrap_or_else(|e| e.into_inner());
    *guard = dir;
}

pub fn log_path() -> PathBuf {
    dir_cell().lock().unwrap_or_else(|e| e.into_inner()).join(FILE_NAME)
}

/// Append one entry. Never panics: this runs inside the panic hook.
pub fn append(source: &str, body: &str) {
    let path = log_path();
    let _ = write_entry(&path, source, body);
}

fn write_entry(path: &Path, source: &str, body: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if std::fs::metadata(path).map(|m| m.len() > MAX_LOG_BYTES).unwrap_or(false) {
        let _ = std::fs::rename(path, path.with_extension("log.1"));
    }
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    let stamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f %:z");
    writeln!(
        file,
        "=== {stamp} [{source}] v{} ===\n{}\n",
        env!("CARGO_PKG_VERSION"),
        body.trim_end()
    )
}

/// Install the panic hook. The previous hook still runs, so a panic is also
/// printed to stderr as before.
pub fn install() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let thread = std::thread::current();
        let name = thread.name().unwrap_or("<unnamed>");
        let message = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "<non-string panic payload>".into());
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown>".into());
        let backtrace = std::backtrace::Backtrace::force_capture();
        append(
            "panic",
            &format!("thread '{name}' panicked at {location}:\n{message}\n\nbacktrace:\n{backtrace}"),
        );
        previous(info);
    }));
}

/// Short text for a caught panic payload.
pub fn payload_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_are_appended_with_source_and_body() {
        let dir = std::env::temp_dir().join(format!("crashlog-{}", uuid::Uuid::new_v4()));
        let path = dir.join(FILE_NAME);
        write_entry(&path, "panic", "boom — çğıöşü 🎵").unwrap();
        write_entry(&path, "frontend", "TypeError: x").unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("[panic]") && text.contains("boom — çğıöşü 🎵"), "{text}");
        assert!(text.contains("[frontend]") && text.contains("TypeError: x"), "{text}");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_large_log_is_rotated() {
        let dir = std::env::temp_dir().join(format!("crashlog-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(FILE_NAME);
        std::fs::write(&path, vec![b'x'; (MAX_LOG_BYTES + 1) as usize]).unwrap();
        write_entry(&path, "panic", "fresh").unwrap();
        assert!(std::fs::metadata(&path).unwrap().len() < 1000);
        assert!(dir.join("crash.log.1").exists());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn payload_text_reads_both_string_kinds() {
        let a: Box<dyn std::any::Any + Send> = Box::new("static");
        let b: Box<dyn std::any::Any + Send> = Box::new(String::from("owned"));
        assert_eq!(payload_text(a.as_ref()), "static");
        assert_eq!(payload_text(b.as_ref()), "owned");
    }
}
