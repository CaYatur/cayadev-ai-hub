//! Documents, images, and devices.
//!
//! ## Images work even when the agent's model cannot see
//!
//! `view_image` does not hand pixels to the running model. It sends the image
//! to whichever vision model routing picked and returns the *description* as
//! text. So a `qwen2.5-coder:7b` agent — no vision capability at all — can still
//! act on a screenshot, and the coding model never has to be swapped for a
//! weaker multimodal one just because an image appeared.
//!
//! ## Documents need Python
//!
//! PDF, Word, and Excel extraction runs through `scripts/extract.py`. Bundling
//! an interpreter would roughly double the release, so Python is a declared
//! prerequisite instead: the tools are advertised only when the interpreter and
//! its libraries are actually present, and the model is never offered a tool
//! that cannot run. Plain text and CSV are handled in Rust and always work.

use base64::Engine as _;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use crate::ollama::types::{ChatRequest, Message, Options};
use crate::ollama::OllamaClient;
use crate::ollama::types::Tool;
use crate::perms::Action;

const TOOL_TIMEOUT: Duration = Duration::from_secs(180);
const MAX_TEXT: usize = 60_000;

/// Extensions handled in Rust, with no interpreter needed.
const NATIVE_TEXT: &[&str] = &[
    "txt", "md", "csv", "tsv", "json", "yaml", "yml", "toml", "xml", "html", "log",
];

/// Extensions that need the Python sidecar.
const SIDECAR_DOCS: &[&str] = &["pdf", "docx", "doc", "xlsx", "xls", "pptx", "odt", "rtf"];

const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp", "bmp"];

const AUDIO_EXTS: &[&str] = &["mp3", "wav", "m4a", "ogg", "flac", "aac", "wma", "opus"];
const VIDEO_EXTS: &[&str] = &["mp4", "mkv", "mov", "avi", "webm", "wmv", "flv"];

pub struct MediaTools {
    workspace: PathBuf,
    client: OllamaClient,
    /// Model used to describe images. `None` disables `view_image`.
    vision_model: Option<String>,
    /// Interpreter for document extraction. `None` disables those tools.
    python: Option<String>,
    /// Interpreter with faster-whisper, for transcription.
    transcriber: Option<String>,
    /// Where the bundler put the Python sidecars.
    ///
    /// A packaged build does not lay resources out beside the executable, so
    /// guessing from `current_exe` finds nothing — and the tool would have
    /// advertised itself (Python is installed) and then failed with "no such
    /// file", which is exactly what the availability checks exist to prevent.
    resource_dir: Option<PathBuf>,
    /// ffmpeg and ffprobe, for video.
    ffmpeg_available: bool,
    adb_available: bool,
}

impl MediaTools {
    pub fn new(
        workspace: impl Into<PathBuf>,
        client: OllamaClient,
        vision_model: Option<String>,
        resource_dir: Option<PathBuf>,
    ) -> Self {
        Self {
            workspace: workspace.into(),
            client,
            vision_model,
            resource_dir,
            python: detect_python(),
            transcriber: detect_transcriber(),
            ffmpeg_available: which("ffmpeg") && which("ffprobe"),
            adb_available: which("adb"),
        }
    }

    fn arg(args: &Map<String, Value>, key: &str) -> Option<String> {
        args.get(key).and_then(Value::as_str).map(str::to_string)
    }

    fn required(args: &Map<String, Value>, key: &str) -> Result<String, String> {
        Self::arg(args, key).ok_or_else(|| format!("missing `{key}`"))
    }

    fn resolve(&self, raw: &str) -> PathBuf {
        let candidate = Path::new(raw);
        let joined = if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            self.workspace.join(candidate)
        };
        crate::perms::normalize(&joined)
    }

    /// Locate a sidecar script.
    ///
    /// The bundled copy first, since that is what a released build has; then
    /// the source tree, so `tauri dev` works; then a bare relative path, which
    /// fails with a readable message rather than silently doing nothing.
    fn sidecar(&self, name: &str) -> PathBuf {
        if let Some(dir) = &self.resource_dir {
            for candidate in resource_candidates(dir, name) {
                if candidate.exists() {
                    return candidate;
                }
            }
        }
        let local = self.workspace.join("scripts").join(name);
        if local.exists() {
            return local;
        }
        // Running from the repository root during development.
        let repo = PathBuf::from("scripts").join(name);
        if repo.exists() {
            return repo;
        }
        PathBuf::from("..").join("scripts").join(name)
    }

    /// Ask the vision model to describe an image, returning text.
    async fn describe_image(&self, path: &Path, question: &str) -> Result<String, String> {
        let model = self
            .vision_model
            .clone()
            .ok_or("no vision-capable model is installed")?;

        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);

        let request = ChatRequest {
            model,
            messages: vec![Message {
                role: "user".into(),
                content: question.to_string(),
                images: Some(vec![encoded]),
                ..Default::default()
            }],
            stream: false,
            tools: None,
            think: None,
            format: None,
            options: Some(Options { num_ctx: Some(8192), num_predict: Some(600), ..Default::default() }),
            keep_alive: Some("10m".into()),
        };

        let response = self.client.chat(request).await.map_err(|e| e.to_string())?;
        Ok(response
            .message
            .map(|m| m.content)
            .filter(|c| !c.trim().is_empty())
            .unwrap_or_else(|| "(the vision model returned nothing)".into()))
    }

    async fn run(&self, program: &str, argv: &[String]) -> Result<String, String> {
        let output = tokio::time::timeout(
            TOOL_TIMEOUT,
            crate::proc::command(program)
                .args(argv)
                .current_dir(&self.workspace)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output(),
        )
        .await
        .map_err(|_| format!("`{program}` timed out"))?
        .map_err(|e| format!("cannot run `{program}`: {e}"))?;

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if output.status.success() {
            Ok(if stdout.is_empty() { stderr } else { stdout })
        } else {
            Err(if stderr.is_empty() { stdout } else { stderr })
        }
    }

    /// Schemas for the tools this machine can actually run.
    pub fn schemas(&self) -> Vec<Tool> {
        let mut tools = vec![Tool::function(
            "read_document",
            "Extract text from a document: PDF, Word, Excel, CSV, or plain text.",
            json!({
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path to the document"}},
                "required": ["path"]
            }),
        )];

        if self.vision_model.is_some() {
            tools.push(Tool::function(
                "view_image",
                "Look at an image and answer a question about it.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the image"},
                        "question": {"type": "string", "description": "What to look for"}
                    },
                    "required": ["path"]
                }),
            ));
            tools.push(Tool::function(
                "screenshot",
                "Capture the screen and answer a question about what is on it.",
                json!({
                    "type": "object",
                    "properties": {"question": {"type": "string", "description": "What to look for"}}
                }),
            ));
        }

        if self.transcriber.is_some() {
            tools.push(Tool::function(
                "transcribe",
                "Transcribe speech from an audio or video file.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the media file"},
                        "language": {"type": "string", "description": "Language code, e.g. tr. Detected if omitted"}
                    },
                    "required": ["path"]
                }),
            ));
        }

        if self.ffmpeg_available && self.vision_model.is_some() {
            tools.push(Tool::function(
                "watch_video",
                "Look at a video and answer a question about what happens in it.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the video"},
                        "question": {"type": "string", "description": "What to look for"}
                    },
                    "required": ["path"]
                }),
            ));
        }

        if self.adb_available {
            tools.push(Tool::function(
                "adb",
                "Inspect a connected Android device: devices, screen, ui, or logcat.",
                json!({
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "description": "devices, screen, ui, or logcat"},
                        "question": {"type": "string", "description": "For screen: what to look for"}
                    },
                    "required": ["action"]
                }),
            ));
        }
        tools
    }

    pub fn handles(&self, name: &str) -> bool {
        self.schemas().iter().any(|t| t.function.name == name)
    }

    pub fn action_for(&self, name: &str, args: &Map<String, Value>) -> Option<Action> {
        match name {
            "read_document" | "view_image" | "transcribe" | "watch_video" => Some(Action::ReadFile {
                path: self.resolve(&Self::arg(args, "path")?),
            }),
            "screenshot" => Some(Action::CaptureScreen { what: "this screen".into() }),
            "adb" => match Self::arg(args, "action").unwrap_or_default().as_str() {
                // Pulling a phone's screen is the same exposure as capturing
                // this one, so it is confirmed the same way.
                "screen" => Some(Action::CaptureScreen { what: "the device screen".into() }),
                other => Some(Action::RunCommand {
                    program: "adb".into(),
                    args: vec![other.to_string()],
                    cwd: Some(self.workspace.clone()),
                }),
            },
            _ => None,
        }
    }

    pub async fn execute(&self, name: &str, args: &Map<String, Value>) -> Result<String, String> {
        match name {
            "read_document" => {
                let raw = Self::required(args, "path")?;
                let path = self.resolve(&raw);
                let ext = path
                    .extension()
                    .map(|e| e.to_string_lossy().to_lowercase())
                    .unwrap_or_default();

                if IMAGE_EXTS.contains(&ext.as_str()) {
                    return Err(format!("{raw} is an image. Use view_image instead."));
                }
                if AUDIO_EXTS.contains(&ext.as_str()) {
                    return Err(format!("{raw} is audio. Use transcribe instead."));
                }
                if VIDEO_EXTS.contains(&ext.as_str()) {
                    return Err(format!(
                        "{raw} is a video. Use watch_video, or transcribe for its speech."
                    ));
                }
                if NATIVE_TEXT.contains(&ext.as_str()) || ext.is_empty() {
                    let text = std::fs::read_to_string(&path)
                        .map_err(|e| format!("cannot read {raw}: {e}"))?;
                    return Ok(text.chars().take(MAX_TEXT).collect());
                }
                if !SIDECAR_DOCS.contains(&ext.as_str()) {
                    return Err(format!("no extractor for .{ext} files"));
                }

                let python = self.python.clone().ok_or(
                    "extracting this format needs Python with pymupdf, python-docx, and openpyxl. \
                     See the README.",
                )?;
                let script = self.sidecar("extract.py");

                let out = self
                    .run(&python, &[script.to_string_lossy().to_string(), path.to_string_lossy().to_string()])
                    .await?;
                Ok(out.chars().take(MAX_TEXT).collect())
            }

            "transcribe" => {
                let raw = Self::required(args, "path")?;
                let path = self.resolve(&raw);
                let python = self
                    .transcriber
                    .clone()
                    .ok_or("transcription needs Python with faster-whisper. See the README.")?;

                let mut argv = vec![
                    self.sidecar("media.py").to_string_lossy().to_string(),
                    "transcribe".into(),
                    path.to_string_lossy().to_string(),
                ];
                if let Some(language) = Self::arg(args, "language") {
                    argv.push(language);
                }
                let out = self.run(&python, &argv).await?;
                Ok(out.chars().take(MAX_TEXT).collect())
            }

            "watch_video" => {
                let raw = Self::required(args, "path")?;
                let path = self.resolve(&raw);
                let question = Self::arg(args, "question")
                    .unwrap_or_else(|| "Describe what is happening.".into());
                let python = self
                    .python
                    .clone()
                    .or_else(|| self.transcriber.clone())
                    .ok_or("video needs Python. See the README.")?;

                let out = self
                    .run(
                        &python,
                        &[
                            self.sidecar("media.py").to_string_lossy().to_string(),
                            "video".into(),
                            path.to_string_lossy().to_string(),
                        ],
                    )
                    .await?;

                let parsed: serde_json::Value =
                    serde_json::from_str(out.lines().last().unwrap_or_default())
                        .map_err(|e| format!("could not read frame data: {e}"))?;

                let summary = parsed
                    .get("summary")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();

                // Each frame is described separately and the answers stitched
                // together. Sending one frame at a time keeps this working with
                // any vision model rather than needing one that accepts video.
                let mut described = vec![summary];
                if let Some(frames) = parsed.get("frames").and_then(Value::as_array) {
                    for frame in frames {
                        let Some(frame_path) = frame.get("path").and_then(Value::as_str) else {
                            continue;
                        };
                        let at = frame.get("at").and_then(Value::as_f64).unwrap_or(0.0);
                        match self.describe_image(Path::new(frame_path), &question).await {
                            Ok(text) => described.push(format!("[{at:.1}s] {text}")),
                            Err(e) => described.push(format!("[{at:.1}s] could not read frame: {e}")),
                        }
                        let _ = std::fs::remove_file(frame_path);
                    }
                }
                Ok(described.join("\n\n").chars().take(MAX_TEXT).collect())
            }

            "view_image" => {
                let raw = Self::required(args, "path")?;
                let path = self.resolve(&raw);
                let question = Self::arg(args, "question")
                    .unwrap_or_else(|| "Describe this image in detail.".into());
                self.describe_image(&path, &question).await
            }

            "screenshot" => {
                let question = Self::arg(args, "question")
                    .unwrap_or_else(|| "Describe what is on this screen.".into());
                let shot = capture_screen().await?;
                let result = self.describe_image(&shot, &question).await;
                let _ = std::fs::remove_file(&shot);
                result
            }

            "adb" => {
                let action = Self::required(args, "action")?;
                match action.as_str() {
                    "devices" => self.run("adb", &["devices".into(), "-l".into()]).await,
                    "ui" => {
                        // Dump the view hierarchy and read it back as text; far
                        // cheaper and more precise than describing a screenshot.
                        self.run(
                            "adb",
                            &["shell".into(), "uiautomator".into(), "dump".into(), "/sdcard/ui.xml".into()],
                        )
                        .await?;
                        let xml = self
                            .run("adb", &["shell".into(), "cat".into(), "/sdcard/ui.xml".into()])
                            .await?;
                        Ok(xml.chars().take(MAX_TEXT).collect())
                    }
                    "logcat" => {
                        self.run("adb", &["logcat".into(), "-d".into(), "-t".into(), "200".into()])
                            .await
                    }
                    "screen" => {
                        let local = std::env::temp_dir().join("cayahub-adb.png");
                        self.run("adb", &["shell".into(), "screencap".into(), "-p".into(), "/sdcard/s.png".into()])
                            .await?;
                        self.run(
                            "adb",
                            &["pull".into(), "/sdcard/s.png".into(), local.to_string_lossy().to_string()],
                        )
                        .await?;
                        let question = Self::arg(args, "question")
                            .unwrap_or_else(|| "Describe what is on this phone screen.".into());
                        let result = self.describe_image(&local, &question).await;
                        let _ = std::fs::remove_file(&local);
                        result
                    }
                    other => Err(format!("unknown adb action `{other}`. Use devices, screen, ui, or logcat.")),
                }
            }

            other => Err(format!("unknown media tool `{other}`")),
        }
    }
}

/// Places a bundled sidecar can end up, in order of likelihood.
///
/// Tauri rewrites a resource declared as `../scripts/extract.py` into
/// `_up_/scripts/extract.py` beneath the resource directory — the `..` becomes
/// a literal `_up_` segment. Missing that is silent: the tool advertises
/// itself because Python is installed, then fails with "no such file".
fn resource_candidates(dir: &Path, name: &str) -> Vec<PathBuf> {
    vec![
        dir.join("_up_").join("scripts").join(name),
        dir.join("scripts").join(name),
        dir.join(name),
    ]
}

/// Capture the primary display to a temporary PNG.
async fn capture_screen() -> Result<PathBuf, String> {
    let target = std::env::temp_dir().join(format!("cayahub-shot-{}.png", uuid::Uuid::new_v4()));

    #[cfg(windows)]
    {
        // PowerShell with System.Drawing avoids shipping a capture binary.
        let script = format!(
            "Add-Type -AssemblyName System.Windows.Forms,System.Drawing; \
             $b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; \
             $bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height; \
             $g = [System.Drawing.Graphics]::FromImage($bmp); \
             $g.CopyFromScreen($b.X, $b.Y, 0, 0, $bmp.Size); \
             $bmp.Save('{}'); $g.Dispose(); $bmp.Dispose()",
            target.to_string_lossy().replace('\\', "\\\\")
        );
        let status = crate::proc::command("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", &script])
            .output()
            .await
            .map_err(|e| format!("screen capture failed: {e}"))?;
        if !status.status.success() {
            return Err(String::from_utf8_lossy(&status.stderr).to_string());
        }
    }

    #[cfg(target_os = "macos")]
    {
        crate::proc::command("screencapture")
            .args(["-x", &target.to_string_lossy()])
            .output()
            .await
            .map_err(|e| format!("screen capture failed: {e}"))?;
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        crate::proc::command("import")
            .args(["-window", "root", &target.to_string_lossy()])
            .output()
            .await
            .map_err(|e| format!("screen capture needs ImageMagick's `import`: {e}"))?;
    }

    if target.exists() {
        Ok(target)
    } else {
        Err("screen capture produced no file".into())
    }
}

/// Is `program` on PATH?
fn which(program: &str) -> bool {
    let probe = if cfg!(windows) { "where" } else { "which" };
    crate::proc::sync_command(probe)
        .arg(program)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Find a Python that can transcribe.
///
/// Checked separately from the document interpreter because the two need different
/// packages, and a machine may well have one and not the other.
fn detect_transcriber() -> Option<String> {
    for candidate in ["python", "python3"] {
        let ok = crate::proc::sync_command(candidate)
            .args(["-c", "import faster_whisper"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ok {
            return Some(candidate.to_string());
        }
    }
    None
}

/// Find a Python that has the extraction libraries installed.
///
/// Checking for the libraries rather than just the interpreter is the point: a
/// bare Python would let the tool be advertised and then fail at the worst
/// moment, having already cost the model a step.
fn detect_python() -> Option<String> {
    for candidate in ["python", "python3"] {
        let ok = crate::proc::sync_command(candidate)
            .args(["-c", "import fitz, docx, openpyxl"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ok {
            return Some(candidate.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ollama::{Timeouts, DEFAULT_HOST};

    fn tools(vision: Option<&str>) -> MediaTools {
        MediaTools::new(
            std::env::temp_dir(),
            OllamaClient::new(DEFAULT_HOST, Timeouts::default()),
            vision.map(str::to_string),
            None,
        )
    }

    fn args(v: Value) -> Map<String, Value> {
        serde_json::from_value(v).expect("object")
    }

    #[test]
    fn a_bundled_sidecar_is_found_where_tauri_puts_it() {
        // The bundler turns `../scripts/extract.py` into `_up_/scripts/...`,
        // which the first version of this lookup did not check — so the
        // packaged build would have advertised the tool and then failed.
        let dir = std::env::temp_dir().join(format!("cayahub-res-{}", uuid::Uuid::new_v4()));
        let bundled = dir.join("_up_").join("scripts");
        std::fs::create_dir_all(&bundled).unwrap();
        std::fs::write(bundled.join("extract.py"), "# stub").unwrap();

        let found = resource_candidates(&dir, "extract.py")
            .into_iter()
            .find(|p| p.exists())
            .expect("bundled sidecar should be found");
        assert_eq!(found, bundled.join("extract.py"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn image_tools_are_hidden_without_a_vision_model() {
        let names: Vec<String> = tools(None).schemas().into_iter().map(|t| t.function.name).collect();
        assert!(!names.contains(&"view_image".to_string()));
        assert!(!names.contains(&"screenshot".to_string()));
        // Documents do not need vision, so that one stays.
        assert!(names.contains(&"read_document".to_string()));
    }

    #[test]
    fn image_tools_appear_when_a_vision_model_exists() {
        let names: Vec<String> = tools(Some("qwen3-vl:8b"))
            .schemas()
            .into_iter()
            .map(|t| t.function.name)
            .collect();
        assert!(names.contains(&"view_image".to_string()));
        assert!(names.contains(&"screenshot".to_string()));
    }

    #[test]
    fn every_advertised_tool_maps_to_an_action() {
        // A tool the engine cannot classify would run ungated.
        let t = tools(Some("qwen3-vl:8b"));
        for tool in t.schemas() {
            let name = &tool.function.name;
            let probe = args(json!({"path": "a.png", "action": "devices"}));
            assert!(
                t.action_for(name, &probe).is_some(),
                "{name} has no Action and would bypass the permission engine"
            );
        }
    }

    #[tokio::test]
    async fn audio_and_video_are_redirected_from_read_document() {
        let t = tools(Some("qwen3-vl:8b"));
        let audio = t
            .execute("read_document", &args(json!({"path": "talk.mp3"})))
            .await
            .unwrap_err();
        assert!(audio.contains("transcribe"), "got: {audio}");

        let video = t
            .execute("read_document", &args(json!({"path": "clip.mp4"})))
            .await
            .unwrap_err();
        assert!(video.contains("watch_video"), "got: {video}");
    }

    #[test]
    fn capturing_a_screen_is_a_distinct_decision() {
        // Both screen captures map to `CaptureScreen` rather than an ordinary
        // file read, so the mode can decide about them as a category. Before
        // this they evaluated against a fake `<screen>` path and slipped
        // through as unremarkable reads.
        use crate::perms::{Action, Decision, PermissionEngine, PermissionMode};
        let t = tools(Some("qwen3-vl:8b"));
        let cautious = PermissionEngine::new(std::env::temp_dir(), PermissionMode::ProjectAuto);

        for (name, payload) in [("screenshot", json!({})), ("adb", json!({"action": "screen"}))] {
            let action = t.action_for(name, &args(payload)).unwrap();
            assert!(
                matches!(action, Action::CaptureScreen { .. }),
                "{name} should be a capture, not a file read"
            );
            assert_eq!(cautious.evaluate(&action).decision, Decision::Ask, "{name}");
        }
    }

    #[tokio::test]
    async fn plain_text_is_extracted_without_python() {
        let dir = std::env::temp_dir().join(format!("cayahub-doc-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("notes.md"), "# Title\nbody text").unwrap();

        let t = MediaTools::new(&dir, OllamaClient::new(DEFAULT_HOST, Timeouts::default()), None, None);
        let out = t.execute("read_document", &args(json!({"path": "notes.md"}))).await.unwrap();
        assert!(out.contains("body text"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn an_image_passed_to_read_document_is_redirected() {
        let t = tools(Some("qwen3-vl:8b"));
        let err = t
            .execute("read_document", &args(json!({"path": "photo.png"})))
            .await
            .unwrap_err();
        assert!(err.contains("view_image"), "got: {err}");
    }

    #[tokio::test]
    async fn an_unsupported_format_says_so() {
        let t = tools(None);
        let err = t
            .execute("read_document", &args(json!({"path": "archive.zip"})))
            .await
            .unwrap_err();
        assert!(err.contains("no extractor"), "got: {err}");
    }

    #[tokio::test]
    async fn view_image_without_a_vision_model_is_a_clear_error() {
        let t = tools(None);
        let err = t
            .execute("view_image", &args(json!({"path": "a.png"})))
            .await
            .unwrap_err();
        assert!(err.contains("vision"), "got: {err}");
    }

    #[tokio::test]
    async fn an_unknown_adb_action_lists_the_valid_ones() {
        let t = tools(None);
        let err = t.execute("adb", &args(json!({"action": "fly"}))).await.unwrap_err();
        assert!(err.contains("devices"), "got: {err}");
    }
}
