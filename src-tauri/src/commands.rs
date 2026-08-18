//! Tauri command layer.
//!
//! The frontend talks to Ollama exclusively through these commands rather than
//! calling the daemon directly. Two reasons, and the second is the load-bearing
//! one:
//!
//! * **No cross-origin problem.** A webview fetch to `localhost:11434` is
//!   subject to `OLLAMA_ORIGINS`, and the packaged app's origin
//!   (`tauri://localhost`) is not allowlisted by default — so a direct fetch
//!   works in `tauri dev` and silently fails in the shipped build. Requests
//!   from Rust have no origin and no such trap.
//! * **One source of truth.** Placement estimation depends on GGUF metadata and
//!   the hybrid-attention KV correction. Duplicating that in TypeScript means
//!   two implementations that drift.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, State};
use tokio::sync::{Mutex, RwLock};

use crate::agent::runner::{AgentEvent, AgentRunner, RunConfig, StopReason, Turn};
use crate::ollama::TaskClass;
use crate::perms::{Outcome, PermissionEngine, PermissionMode};
use crate::session::{self, PendingApprovals};
use crate::tools::WorkspaceTools;

use crate::ollama::{
    load_profiles, ModelProfile, OllamaClient, RoutingPolicy, Timeouts, VramBudget, DEFAULT_HOST,
};

pub struct AppState {
    pub client: OllamaClient,
    pub budget: RwLock<VramBudget>,
    pub profiles: RwLock<Vec<ModelProfile>>,
    /// Root the agent is confined to.
    pub workspace: RwLock<Option<PathBuf>>,
    pub mode: RwLock<PermissionMode>,
    /// Whether the classifier may settle escalated calls.
    pub assisted_approval: RwLock<bool>,
    pub pending: PendingApprovals,
    /// Files dragged or pasted into the chat.
    ///
    /// They usually live outside the workspace, which the permission engine
    /// would otherwise escalate on every read. Registering each file as an
    /// extra root lets the agent open exactly what the user handed it — not
    /// the rest of the folder it came from.
    pub attachments: RwLock<Vec<PathBuf>>,
    /// Set by `cancel_run`; the in-flight runner and its HTTP stream both
    /// watch it, so Stop takes effect immediately rather than at the end of
    /// the current step.
    pub cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl AppState {
    pub fn new() -> Self {
        // Read the real budget from the driver. The default is only a
        // fallback for hosts with no supported GPU tool, and it is a guess
        // pessimistic enough to have caused a wrong routing decision.
        let budget = crate::ollama::probe_vram().unwrap_or_default();
        Self {
            client: OllamaClient::new(DEFAULT_HOST, Timeouts::default()),
            budget: RwLock::new(budget),
            profiles: RwLock::new(Vec::new()),
            workspace: RwLock::new(None),
            mode: RwLock::new(PermissionMode::default()),
            assisted_approval: RwLock::new(true),
            pending: Arc::new(Mutex::new(HashMap::new())),
            attachments: RwLock::new(Vec::new()),
            cancel: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

/// Daemon version, or an error the UI can show verbatim.
#[tauri::command]
pub async fn ollama_version(state: State<'_, AppState>) -> Result<String, String> {
    state.client.version().await.map_err(|e| e.to_string())
}

/// Every installed model with capabilities, true context limit, and placement.
#[tauri::command]
pub async fn list_models(state: State<'_, AppState>) -> Result<Vec<ModelProfile>, String> {
    let budget = *state.budget.read().await;
    let profiles = load_profiles(&state.client, budget)
        .await
        .map_err(|e| e.to_string())?;
    *state.profiles.write().await = profiles.clone();
    Ok(profiles)
}

/// Routing policy derived from what is installed.
#[tauri::command]
pub async fn routing_policy(state: State<'_, AppState>) -> Result<RoutingPolicy, String> {
    let profiles = state.profiles.read().await;
    if profiles.is_empty() {
        return Err("No models loaded yet. Call list_models first.".into());
    }
    Ok(RoutingPolicy::derive(&profiles))
}

/// Override the VRAM budget used for placement estimates.
#[tauri::command]
pub async fn set_vram_budget(
    state: State<'_, AppState>,
    total_bytes: u64,
    reserved_bytes: u64,
) -> Result<(), String> {
    *state.budget.write().await = VramBudget { total_bytes, reserved_bytes };
    Ok(())
}

/// Re-read GPU memory from the driver and adopt it as the budget.
///
/// Worth calling when the user closes a game or a browser: available VRAM
/// changes, and with it which models are worth routing to.
#[tauri::command]
pub async fn refresh_vram_budget(state: State<'_, AppState>) -> Result<VramBudget, String> {
    let probed = crate::ollama::probe_vram()
        .ok_or("no supported GPU found; set the budget manually")?;
    *state.budget.write().await = probed;
    Ok(probed)
}

// ---------------------------------------------------------------------------
// Session control
// ---------------------------------------------------------------------------

/// Point the agent at a directory. Everything it does is confined to this root.
#[tauri::command]
pub async fn set_workspace(state: State<'_, AppState>, path: String) -> Result<String, String> {
    let path = PathBuf::from(path);
    if !path.is_dir() {
        return Err(format!("{} is not a directory", path.display()));
    }
    let canonical = path.canonicalize().unwrap_or(path);
    let normalized = crate::perms::normalize(&canonical);
    *state.workspace.write().await = Some(normalized.clone());
    Ok(normalized.display().to_string())
}

#[tauri::command]
pub async fn set_permission_mode(
    state: State<'_, AppState>,
    mode: PermissionMode,
) -> Result<(), String> {
    *state.mode.write().await = mode;
    Ok(())
}

#[tauri::command]
pub async fn set_assisted_approval(
    state: State<'_, AppState>,
    enabled: bool,
) -> Result<(), String> {
    *state.assisted_approval.write().await = enabled;
    Ok(())
}

/// Answer a call the agent asked about.
#[tauri::command]
pub async fn resolve_approval(
    state: State<'_, AppState>,
    id: String,
    approved: bool,
) -> Result<bool, String> {
    let outcome = if approved { Outcome::UserApproved } else { Outcome::UserRejected };
    Ok(session::resolve(&state.pending, &id, outcome).await)
}

/// Stop the current run at once.
///
/// Sets the flag the runner and the HTTP stream both poll, then refuses every
/// approval still waiting so nothing is left blocked on an answer.
#[tauri::command]
pub async fn cancel_run(state: State<'_, AppState>) -> Result<(), String> {
    state.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    session::reject_all(&state.pending).await;
    Ok(())
}

/// Run the agent on one instruction, streaming events to the frontend.
///
/// Events arrive on the `agent://event` channel; approval questions on
/// `agent://approval`.
#[tauri::command]
pub async fn run_agent(
    app: AppHandle,
    state: State<'_, AppState>,
    prompt: String,
    model: Option<String>,
    num_ctx: Option<u32>,
    effort: Option<Effort>,
    history: Option<Vec<Turn>>,
) -> Result<String, String> {
    let workspace = state
        .workspace
        .read()
        .await
        .clone()
        .ok_or("No workspace selected. Choose a folder first.")?;

    let profiles = state.profiles.read().await.clone();
    if profiles.is_empty() {
        return Err("No models loaded. Refresh the model list first.".into());
    }
    let policy = RoutingPolicy::derive(&profiles);

    // An explicit choice wins; otherwise routing picks the coding model.
    let chosen = model
        .or_else(|| policy.get(TaskClass::Coding).map(str::to_string))
        .ok_or("No model available for coding tasks.")?;

    let profile = profiles.iter().find(|p| p.name == chosen);
    let params = profile.and_then(|p| p.params_billions());

    // Fall back to what the registry computed for this model rather than a
    // constant: `recommended_context` is the largest window that still leaves
    // the model resident in VRAM.
    let context = num_ctx
        .or_else(|| profile.and_then(|p| p.recommended_context).map(|c| c as u32))
        .unwrap_or(8192);

    let effort = effort.unwrap_or_default();
    // Thinking is only meaningful where the model advertises it; forcing it on
    // a model without the capability wastes tokens and confuses the parser.
    let think = effort.wants_thinking()
        && profile.map(|p| p.capabilities.thinking).unwrap_or(false);

    let classifier = policy.get(TaskClass::Classify).map(str::to_string);
    let assisted = *state.assisted_approval.read().await;
    let mode = *state.mode.read().await;

    let mut tools = WorkspaceTools::new(&workspace)
        .with_media(
            state.client.clone(),
            policy.get(TaskClass::Vision).map(str::to_string),
            tauri::Manager::path(&app).resource_dir().ok(),
        );
    let mut perms = PermissionEngine::new(&workspace, mode);
    for file in state.attachments.read().await.iter() {
        // The file itself, not its directory — dropping one file from Downloads
        // must not open the rest of Downloads.
        perms.allow_root(file);
    }

    // A new run starts uncancelled, whatever the last one ended as.
    state.cancel.store(false, std::sync::atomic::Ordering::Relaxed);

    let runner = AgentRunner::new(
        state.client.clone(),
        perms,
        RunConfig {
            model: chosen.clone(),
            num_ctx: Some(context),
            params_billions: params,
            think,
            max_steps: effort.max_steps(),
            // What was already said in this conversation. The frontend owns the
            // transcript, so it decides what is worth replaying — the engine
            // holding its own copy is what leaked attachments between sessions.
            history: history.unwrap_or_default(),
            ..Default::default()
        },
    )
    .with_cancel(state.cancel.clone());

    let gate = session::build_gate(
        app.clone(),
        state.pending.clone(),
        state.client.clone(),
        classifier,
        assisted,
    );

    let (tx, mut rx) = tokio::sync::mpsc::channel::<AgentEvent>(256);
    let emitter = app.clone();
    let pump = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            let _ = emitter.emit("agent://event", &event);
        }
    });

    let reason: StopReason = runner.run(&prompt, &mut tools, gate.as_ref(), tx).await;
    let _ = pump.await;

    Ok(format!("{reason:?}"))
}

/// How much the agent should spend on a task.
///
/// Maps to real levers rather than a vague dial: whether to let a
/// thinking-capable model reason before answering, and how many tool steps it
/// may take before the run is cut off. Both change wall-clock time
/// substantially on a local model, which is the point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effort {
    /// No thinking, short leash. For edits and lookups.
    Fast,
    #[default]
    Balanced,
    /// Thinking on where supported, long leash. For debugging and design.
    Thorough,
}

impl Effort {
    pub fn wants_thinking(self) -> bool {
        matches!(self, Self::Thorough)
    }

    pub fn max_steps(self) -> usize {
        match self {
            Self::Fast => 8,
            Self::Balanced => 24,
            Self::Thorough => 48,
        }
    }
}

/// Context sizes worth offering for a model, plus what it was trained for.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ContextOptions {
    pub recommended: Option<u64>,
    pub trained_max: Option<u64>,
    /// Steps the UI offers, capped at the trained maximum.
    pub choices: Vec<u64>,
}

/// What context sizes this model can be run at.
#[tauri::command]
pub async fn context_options(
    state: State<'_, AppState>,
    model: String,
) -> Result<ContextOptions, String> {
    let profiles = state.profiles.read().await;
    let profile = profiles
        .iter()
        .find(|p| p.name == model)
        .ok_or_else(|| format!("unknown model {model}"))?;

    let ceiling = profile.max_context.unwrap_or(32_768);
    let choices = [2_048u64, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144]
        .into_iter()
        .filter(|c| *c <= ceiling)
        .collect();

    Ok(ContextOptions {
        recommended: profile.recommended_context,
        trained_max: profile.max_context,
        choices,
    })
}

// ---------------------------------------------------------------------------
// Checkpoints
// ---------------------------------------------------------------------------

/// Restorable points for the current workspace, newest first.
#[tauri::command]
pub async fn checkpoint_list(
    state: State<'_, AppState>,
) -> Result<Vec<crate::checkpoint::Checkpoint>, String> {
    let workspace = state
        .workspace
        .read()
        .await
        .clone()
        .ok_or("No workspace selected.")?;
    let mut all = crate::checkpoint::CheckpointStore::new(workspace)
        .list()
        .map_err(|e| e.to_string())?;
    all.reverse();
    Ok(all)
}

/// Put the files in a checkpoint back to their captured contents.
#[tauri::command]
pub async fn checkpoint_restore(
    state: State<'_, AppState>,
    id: String,
) -> Result<Vec<String>, String> {
    let workspace = state
        .workspace
        .read()
        .await
        .clone()
        .ok_or("No workspace selected.")?;
    crate::checkpoint::CheckpointStore::new(workspace).restore(&id)
}

/// Bytes the undo history currently occupies.
#[tauri::command]
pub async fn checkpoint_size(state: State<'_, AppState>) -> Result<u64, String> {
    let workspace = state
        .workspace
        .read()
        .await
        .clone()
        .ok_or("No workspace selected.")?;
    Ok(crate::checkpoint::CheckpointStore::new(workspace).size_bytes())
}

// ---------------------------------------------------------------------------
// Attachments
// ---------------------------------------------------------------------------

/// One file the user handed to the agent.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Attachment {
    /// The copy the agent sees.
    pub path: String,
    /// Where it came from, so the UI can show the real location.
    pub source: String,
    pub name: String,
    pub bytes: u64,
    /// Which tool will open it, so the UI can say so.
    pub tool: String,
}

fn tool_for(path: &std::path::Path) -> &'static str {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" => "view_image",
        "pdf" | "docx" | "doc" | "xlsx" | "xls" | "pptx" => "read_document",
        _ => "read_file",
    }
}

/// Where copies are staged.
///
/// Outside the workspace, so attachments never end up committed to the user's
/// project by accident.
fn attachment_dir() -> Result<PathBuf, String> {
    let dir = std::env::temp_dir().join("cayahub-attachments");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// Copy dropped or pasted files somewhere the agent may read them.
///
/// The agent gets a *copy*, never the original. It can edit, rewrite, or ruin
/// an attachment without the user's own file being touched — which matters
/// because attachments usually come from Downloads or the Desktop, where a
/// mistake is not recoverable from the project's git history.
///
/// Each copy is registered individually as an allowed root: dropping one file
/// from Downloads must not open the rest of Downloads.
#[tauri::command]
pub async fn attach_files(
    state: State<'_, AppState>,
    paths: Vec<String>,
) -> Result<Vec<Attachment>, String> {
    let staging = attachment_dir()?;
    let mut accepted = Vec::new();
    let mut roots = state.attachments.write().await;

    for raw in paths {
        let source = crate::perms::normalize(std::path::Path::new(&raw));
        let meta = std::fs::metadata(&source)
            .map_err(|e| format!("cannot read {}: {e}", source.display()))?;
        if !meta.is_file() {
            return Err(format!("{} is not a file", source.display()));
        }
        // Credentials stay refused however they arrive.
        if crate::perms::is_sensitive(&source) {
            return Err(format!(
                "{} holds credentials and cannot be attached",
                source.display()
            ));
        }

        let name = source
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "attachment".into());

        // A per-file subdirectory keeps two attachments with the same name apart.
        let slot = staging.join(uuid::Uuid::new_v4().to_string());
        std::fs::create_dir_all(&slot).map_err(|e| e.to_string())?;
        let copy = slot.join(&name);
        std::fs::copy(&source, &copy)
            .map_err(|e| format!("cannot copy {}: {e}", source.display()))?;

        roots.push(copy.clone());
        accepted.push(Attachment {
            name,
            bytes: meta.len(),
            tool: tool_for(&copy).into(),
            path: copy.display().to_string(),
            source: source.display().to_string(),
        });
    }
    Ok(accepted)
}

/// Forget every attachment and delete the staged copies.
#[tauri::command]
pub async fn clear_attachments(state: State<'_, AppState>) -> Result<(), String> {
    let mut roots = state.attachments.write().await;
    for copy in roots.iter() {
        // Remove the per-file slot, not just the file, so nothing is left behind.
        if let Some(slot) = copy.parent() {
            let _ = std::fs::remove_dir_all(slot);
        }
    }
    roots.clear();
    Ok(())
}

/// Stage raw bytes as an attachment — used for images pasted from the clipboard.
///
/// A pasted image has no path, so the frontend hands over the bytes and this
/// gives them one. Same staging directory and same per-file root registration
/// as a dropped file, so nothing downstream has to know the difference.
#[tauri::command]
pub async fn attach_bytes(
    state: State<'_, AppState>,
    name: String,
    data: Vec<u8>,
) -> Result<Attachment, String> {
    const MAX_PASTE_BYTES: usize = 32 * 1024 * 1024;
    if data.len() > MAX_PASTE_BYTES {
        return Err(format!(
            "pasted content is {} MB; the limit is {} MB",
            data.len() / 1024 / 1024,
            MAX_PASTE_BYTES / 1024 / 1024
        ));
    }

    // Take only the file name: a crafted name must not write outside staging.
    let safe = std::path::Path::new(&name)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "pasted.png".into());

    let slot = attachment_dir()?.join(uuid::Uuid::new_v4().to_string());
    std::fs::create_dir_all(&slot).map_err(|e| e.to_string())?;
    let target = slot.join(&safe);
    let bytes = data.len() as u64;
    std::fs::write(&target, data).map_err(|e| format!("cannot stage {safe}: {e}"))?;

    state.attachments.write().await.push(target.clone());
    Ok(Attachment {
        name: safe,
        bytes,
        tool: tool_for(&target).into(),
        path: target.display().to_string(),
        source: "clipboard".into(),
    })
}
