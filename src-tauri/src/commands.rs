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

use crate::agent::runner::{AgentEvent, AgentRunner, RunConfig, StopReason};
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

/// Refuse everything still waiting, ending the current run's questions.
#[tauri::command]
pub async fn cancel_run(state: State<'_, AppState>) -> Result<(), String> {
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

    let tools = WorkspaceTools::new(&workspace);
    let perms = PermissionEngine::new(&workspace, mode);
    let runner = AgentRunner::new(
        state.client.clone(),
        perms,
        RunConfig {
            model: chosen.clone(),
            num_ctx: Some(context),
            params_billions: params,
            think,
            max_steps: effort.max_steps(),
            ..Default::default()
        },
    );

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

    let reason: StopReason = runner.run(&prompt, &tools, gate.as_ref(), tx).await;
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
