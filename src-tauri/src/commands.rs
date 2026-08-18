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

use tauri::State;
use tokio::sync::RwLock;

use crate::ollama::{
    load_profiles, ModelProfile, OllamaClient, RoutingPolicy, Timeouts, VramBudget, DEFAULT_HOST,
};

pub struct AppState {
    pub client: OllamaClient,
    pub budget: RwLock<VramBudget>,
    pub profiles: RwLock<Vec<ModelProfile>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            client: OllamaClient::new(DEFAULT_HOST, Timeouts::default()),
            budget: RwLock::new(VramBudget::default()),
            profiles: RwLock::new(Vec::new()),
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
