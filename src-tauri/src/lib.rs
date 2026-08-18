//! CaYaDev AI Hub — a local-first agent runtime for Ollama.
//!
//! The organising idea: on consumer hardware the dominant cost is *which model
//! runs a step*, not how any single model is tuned. See `docs/BENCHMARKS.md`
//! for the measurements this is built on.

pub mod agent;
pub mod commands;
pub mod ollama;
pub mod perms;
pub mod tools;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_os::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_store::Builder::new().build())
        .plugin(tauri_plugin_clipboard_manager::init())
        .manage(commands::AppState::new())
        .invoke_handler(tauri::generate_handler![
            commands::ollama_version,
            commands::list_models,
            commands::routing_policy,
            commands::set_vram_budget,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
