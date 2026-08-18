//! CaYaDev AI Hub — a local-first agent runtime for Ollama.
//!
//! The organising idea: on consumer hardware the dominant cost is *which model
//! runs a step*, not how any single model is tuned. See `docs/BENCHMARKS.md`
//! for the measurements this is built on.

pub mod agent;
pub mod checkpoint;
pub mod commands;
pub mod ollama;
pub mod perms;
pub mod session;
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
            commands::refresh_vram_budget,
            commands::set_workspace,
            commands::set_permission_mode,
            commands::set_assisted_approval,
            commands::resolve_approval,
            commands::cancel_run,
            commands::run_agent,
            commands::context_options,
            commands::checkpoint_list,
            commands::checkpoint_restore,
            commands::checkpoint_size,
            commands::attach_files,
            commands::attach_bytes,
            commands::clear_attachments,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
