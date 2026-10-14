//! CaYaDev AI Hub — a local-first agent runtime for Ollama.
//!
//! The organising idea: on consumer hardware the dominant cost is *which model
//! runs a step*, not how any single model is tuned. See `docs/BENCHMARKS.md`
//! for the measurements this is built on.

pub mod agent;
pub mod checkpoint;
pub mod mcp;
pub mod commands;
pub mod crashlog;
pub mod ollama;
pub mod perms;
pub mod proc;
pub mod session;
pub mod tools;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // First, so even a panic during startup leaves a report.
    crashlog::install();
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
        .setup(|app| {
            // MCP servers live in the app's config directory, next to the
            // store the frontend uses.
            use tauri::Manager;
            if let Ok(dir) = app.path().app_log_dir() {
                crashlog::set_dir(dir);
            }
            if let Ok(dir) = app.path().app_config_dir() {
                let state = app.state::<commands::AppState>();
                let mcp = state.mcp.clone();
                tauri::async_runtime::block_on(mcp.load_from(dir.join("mcp.json")));
            }
            Ok(())
        })
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
            commands::answer_question,
            commands::run_agent,
            commands::context_options,
            commands::checkpoint_list,
            commands::checkpoint_restore,
            commands::checkpoint_size,
            commands::attach_files,
            commands::attach_bytes,
            commands::clear_attachments,
            commands::mcp_get_config,
            commands::mcp_save_config,
            commands::mcp_test_server,
            commands::mcp_status,
            commands::mcp_list_tools,
            commands::report_frontend_error,
            commands::crash_log_path,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            // No MCP server process may outlive the app.
            if let tauri::RunEvent::Exit = event {
                use tauri::Manager;
                let mcp = app.state::<commands::AppState>().mcp.clone();
                tauri::async_runtime::block_on(mcp.shutdown_all());
            }
        });
}
