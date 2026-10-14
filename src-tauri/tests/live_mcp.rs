//! MCP against real servers, and one agent run against a live Ollama.
//!
//! Needs `npx` (Node.js) and network access for the first download of
//! `@modelcontextprotocol/server-everything`; the agent test also needs
//! Ollama with a tool-capable model (`MCP_TEST_MODEL`, default qwen3.5:4b).
//!
//! ```text
//! cargo test --test live_mcp -- --ignored --nocapture --test-threads=1
//! ```

use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

use cayadev_ai_hub_lib::agent::runner::{
    AgentEvent, AgentRunner, ApprovalGate, ProposedCall, RunConfig, StopReason, ToolExecutor,
};
use cayadev_ai_hub_lib::mcp::tools::WithMcp;
use cayadev_ai_hub_lib::mcp::{McpConfig, McpManager, ServerConfig, Transport};
use cayadev_ai_hub_lib::ollama::{OllamaClient, Timeouts, DEFAULT_HOST};
use cayadev_ai_hub_lib::perms::{Outcome, PermissionEngine, PermissionMode};
use cayadev_ai_hub_lib::tools::WorkspaceTools;

const PACKAGE: &str = "@modelcontextprotocol/server-everything";

fn scratch(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cayahub-mcp-{label}-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.canonicalize().map(|p| cayadev_ai_hub_lib::perms::normalize(&p)).unwrap_or(dir)
}

fn everything_stdio() -> ServerConfig {
    ServerConfig {
        id: "everything".into(),
        name: "Everything".into(),
        enabled: true,
        transport: Transport::Stdio {
            command: "npx".into(),
            args: vec!["-y".into(), PACKAGE.into(), "stdio".into()],
            env: BTreeMap::new(),
            cwd: None,
        },
        disabled_tools: vec![],
        auto_approve: true,
        timeout_secs: 60,
    }
}

fn args(v: Value) -> Map<String, Value> {
    serde_json::from_value(v).unwrap()
}

async fn manager_with(server: ServerConfig, label: &str) -> (Arc<McpManager>, PathBuf) {
    let dir = scratch(label);
    let manager = Arc::new(McpManager::new());
    manager.load_from(dir.join("mcp.json")).await;
    manager.save(McpConfig { servers: vec![server] }).await.expect("save");
    (manager, dir)
}

#[tokio::test]
#[ignore]
async fn stdio_server_lists_and_calls_tools() {
    let (manager, dir) = manager_with(everything_stdio(), "stdio").await;

    let listed = manager.test(everything_stdio()).await.expect("test connection");
    let names: Vec<&str> = listed.iter().map(|t| t.name.as_str()).collect();
    println!("tools/list: {names:?}");
    assert!(names.contains(&"echo"), "{names:?}");

    let (tools, errors) = manager.tools_for_agent().await;
    assert!(errors.is_empty(), "{errors:?}");
    let echo = tools.iter().find(|t| t.name == "echo").unwrap();
    assert_eq!(echo.exposed, "mcp__everything__echo");

    let out = manager.call("everything", "echo", &args(json!({"message": "merhaba"}))).await.expect("echo");
    println!("tools/call echo -> {out}");
    assert!(out.contains("merhaba"));

    // A server-side failure comes back as a tool error, not a panic.
    let err = manager.call("everything", "no-such-tool", &Map::new()).await;
    println!("unknown tool -> {err:?}");
    assert!(err.is_err());

    // The saved file is what the page will read back.
    let saved = McpConfig::load(&dir.join("mcp.json"));
    assert_eq!(saved.servers[0].id, "everything");

    manager.shutdown_all().await;
    assert!(manager.status().await.iter().all(|s| !s.connected));
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
#[ignore]
async fn a_server_that_cannot_start_is_reported_not_fatal() {
    let mut bad = everything_stdio();
    bad.transport = Transport::Stdio {
        command: "definitely-not-a-real-mcp-binary".into(),
        args: vec![],
        env: BTreeMap::new(),
        cwd: None,
    };
    let (manager, dir) = manager_with(bad, "bad").await;
    let started = std::time::Instant::now();
    let (tools, errors) = manager.tools_for_agent().await;
    println!("errors: {errors:?}");
    assert!(tools.is_empty());
    assert_eq!(errors.len(), 1);
    assert!(started.elapsed() < Duration::from_secs(100));
    std::fs::remove_dir_all(&dir).ok();
}

/// Start `server-everything` in Streamable HTTP mode on `port`.
fn spawn_http(port: u16) -> std::process::Child {
    let mut cmd = if cfg!(windows) {
        let mut c = std::process::Command::new("cmd");
        c.args(["/C", "npx", "-y", PACKAGE, "streamableHttp"]);
        c
    } else {
        let mut c = std::process::Command::new("npx");
        c.args(["-y", PACKAGE, "streamableHttp"]);
        c
    };
    cmd.env("PORT", port.to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn npx")
}

fn kill_tree(child: &mut std::process::Child) {
    if cfg!(windows) {
        let _ = std::process::Command::new("taskkill")
            .args(["/F", "/T", "/PID", &child.id().to_string()])
            .output();
    }
    let _ = child.kill();
    let _ = child.wait();
}

async fn wait_for_port(port: u16) {
    for _ in 0..120 {
        if tokio::net::TcpStream::connect(("127.0.0.1", port)).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    panic!("port {port} never opened");
}

#[tokio::test]
#[ignore]
async fn http_server_lists_calls_and_survives_a_restart() {
    let port = 3917;
    let mut child = spawn_http(port);
    wait_for_port(port).await;

    let server = ServerConfig {
        id: "web".into(),
        name: "Everything HTTP".into(),
        transport: Transport::Http {
            url: format!("http://127.0.0.1:{port}/mcp"),
            headers: BTreeMap::from([("X-Test".to_string(), "1".to_string())]),
            bearer: Some("unused-token".into()),
        },
        ..everything_stdio()
    };
    let (manager, dir) = manager_with(server, "http").await;

    let (tools, errors) = manager.tools_for_agent().await;
    assert!(errors.is_empty(), "{errors:?}");
    println!("http tools/list: {} tools", tools.len());
    let out = manager.call("web", "echo", &args(json!({"message": "http-ok"}))).await.expect("echo");
    assert!(out.contains("http-ok"), "{out}");

    // Restart the server: the old session is gone, so the next call has to
    // reconnect on its own.
    kill_tree(&mut child);
    let mut child = spawn_http(port);
    wait_for_port(port).await;
    let out = manager.call("web", "echo", &args(json!({"message": "after-restart"}))).await;
    println!("after restart -> {out:?}");
    kill_tree(&mut child);
    manager.shutdown_all().await;
    std::fs::remove_dir_all(&dir).ok();
    assert!(out.expect("reconnect").contains("after-restart"));
}

struct Approve;
#[async_trait::async_trait]
impl ApprovalGate for Approve {
    async fn decide(&self, _: &ProposedCall) -> Outcome {
        Outcome::UserApproved
    }
}

/// One real agent run: the model has to pick the MCP tool and use its output.
async fn agent_run(native_tools: bool) {
    let model = std::env::var("MCP_TEST_MODEL").unwrap_or_else(|_| "qwen3.5:4b".into());
    let (manager, dir) = manager_with(everything_stdio(), "agent").await;
    let (mcp_tools, errors) = manager.tools_for_agent().await;
    assert!(errors.is_empty(), "{errors:?}");

    let workspace = scratch("ws");
    let mut tools = WithMcp::new(
        WorkspaceTools::without_checkpoints(&workspace),
        manager.clone(),
        mcp_tools,
        vec!["everything".into()],
    );
    println!("offered {} tools", tools.schemas().len());

    let runner = AgentRunner::new(
        OllamaClient::new(DEFAULT_HOST, Timeouts::default()),
        PermissionEngine::new(&workspace, PermissionMode::Balanced),
        RunConfig {
            model: model.clone(),
            num_ctx: Some(8192),
            think: Some(false),
            params_billions: Some(4.0),
            max_steps: 6,
            native_tools,
            ..Default::default()
        },
    );
    let (tx, mut rx) = mpsc::channel(512);
    let collector = tokio::spawn(async move {
        let mut calls = Vec::new();
        let mut outputs = Vec::new();
        let mut text = String::new();
        while let Some(ev) = rx.recv().await {
            match ev {
                AgentEvent::ToolProposed { call } => {
                    println!("-> {} {:?} via {:?}", call.name, call.arguments, call.recovery);
                    calls.push(call.name);
                }
                AgentEvent::ToolSettled { output, .. } => {
                    println!("<- {output}");
                    outputs.push(output);
                }
                AgentEvent::Token { text: t } => text.push_str(&t),
                AgentEvent::Error { message } => println!("!! {message}"),
                _ => {}
            }
        }
        (calls, outputs, text)
    });
    let reason = runner
        .run(
            "Use the MCP echo tool to echo the exact text ZEBRA-4417, then tell me what it returned.",
            &mut tools,
            &Approve,
            tx,
        )
        .await;
    let (calls, outputs, text) = collector.await.unwrap();
    println!("model {model}, native_tools={native_tools}: {reason:?}\nfinal text: {text}");
    manager.shutdown_all().await;
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&workspace).ok();

    assert_eq!(reason, StopReason::Completed);
    assert!(calls.iter().any(|c| c == "mcp__everything__echo"), "{calls:?}");
    assert!(outputs.iter().any(|o| o.contains("ZEBRA-4417")), "{outputs:?}");
}

#[tokio::test]
#[ignore]
async fn agent_calls_an_mcp_tool_natively() {
    agent_run(true).await;
}

#[tokio::test]
#[ignore]
async fn agent_calls_an_mcp_tool_in_text_mode() {
    agent_run(false).await;
}
