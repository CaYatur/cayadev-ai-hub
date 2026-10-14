//! Connections to MCP servers, on the official Rust SDK (`rmcp`).
//!
//! One live connection per server, opened on first use and kept for the
//! session. Every network or process step is bounded by a timeout, because a
//! server that never answers must not hang an agent run. A call that fails
//! because the transport dropped (the process exited, the HTTP session
//! expired) reconnects once and retries; anything else is reported to the
//! model as a tool error.

use rmcp::model::CallToolRequestParams;
use rmcp::service::{RoleClient, RunningService, ServiceError};
use rmcp::ServiceExt;
use serde::Serialize;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, RwLock};

use super::config::{McpConfig, ServerConfig, Transport};
use super::TOOL_PREFIX;

/// Starting a stdio server through `npx` can mean downloading it first.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(90);
const LIST_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(3);
/// Largest tool result handed back to the model, in bytes.
const MAX_RESULT_BYTES: usize = 20_000;
/// Tool names are limited to 64 characters by most providers.
const MAX_TOOL_NAME: usize = 64;
/// Last bytes of a stdio server's stderr kept for error messages.
const STDERR_TAIL: usize = 2_000;

type Client = RunningService<RoleClient, ()>;

/// One tool a server offers, as the agent will see it.
#[derive(Debug, Clone, Serialize)]
pub struct McpTool {
    pub server_id: String,
    pub server_name: String,
    /// The server's own name for the tool.
    pub name: String,
    /// The name offered to the model: `mcp__<server>__<tool>`.
    pub exposed: String,
    pub description: String,
    pub input_schema: Value,
    /// The server says the tool does not change anything (`readOnlyHint`).
    pub read_only: bool,
    pub enabled: bool,
}

struct Connection {
    client: Client,
    tools: Vec<McpTool>,
    stderr: Arc<std::sync::Mutex<String>>,
}

/// Status of one configured server, for the MCP page.
#[derive(Debug, Clone, Serialize)]
pub struct ServerStatus {
    pub id: String,
    pub connected: bool,
    pub tool_count: usize,
    pub error: Option<String>,
}

pub struct McpManager {
    path: RwLock<Option<PathBuf>>,
    config: RwLock<McpConfig>,
    connections: Mutex<HashMap<String, Arc<Connection>>>,
    errors: Mutex<HashMap<String, String>>,
}

impl Default for McpManager {
    fn default() -> Self {
        Self::new()
    }
}

impl McpManager {
    pub fn new() -> Self {
        Self {
            path: RwLock::new(None),
            config: RwLock::new(McpConfig::default()),
            connections: Mutex::new(HashMap::new()),
            errors: Mutex::new(HashMap::new()),
        }
    }

    /// Load the saved config from `path`, which later saves also go to.
    pub async fn load_from(&self, path: PathBuf) {
        let cfg = McpConfig::load(&path);
        *self.config.write().await = cfg.normalize().unwrap_or_default();
        *self.path.write().await = Some(path);
    }

    pub async fn config(&self) -> McpConfig {
        self.config.read().await.clone()
    }

    /// Replace the config, persist it, and drop connections whose settings
    /// changed so the next use reconnects with the new ones.
    pub async fn save(&self, cfg: McpConfig) -> Result<McpConfig, String> {
        let cfg = cfg.normalize()?;
        if let Some(path) = self.path.read().await.as_ref() {
            cfg.save(path)?;
        }
        let old = std::mem::replace(&mut *self.config.write().await, cfg.clone());
        let mut stale = Vec::new();
        for s in &old.servers {
            let same = cfg.server(&s.id).is_some_and(|n| n.transport == s.transport && n.enabled);
            if !same {
                stale.push(s.id.clone());
            }
        }
        for id in stale {
            self.disconnect(&id).await;
        }
        Ok(cfg)
    }

    pub async fn status(&self) -> Vec<ServerStatus> {
        let cfg = self.config.read().await.clone();
        let conns = self.connections.lock().await;
        let errors = self.errors.lock().await;
        cfg.servers
            .iter()
            .map(|s| ServerStatus {
                id: s.id.clone(),
                connected: conns.get(&s.id).is_some_and(|c| !c.client.is_closed()),
                tool_count: conns.get(&s.id).map_or(0, |c| c.tools.len()),
                error: errors.get(&s.id).cloned(),
            })
            .collect()
    }

    /// Connect to `server` from scratch and list its tools, without keeping
    /// the connection. Used by the page's "Test connection" button, so it
    /// works on settings that have not been saved yet.
    pub async fn test(&self, mut server: ServerConfig) -> Result<Vec<McpTool>, String> {
        server.check()?;
        if server.id.trim().is_empty() {
            server.id = super::config::slug(&server.name);
        }
        let conn = connect(&server).await?;
        let tools = conn.tools.clone();
        shutdown(conn).await;
        Ok(tools)
    }

    /// Tools of every enabled server, connecting where needed.
    ///
    /// A server that cannot start does not stop the run; its error comes back
    /// alongside the tools that did load.
    pub async fn tools_for_agent(&self) -> (Vec<McpTool>, Vec<String>) {
        let cfg = self.config.read().await.clone();
        let mut tools = Vec::new();
        let mut errors = Vec::new();
        for server in cfg.servers.iter().filter(|s| s.enabled) {
            match self.connection(server).await {
                Ok(conn) => tools.extend(
                    conn.tools.iter().filter(|t| server.tool_enabled(&t.name)).cloned(),
                ),
                Err(e) => errors.push(format!("MCP server `{}`: {e}", server.name)),
            }
        }
        (tools, errors)
    }

    async fn connection(&self, server: &ServerConfig) -> Result<Arc<Connection>, String> {
        if let Some(c) = self.connections.lock().await.get(&server.id) {
            if !c.client.is_closed() {
                return Ok(c.clone());
            }
        }
        // Connect without holding the lock: a slow server must not block the
        // others.
        match connect(server).await {
            Ok(conn) => {
                let conn = Arc::new(conn);
                self.errors.lock().await.remove(&server.id);
                let previous = self.connections.lock().await.insert(server.id.clone(), conn.clone());
                if let Some(old) = previous {
                    if let Ok(old) = Arc::try_unwrap(old) {
                        shutdown(old).await;
                    }
                }
                Ok(conn)
            }
            Err(e) => {
                self.errors.lock().await.insert(server.id.clone(), e.clone());
                Err(e)
            }
        }
    }

    async fn disconnect(&self, id: &str) {
        let conn = self.connections.lock().await.remove(id);
        if let Some(conn) = conn {
            if let Ok(conn) = Arc::try_unwrap(conn) {
                shutdown(conn).await;
            }
        }
    }

    /// Call one tool. `Err` is a message for the model, not a crash.
    pub async fn call(&self, server_id: &str, tool: &str, args: &Map<String, Value>) -> Result<String, String> {
        let server = self
            .config
            .read()
            .await
            .server(server_id)
            .cloned()
            .ok_or_else(|| format!("MCP server `{server_id}` is no longer configured"))?;
        let timeout = Duration::from_secs(server.timeout_secs);

        let mut attempt = 0;
        loop {
            let conn = self.connection(&server).await?;
            let mut params = CallToolRequestParams::new(tool.to_string());
            params.arguments = Some(args.clone());
            match tokio::time::timeout(timeout, conn.client.call_tool(params)).await {
                Err(_) => return Err(format!("`{tool}` timed out after {}s", timeout.as_secs())),
                Ok(Ok(result)) => {
                    let value = serde_json::to_value(&result).unwrap_or(Value::Null);
                    return render_result(&value);
                }
                Ok(Err(e)) if attempt == 0 && is_transport_error(&e) => {
                    tracing::warn!("MCP `{}` dropped ({e}); reconnecting", server.name);
                    drop(conn);
                    self.disconnect(&server.id).await;
                    attempt += 1;
                }
                Ok(Err(ServiceError::McpError(e))) => return Err(format!("server error: {}", e.message)),
                Ok(Err(e)) => {
                    let tail = conn.stderr.lock().map(|s| s.clone()).unwrap_or_default();
                    return Err(with_stderr(format!("call failed: {e}"), &tail));
                }
            }
        }
    }

    /// Close every connection. Called when the app exits so no server
    /// process outlives it.
    pub async fn shutdown_all(&self) {
        let conns: Vec<_> = self.connections.lock().await.drain().map(|(_, c)| c).collect();
        for conn in conns {
            if let Ok(conn) = Arc::try_unwrap(conn) {
                shutdown(conn).await;
            }
        }
    }
}

fn is_transport_error(e: &ServiceError) -> bool {
    matches!(e, ServiceError::TransportClosed | ServiceError::TransportSend(_) | ServiceError::Cancelled { .. })
}

async fn shutdown(conn: Connection) {
    // Dropping the child-process transport kills the process anyway; this is
    // the polite path first.
    let _ = tokio::time::timeout(SHUTDOWN_TIMEOUT, conn.client.cancel()).await;
}

fn with_stderr(msg: String, tail: &str) -> String {
    let tail = tail.trim();
    if tail.is_empty() { msg } else { format!("{msg}\nserver stderr:\n{tail}") }
}

async fn connect(server: &ServerConfig) -> Result<Connection, String> {
    let stderr = Arc::new(std::sync::Mutex::new(String::new()));
    let client: Client = match &server.transport {
        Transport::Stdio { command, args, env, cwd } => {
            let mut cmd = stdio_command(command, args);
            cmd.envs(env);
            if let Some(dir) = cwd.as_deref().filter(|d| !d.trim().is_empty()) {
                cmd.current_dir(dir);
            }
            let (transport, err) = rmcp::transport::TokioChildProcess::builder(cmd)
                .stderr(std::process::Stdio::piped())
                .spawn()
                .map_err(|e| format!("cannot start `{command}`: {e}"))?;
            if let Some(err) = err {
                tokio::spawn(collect_stderr(err, stderr.clone()));
            }
            let served = tokio::time::timeout(CONNECT_TIMEOUT, ().serve(transport)).await;
            let tail = || stderr.lock().map(|s| s.clone()).unwrap_or_default();
            match served {
                Err(_) => return Err(with_stderr(format!("no answer within {}s", CONNECT_TIMEOUT.as_secs()), &tail())),
                Ok(Err(e)) => return Err(with_stderr(format!("handshake failed: {e}"), &tail())),
                Ok(Ok(c)) => c,
            }
        }
        Transport::Http { url, headers, bearer } => {
            use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
            let mut custom = HashMap::new();
            for (k, v) in headers {
                let name = http::HeaderName::from_bytes(k.trim().as_bytes())
                    .map_err(|_| format!("invalid header name `{k}`"))?;
                let value = http::HeaderValue::from_str(v.trim())
                    .map_err(|_| format!("invalid value for header `{k}`"))?;
                custom.insert(name, value);
            }
            let mut cfg = StreamableHttpClientTransportConfig::with_uri(url.clone()).custom_headers(custom);
            if let Some(token) = bearer.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
                cfg = cfg.auth_header(token.trim_start_matches("Bearer ").to_string());
            }
            let transport = rmcp::transport::StreamableHttpClientTransport::from_config(cfg);
            match tokio::time::timeout(CONNECT_TIMEOUT, ().serve(transport)).await {
                Err(_) => return Err(format!("no answer from {url} within {}s", CONNECT_TIMEOUT.as_secs())),
                Ok(Err(e)) => return Err(format!("handshake with {url} failed: {e}")),
                Ok(Ok(c)) => c,
            }
        }
    };

    let listed = match tokio::time::timeout(LIST_TIMEOUT, client.list_all_tools()).await {
        Ok(Ok(t)) => t,
        Ok(Err(e)) => {
            let _ = tokio::time::timeout(SHUTDOWN_TIMEOUT, client.cancel()).await;
            return Err(format!("tools/list failed: {e}"));
        }
        Err(_) => {
            let _ = tokio::time::timeout(SHUTDOWN_TIMEOUT, client.cancel()).await;
            return Err("tools/list timed out".into());
        }
    };
    let raw: Vec<Value> = listed.iter().filter_map(|t| serde_json::to_value(t).ok()).collect();
    let tools = describe_tools(server, &raw);
    Ok(Connection { client, tools, stderr })
}

/// Build the command for a stdio server.
///
/// On Windows `npx`, `npm` and friends are `.cmd` scripts, which
/// `CreateProcess` cannot start directly, so those run through `cmd /C`.
fn stdio_command(command: &str, args: &[String]) -> tokio::process::Command {
    #[cfg(windows)]
    {
        if needs_cmd_shell(command) {
            let mut cmd = crate::proc::command("cmd");
            cmd.arg("/C").arg(command).args(args);
            return cmd;
        }
    }
    let mut cmd = crate::proc::command(command);
    cmd.args(args);
    cmd
}

#[cfg(windows)]
fn needs_cmd_shell(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    if lower.ends_with(".cmd") || lower.ends_with(".bat") {
        return true;
    }
    if std::path::Path::new(command).extension().is_some() || command.contains(['/', '\\']) {
        return false;
    }
    // A bare name: find what PATH resolves it to.
    let Some(path) = std::env::var_os("PATH") else { return false };
    for dir in std::env::split_paths(&path) {
        if dir.join(format!("{command}.exe")).is_file() || dir.join(format!("{command}.com")).is_file() {
            return false;
        }
        if dir.join(format!("{command}.cmd")).is_file() || dir.join(format!("{command}.bat")).is_file() {
            return true;
        }
    }
    false
}

async fn collect_stderr(err: tokio::process::ChildStderr, sink: Arc<std::sync::Mutex<String>>) {
    use tokio::io::AsyncBufReadExt;
    let mut lines = tokio::io::BufReader::new(err).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        if let Ok(mut s) = sink.lock() {
            s.push_str(&line);
            s.push('\n');
            if s.len() > STDERR_TAIL {
                let mut cut = s.len() - STDERR_TAIL;
                while !s.is_char_boundary(cut) {
                    cut += 1;
                }
                s.drain(..cut);
            }
        }
    }
}

/// Make a server's tool name safe for the model: `[A-Za-z0-9_-]`, prefixed
/// with the server id, at most 64 characters.
pub fn exposed_name(server_id: &str, tool: &str) -> String {
    let clean: String = tool
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    let mut name = format!("{TOOL_PREFIX}{server_id}__{clean}");
    name.truncate(MAX_TOOL_NAME);
    name
}

/// Turn the server's `tools/list` entries into what the agent sees.
pub fn describe_tools(server: &ServerConfig, raw: &[Value]) -> Vec<McpTool> {
    let mut out: Vec<McpTool> = Vec::new();
    for t in raw {
        let Some(name) = t.get("name").and_then(Value::as_str) else { continue };
        let mut exposed = exposed_name(&server.id, name);
        // Truncation or sanitising can make two names collide.
        let mut n = 2;
        while out.iter().any(|o| o.exposed == exposed) {
            let suffix = format!("_{n}");
            exposed = exposed_name(&server.id, name);
            exposed.truncate(MAX_TOOL_NAME - suffix.len());
            exposed.push_str(&suffix);
            n += 1;
        }
        let description = t
            .get("description")
            .and_then(Value::as_str)
            .or_else(|| t.get("title").and_then(Value::as_str))
            .unwrap_or(name)
            .trim()
            .to_string();
        let read_only = t
            .pointer("/annotations/readOnlyHint")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        out.push(McpTool {
            server_id: server.id.clone(),
            server_name: server.name.clone(),
            name: name.to_string(),
            exposed,
            description,
            input_schema: clean_schema(t.get("inputSchema").cloned().unwrap_or(Value::Null)),
            read_only,
            enabled: server.tool_enabled(name),
        });
    }
    out
}

/// Ollama and OpenAI-compatible APIs want an object schema with
/// `properties`; some servers send none, or decorate it with keys that only
/// cost tokens.
fn clean_schema(schema: Value) -> Value {
    let mut obj = match schema {
        Value::Object(o) => o,
        _ => Map::new(),
    };
    obj.remove("$schema");
    obj.remove("$id");
    obj.insert("type".into(), Value::String("object".into()));
    obj.entry("properties").or_insert_with(|| Value::Object(Map::new()));
    Value::Object(obj)
}

/// Flatten a `tools/call` result into text for the model.
pub fn render_result(result: &Value) -> Result<String, String> {
    let mut parts: Vec<String> = Vec::new();
    for block in result.get("content").and_then(Value::as_array).into_iter().flatten() {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => parts.push(block.get("text").and_then(Value::as_str).unwrap_or("").to_string()),
            Some("image") | Some("audio") => {
                let kind = block.get("type").and_then(Value::as_str).unwrap_or("media");
                let mime = block.get("mimeType").and_then(Value::as_str).unwrap_or("unknown");
                let size = block.get("data").and_then(Value::as_str).map_or(0, |d| d.len() * 3 / 4);
                parts.push(format!("[{kind}: {mime}, about {size} bytes, not shown]"));
            }
            Some("resource") => {
                let r = block.get("resource").cloned().unwrap_or(Value::Null);
                match r.get("text").and_then(Value::as_str) {
                    Some(text) => parts.push(text.to_string()),
                    None => parts.push(format!(
                        "[resource {}]",
                        r.get("uri").and_then(Value::as_str).unwrap_or("?")
                    )),
                }
            }
            Some("resource_link") => parts.push(format!(
                "[link {}]",
                block.get("uri").and_then(Value::as_str).unwrap_or("?")
            )),
            _ => parts.push(block.to_string()),
        }
    }
    if parts.iter().all(|p| p.trim().is_empty()) {
        if let Some(s) = result.get("structuredContent").filter(|v| !v.is_null()) {
            parts = vec![s.to_string()];
        }
    }
    let mut text = parts.join("\n");
    if text.trim().is_empty() {
        text = "(no output)".into();
    }
    if text.len() > MAX_RESULT_BYTES {
        let mut end = MAX_RESULT_BYTES;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        let total = text.len();
        text.truncate(end);
        text.push_str(&format!("\n\n[truncated — {end} of {total} bytes shown]"));
    }
    if result.get("isError").and_then(Value::as_bool).unwrap_or(false) {
        Err(text)
    } else {
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::BTreeMap;

    fn server() -> ServerConfig {
        ServerConfig {
            id: "fs".into(),
            name: "Files".into(),
            enabled: true,
            transport: Transport::Stdio { command: "x".into(), args: vec![], env: BTreeMap::new(), cwd: None },
            disabled_tools: vec!["write".into()],
            auto_approve: false,
            timeout_secs: 60,
        }
    }

    #[test]
    fn exposed_names_are_prefixed_sanitised_and_capped() {
        assert_eq!(exposed_name("fs", "read_file"), "mcp__fs__read_file");
        assert_eq!(exposed_name("fs", "get.thing/now"), "mcp__fs__get_thing_now");
        assert!(exposed_name("fs", &"x".repeat(200)).len() <= 64);
    }

    #[test]
    fn listed_tools_are_described_with_hints_and_clean_schemas() {
        let raw = vec![
            json!({"name": "read", "description": "Read it", "annotations": {"readOnlyHint": true},
                   "inputSchema": {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
                                   "properties": {"p": {"type": "string"}}}}),
            json!({"name": "write", "inputSchema": {}}),
            json!({"description": "no name, skipped"}),
        ];
        let tools = describe_tools(&server(), &raw);
        assert_eq!(tools.len(), 2);
        assert!(tools[0].read_only && tools[0].enabled);
        assert!(tools[0].input_schema.get("$schema").is_none());
        assert_eq!(tools[1].description, "write");
        assert_eq!(tools[1].input_schema, json!({"type": "object", "properties": {}}));
        assert!(!tools[1].enabled, "disabled_tools must be honoured");
    }

    #[test]
    fn colliding_names_get_a_suffix() {
        let raw = vec![json!({"name": "a.b"}), json!({"name": "a/b"})];
        let tools = describe_tools(&server(), &raw);
        assert_ne!(tools[0].exposed, tools[1].exposed);
    }

    #[test]
    fn results_are_flattened_and_errors_are_errors() {
        let ok = json!({"content": [{"type": "text", "text": "hello"},
                                    {"type": "image", "mimeType": "image/png", "data": "AAAA"}]});
        let text = render_result(&ok).unwrap();
        assert!(text.starts_with("hello\n[image: image/png"), "{text}");

        let structured = json!({"content": [], "structuredContent": {"sum": 3}});
        assert_eq!(render_result(&structured).unwrap(), "{\"sum\":3}");

        let err = json!({"content": [{"type": "text", "text": "boom"}], "isError": true});
        assert_eq!(render_result(&err).unwrap_err(), "boom");

        let big = json!({"content": [{"type": "text", "text": "a".repeat(MAX_RESULT_BYTES + 10)}]});
        assert!(render_result(&big).unwrap().contains("[truncated"));
    }
}
