//! MCP server configuration, as saved by the MCP page.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// Seconds a single tool call may take before it is abandoned.
pub const DEFAULT_CALL_TIMEOUT_SECS: u64 = 60;

fn default_true() -> bool {
    true
}

fn default_timeout() -> u64 {
    DEFAULT_CALL_TIMEOUT_SECS
}

/// How the app talks to a server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Transport {
    /// A local process speaking JSON-RPC over stdin/stdout.
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: BTreeMap<String, String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cwd: Option<String>,
    },
    /// A remote server speaking Streamable HTTP (responses may stream as SSE).
    Http {
        url: String,
        #[serde(default)]
        headers: BTreeMap<String, String>,
        /// Token sent as `Authorization: Bearer <token>`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bearer: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Stable slug, used in tool names (`mcp__<id>__<tool>`).
    #[serde(default)]
    pub id: String,
    pub name: String,
    /// Whether the agent may use this server at all.
    #[serde(default = "default_true")]
    pub enabled: bool,
    pub transport: Transport,
    /// Tools the user switched off for the agent, by their original name.
    #[serde(default)]
    pub disabled_tools: Vec<String>,
    /// Run this server's tools without asking first.
    #[serde(default)]
    pub auto_approve: bool,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

impl ServerConfig {
    pub fn tool_enabled(&self, tool: &str) -> bool {
        !self.disabled_tools.iter().any(|t| t == tool)
    }

    /// Reject a config that can never connect, with a message for the UI.
    pub fn check(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("Server name is empty.".into());
        }
        match &self.transport {
            Transport::Stdio { command, .. } if command.trim().is_empty() => {
                Err(format!("`{}`: command is empty.", self.name))
            }
            Transport::Http { url, .. }
                if !(url.starts_with("http://") || url.starts_with("https://")) =>
            {
                Err(format!("`{}`: URL must start with http:// or https://.", self.name))
            }
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpConfig {
    #[serde(default)]
    pub servers: Vec<ServerConfig>,
}

/// Lowercase ASCII slug, short enough to keep tool names under 64 chars.
pub fn slug(name: &str) -> String {
    let mut out = String::new();
    for c in name.chars() {
        let c = c.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if !out.ends_with('_') && !out.is_empty() {
            out.push('_');
        }
        if out.len() >= 20 {
            break;
        }
    }
    let out = out.trim_end_matches('_').to_string();
    if out.is_empty() { "server".into() } else { out }
}

impl McpConfig {
    /// Fill in missing ids, make them unique, clamp timeouts, and check
    /// every server. Called on every save so the file on disk is always sane.
    pub fn normalize(mut self) -> Result<Self, String> {
        let mut seen: Vec<String> = Vec::new();
        for s in self.servers.iter_mut() {
            s.check()?;
            s.name = s.name.trim().to_string();
            let base = if s.id.trim().is_empty() { slug(&s.name) } else { slug(&s.id) };
            let mut id = base.clone();
            let mut n = 2;
            while seen.contains(&id) {
                id = format!("{base}{n}");
                n += 1;
            }
            seen.push(id.clone());
            s.id = id;
            s.timeout_secs = s.timeout_secs.clamp(5, 600);
        }
        Ok(self)
    }

    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(text.trim_start_matches('\u{feff}')).unwrap_or_else(|e| {
                tracing::warn!("ignoring unreadable {}: {e}", path.display());
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    /// Write through a temporary file so a crash mid-write cannot leave a
    /// half-written config behind.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        }
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, text).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("cannot replace {}: {e}", path.display()))
    }

    pub fn server(&self, id: &str) -> Option<&ServerConfig> {
        self.servers.iter().find(|s| s.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stdio(name: &str) -> ServerConfig {
        ServerConfig {
            id: String::new(),
            name: name.into(),
            enabled: true,
            transport: Transport::Stdio {
                command: "npx".into(),
                args: vec![],
                env: BTreeMap::new(),
                cwd: None,
            },
            disabled_tools: vec![],
            auto_approve: false,
            timeout_secs: 60,
        }
    }

    #[test]
    fn slugs_are_short_ascii_and_never_empty() {
        assert_eq!(slug("My Files!"), "my_files");
        assert_eq!(slug("Dosya Sistemi"), "dosya_sistemi");
        assert_eq!(slug("***"), "server");
        assert!(slug(&"a".repeat(100)).len() <= 20);
    }

    #[test]
    fn normalize_assigns_unique_ids() {
        let cfg = McpConfig { servers: vec![stdio("fs"), stdio("fs")] }.normalize().unwrap();
        assert_eq!(cfg.servers[0].id, "fs");
        assert_eq!(cfg.servers[1].id, "fs2");
    }

    #[test]
    fn normalize_rejects_an_empty_command_and_a_bad_url() {
        let mut s = stdio("x");
        s.transport = Transport::Stdio { command: " ".into(), args: vec![], env: BTreeMap::new(), cwd: None };
        assert!(McpConfig { servers: vec![s] }.normalize().is_err());

        let mut h = stdio("y");
        h.transport = Transport::Http { url: "localhost:3000".into(), headers: BTreeMap::new(), bearer: None };
        assert!(McpConfig { servers: vec![h] }.normalize().is_err());
    }

    #[test]
    fn config_round_trips_and_tolerates_missing_fields() {
        let json = r#"{"servers":[{"name":"Everything","transport":{"kind":"http","url":"http://127.0.0.1:3001/mcp"}}]}"#;
        let cfg: McpConfig = serde_json::from_str(json).unwrap();
        let s = &cfg.servers[0];
        assert!(s.enabled);
        assert_eq!(s.timeout_secs, DEFAULT_CALL_TIMEOUT_SECS);

        let dir = std::env::temp_dir().join(format!("cayahub-mcp-{}", uuid::Uuid::new_v4()));
        let path = dir.join("mcp.json");
        let cfg = cfg.normalize().unwrap();
        cfg.save(&path).unwrap();
        assert_eq!(McpConfig::load(&path), cfg);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_missing_or_corrupt_file_loads_as_empty() {
        let dir = std::env::temp_dir().join(format!("cayahub-mcp-{}", uuid::Uuid::new_v4()));
        assert!(McpConfig::load(&dir.join("none.json")).servers.is_empty());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("bad.json"), "{not json").unwrap();
        assert!(McpConfig::load(&dir.join("bad.json")).servers.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }
}
