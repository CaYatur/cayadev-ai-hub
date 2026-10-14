//! Model Context Protocol support.
//!
//! The agent reaches MCP servers the user configured on the MCP page. Three
//! pieces, kept apart so each can be tested on its own:
//!
//! * [`config`]: what the user saved, persisted as `mcp.json` in the app's
//!   config directory.
//! * [`client`]: connections, built on the official Rust SDK (`rmcp`), with
//!   connect/call timeouts, one reconnect on a dropped transport, and a clean
//!   shutdown when the app exits.
//! * [`tools`]: the bridge that offers MCP tools to the agent loop next to the
//!   built-in ones.
//!
//! Argument validation lives in [`schema`] because it is shared with the
//! built-in tools: a small model gets the same precise, retryable error
//! whichever kind of tool it called badly.

pub mod client;
pub mod config;
pub mod sanitize;
pub mod schema;
pub mod tools;

pub use client::{McpManager, McpTool};
pub use config::{McpConfig, ServerConfig, Transport};

/// Prefix that marks a tool name as coming from an MCP server.
pub const TOOL_PREFIX: &str = "mcp__";
