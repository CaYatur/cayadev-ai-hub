//! Ollama integration: wire types, HTTP client, and the model registry.

pub mod client;
pub mod registry;
pub mod types;

pub use client::{OllamaClient, OllamaError, StreamEvent, Timeouts};
pub use registry::{
    load_profiles, Capabilities, ModelProfile, Placement, RoutingPolicy, TaskClass, VramBudget,
};
pub use types::{ChatRequest, ChatResponse, Message, Metrics, Options, Tool, ToolCall};

/// Default endpoint for a local daemon.
pub const DEFAULT_HOST: &str = "http://localhost:11434";
