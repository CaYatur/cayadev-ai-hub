//! Ollama integration: wire types, HTTP client, and the model registry.

pub mod client;
pub mod gpu;
pub mod registry;
pub mod types;

pub use gpu::probe as probe_vram;
pub use client::{reply_budget, OllamaClient, OllamaError, StreamEvent, Timeouts};
pub use registry::{
    load_profiles, Capabilities, ModelProfile, Placement, RoutingPolicy, TaskClass, VramBudget,
};
pub use types::{ChatRequest, ChatResponse, Message, Metrics, Options, Tool, ToolCall};

/// Default endpoint for a local daemon.
pub const DEFAULT_HOST: &str = "http://localhost:11434";
