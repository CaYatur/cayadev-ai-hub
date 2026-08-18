//! The agent loop and the machinery that keeps it running on weak models.

pub mod approver;
pub mod context;
pub mod prompt;
pub mod runner;
pub mod toolcall;

pub use context::{fit, total_tokens};
pub use approver::{ApprovalFacts, AssistedGate, ModelOracle, RiskOracle, Verdict};
pub use prompt::{compact_tools, estimate_tokens, select_tools, system_prompt, PromptTier};
pub use runner::{AgentEvent, AgentRunner, ApprovalGate, ProposedCall, RunConfig, StopReason, ToolExecutor};
pub use toolcall::{extract, unknown_tool_name, Extracted, Recovery};
