//! The agent loop and the machinery that keeps it running on weak models.

pub mod approver;
pub mod prompt;
pub mod runner;
pub mod toolcall;

pub use approver::{ApprovalFacts, AssistedGate, ModelOracle, RiskOracle, Verdict};
pub use prompt::{compact_tools, estimate_tokens, system_prompt, PromptTier};
pub use runner::{AgentEvent, AgentRunner, ApprovalGate, ProposedCall, RunConfig, StopReason, ToolExecutor};
pub use toolcall::{extract, Extracted, Recovery};
