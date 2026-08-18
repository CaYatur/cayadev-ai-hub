//! The agent loop and the machinery that keeps it running on weak models.

pub mod prompt;
pub mod toolcall;

pub use prompt::{compact_tools, estimate_tokens, system_prompt, PromptTier};
pub use toolcall::{extract, Extracted, Recovery};
