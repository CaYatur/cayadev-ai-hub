//! Keeping a conversation inside the model's window.
//!
//! Left alone, an agent conversation grows without limit — every step appends
//! an assistant turn and a tool result, and tool results are the long ones. Once
//! it passes `num_ctx`, Ollama truncates from the *front*, which is where the
//! system prompt and the user's actual instruction live. Two things then happen,
//! both observed:
//!
//! * The chat template finds no user turn and the daemon answers
//!   `500 {"error":"no user query found in messages"}`.
//! * Or it survives but the model has lost the task and starts improvising.
//!
//! So the trimming happens here instead, and it drops from the middle. The
//! system prompt and the first user message are never removed — they are the
//! task — and the most recent exchanges are kept because they are what the next
//! step reasons from. What goes is the long stretch of old tool output nobody
//! needs any more, replaced by a note saying how much was dropped so the model
//! knows its history is incomplete rather than silently misremembering.

use crate::agent::prompt::estimate_tokens;
use crate::ollama::types::Message;

/// Fraction of the window the conversation may occupy before trimming.
///
/// The rest is headroom for the reply. Filling the window entirely leaves the
/// model no room to answer, which reads as a stall.
const OCCUPANCY: f64 = 0.65;

/// Exchanges at the end that are never dropped.
///
/// One step is an assistant turn plus its tool result, so six messages is
/// roughly the last three steps — enough to see what was just tried and what
/// came back.
const KEEP_RECENT: usize = 6;

/// Rough token cost of a message, including the per-message framing the
/// template adds.
fn cost(message: &Message) -> usize {
    let body = estimate_tokens(&message.content)
        + message.thinking.as_deref().map(estimate_tokens).unwrap_or(0)
        + message
            .tool_calls
            .as_ref()
            .map(|calls| {
                calls
                    .iter()
                    .map(|c| estimate_tokens(&c.function.arguments.to_string()) + 8)
                    .sum::<usize>()
            })
            .unwrap_or(0);
    body + 8
}

pub fn total_tokens(messages: &[Message]) -> usize {
    messages.iter().map(cost).sum()
}

/// Trim a conversation to fit, dropping from the middle.
///
/// Returns the number of messages removed, and mutates `messages` in place.
pub fn fit(messages: &mut Vec<Message>, num_ctx: u32) -> usize {
    let budget = (num_ctx as f64 * OCCUPANCY) as usize;
    if total_tokens(messages) <= budget {
        return 0;
    }

    // The head is what states the task: the system prompt and the first user
    // turn. Everything downstream is meaningless without them.
    let head = messages
        .iter()
        .position(|m| m.role == "user")
        .map(|i| i + 1)
        .unwrap_or(1)
        .min(messages.len());

    let mut dropped = 0usize;
    // Never cut into the head or the recent tail, however tight the budget —
    // a conversation that cannot fit is better sent slightly oversized than
    // sent without its instruction.
    while total_tokens(messages) > budget && messages.len() > head + KEEP_RECENT {
        messages.remove(head);
        dropped += 1;
    }

    if dropped > 0 {
        // Say what was lost. A model that knows its history is partial asks or
        // re-reads; one that does not will confidently misremember.
        messages.insert(
            head,
            Message::user(format!(
                "[{dropped} earlier steps were dropped to stay within the context \
                 window. Re-read any file you need rather than relying on memory \
                 of it.]"
            )),
        );
    }
    dropped
}

#[cfg(test)]
mod tests {
    use super::*;

    fn convo(steps: usize, filler: usize) -> Vec<Message> {
        let mut messages = vec![
            Message::system("You are a coding assistant."),
            Message::user("Refactor the parser."),
        ];
        for i in 0..steps {
            messages.push(Message::assistant(format!("step {i}")));
            messages.push(Message::tool_result("read_file", "x".repeat(filler)));
        }
        messages
    }

    #[test]
    fn a_short_conversation_is_untouched() {
        let mut messages = convo(2, 100);
        let before = messages.len();
        assert_eq!(fit(&mut messages, 8192), 0);
        assert_eq!(messages.len(), before);
    }

    #[test]
    fn a_long_conversation_is_trimmed_to_fit() {
        let mut messages = convo(40, 2_000);
        assert!(total_tokens(&messages) > 8192);
        let dropped = fit(&mut messages, 8192);

        assert!(dropped > 0, "nothing was dropped");
        assert!(
            total_tokens(&messages) <= (8192.0 * OCCUPANCY) as usize + 200,
            "still over budget: {}",
            total_tokens(&messages)
        );
    }

    #[test]
    fn the_task_is_never_dropped() {
        // The failure this exists to prevent: Ollama truncating from the front,
        // leaving no user turn, and answering 500.
        let mut messages = convo(60, 3_000);
        fit(&mut messages, 4096);

        assert_eq!(messages[0].role, "system", "system prompt was dropped");
        assert!(
            messages.iter().any(|m| m.role == "user" && m.content.contains("Refactor")),
            "the original instruction was dropped"
        );
    }

    #[test]
    fn the_most_recent_steps_survive() {
        let mut messages = convo(40, 2_000);
        let last = messages.last().cloned().unwrap();
        fit(&mut messages, 8192);

        assert_eq!(
            messages.last().unwrap().content,
            last.content,
            "the newest tool result was dropped"
        );
        assert!(messages.len() >= KEEP_RECENT);
    }

    #[test]
    fn the_model_is_told_that_history_was_dropped() {
        let mut messages = convo(40, 2_000);
        fit(&mut messages, 8192);
        assert!(
            messages.iter().any(|m| m.content.contains("were dropped")),
            "trimming happened silently"
        );
    }

    #[test]
    fn an_impossibly_small_window_still_keeps_the_instruction() {
        // Better to send something slightly oversized than something the model
        // cannot act on at all.
        let mut messages = convo(30, 5_000);
        fit(&mut messages, 512);
        assert_eq!(messages[0].role, "system");
        assert!(messages.iter().any(|m| m.content.contains("Refactor")));
    }

    #[test]
    fn trimming_is_idempotent() {
        let mut messages = convo(40, 2_000);
        fit(&mut messages, 8192);
        let after_first = messages.len();
        assert_eq!(fit(&mut messages, 8192), 0);
        assert_eq!(messages.len(), after_first);
    }
}
