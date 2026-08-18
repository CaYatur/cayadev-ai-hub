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
//! So the trimming happens here instead, and it drops from the middle. Three
//! things are never removed: the system prompt, the current instruction, and
//! the most recent exchanges — the first two state the task, the last is what
//! the next step reasons from. What goes is the long stretch of earlier
//! conversation and old tool output nobody needs any more, replaced by a note
//! saying how much was dropped so the model knows its history is incomplete
//! rather than silently misremembering.
//!
//! The current instruction is pinned by index rather than by "the first user
//! message", because a conversation carries earlier turns before it. Pinning
//! the first user message would protect something the user wrote an hour ago
//! and drop what they just asked for.

use crate::agent::prompt::estimate_tokens;
use crate::ollama::types::Message;

/// Fraction of the window the conversation may occupy before trimming.
///
/// The rest is headroom for the reply. Filling the window entirely leaves the
/// model no room to answer, which reads as a stall.
const OCCUPANCY: f64 = 0.65;

/// What the conversation is trimmed *down* to, once it has to be trimmed.
///
/// Not the same number as [`OCCUPANCY`], deliberately. Trimming to exactly the
/// threshold means the next step crosses it again and trims again, so every
/// single step changes the messages behind the system prompt and the daemon
/// throws away its KV cache each time. Cutting well under the line buys several
/// steps that only append, and appending is what the cache is built for.
const TARGET: f64 = 0.45;

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

/// What a trim did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Trim {
    /// Messages removed.
    pub dropped: usize,
    /// Where the current instruction ended up. Indices before it shift when
    /// messages are removed, so the caller has to be told.
    pub task: usize,
}

/// Trim a conversation to fit, dropping from the middle.
///
/// `task` is the index of the message holding the current instruction. It
/// survives regardless of how old it gets — without that, a long run drops the
/// thing it is trying to do and the model starts improvising.
pub fn fit(messages: &mut Vec<Message>, num_ctx: u32, task: usize) -> Trim {
    let ceiling = (num_ctx as f64 * OCCUPANCY) as usize;
    let mut task = task.min(messages.len().saturating_sub(1));
    if total_tokens(messages) <= ceiling {
        return Trim { dropped: 0, task };
    }
    let budget = (num_ctx as f64 * TARGET) as usize;

    // The system prompt states how to behave; everything after it is
    // conversation and can go, except the instruction and the recent tail.
    let head = usize::from(messages.first().is_some_and(|m| m.role == "system"));

    let mut dropped = 0usize;
    let mut cut = head;
    while total_tokens(messages) > budget {
        let floor = messages.len().saturating_sub(KEEP_RECENT);
        if cut >= floor {
            // Nothing left that may be dropped. Better slightly oversized than
            // missing the instruction.
            break;
        }
        if cut == task {
            cut += 1;
            continue;
        }
        messages.remove(cut);
        if task > cut {
            task -= 1;
        }
        dropped += 1;
    }

    if dropped > 0 {
        // Say what was lost. A model that knows its history is partial asks or
        // re-reads; one that does not will confidently misremember.
        messages.insert(
            head,
            Message::user(format!(
                "[{dropped} earlier steps were dropped to stay within the context                  window. Re-read any file you need rather than relying on memory                  of it.]"
            )),
        );
        if task >= head {
            task += 1;
        }
    }
    Trim { dropped, task }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A run with no prior conversation: system, task, then steps.
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

    /// A run that carries prior turns: system, history, then the task.
    /// Returns the conversation and the index of the current instruction.
    fn with_history(turns: usize, filler: usize, steps: usize) -> (Vec<Message>, usize) {
        let mut messages = vec![Message::system("You are a coding assistant.")];
        for i in 0..turns {
            messages.push(Message::user(format!("older question {i} {}", "y".repeat(filler))));
            messages.push(Message::assistant(format!("older answer {i} {}", "y".repeat(filler))));
        }
        messages.push(Message::user("Refactor the parser."));
        let task = messages.len() - 1;
        for i in 0..steps {
            messages.push(Message::assistant(format!("step {i}")));
            messages.push(Message::tool_result("read_file", "x".repeat(filler)));
        }
        (messages, task)
    }

    #[test]
    fn a_short_conversation_is_untouched() {
        let mut messages = convo(2, 100);
        let before = messages.len();
        assert_eq!(fit(&mut messages, 8192, 1).dropped, 0);
        assert_eq!(messages.len(), before);
    }

    #[test]
    fn a_long_conversation_is_trimmed_to_fit() {
        let mut messages = convo(40, 2_000);
        assert!(total_tokens(&messages) > 8192);
        let trim = fit(&mut messages, 8192, 1);

        assert!(trim.dropped > 0, "nothing was dropped");
        assert!(
            total_tokens(&messages) <= (8192.0 * OCCUPANCY) as usize + 200,
            "still over budget: {}",
            total_tokens(&messages)
        );
    }

    #[test]
    fn trimming_leaves_room_to_grow_before_the_next_trim() {
        // Cutting to exactly the threshold would trim again on the very next
        // step, and a conversation that changes behind the system prompt every
        // step is a KV cache thrown away every step.
        let mut messages = convo(40, 2_000);
        let mut trim = fit(&mut messages, 8192, 1);
        assert!(trim.dropped > 0);

        // Several more steps of ordinary size must fit without another trim.
        for i in 0..3 {
            messages.push(Message::assistant(format!("later {i}")));
            messages.push(Message::tool_result("read_file", "z".repeat(1_000)));
            trim = fit(&mut messages, 8192, trim.task);
            assert_eq!(trim.dropped, 0, "trimmed again after only {i} more steps");
        }
    }

    #[test]
    fn the_task_is_never_dropped() {
        // The failure this exists to prevent: Ollama truncating from the front,
        // leaving no user turn, and answering 500.
        let mut messages = convo(60, 3_000);
        let trim = fit(&mut messages, 4096, 1);

        assert_eq!(messages[0].role, "system", "system prompt was dropped");
        assert!(messages[trim.task].content.contains("Refactor"));
    }

    #[test]
    fn the_current_instruction_outlives_the_conversation_it_arrived_in() {
        // The discriminating case. With prior turns in front of it the task is
        // no longer the first user message, so protecting "the first user
        // message" would protect an hour-old question and drop what was just
        // asked. Fifty turns of history plus a long run, into a small window.
        let (mut messages, task) = with_history(50, 1_500, 20);
        assert!(total_tokens(&messages) > 8192);

        let trim = fit(&mut messages, 8192, task);

        assert!(trim.dropped > 0);
        assert_eq!(messages[0].role, "system");
        assert_eq!(
            messages[trim.task].content, "Refactor the parser.",
            "the reported task index does not point at the instruction"
        );
        assert!(
            messages.iter().any(|m| m.content == "Refactor the parser."),
            "the current instruction was dropped"
        );
    }

    #[test]
    fn old_turns_go_before_recent_ones() {
        let (mut messages, task) = with_history(30, 1_500, 4);
        let trim = fit(&mut messages, 8192, task);
        assert!(trim.dropped > 0);
        assert!(
            !messages.iter().any(|m| m.content.starts_with("older question 0")),
            "the oldest turn survived while newer ones were dropped"
        );
        assert!(
            messages.iter().any(|m| m.content.starts_with("step 3")),
            "the newest step was dropped"
        );
    }

    #[test]
    fn the_most_recent_steps_survive() {
        let mut messages = convo(40, 2_000);
        let last = messages.last().cloned().unwrap();
        fit(&mut messages, 8192, 1);

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
        fit(&mut messages, 8192, 1);
        assert!(
            messages.iter().any(|m| m.content.contains("were dropped")),
            "trimming happened silently"
        );
    }

    #[test]
    fn an_impossibly_small_window_still_keeps_the_instruction() {
        // Better to send something slightly oversized than something the model
        // cannot act on at all.
        let (mut messages, task) = with_history(20, 4_000, 10);
        let trim = fit(&mut messages, 512, task);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[trim.task].content, "Refactor the parser.");
    }

    #[test]
    fn trimming_is_idempotent() {
        let mut messages = convo(40, 2_000);
        let first = fit(&mut messages, 8192, 1);
        let after_first = messages.len();
        let second = fit(&mut messages, 8192, first.task);
        assert_eq!(second.dropped, 0);
        assert_eq!(messages.len(), after_first);
    }

    #[test]
    fn a_task_index_past_the_end_does_not_panic() {
        // Defensive: the caller tracks the index across appends and removals,
        // and a wrong one must degrade rather than crash a run.
        let mut messages = convo(40, 2_000);
        let trim = fit(&mut messages, 8192, 9_999);
        assert!(trim.task < messages.len());
    }
}
