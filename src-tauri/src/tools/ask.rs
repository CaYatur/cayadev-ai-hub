//! Asking the user a question mid-run.
//!
//! An agent that guesses when it should ask produces confident wrong work. The
//! usual failure is small and expensive: which of three config files did you
//! mean, should this go in the existing module or a new one, what should the
//! command be called. Guessing costs a whole run; asking costs a click.
//!
//! This is deliberately not the approval gate. Approval answers "may I do
//! this?" about something the model already decided; this answers "which?" or
//! "what?" about something it has not. The plumbing is the same shape — a
//! question goes to the UI, the run waits, an answer comes back — but the
//! question is the model's, not the permission engine's.

use serde::Serialize;

/// A question put to the user.
#[derive(Debug, Clone, Serialize)]
pub struct Question {
    pub id: String,
    pub question: String,
    /// Choices to offer. Empty means the answer is free text.
    pub options: Vec<String>,
    /// Whether more than one option may be picked.
    pub multiple: bool,
}

/// What the user said back.
///
/// Skipping is a real answer and is reported as one: the model needs to know
/// it was declined so it can proceed on its own judgement, rather than reading
/// an empty string as agreement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Answer {
    Given(String),
    Skipped,
}

impl Answer {
    /// How the answer reads to the model, as tool output.
    pub fn as_tool_output(&self) -> String {
        match self {
            Self::Given(text) => text.clone(),
            Self::Skipped => {
                "The user skipped the question. Decide for yourself and say what you assumed."
                    .into()
            }
        }
    }
}

/// Somewhere to put a question. Implemented by the UI bridge.
#[async_trait::async_trait]
pub trait Asker: Send + Sync {
    async fn ask(&self, question: Question) -> Answer;
}

/// Read the options out of whatever the model produced.
///
/// Models are inconsistent here in the ways they are inconsistent everywhere:
/// a JSON array, a comma-separated string, a single string. All three mean the
/// same thing to the user, so all three are accepted.
pub fn options_from(value: Option<&serde_json::Value>) -> Vec<String> {
    /// More than this is a list nobody wants to read in a dialogue.
    const MAX_OPTIONS: usize = 8;
    /// A whole paragraph as a "choice" is the model misusing the tool.
    const MAX_OPTION_CHARS: usize = 120;

    let raw: Vec<String> = match value {
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .map(|i| match i {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .collect(),
        Some(serde_json::Value::String(text)) => {
            text.split(&[',', '|', '\n'][..]).map(str::to_string).collect()
        }
        _ => Vec::new(),
    };

    raw.into_iter()
        .map(|s| s.trim().trim_matches('"').to_string())
        .filter(|s| !s.is_empty() && s.chars().count() <= MAX_OPTION_CHARS)
        .take(MAX_OPTIONS)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_json_array_is_the_expected_shape() {
        let value = json!(["src/main.rs", "src/lib.rs"]);
        assert_eq!(options_from(Some(&value)), vec!["src/main.rs", "src/lib.rs"]);
    }

    #[test]
    fn a_comma_separated_string_is_accepted_too() {
        let value = json!("keep it, rewrite it, ask me later");
        assert_eq!(
            options_from(Some(&value)),
            vec!["keep it", "rewrite it", "ask me later"]
        );
    }

    #[test]
    fn a_single_choice_still_works() {
        assert_eq!(options_from(Some(&json!("yes"))), vec!["yes"]);
    }

    #[test]
    fn no_options_means_free_text() {
        assert!(options_from(None).is_empty());
        assert!(options_from(Some(&json!([]))).is_empty());
        assert!(options_from(Some(&json!(""))).is_empty());
    }

    #[test]
    fn a_runaway_list_is_cut_to_something_readable() {
        let many: Vec<String> = (0..40).map(|i| format!("option {i}")).collect();
        assert_eq!(options_from(Some(&json!(many))).len(), 8);
    }

    #[test]
    fn a_paragraph_is_not_a_choice() {
        let value = json!(["fine", "x".repeat(400)]);
        assert_eq!(options_from(Some(&value)), vec!["fine"]);
    }

    #[test]
    fn quotes_and_padding_are_stripped() {
        let value = json!("  \"first\" ,  second  ");
        assert_eq!(options_from(Some(&value)), vec!["first", "second"]);
    }

    #[test]
    fn skipping_tells_the_model_to_proceed() {
        let output = Answer::Skipped.as_tool_output();
        assert!(output.contains("skipped"));
        assert!(!output.is_empty(), "an empty answer reads as agreement");
    }
}
