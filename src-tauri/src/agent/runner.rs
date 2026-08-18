//! The agent loop.
//!
//! Holds the model, the tools, and the permission engine together for one task.
//! Most of the code here exists for failure modes that only show up on small
//! models — a 7B recovers from a confusing tool result on its own; a 0.5B will
//! call the same tool with the same arguments until the step budget runs out.
//!
//! Three guards, in the order they fire:
//!
//! 1. **Schema validation** before execution. A model that omits a required
//!    argument is told which one, by name, instead of receiving a generic tool
//!    failure it cannot act on.
//! 2. **Loop detection** on the call signature. Identical calls repeated past a
//!    threshold get one corrective message, then the run is abandoned.
//! 3. **Step budget.** A hard ceiling so a confused session ends in seconds
//!    rather than minutes.

use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::ollama::client::{OllamaClient, StreamEvent};
use crate::ollama::types::{ChatRequest, Message, Metrics, Options, Tool};
use crate::perms::{Action, Decision, Evaluation, Outcome, PermissionEngine};

use super::prompt::{compact_tools, select_tools, system_prompt, PromptTier};
use super::toolcall::{extract, unknown_tool_name, Recovery};

/// Executes tool calls. Implemented by the real tool surface and by test doubles.
#[async_trait::async_trait]
pub trait ToolExecutor: Send + Sync {
    /// Schemas advertised to the model.
    fn schemas(&self) -> Vec<Tool>;

    /// What this call would touch, so permissions can be judged before it runs.
    /// `None` means the call has no effect outside the process.
    fn action_for(&self, name: &str, args: &Map<String, Value>) -> Option<Action>;

    /// Run the call. `Err` is a tool-level failure the model should see and react
    /// to, not a transport error.
    async fn execute(&self, name: &str, args: &Map<String, Value>) -> Result<String, String>;

    /// Where a long-running tool sends output before it finishes.
    ///
    /// Default is a no-op, so a tool surface that has nothing to stream needs
    /// no code. `WorkspaceTools` uses it for `run_command`, which otherwise
    /// says nothing at all for the length of a build.
    fn set_progress(&mut self, _tx: mpsc::Sender<String>) {}
}

/// Settles `Ask` decisions. The UI implementation prompts the user; the
/// model-assisted approver may soften some to `Allow`; tests use fixed answers.
#[async_trait::async_trait]
pub trait ApprovalGate: Send + Sync {
    async fn decide(&self, call: &ProposedCall) -> Outcome;
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ProposedCall {
    pub id: String,
    pub step: usize,
    pub name: String,
    pub arguments: Map<String, Value>,
    pub evaluation: Evaluation,
    /// How the call was parsed. Anything but `Native` means the model did not
    /// use the API properly and the parser rescued it — surfaced in the UI.
    pub recovery: Recovery,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AgentEvent {
    Token { text: String },
    Thinking { text: String },
    ToolProposed { call: ProposedCall },
    ToolSettled { id: String, outcome: Outcome, output: String },
    /// Output from a command that is still running.
    ToolProgress { text: String },
    StepDone { step: usize, metrics: Metrics },
    Done { reason: StopReason },
    Error { message: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Model answered without calling a tool.
    Completed,
    /// Hit `max_steps`.
    BudgetExhausted,
    /// Same call repeated past the threshold.
    LoopDetected,
    /// Transport or daemon failure.
    Failed,
    /// The user pressed Stop.
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct RunConfig {
    pub model: String,
    pub max_steps: usize,
    pub num_ctx: Option<u32>,
    pub think: bool,
    pub keep_alive: String,
    /// Identical call signatures tolerated before the run is abandoned.
    pub loop_threshold: usize,
    /// Parameter count in billions, used to size the system prompt.
    pub params_billions: Option<f64>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            max_steps: 24,
            num_ctx: None,
            think: false,
            // Long enough that a multi-step task never pays a reload. A cold
            // 27B load costs ~30 s and dwarfs everything else in a turn.
            keep_alive: "15m".into(),
            loop_threshold: 3,
            params_billions: None,
        }
    }
}

pub struct AgentRunner {
    client: OllamaClient,
    perms: PermissionEngine,
    config: RunConfig,
    cancel: Arc<AtomicBool>,
}

impl AgentRunner {
    pub fn new(client: OllamaClient, perms: PermissionEngine, config: RunConfig) -> Self {
        Self { client, perms, config, cancel: Arc::new(AtomicBool::new(false)) }
    }

    /// Share a cancellation flag with the caller, so Stop can reach a run that
    /// is already in flight.
    pub fn with_cancel(mut self, cancel: Arc<AtomicBool>) -> Self {
        self.cancel = cancel;
        self
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// Run until the model stops calling tools, the budget runs out, or it loops.
    pub async fn run(
        &self,
        user_message: &str,
        tools: &mut dyn ToolExecutor,
        gate: &dyn ApprovalGate,
        events: mpsc::Sender<AgentEvent>,
    ) -> StopReason {
        // Relay command output as it arrives.
        let (progress_tx, mut progress_rx) = mpsc::channel::<String>(256);
        let progress_events = events.clone();
        let progress_pump = tokio::spawn(async move {
            while let Some(text) = progress_rx.recv().await {
                let _ = progress_events.send(AgentEvent::ToolProgress { text }).await;
            }
        });

        let tier = PromptTier::for_params(self.config.params_billions);
        // Trim to what this model can choose between, then shorten what is left.
        let schemas = compact_tools(&select_tools(&tools.schemas(), tier), tier);
        let known: Vec<String> = schemas.iter().map(|t| t.function.name.clone()).collect();

        tools.set_progress(progress_tx);
        let _ = &progress_pump;

        let workspace = self.perms.workspace().display().to_string();
        let mut messages = vec![
            Message::system(system_prompt(tier, &workspace, &schemas)),
            Message::user(user_message),
        ];

        let mut signature_counts: HashMap<String, usize> = HashMap::new();

        for step in 0..self.config.max_steps {
            if self.cancelled() {
                let _ = events.send(AgentEvent::Done { reason: StopReason::Cancelled }).await;
                return StopReason::Cancelled;
            }

            let request = ChatRequest {
                model: self.config.model.clone(),
                messages: messages.clone(),
                stream: true,
                tools: Some(schemas.clone()),
                think: self.config.think.then_some(true),
                format: None,
                options: Some(Options { num_ctx: self.config.num_ctx, ..Default::default() }),
                keep_alive: Some(self.config.keep_alive.clone()),
            };

            let (tx, mut rx) = mpsc::channel::<StreamEvent>(64);
            let forward = events.clone();
            // Relay tokens to the caller while the request is still in flight.
            let pump = tokio::spawn(async move {
                while let Some(ev) = rx.recv().await {
                    let out = match ev {
                        StreamEvent::Token { text } => Some(AgentEvent::Token { text }),
                        StreamEvent::Thinking { text } => Some(AgentEvent::Thinking { text }),
                        StreamEvent::Error { message } => Some(AgentEvent::Error { message }),
                        StreamEvent::Done { .. } => None,
                    };
                    if let Some(out) = out {
                        let _ = forward.send(out).await;
                    }
                }
            });

            let response = self
                .client
                .chat_stream_cancellable(request, tx, self.cancel.clone())
                .await;
            let _ = pump.await;

            let response = match response {
                Ok(r) => r,
                Err(crate::ollama::OllamaError::Cancelled) => {
                    let _ = events.send(AgentEvent::Done { reason: StopReason::Cancelled }).await;
                    return StopReason::Cancelled;
                }
                Err(e) => {
                    let _ = events.send(AgentEvent::Error { message: e.to_string() }).await;
                    let _ = events.send(AgentEvent::Done { reason: StopReason::Failed }).await;
                    return StopReason::Failed;
                }
            };

            let assistant = response.message.clone().unwrap_or_default();
            let _ = events
                .send(AgentEvent::StepDone { step, metrics: response.metrics() })
                .await;

            let Some(found) = extract(&assistant.content, assistant.tool_calls.as_deref(), &known)
            else {
                // A call-shaped reply naming a tool that does not exist is a
                // mistake to correct, not an answer. Treating it as an answer
                // is how a session ends having done nothing.
                if let Some(invented) = unknown_tool_name(&assistant.content) {
                    let msg = format!(
                        "There is no tool called `{invented}`. The tools you can use are: \
                         {}. Call one of those, or answer without a tool.",
                        known.join(", ")
                    );
                    messages.push(Message {
                        role: "assistant".into(),
                        content: assistant.content.clone(),
                        ..Default::default()
                    });
                    messages.push(Message::user(&msg));
                    let _ = events
                        .send(AgentEvent::ToolSettled {
                            id: String::new(),
                            outcome: Outcome::Failed,
                            output: msg,
                        })
                        .await;
                    continue;
                }

                // No tool call at all: the model is answering, so the task is done.
                messages.push(assistant);
                let _ = events.send(AgentEvent::Done { reason: StopReason::Completed }).await;
                return StopReason::Completed;
            };

            // Record what the model said with the call syntax stripped out, so
            // the transcript does not carry raw JSON as if it were prose.
            messages.push(Message {
                role: "assistant".into(),
                content: found.residual_text.clone(),
                tool_calls: Some(found.calls.clone()),
                ..Default::default()
            });

            for call in &found.calls {
                if self.cancelled() {
                    let _ = events.send(AgentEvent::Done { reason: StopReason::Cancelled }).await;
                    return StopReason::Cancelled;
                }
                let name = call.function.name.clone();
                let args = call.function.arguments_object();

                // --- Guard 1: required arguments ----------------------------
                if let Some(missing) = missing_required(&schemas, &name, &args) {
                    let msg = format!(
                        "Error: `{name}` requires the argument `{missing}`, which was not \
                         provided. Call it again with `{missing}` set."
                    );
                    messages.push(Message::tool_result(&name, &msg));
                    let _ = events
                        .send(AgentEvent::ToolSettled {
                            id: String::new(),
                            outcome: Outcome::Failed,
                            output: msg,
                        })
                        .await;
                    continue;
                }

                // --- Guard 2: repetition ------------------------------------
                let signature = call_signature(&name, &args);
                let seen = signature_counts.entry(signature).or_insert(0);
                *seen += 1;
                if *seen > self.config.loop_threshold {
                    let msg = format!(
                        "You have called `{name}` with identical arguments {seen} times. \
                         The result will not change. Use the result you already have, or \
                         answer without calling a tool."
                    );
                    messages.push(Message::tool_result(&name, &msg));
                    let _ = events
                        .send(AgentEvent::Done { reason: StopReason::LoopDetected })
                        .await;
                    return StopReason::LoopDetected;
                }

                // --- Guard 3: permission ------------------------------------
                let evaluation = match tools.action_for(&name, &args) {
                    Some(action) => self.perms.evaluate(&action),
                    None => Evaluation::allow_internal("Tool has no effect outside the app."),
                };

                let proposed = ProposedCall {
                    id: uuid::Uuid::new_v4().to_string(),
                    step,
                    name: name.clone(),
                    arguments: args.clone(),
                    evaluation: evaluation.clone(),
                    recovery: found.recovery,
                };
                let _ = events
                    .send(AgentEvent::ToolProposed { call: proposed.clone() })
                    .await;

                let outcome = match evaluation.decision {
                    Decision::Allow => Outcome::AutoApproved,
                    Decision::Deny => Outcome::Denied,
                    Decision::Ask => gate.decide(&proposed).await,
                };

                let output = match outcome {
                    Outcome::AutoApproved | Outcome::UserApproved => {
                        match tools.execute(&name, &args).await {
                            Ok(out) => out,
                            Err(e) => format!("Error: {e}"),
                        }
                    }
                    Outcome::Denied => {
                        format!("Refused: {}. Do not retry this call.", evaluation.reason)
                    }
                    Outcome::UserRejected => {
                        "The user declined this action. Ask what to do instead.".into()
                    }
                    Outcome::Failed => "The tool failed to run.".into(),
                };

                messages.push(Message::tool_result(&name, &output));
                let _ = events
                    .send(AgentEvent::ToolSettled {
                        id: proposed.id,
                        outcome,
                        output: output.clone(),
                    })
                    .await;
            }
        }

        let _ = events
            .send(AgentEvent::Done { reason: StopReason::BudgetExhausted })
            .await;
        StopReason::BudgetExhausted
    }
}

/// Stable identity for a call, used to spot repetition.
fn call_signature(name: &str, args: &Map<String, Value>) -> String {
    // Ordering the keys makes the signature independent of emission order, so
    // the same call written two ways still counts as a repeat.
    let ordered: std::collections::BTreeMap<_, _> = args.iter().collect();
    format!("{name}::{}", serde_json::to_string(&ordered).unwrap_or_default())
}

/// First required argument the call omitted, if any.
fn missing_required(schemas: &[Tool], name: &str, args: &Map<String, Value>) -> Option<String> {
    let tool = schemas.iter().find(|t| t.function.name == name)?;
    let required = tool.function.parameters.get("required")?.as_array()?;

    required
        .iter()
        .filter_map(Value::as_str)
        .find(|key| match args.get(*key) {
            None | Some(Value::Null) => true,
            // An empty string is almost always a model failing to fill a slot
            // rather than a deliberate value.
            Some(Value::String(s)) => s.is_empty(),
            _ => false,
        })
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn args(v: Value) -> Map<String, Value> {
        serde_json::from_value(v).expect("object")
    }

    fn schema() -> Vec<Tool> {
        vec![Tool::function(
            "read_file",
            "Read a file.",
            json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
        )]
    }

    #[test]
    fn signature_ignores_key_order() {
        let a = args(json!({"path": "a", "mode": "r"}));
        let b = args(json!({"mode": "r", "path": "a"}));
        assert_eq!(call_signature("read", &a), call_signature("read", &b));
    }

    #[test]
    fn signature_separates_different_arguments() {
        let a = args(json!({"path": "a"}));
        let b = args(json!({"path": "b"}));
        assert_ne!(call_signature("read", &a), call_signature("read", &b));
    }

    #[test]
    fn signature_separates_different_tools() {
        let a = args(json!({"path": "a"}));
        assert_ne!(call_signature("read", &a), call_signature("write", &a));
    }

    #[test]
    fn missing_argument_is_named() {
        assert_eq!(
            missing_required(&schema(), "read_file", &Map::new()).as_deref(),
            Some("path")
        );
    }

    #[test]
    fn empty_string_counts_as_missing() {
        let a = args(json!({"path": ""}));
        assert_eq!(missing_required(&schema(), "read_file", &a).as_deref(), Some("path"));
    }

    #[test]
    fn a_complete_call_passes_validation() {
        let a = args(json!({"path": "a.rs"}));
        assert_eq!(missing_required(&schema(), "read_file", &a), None);
    }

    #[test]
    fn unknown_tool_reports_no_missing_argument() {
        // Name resolution already rejected it; this must not panic.
        assert_eq!(missing_required(&schema(), "nope", &Map::new()), None);
    }
}
