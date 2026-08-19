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
//! 2. **Loop detection** on the call signature, remembered only until the run
//!    changes something. Identical calls repeated with nothing changed in
//!    between get one corrective message, then the run is abandoned.
//! 3. **Step ceiling.** A bound on a run nobody is watching. Not a judgement
//!    about whether the work is finished, so it sits far above what a real
//!    repair costs.

use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::ollama::client::{reply_budget, OllamaClient, StreamEvent};
use crate::ollama::types::{ChatRequest, Message, Metrics, Options, Tool};
use crate::perms::{Action, Decision, Evaluation, Outcome, PermissionEngine};

use super::context;
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
    /// The model has sent nothing for `seconds`, and the run is still waiting.
    ///
    /// Emitted so a long silence reads as work rather than as a hang: a big
    /// model loading, or one generating a tool call its template will not send
    /// until it is complete.
    Waiting { seconds: u64, first_token: bool },
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

/// One earlier exchange, replayed so the model can refer back to it.
///
/// Only the text: reasoning is left out because replaying chain-of-thought
/// burns window and confuses the templates that emit it, and tool output is
/// folded into the assistant's text by the caller as one summary line per call.
/// What the model needs from three questions ago is what was said and what was
/// done, not the four thousand characters a `read_file` returned.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Turn {
    pub role: String,
    pub text: String,
}

impl Turn {
    fn to_message(&self) -> Message {
        if self.role == "assistant" {
            Message::assistant(&self.text)
        } else {
            Message::user(&self.text)
        }
    }
}

#[derive(Debug, Clone)]
pub struct RunConfig {
    pub model: String,
    /// Tool steps before the run is cut off.
    ///
    /// A bound on a run nobody is watching, not a guess at how much work the
    /// task needs. Repairing a file is thirty to eighty steps of read, edit,
    /// run, read the traceback, edit again; sized under that, the ceiling
    /// stops runs that are working, which is exactly what it used to do.
    pub max_steps: usize,
    pub num_ctx: Option<u32>,
    pub think: bool,
    pub keep_alive: String,
    /// Identical call signatures tolerated before the run is abandoned.
    pub loop_threshold: usize,
    /// Parameter count in billions, used to size the system prompt.
    pub params_billions: Option<f64>,
    /// Earlier exchanges in this conversation, oldest first.
    ///
    /// Sent between the system prompt and the instruction, so the system
    /// prefix stays byte-stable and the daemon keeps reusing its KV cache.
    pub history: Vec<Turn>,
    /// Base64 images attached to this turn, carried on the instruction itself.
    pub images: Vec<String>,
    /// Whether to offer tools at all.
    ///
    /// Several vision models cannot call them, and Ollama refuses a request
    /// that offers tools to a model without the capability.
    pub tools_enabled: bool,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            max_steps: 80,
            num_ctx: None,
            think: false,
            // Long enough that a multi-step task never pays a reload. A cold
            // 27B load costs ~30 s and dwarfs everything else in a turn.
            keep_alive: "15m".into(),
            loop_threshold: 3,
            params_billions: None,
            history: Vec::new(),
            images: Vec::new(),
            tools_enabled: true,
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
        //
        // The pump holds a clone of `events`, so it has to be stopped when the
        // run ends — otherwise the event channel never closes, whoever is
        // draining it never sees the end of the stream, and the caller waits
        // forever. `run` returns from a dozen places, so the handle is dropped
        // by a guard rather than aborted at each one.
        let (progress_tx, mut progress_rx) = mpsc::channel::<String>(256);
        let progress_events = events.clone();
        let _progress = AbortOnDrop(tokio::spawn(async move {
            while let Some(text) = progress_rx.recv().await {
                let _ = progress_events.send(AgentEvent::ToolProgress { text }).await;
            }
        }));

        let tier = PromptTier::for_params(self.config.params_billions);
        // Trim to what this model can choose between, then shorten what is left.
        let schemas = if self.config.tools_enabled {
            compact_tools(&select_tools(&tools.schemas(), tier), tier)
        } else {
            Vec::new()
        };
        let known: Vec<String> = schemas.iter().map(|t| t.function.name.clone()).collect();

        tools.set_progress(progress_tx);

        let workspace = self.perms.workspace().display().to_string();
        let mut messages = vec![Message::system(system_prompt(tier, &workspace, &schemas))];
        messages.extend(self.config.history.iter().map(Turn::to_message));
        let mut instruction = Message::user(user_message);
        if !self.config.images.is_empty() {
            instruction.images = Some(self.config.images.clone());
        }
        messages.push(instruction);
        // Which message holds the instruction. Trimming removes from in front
        // of it, so the index moves and has to be carried rather than assumed.
        let mut task = messages.len() - 1;

        let mut repeats = Repetition::default();
        // Retries count consecutively, not per run. Two stalls in a row is a
        // real failure; a stall at step five and another at step seventy are
        // unrelated events, and an hour of finished work should not turn on
        // whether the first one happened.
        let mut stalled_in_a_row = 0usize;

        for step in 0..self.config.max_steps {
            if self.cancelled() {
                let _ = events.send(AgentEvent::Done { reason: StopReason::Cancelled }).await;
                return StopReason::Cancelled;
            }

            // Trim before sending, not after overflowing. Ollama truncates
            // from the front, which is where the system prompt and the task
            // are — and then its template reports "no user query found".
            if let Some(num_ctx) = self.config.num_ctx {
                let trim = context::fit(&mut messages, num_ctx, task);
                task = trim.task;
                if trim.dropped > 0 {
                    tracing::info!("dropped {} messages to fit {num_ctx} tokens", trim.dropped);
                }
            }

            let request = ChatRequest {
                model: self.config.model.clone(),
                messages: messages.clone(),
                stream: true,
                // An empty list is not the same as no list: Ollama rejects a
                // request that offers tools to a model without the capability.
                tools: (!schemas.is_empty()).then(|| schemas.clone()),
                think: self.config.think.then_some(true),
                format: None,
                options: Some(Options {
                    num_ctx: self.config.num_ctx,
                    // Capped so the wait for a silent model is bounded by
                    // something real rather than by a guess. See
                    // `ollama::reply_budget`.
                    num_predict: Some(reply_budget(self.config.num_ctx) as i32),
                    ..Default::default()
                }),
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
                        StreamEvent::Silence { seconds, first_token } => {
                            Some(AgentEvent::Waiting { seconds, first_token })
                        }
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
                // A stall is one of two things, and they want different
                // handling. Before any output, it is usually the conversation
                // having grown too long for the model to start producing, and
                // trimming recovers it. After output has already streamed, the
                // conversation is demonstrably fine — the model was generating
                // and did not finish in time — so trimming would only make the
                // retry say something different for no reason.
                Err(crate::ollama::OllamaError::Stalled { .. }) if stalled_in_a_row == 0 => {
                    stalled_in_a_row += 1;
                    let long = self
                        .config
                        .num_ctx
                        .is_some_and(|ctx| context::total_tokens(&messages) > (ctx / 2) as usize);
                    let message = if long {
                        if let Some(num_ctx) = self.config.num_ctx {
                            task = context::fit(&mut messages, num_ctx / 2, task).task;
                        }
                        "The model stopped producing output. Shortening the conversation \
                         and trying once more."
                    } else {
                        "The model stopped producing output. Trying once more."
                    };
                    let _ = events.send(AgentEvent::Error { message: message.into() }).await;
                    continue;
                }
                Err(e) => {
                    let _ = events.send(AgentEvent::Error { message: e.to_string() }).await;
                    let _ = events.send(AgentEvent::Done { reason: StopReason::Failed }).await;
                    return StopReason::Failed;
                }
            };

            // Any response at all counts, including a reply truncated at the
            // length cap: the daemon answered, so the previous stall was not
            // the start of a dead run. Alternating stall and truncation would
            // then retry indefinitely, and the step ceiling is what stops it.
            stalled_in_a_row = 0;

            let assistant = response.message.clone().unwrap_or_default();
            let _ = events
                .send(AgentEvent::StepDone { step, metrics: response.metrics() })
                .await;

            // A reply cut off at the length cap comes back empty — measured on
            // qwen3.8:27b with `num_predict: 300`: no content, no reasoning and
            // no tool call, the half-written call simply discarded. Silently
            // treating that as "the model had nothing to say" ends the run
            // having done nothing, so it is named instead.
            if response.done_reason.as_deref() == Some("length")
                && assistant.content.trim().is_empty()
                && assistant.tool_calls.as_ref().is_none_or(|c| c.is_empty())
            {
                let _ = events
                    .send(AgentEvent::Error {
                        message: format!(
                            "The reply hit the {} token limit before the model finished,                              so nothing usable came back. Ask for a smaller piece of work,                              or raise the context window.",
                            reply_budget(self.config.num_ctx)
                        ),
                    })
                    .await;
                let _ = events.send(AgentEvent::Done { reason: StopReason::Failed }).await;
                return StopReason::Failed;
            }

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
                let seen = repeats.record(&signature);
                if seen > self.config.loop_threshold {
                    let msg = format!(
                        "You have called `{name}` with identical arguments {seen} times and \
                         nothing has changed in between. The result will not change either. \
                         Use the result you already have, or answer without calling a tool."
                    );
                    messages.push(Message::tool_result(&name, &msg));
                    let _ = events
                        .send(AgentEvent::Done { reason: StopReason::LoopDetected })
                        .await;
                    return StopReason::LoopDetected;
                }

                // --- Guard 3: permission ------------------------------------
                let action = tools.action_for(&name, &args);
                let evaluation = match &action {
                    Some(action) => self.perms.evaluate(action),
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
                    // Waiting on a human is unbounded, so Stop has to reach
                    // into this too — otherwise cancelling an unanswered
                    // prompt does nothing.
                    Decision::Ask => tokio::select! {
                        biased;
                        _ = crate::ollama::client::cancelled(&self.cancel) => {
                            let _ = events
                                .send(AgentEvent::Done { reason: StopReason::Cancelled })
                                .await;
                            return StopReason::Cancelled;
                        }
                        answer = gate.decide(&proposed) => answer,
                    },
                };

                let output = match outcome {
                    Outcome::AutoApproved | Outcome::UserApproved => {
                        // A build, a transcription, or a vision call is often the
                        // longest part of a step, and Stop has to reach into it.
                        let ran = tokio::select! {
                            biased;
                            _ = crate::ollama::client::cancelled(&self.cancel) => None,
                            result = tools.execute(&name, &args) => Some(result),
                        };
                        match ran {
                            None => {
                                let _ = events
                                    .send(AgentEvent::Done { reason: StopReason::Cancelled })
                                    .await;
                                return StopReason::Cancelled;
                            }
                            Some(Ok(out)) => {
                                // The world moved. A command that failed a
                                // moment ago may well succeed now, so what the
                                // repetition guard remembers starts again here.
                                if action.as_ref().is_some_and(Action::changes_something) {
                                    repeats.changed(&signature);
                                }
                                out
                            }
                            Some(Err(e)) => format!("Error: {e}"),
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

/// Stops a background task when it goes out of scope.
///
/// Used for the progress relay, which holds a clone of the event sender. A
/// relay that outlives its run keeps the event channel open, and the caller
/// draining that channel then waits for an end that never arrives.
struct AbortOnDrop(tokio::task::JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Which calls have been made, so a model going in circles is caught.
///
/// The guard tells the model "the result will not change", and that is only
/// true while nothing has changed. Counted cumulatively across a whole run it
/// ended runs that were working: repairing a file means running the same
/// failing command after every edit, and the fourth run tripped a threshold
/// meant for a model asking the same question over and over.
///
/// So a call that changes something clears the memory — apart from its own
/// count, which would otherwise let a call excuse its own repetition forever.
///
/// The limit this leaves: two calls that both change something can alternate
/// without either count climbing — edit, run, edit, run. That is also exactly
/// what repairing a file looks like, so the guard cannot tell the two apart
/// and does not try. The step ceiling is what bounds it.
#[derive(Default)]
struct Repetition {
    counts: HashMap<String, usize>,
}

impl Repetition {
    /// Count this call, and report how many times it has now been made.
    fn record(&mut self, signature: &str) -> usize {
        let seen = self.counts.entry(signature.to_string()).or_insert(0);
        *seen += 1;
        *seen
    }

    /// Something changed, by way of the call `signature` names.
    fn changed(&mut self, signature: &str) {
        self.counts.retain(|call, _| call == signature);
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

    /// The run this came from: a 20B repairing a Python file, editing it and
    /// running it again after each edit. Counted cumulatively, the fourth run
    /// of the same command was a loop and the repair was abandoned half done.
    #[test]
    fn re_running_a_command_after_an_edit_is_not_a_loop() {
        let mut repeats = Repetition::default();
        let run = "run_command::python ringtone_generator.py";
        for attempt in 0..12 {
            assert_eq!(repeats.record(run), 1, "run {attempt} counted as a repeat");
            let edit = format!("write_file::attempt {attempt}");
            repeats.record(&edit);
            repeats.changed(&edit);
        }
    }

    #[test]
    fn the_same_call_with_nothing_changed_still_counts_up() {
        let mut repeats = Repetition::default();
        let call = "read_file::a.py";
        assert_eq!(repeats.record(call), 1);
        assert_eq!(repeats.record(call), 2);
        assert_eq!(repeats.record(call), 3);
        assert_eq!(repeats.record(call), 4);
    }

    #[test]
    fn a_change_does_not_excuse_its_own_repetition() {
        // Writing the identical file over and over is a loop even though every
        // write changes something; it must not be able to clear its own count.
        let mut repeats = Repetition::default();
        let write = "write_file::same content";
        for expected in 1..=4 {
            assert_eq!(repeats.record(write), expected);
            repeats.changed(write);
        }
    }

    #[test]
    fn reading_around_does_not_clear_the_memory() {
        // Only a change clears it, and the runner only reports one for a call
        // that was more than read-only.
        let mut repeats = Repetition::default();
        assert_eq!(repeats.record("read::a"), 1);
        repeats.record("read::b");
        repeats.record("search::todo");
        assert_eq!(repeats.record("read::a"), 2);
    }

    /// The ceiling is a bound on an unattended run, not a guess at the size of
    /// the job. The run that prompted this spent twenty-four steps in fifteen
    /// minutes and was not close to done.
    #[test]
    fn the_ceiling_is_well_above_what_a_repair_costs() {
        assert!(RunConfig::default().max_steps >= 60);
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
