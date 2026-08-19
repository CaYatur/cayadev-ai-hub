//! HTTP client for the Ollama daemon.
//!
//! ## Why the timeout handling is unusual
//!
//! The obvious implementation gives the request a total timeout. That is wrong
//! for local inference and is a leading cause of "the agent died before it did
//! anything": a 27B model that spilled to system RAM can legitimately spend 90
//! seconds on prefill before emitting its first token, and a total timeout kills
//! it mid-thought. Worse, the failure looks like a hang rather than a timeout,
//! so users blame the model.
//!
//! Instead we bound the *gap between chunks*. A stream that is still producing
//! tokens is never cancelled no matter how long the whole turn takes; a stream
//! that has genuinely stalled is cut promptly. First-token and inter-token gaps
//! get separate budgets, because the first is expected to be slow.

use futures_util::StreamExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

use super::types::*;

#[derive(Debug, thiserror::Error)]
pub enum OllamaError {
    #[error("cannot reach Ollama at {url} — is the daemon running? ({source})")]
    Unreachable { url: String, source: reqwest::Error },

    #[error("Ollama returned {status}: {body}")]
    Status { status: u16, body: String },

    #[error("no output for {seconds}s (model: {model}) — the stream stalled")]
    Stalled { seconds: u64, model: String },

    #[error("transport error: {0}")]
    Transport(#[from] reqwest::Error),

    #[error("malformed response: {0}")]
    Decode(String),

    #[error("cancelled")]
    Cancelled,
}

pub type Result<T> = std::result::Result<T, OllamaError>;

/// Resolve as soon as `cancel` is set.
///
/// Checking the flag only between chunks is not enough: the longest waits are
/// the ones with nothing arriving — a cold model load, or a prefill on a
/// spilled 27B — and those are exactly when a person reaches for Stop. Racing
/// this against the read makes cancellation land within a tick instead of
/// whenever the next token happens to show up.
pub async fn cancelled(cancel: &AtomicBool) {
    while !cancel.load(Ordering::Relaxed) {
        tokio::time::sleep(Duration::from_millis(80)).await;
    }
}

/// Timeout budgets, all in seconds.
#[derive(Debug, Clone)]
pub struct Timeouts {
    /// TCP/handshake budget. Short — the daemon is on localhost.
    pub connect: u64,
    /// Budget for the *first* chunk. Covers model load plus prefill, both of
    /// which are slow and legitimate on a cold big model.
    pub first_token: u64,
    /// Budget for each subsequent chunk. Once decoding starts, tokens arrive
    /// steadily; a long gap here really is a stall.
    pub idle: u64,
}

impl Default for Timeouts {
    fn default() -> Self {
        // Sized for a model that fits in VRAM. A model that does not gets
        // [`Timeouts::for_throughput`] instead — see there for why the
        // difference is nearly an order of magnitude.
        Self { connect: 5, first_token: 600, idle: 180 }
    }
}

/// How often a wait reports that it is still waiting.
///
/// Short enough that the status line moves, long enough not to flood the
/// channel during a ten-minute generation.
const HEARTBEAT: Duration = Duration::from_secs(10);

/// Silence tolerated before the daemon is checked at all.
///
/// Short gaps are ordinary. Asking about every one of them would put an HTTP
/// request between every few tokens of a slow model.
const LIVENESS_AFTER: Duration = Duration::from_secs(60);

/// How long between those checks once they start.
const LIVENESS_EVERY: Duration = Duration::from_secs(60);

impl Timeouts {
    /// Budgets that cover the longest reply this model is allowed to produce.
    ///
    /// A silence is not evidence of failure. Some templates buffer a whole tool
    /// call and send it in one piece — measured on `qwen3.8:27b-mtp-q4_K_M`,
    /// which streams its reasoning and then says nothing at all while it writes
    /// the file. So the question is not "how long is too long" but "how long
    /// *can* this legitimately take", and that has an exact answer: the reply
    /// is capped at `max_tokens`, and the model decodes at `tokens_per_second`.
    ///
    /// Guessing a constant instead does not work. 4096 tokens' worth was tried
    /// and the same model was still cut off at 623 s, because nothing bounded
    /// how much it could generate — without an explicit `num_predict` a reply
    /// may run until the context window fills, which at 65k tokens and 5.7
    /// tok/s is over three hours.
    ///
    /// `tokens_per_second` is what the model was measured at, or an estimate
    /// from its placement. The spread is the reason this is a function at all:
    /// `qwen2.5-coder:7b` decodes at 42 tok/s and three minutes of silence from
    /// it means something is broken, while the same silence from a 27B running
    /// out of system RAM is a third of one reply.
    pub fn for_generation(tokens_per_second: f64, max_tokens: u32) -> Self {
        /// Slack over the rate we think the model runs at.
        ///
        /// Deliberately large, because the two errors are not symmetric. Being
        /// generous costs a wait the user can cancel at any moment — the status
        /// line says what is happening and Stop works. Being tight costs an
        /// hour of finished work, and that has now happened twice: once at 623 s
        /// and again at 1800 s, both on a model that was working the whole time.
        ///
        /// It is a factor rather than a better estimate because the rate is not
        /// knowable in advance. It varies within a single run — the same model
        /// measured 6.4 tok/s on its short steps and 4.0 while writing a long
        /// file — and it varies with whatever else is using the card. Three
        /// times covers that spread without pretending to predict it.
        const MARGIN: f64 = 3.0;
        /// Never shorter than this, however fast the model is.
        const MIN_IDLE: u64 = 180;
        /// Never longer than this.
        ///
        /// This is a guard against a nonsensical rate, not a policy about how
        /// long a user should wait. It was 2400 s, and that cut off a real run:
        /// a 27B decoding at 4 tok/s cannot finish the reply it is allowed to
        /// produce in that time, so the cap fired before the model could
        /// possibly have been done. A cap below the honest bound is the worst of
        /// both — the user waits *and* loses the work.
        ///
        /// Detecting a genuinely dead run is not this number's job. The daemon
        /// is asked every minute whether it still has the model loaded, and a
        /// run whose model has gone fails at once regardless of the budget.
        ///
        /// Set to cover the slowest rate worth calling a working model: one
        /// token a second, producing the whole allowance, is 8192 s. Below that
        /// the machine cannot run this model usefully at all and the cap is
        /// academic.
        const MAX_IDLE: u64 = 9000;

        let rate = tokens_per_second.max(0.5);
        let idle = (f64::from(max_tokens) / rate * MARGIN).round() as u64;
        Self {
            connect: 5,
            // Loading 17 GB of weights from disk and prefilling a full window is
            // minutes on a spilled model, and it is not a failure. Measured at
            // 424 s for a cold load plus 300 tokens.
            first_token: 900,
            idle: idle.clamp(MIN_IDLE, MAX_IDLE),
        }
    }
}

/// The reply length to allow, given the window.
///
/// Bounding this is what makes a silence measurable at all — with no cap a
/// reply may run until the context fills. It also has to be generous: a
/// truncated tool call comes back as *nothing*. Measured on
/// `qwen3.8:27b-mtp-q4_K_M` with `num_predict: 300`: `done_reason: "length"`,
/// `eval_count: 300`, and no content, no reasoning and no tool call at all —
/// the half-written call is discarded rather than repaired. So the cap has to
/// sit above any file the model would realistically write in one go; a
/// 199-line HTML page measured about 2500 tokens.
pub fn reply_budget(num_ctx: Option<u32>) -> u32 {
    /// Enough for a long single-file page with inline styles.
    const MAX_REPLY: u32 = 8192;
    /// Below this the cap starts truncating ordinary work.
    const MIN_REPLY: u32 = 1024;

    // Half the window, so the conversation still has room to be sent.
    num_ctx.map_or(MAX_REPLY, |ctx| (ctx / 2).clamp(MIN_REPLY, MAX_REPLY))
}

/// One event from a streaming chat turn.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Incremental visible text.
    Token { text: String },
    /// Incremental reasoning text, on thinking-capable models.
    Thinking { text: String },
    /// Nothing has arrived for a while, and the run is still waiting.
    ///
    /// Not an error. Emitted so a silence has a visible reason: a model loading,
    /// or one generating a tool call it will not send until it is complete.
    Silence { seconds: u64, first_token: bool },
    /// Terminal event carrying the assembled message and timings.
    Done { message: Message, metrics: Metrics, done_reason: Option<String> },
    Error { message: String },
}

#[derive(Clone)]
pub struct OllamaClient {
    base_url: String,
    http: reqwest::Client,
    timeouts: Timeouts,
}

impl OllamaClient {
    pub fn new(base_url: impl Into<String>, timeouts: Timeouts) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(timeouts.connect))
            // Deliberately no global .timeout(): streams are bounded per-chunk.
            .pool_idle_timeout(Duration::from_secs(300))
            .build()
            .expect("reqwest client construction cannot fail with these options");
        Self { base_url: base_url.into(), http, timeouts }
    }

    /// The same client with different budgets.
    ///
    /// The connection pool is shared — `reqwest::Client` is a handle — so this
    /// is cheap enough to do per run, which is what sizing budgets to the
    /// chosen model requires.
    pub fn with_timeouts(&self, timeouts: Timeouts) -> Self {
        Self { base_url: self.base_url.clone(), http: self.http.clone(), timeouts }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    async fn post_json<B: serde::Serialize, R: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<R> {
        let url = format!("{}{path}", self.base_url);
        let resp = self
            .http
            .post(&url)
            .json(body)
            .timeout(Duration::from_secs(120))
            .send()
            .await
            .map_err(|e| OllamaError::Unreachable { url: url.clone(), source: e })?;

        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(OllamaError::Status { status: status.as_u16(), body: text });
        }
        serde_json::from_str(&text).map_err(|e| OllamaError::Decode(format!("{e}: {text}")))
    }

    pub async fn version(&self) -> Result<String> {
        let url = format!("{}/api/version", self.base_url);
        let resp = self
            .http
            .get(&url)
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .map_err(|e| OllamaError::Unreachable { url, source: e })?;
        let v: serde_json::Value = resp.json().await?;
        Ok(v.get("version").and_then(|s| s.as_str()).unwrap_or("unknown").to_string())
    }

    pub async fn list(&self) -> Result<ListResponse> {
        let url = format!("{}/api/tags", self.base_url);
        let resp = self
            .http
            .get(&url)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| OllamaError::Unreachable { url, source: e })?;
        Ok(resp.json().await?)
    }

    pub async fn show(&self, model: &str) -> Result<ShowResponse> {
        self.post_json("/api/show", &serde_json::json!({ "model": model })).await
    }

    /// Currently loaded models, with the VRAM/RAM split the UI reports.
    pub async fn ps(&self) -> Result<PsResponse> {
        let url = format!("{}/api/ps", self.base_url);
        let resp = self
            .http
            .get(&url)
            .timeout(Duration::from_secs(15))
            .send()
            .await
            .map_err(|e| OllamaError::Unreachable { url, source: e })?;
        Ok(resp.json().await?)
    }

    /// Non-streaming chat. Used for short internal calls (routing, classifying)
    /// where incremental output has no value.
    pub async fn chat(&self, req: ChatRequest) -> Result<ChatResponse> {
        self.chat_cancellable(req, &AtomicBool::new(false)).await
    }

    /// Non-streaming chat that gives up as soon as `cancel` is set.
    ///
    /// Used for the classifier and for vision calls, both of which can take
    /// long enough on a busy GPU that a user presses Stop during one.
    pub async fn chat_cancellable(
        &self,
        mut req: ChatRequest,
        cancel: &AtomicBool,
    ) -> Result<ChatResponse> {
        req.stream = false;
        let url = format!("{}/api/chat", self.base_url);

        // Nothing streams back from a non-streaming call, so without racing the
        // request there is no point at which Stop could be noticed.
        let resp = tokio::select! {
            biased;
            _ = cancelled(cancel) => return Err(OllamaError::Cancelled),
            sent = self
                .http
                .post(&url)
                .json(&req)
                // A whole non-streaming turn still needs the first-token budget.
                .timeout(Duration::from_secs(self.timeouts.first_token))
                .send() =>
            {
                sent.map_err(|e| OllamaError::Unreachable { url, source: e })?
            }
        };

        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(OllamaError::Status { status: status.as_u16(), body: text });
        }
        serde_json::from_str(&text).map_err(|e| OllamaError::Decode(format!("{e}: {text}")))
    }

    /// Streaming chat. Events are pushed to `tx` as they arrive.
    ///
    /// The returned future completes when the stream ends; the caller drives the
    /// receiver. Each chunk read is wrapped in its own timeout so a stalled
    /// stream fails fast while a slow-but-alive one runs as long as it needs.
    pub async fn chat_stream(
        &self,
        req: ChatRequest,
        tx: mpsc::Sender<StreamEvent>,
    ) -> Result<ChatResponse> {
        self.chat_stream_cancellable(req, tx, Arc::new(AtomicBool::new(false)))
            .await
    }

    /// Streaming chat that stops as soon as `cancel` is set.
    ///
    /// The flag is checked before each chunk rather than only between steps.
    /// Dropping the response body closes the connection, so the daemon stops
    /// generating too — a Stop button that only takes effect at the end of the
    /// current turn is not a Stop button on a model producing five tokens a
    /// second.
    pub async fn chat_stream_cancellable(
        &self,
        mut req: ChatRequest,
        tx: mpsc::Sender<StreamEvent>,
        cancel: Arc<AtomicBool>,
    ) -> Result<ChatResponse> {
        req.stream = true;
        let model = req.model.clone();
        let url = format!("{}/api/chat", self.base_url);

        let resp = tokio::select! {
            biased;
            _ = cancelled(&cancel) => return Err(OllamaError::Cancelled),
            sent = self.http.post(&url).json(&req).send() => {
                sent.map_err(|e| OllamaError::Unreachable { url, source: e })?
            }
        };

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(OllamaError::Status { status: status.as_u16(), body });
        }

        let mut stream = resp.bytes_stream();
        let mut buf = Vec::<u8>::new();
        let mut content = String::new();
        let mut thinking = String::new();
        let mut tool_calls: Option<Vec<ToolCall>> = None;
        let mut final_resp = ChatResponse::default();
        let mut seen_any = false;
        let started = Instant::now();

        loop {
            let budget = if seen_any { self.timeouts.idle } else { self.timeouts.first_token };

            // Wait for the next chunk, saying so while it does not come.
            //
            // A silence here is usually the model working, not the model dying:
            // some templates buffer a whole tool call and send it in one piece,
            // so a file being written produces nothing at all until it is
            // finished. Waiting the entire budget in silence makes a working
            // app look hung, so the wait is broken into ticks and each one
            // reports how long it has been quiet.
            let silent_since = Instant::now();
            // When the daemon was last asked whether it still has the model.
            let mut checked: Option<Instant> = None;
            let next = loop {
                // `biased` so cancellation wins a tie rather than being subject
                // to random branch order.
                let tick = tokio::select! {
                    biased;
                    _ = cancelled(&cancel) => {
                        // Dropping `stream` here closes the connection, so the
                        // daemon stops generating too.
                        return Err(OllamaError::Cancelled);
                    }
                    read = tokio::time::timeout(HEARTBEAT, stream.next()) => read,
                };
                match tick {
                    Ok(chunk) => break Some(chunk),
                    Err(_) => {
                        let quiet = silent_since.elapsed().as_secs();
                        if quiet >= budget {
                            break None;
                        }
                        // Waiting out the whole budget is right when the model
                        // is working and wrong when the daemon has gone. It
                        // cannot be asked whether it is generating — a loaded
                        // model looks the same either way — but it can be asked
                        // whether the model is loaded at all, and while a
                        // request is in flight it must be. If it has vanished,
                        // waiting another twenty minutes proves nothing.
                        let due = quiet >= LIVENESS_AFTER.as_secs()
                            && checked.is_none_or(|at: Instant| at.elapsed() >= LIVENESS_EVERY);
                        if due {
                            checked = Some(Instant::now());
                            // Raced against cancellation as well: a fifteen
                            // second HTTP timeout in here would be fifteen
                            // seconds where Stop does nothing.
                            let alive = tokio::select! {
                                biased;
                                _ = cancelled(&cancel) => return Err(OllamaError::Cancelled),
                                probe = self.ps() => probe,
                            };
                            if let Ok(ps) = alive {
                                if !ps.models.iter().any(|m| m.name == model) {
                                    let _ = tx
                                        .send(StreamEvent::Error {
                                            message: format!(
                                                "The daemon no longer has {model} loaded, so \
                                                 nothing is generating. It was probably \
                                                 unloaded to make room, or the daemon restarted."
                                            ),
                                        })
                                        .await;
                                    return Err(OllamaError::Stalled { seconds: quiet, model });
                                }
                            }
                        }
                        let _ = tx
                            .send(StreamEvent::Silence { seconds: quiet, first_token: !seen_any })
                            .await;
                    }
                }
            };

            let chunk = match next {
                None => {
                    // The two cases want different advice. Nothing at all means
                    // the weights are still being loaded, or do not fit; a gap
                    // part-way through means the model is generating something
                    // it does not stream, or it is being paged.
                    let message = if seen_any {
                        format!(
                            "Nothing further for {budget}s. Some models send a whole \
                             tool call in one piece, so writing a long file is silent \
                             — but not this long. A model that does not fit in VRAM \
                             is the usual reason."
                        )
                    } else {
                        format!(
                            "Nothing back after {budget}s. The model is still loading, \
                             or it does not fit in this machine's VRAM and is running \
                             from system memory."
                        )
                    };
                    let _ = tx.send(StreamEvent::Error { message }).await;
                    return Err(OllamaError::Stalled { seconds: budget, model });
                }
                Some(None) => break,
                Some(Some(Err(e))) => return Err(OllamaError::Transport(e)),
                Some(Some(Ok(bytes))) => bytes,
            };

            seen_any = true;
            buf.extend_from_slice(&chunk);

            // Ollama streams newline-delimited JSON; a chunk may split a line.
            while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=nl).collect();
                let line = String::from_utf8_lossy(&line[..line.len() - 1]);
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                let parsed: ChatResponse = match serde_json::from_str(line) {
                    Ok(v) => v,
                    // A malformed line is not worth killing a working stream over.
                    Err(_) => continue,
                };

                if let Some(msg) = &parsed.message {
                    if !msg.content.is_empty() {
                        content.push_str(&msg.content);
                        let _ = tx.send(StreamEvent::Token { text: msg.content.clone() }).await;
                    }
                    if let Some(t) = &msg.thinking {
                        if !t.is_empty() {
                            thinking.push_str(t);
                            let _ = tx.send(StreamEvent::Thinking { text: t.clone() }).await;
                        }
                    }
                    if let Some(tc) = &msg.tool_calls {
                        if !tc.is_empty() {
                            tool_calls.get_or_insert_with(Vec::new).extend(tc.clone());
                        }
                    }
                }

                if parsed.done {
                    final_resp = parsed;
                }
            }
        }

        let message = Message {
            role: "assistant".into(),
            content: content.clone(),
            thinking: (!thinking.is_empty()).then_some(thinking),
            tool_calls: tool_calls.clone(),
            ..Default::default()
        };

        let mut metrics = final_resp.metrics();
        // Some builds omit total_duration on streamed turns; wall time is a
        // truthful stand-in and keeps the UI from showing 0.00s.
        if metrics.total_seconds == 0.0 {
            metrics.total_seconds = started.elapsed().as_secs_f64();
        }

        final_resp.message = Some(message.clone());
        let _ = tx
            .send(StreamEvent::Done {
                message,
                metrics,
                done_reason: final_resp.done_reason.clone(),
            })
            .await;

        Ok(final_resp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_working_model_is_never_cut_off_before_it_could_have_finished() {
        // The reply is capped at `reply_budget`, so the longest a legitimate
        // silence can last is that many tokens at whatever rate the model
        // manages. A budget below that number cannot be right — it fires while
        // the model is still allowed to be producing.
        // From one token a second — below which the model is unusable anyway,
        // and the cap stops trying to cover it — up to a resident model.
        for rate in [1.0, 2.0, 4.0, 6.4, 12.0, 42.0] {
            let cap = reply_budget(Some(65_536));
            let needed = (f64::from(cap) / rate) as u64;
            let t = Timeouts::for_generation(rate, cap);
            assert!(
                t.idle >= needed,
                "at {rate} tok/s the model may legitimately take {needed}s,                  budget was {}",
                t.idle
            );
        }
    }

    #[test]
    fn the_two_runs_that_were_cut_off_would_now_survive() {
        // Both reported by the user, both on qwen3.8:27b at 65k context. The
        // first died after 623 s of silence, the second after about 1800 s.
        let cap = reply_budget(Some(65_536));
        // The rate its long step actually ran at, from the status line.
        let t = Timeouts::for_generation(4.0, cap);
        assert!(t.idle > 1800, "still under the 1800s that failed: {}", t.idle);
        assert!(t.idle > 623, "still under the 623s that failed: {}", t.idle);
    }

    #[test]
    fn a_resident_model_is_not_given_two_hours() {
        // Ten minutes of silence from a 42 tok/s model is a fault, not work.
        let t = Timeouts::for_generation(42.3, reply_budget(Some(32_768)));
        assert!(t.idle <= 900, "budget {} is far longer than this model needs", t.idle);
    }

    #[test]
    fn the_budget_is_bounded_at_both_ends() {
        // A nonsensical rate must not produce a zero budget or an endless one.
        assert_eq!(Timeouts::for_generation(0.0, 8192).idle, 9000);
        assert_eq!(Timeouts::for_generation(-5.0, 8192).idle, 9000);
        assert_eq!(Timeouts::for_generation(f64::MAX, 8192).idle, 180);
    }

    #[test]
    fn slower_models_always_get_at_least_as_long() {
        let mut previous = 0;
        for rate in [200.0, 60.0, 40.0, 20.0, 10.0, 5.0, 2.0, 1.0] {
            let idle = Timeouts::for_generation(rate, 8192).idle;
            assert!(idle >= previous, "{rate} tok/s got {idle}s, less than a faster model");
            previous = idle;
        }
    }

    #[test]
    fn loading_a_large_model_is_never_the_thing_that_fails() {
        // Measured: a cold 17 GB load plus 300 tokens took 424 s.
        assert!(Timeouts::for_generation(5.0, 8192).first_token >= 900);
    }

    #[test]
    fn the_reply_cap_leaves_room_for_the_conversation() {
        // Half the window, so what is sent and what comes back both fit.
        assert_eq!(reply_budget(Some(32_768)), 8192);
        assert_eq!(reply_budget(Some(16_384)), 8192);
        assert_eq!(reply_budget(Some(8_192)), 4096);
        assert!(reply_budget(Some(8_192)) < 8_192);
    }

    #[test]
    fn the_reply_cap_never_gets_small_enough_to_truncate_ordinary_work() {
        // A truncated tool call comes back as nothing at all — measured with
        // `num_predict: 300`, which returned no content, no reasoning and no
        // call. Cutting a reply short is worse than taking a while.
        assert_eq!(reply_budget(Some(512)), 1024);
        assert_eq!(reply_budget(Some(1)), 1024);
        assert_eq!(reply_budget(None), 8192);
    }
}
