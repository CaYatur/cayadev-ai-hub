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

impl Timeouts {
    /// Budgets scaled to how fast the model actually runs.
    ///
    /// `tokens_per_second` is what the model was measured at, or an estimate
    /// from its placement. This exists because one number cannot serve both
    /// ends of the range: `qwen2.5-coder:7b` decodes at 42 tok/s and a
    /// three-minute silence from it means something is broken, while
    /// `qwen3.8:27b` spilled to system RAM decodes at 5.4 and a single long
    /// reply is a quarter of an hour of work — during which a fixed 180 s
    /// budget cut it off and lost everything it had done.
    ///
    /// The idle budget is the time to produce [`STALL_TOKENS`] tokens at that
    /// rate: enough that a working model is never mistaken for a dead one, and
    /// short enough that a genuinely dead connection is still noticed.
    pub fn for_throughput(tokens_per_second: f64) -> Self {
        /// Tokens' worth of silence tolerated before calling it a stall.
        ///
        /// The unit is work, not time, because the silence is not really "no
        /// tokens" — it is tokens being generated and not sent. Measured on
        /// `qwen3.8:27b-mtp-q4_K_M`: reasoning streams normally, then the whole
        /// tool call is buffered and arrives as one chunk. Asked for a
        /// single-file HTML page it went **547 seconds** between the last
        /// reasoning chunk and that one, and a 180 s budget threw the work away
        /// twice. 4096 covers a long file written in one call.
        const STALL_TOKENS: f64 = 4096.0;
        /// Never shorter than this, however fast the model is.
        const MIN_IDLE: u64 = 180;
        /// Never longer than this, however slow: past half an hour of silence
        /// the run is not worth holding open.
        const MAX_IDLE: u64 = 1800;

        let rate = tokens_per_second.max(0.5);
        let idle = (STALL_TOKENS / rate).round() as u64;
        Self {
            connect: 5,
            // Loading 17 GB of weights from disk and prefilling a full window
            // is minutes on a spilled model, and it is not a failure.
            first_token: 900,
            idle: idle.clamp(MIN_IDLE, MAX_IDLE),
        }
    }
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
    fn a_spilled_model_gets_the_silence_it_actually_needs() {
        // The measured failure: qwen3.8:27b spilled to system RAM decodes at
        // 5.4 tok/s and went 547 seconds between its last reasoning chunk and
        // the buffered tool call carrying a whole HTML file. A budget under
        // that throws the work away.
        let t = Timeouts::for_throughput(5.4);
        assert!(t.idle > 547, "budget {} would have cut off a real run", t.idle);
    }

    #[test]
    fn a_resident_model_is_not_given_a_quarter_of_an_hour() {
        // At 42 tok/s a three-minute silence is a fault, not work in progress.
        let t = Timeouts::for_throughput(42.3);
        assert_eq!(t.idle, 180);
    }

    #[test]
    fn the_budget_is_bounded_at_both_ends() {
        // A nonsensical rate must not produce a zero budget or an endless one.
        assert_eq!(Timeouts::for_throughput(0.0).idle, 1800);
        assert_eq!(Timeouts::for_throughput(-5.0).idle, 1800);
        assert_eq!(Timeouts::for_throughput(f64::MAX).idle, 180);
        assert!(Timeouts::for_throughput(1.0).idle <= 1800);
    }

    #[test]
    fn slower_models_always_get_at_least_as_long() {
        let mut previous = 0;
        for rate in [200.0, 60.0, 40.0, 20.0, 10.0, 5.0, 2.0, 1.0] {
            let idle = Timeouts::for_throughput(rate).idle;
            assert!(idle >= previous, "{rate} tok/s got {idle}s, less than a faster model");
            previous = idle;
        }
    }

    #[test]
    fn loading_a_large_model_is_never_the_thing_that_fails() {
        // Weights come off disk once per run at worst; fifteen minutes covers a
        // 17 GB model and a full prefill.
        assert!(Timeouts::for_throughput(5.0).first_token >= 900);
    }
}
