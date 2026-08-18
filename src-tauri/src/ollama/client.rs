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
        // 10 min covers a cold 27B load (~30 s) plus a long prefill with wide
        // margin. The inter-token budget is 180 s rather than 60 because a
        // spilled 27B running at 4 tok/s with a full context pauses longer than
        // expected between chunks, and cutting it there lost real work.
        Self { connect: 5, first_token: 600, idle: 180 }
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

            // `biased` so cancellation wins a tie rather than being subject to
            // random branch order.
            let next = tokio::select! {
                biased;
                _ = cancelled(&cancel) => {
                    // Dropping `stream` here closes the connection, so the
                    // daemon stops generating too.
                    return Err(OllamaError::Cancelled);
                }
                read = tokio::time::timeout(Duration::from_secs(budget), stream.next()) => read,
            };

            let chunk = match next {
                Err(_) => {
                    let _ = tx
                        .send(StreamEvent::Error {
                            message: format!(
                                "No output for {budget}s. The model may be loading or the \
                                 request may exceed available memory."
                            ),
                        })
                        .await;
                    return Err(OllamaError::Stalled { seconds: budget, model });
                }
                Ok(None) => break,
                Ok(Some(Err(e))) => return Err(OllamaError::Transport(e)),
                Ok(Some(Ok(bytes))) => bytes,
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
