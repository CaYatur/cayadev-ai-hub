//! Wire types for the Ollama HTTP API.
//!
//! Response structs are deliberately permissive: every field Ollama may omit is
//! `Option` or defaulted, because older daemons and cloud-hosted models leave
//! different subsets populated. A missing counter must never fail a stream.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Message {
    pub role: String,
    pub content: String,

    /// Base64-encoded images, for vision-capable models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,

    /// Separated reasoning trace on thinking-capable models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Set on `role: "tool"` replies so the model can match result to call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into(), ..Default::default() }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into(), ..Default::default() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into(), ..Default::default() }
    }

    pub fn tool_result(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: content.into(),
            tool_name: Some(name.into()),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolFunction,
}

impl Tool {
    pub fn function(name: &str, description: &str, parameters: serde_json::Value) -> Self {
        Self {
            kind: "function".into(),
            function: ToolFunction {
                name: name.into(),
                description: description.into(),
                parameters,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// Ollama sends an object here, but weaker models sometimes emit a JSON
    /// *string*. `arguments_object` normalises both into a map.
    pub arguments: serde_json::Value,
}

impl ToolCallFunction {
    /// Best-effort coercion of `arguments` into an object.
    ///
    /// Handles the three shapes seen in practice: a real object, a JSON string
    /// holding an object, and a bare scalar (wrapped under `"input"`).
    pub fn arguments_object(&self) -> serde_json::Map<String, serde_json::Value> {
        match &self.arguments {
            serde_json::Value::Object(map) => map.clone(),
            serde_json::Value::String(s) => serde_json::from_str(s)
                .ok()
                .and_then(|v: serde_json::Value| v.as_object().cloned())
                .unwrap_or_default(),
            serde_json::Value::Null => serde_json::Map::new(),
            other => {
                let mut m = serde_json::Map::new();
                m.insert("input".into(), other.clone());
                m
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Generation options
// ---------------------------------------------------------------------------

/// Sampling and runtime knobs passed through to the model runner.
///
/// Every field is optional so a partially-specified profile leaves the rest at
/// the daemon's own defaults rather than overriding them with zeros.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Options {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_gpu: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_thread: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_batch: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_last_n: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flash_attn: Option<bool>,
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub stream: bool,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<Options>,
    /// How long to hold the model in memory after the request. Keeping this
    /// generous is the difference between a 0.3s and a 30s turn on big models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ChatResponse {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub message: Option<Message>,
    #[serde(default)]
    pub done: bool,
    #[serde(default)]
    pub done_reason: Option<String>,

    // Timing counters, nanoseconds. Absent on intermediate stream chunks.
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u32>,
    #[serde(default)]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default)]
    pub eval_count: Option<u32>,
    #[serde(default)]
    pub eval_duration: Option<u64>,
}

/// Throughput figures derived from a completed response, in the units a user
/// actually reads.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metrics {
    pub decode_tps: f64,
    pub prefill_tps: f64,
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    pub load_seconds: f64,
    pub ttft_seconds: f64,
    pub total_seconds: f64,
}

impl ChatResponse {
    pub fn metrics(&self) -> Metrics {
        let ns = |v: Option<u64>| v.unwrap_or(0) as f64 / 1e9;
        let load = ns(self.load_duration);
        let prefill_s = ns(self.prompt_eval_duration);
        let decode_s = ns(self.eval_duration);
        let out = self.eval_count.unwrap_or(0);
        let prompt = self.prompt_eval_count.unwrap_or(0);

        Metrics {
            decode_tps: if decode_s > 0.0 { out as f64 / decode_s } else { 0.0 },
            prefill_tps: if prefill_s > 0.0 { prompt as f64 / prefill_s } else { 0.0 },
            prompt_tokens: prompt,
            output_tokens: out,
            load_seconds: load,
            ttft_seconds: load + prefill_s,
            total_seconds: ns(self.total_duration),
        }
    }
}

// ---------------------------------------------------------------------------
// Model introspection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ListResponse {
    #[serde(default)]
    pub models: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelEntry {
    pub name: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub digest: String,
    #[serde(default)]
    pub details: Option<ModelDetails>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ModelDetails {
    #[serde(default)]
    pub parent_model: String,
    #[serde(default)]
    pub format: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub parameter_size: String,
    #[serde(default)]
    pub quantization_level: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ShowResponse {
    /// e.g. `["completion", "tools", "vision", "thinking"]`
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub details: Option<ModelDetails>,
    /// Raw GGUF metadata. Keys are architecture-prefixed, so the context length
    /// lives at `"<arch>.context_length"` rather than a fixed key.
    #[serde(default)]
    pub model_info: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub template: String,
}

impl ShowResponse {
    pub fn has(&self, cap: &str) -> bool {
        self.capabilities.iter().any(|c| c == cap)
    }

    pub fn architecture(&self) -> Option<String> {
        self.model_info
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .map(str::to_string)
    }

    /// The model's true maximum context, read from GGUF metadata.
    ///
    /// This is the number users need and almost no UI surfaces — it is what the
    /// weights were trained for, not whatever the daemon happens to allocate.
    pub fn max_context(&self) -> Option<u64> {
        let arch = self.architecture()?;
        self.model_info
            .get(&format!("{arch}.context_length"))
            .and_then(serde_json::Value::as_u64)
    }

    pub fn param_count(&self) -> Option<u64> {
        self.model_info
            .get("general.parameter_count")
            .and_then(serde_json::Value::as_u64)
    }

    fn info_u64(&self, suffix: &str) -> Option<u64> {
        let arch = self.architecture()?;
        self.model_info
            .get(&format!("{arch}.{suffix}"))
            .and_then(serde_json::Value::as_u64)
    }

    /// Bytes of KV cache per token at f16, or `None` when metadata is missing.
    ///
    /// Hybrid architectures interleave attention with SSM/linear layers and
    /// report `full_attention_interval`; only that fraction of layers holds a
    /// KV cache. Ignoring this overestimates the cache by 4x or more and leads
    /// to advice ("shrink your context!") that does nothing.
    pub fn kv_bytes_per_token(&self) -> Option<u64> {
        let layers = self.info_u64("block_count")?;
        let kv_heads = self.info_u64("attention.head_count_kv")?;

        let key_len = self
            .info_u64("attention.key_length")
            .or_else(|| {
                let embed = self.info_u64("embedding_length")?;
                let heads = self.info_u64("attention.head_count")?;
                Some(embed / heads.max(1))
            })
            .unwrap_or(128);
        let val_len = self.info_u64("attention.value_length").unwrap_or(key_len);

        let attn_layers = match self.info_u64("full_attention_interval") {
            Some(interval) if interval > 1 => layers.div_ceil(interval),
            _ => layers,
        };

        // 2 bytes per f16 element, keys + values.
        Some(attn_layers * kv_heads * (key_len + val_len) * 2)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct PsResponse {
    #[serde(default)]
    pub models: Vec<PsEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PsEntry {
    pub name: String,
    #[serde(default)]
    pub model: String,
    /// Total footprint the runner reserved, including KV cache.
    #[serde(default)]
    pub size: u64,
    /// Portion of `size` resident in VRAM. The gap between the two is what
    /// spilled to system RAM — the single best predictor of slow decoding.
    #[serde(default)]
    pub size_vram: u64,
    #[serde(default)]
    pub context_length: Option<u64>,
    #[serde(default)]
    pub expires_at: Option<String>,
}

impl PsEntry {
    /// Fraction of the model resident in VRAM, 0.0–1.0.
    pub fn gpu_fraction(&self) -> f64 {
        if self.size == 0 {
            return 0.0;
        }
        (self.size_vram as f64 / self.size as f64).clamp(0.0, 1.0)
    }
}
