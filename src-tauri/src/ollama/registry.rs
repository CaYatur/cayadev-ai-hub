//! Model registry: capability detection, memory-fit estimation, and routing.
//!
//! The measurements in `docs/BENCHMARKS.md` drive everything here. The headline
//! result — a 7B model resident in VRAM decodes ~8x faster than a 27B model
//! that spilled to system RAM — means the highest-leverage decision the app
//! makes is *which model runs this step*, not how a single model is tuned.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::client::{OllamaClient, Result};
use super::types::{ModelEntry, ShowResponse};

/// What a model can be asked to do, resolved from Ollama's own capability list.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Capabilities {
    pub completion: bool,
    pub tools: bool,
    pub vision: bool,
    pub thinking: bool,
    pub embedding: bool,
}

impl From<&ShowResponse> for Capabilities {
    fn from(s: &ShowResponse) -> Self {
        Self {
            completion: s.has("completion"),
            tools: s.has("tools"),
            vision: s.has("vision"),
            thinking: s.has("thinking"),
            embedding: s.has("embedding"),
        }
    }
}

/// Where a model's weights will actually live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Placement {
    /// Fits in VRAM with headroom. Full speed.
    Resident,
    /// Fits only if little else is using the GPU. Speed depends on the desktop.
    Marginal,
    /// Will spill to system RAM. Expect single-digit tok/s on a laptop GPU.
    Spilled,
    /// Runs on Ollama's servers; local memory is irrelevant.
    Cloud,
    /// Not enough metadata to say.
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub name: String,
    pub family: String,
    pub parameter_size: String,
    pub quantization: String,
    /// On-disk size of the weights, bytes.
    pub weights_bytes: u64,
    pub capabilities: Capabilities,
    /// Context the weights were trained for, from GGUF metadata.
    pub max_context: Option<u64>,
    /// KV cache cost per token at f16, accounting for hybrid attention layers.
    pub kv_bytes_per_token: Option<u64>,
    pub architecture: Option<String>,
    pub placement: Placement,
    /// Largest context that still leaves the model resident in VRAM, if known.
    pub recommended_context: Option<u64>,
    /// Measured decode rate once observed, so the UI stops guessing.
    pub observed_decode_tps: Option<f64>,
}

impl ModelProfile {
    pub fn is_cloud(&self) -> bool {
        self.name.ends_with(":cloud") || self.placement == Placement::Cloud
    }

    /// Rough parameter count parsed from the "27.3B" style label.
    pub fn params_billions(&self) -> Option<f64> {
        let s = self.parameter_size.trim().trim_end_matches(['B', 'b']);
        s.parse::<f64>().ok()
    }
}

/// GPU memory budget used for fit decisions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VramBudget {
    pub total_bytes: u64,
    /// Reserved for the compositor, browsers, and everything else on the
    /// desktop. Ignoring this is why naive estimators promise a fit that then
    /// spills in practice.
    pub reserved_bytes: u64,
}

impl VramBudget {
    pub fn available(&self) -> u64 {
        self.total_bytes.saturating_sub(self.reserved_bytes)
    }
}

impl Default for VramBudget {
    fn default() -> Self {
        // Overridden at runtime by a real GPU probe; this is only a fallback.
        Self { total_bytes: 8 * 1024 * 1024 * 1024, reserved_bytes: 2 * 1024 * 1024 * 1024 }
    }
}

/// Estimate where a model will run and how much context keeps it there.
///
/// Returns `(placement, recommended_context)`.
pub fn estimate_placement(
    weights_bytes: u64,
    kv_per_token: Option<u64>,
    max_context: Option<u64>,
    budget: VramBudget,
) -> (Placement, Option<u64>) {
    let available = budget.available();

    // Runner overhead: compute buffers, the graph, and fragmentation.
    const OVERHEAD: u64 = 600 * 1024 * 1024;
    let weights_with_overhead = weights_bytes.saturating_add(OVERHEAD);

    if weights_with_overhead >= available {
        return (Placement::Spilled, None);
    }

    let kv_room = available - weights_with_overhead;
    let ceiling = max_context.unwrap_or(32_768);

    let recommended = match kv_per_token {
        Some(per_token) if per_token > 0 => {
            let fits = kv_room / per_token;
            // Round down to a power-of-two-ish step so the UI shows tidy values.
            let stepped = [131_072u64, 65_536, 32_768, 16_384, 8_192, 4_096, 2_048]
                .into_iter()
                .find(|&step| step <= fits.min(ceiling));
            stepped.or(Some(2_048))
        }
        // Without KV metadata, stay conservative rather than promise a fit.
        _ => Some(ceiling.min(8_192)),
    };

    // Under 15% headroom, a browser or a game will push this out of VRAM.
    let placement = if weights_with_overhead as f64 > available as f64 * 0.85 {
        Placement::Marginal
    } else {
        Placement::Resident
    };

    (placement, recommended)
}

/// The kind of work a step needs, which decides how much model it deserves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskClass {
    /// Multi-step planning, architecture, hard debugging.
    Reasoning,
    /// Writing and editing code. The common case.
    Coding,
    /// Mechanical transformation with a clear spec.
    Edit,
    /// Yes/no and pick-one decisions: approve this call, classify this intent.
    /// Runs constantly, so it must be nearly free.
    Classify,
    /// Anything involving an image.
    Vision,
    /// Summarising, titling, commit messages.
    Summarize,
}

impl TaskClass {
    /// Does this class require the model to support tool calling?
    pub fn needs_tools(&self) -> bool {
        matches!(self, Self::Reasoning | Self::Coding | Self::Edit)
    }
}

/// Which model to use for each task class.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoutingPolicy {
    pub assignments: HashMap<String, String>,
    /// When true, the router prefers a VRAM-resident model even if a larger
    /// spilled model is nominally more capable.
    pub prefer_resident: bool,
}

impl RoutingPolicy {
    fn key(class: TaskClass) -> &'static str {
        match class {
            TaskClass::Reasoning => "reasoning",
            TaskClass::Coding => "coding",
            TaskClass::Edit => "edit",
            TaskClass::Classify => "classify",
            TaskClass::Vision => "vision",
            TaskClass::Summarize => "summarize",
        }
    }

    pub fn get(&self, class: TaskClass) -> Option<&str> {
        self.assignments.get(Self::key(class)).map(String::as_str)
    }

    pub fn set(&mut self, class: TaskClass, model: impl Into<String>) {
        self.assignments.insert(Self::key(class).into(), model.into());
    }

    /// Build a sensible default policy from what the user actually has
    /// installed, so a fresh install is fast without being configured.
    ///
    /// The rules follow the benchmark: prefer resident models everywhere, spend
    /// the smallest capable model on classification, and reserve a spilled
    /// model for reasoning only when nothing resident can do the job.
    pub fn derive(profiles: &[ModelProfile]) -> Self {
        let mut policy = Self { prefer_resident: true, ..Default::default() };

        let candidates: Vec<&ModelProfile> = profiles.iter().filter(|p| !p.is_cloud()).collect();

        // Rank placement above size. A 4.6 GB model that only just fits is a
        // worse pick than a 4.4 GB one with headroom, and both beat a 17 GB
        // model that spilled — measured at ~8x slower. Sorting by size alone
        // inverts all of that.
        fn score(p: &&ModelProfile) -> (u8, u64) {
            (placement_rank(p.placement), p.weights_bytes)
        }

        let workhorse = candidates
            .iter()
            .copied()
            .filter(|p| p.capabilities.tools)
            .max_by_key(|p| score(&p));

        // Reasoning prefers a thinking model, but only one that is not a large
        // step down in capability. Without that guard a 0.8B thinker outranks a
        // 7B workhorse for planning, which is exactly backwards.
        let reasoner = workhorse
            .and_then(|best| {
                candidates
                    .iter()
                    .copied()
                    .filter(|p| p.capabilities.tools && p.capabilities.thinking)
                    .filter(|p| placement_rank(p.placement) >= placement_rank(best.placement))
                    .filter(|p| p.weights_bytes * 2 >= best.weights_bytes)
                    .max_by_key(|p| score(&p))
            })
            .or(workhorse);

        // Classification runs constantly, so it takes the cheapest model that
        // is resident — speed matters far more than capability here.
        let classifier = candidates
            .iter()
            .copied()
            .filter(|p| p.capabilities.completion && is_resident(p.placement))
            .min_by_key(|p| p.weights_bytes)
            .or(workhorse);

        // Vision is ranked differently from everything else: best placement
        // first, then the *smallest* model rather than the largest.
        //
        // Describing a picture is a bounded job that a 7B VL model does as well
        // as a 27B one, and when nothing fits in VRAM "largest" picks the 17.7
        // GB model — measured at 249 s to read one screenshot, against seconds
        // for a 6 GB one. Size buys nothing here and costs everything.
        let vision = candidates
            .iter()
            .copied()
            .filter(|p| p.capabilities.vision)
            .min_by_key(|p| (std::cmp::Reverse(placement_rank(p.placement)), p.weights_bytes));

        if let Some(m) = reasoner {
            policy.set(TaskClass::Reasoning, &m.name);
        }
        if let Some(m) = workhorse {
            policy.set(TaskClass::Coding, &m.name);
            policy.set(TaskClass::Edit, &m.name);
        }
        if let Some(m) = classifier {
            policy.set(TaskClass::Classify, &m.name);
            policy.set(TaskClass::Summarize, &m.name);
        }
        if let Some(m) = vision {
            policy.set(TaskClass::Vision, &m.name);
        }
        policy
    }
}

impl ModelProfile {
    /// Decode rate to plan around, in tokens per second.
    ///
    /// A measurement when there is one, otherwise what placement implies.
    /// The estimates are the measured figures from `docs/BENCHMARKS.md` on an
    /// 8 GB card: a resident 7B decodes at 42, and the same size spilled to
    /// system RAM manages 5. That eight-fold spread is why anything timing the
    /// model has to ask rather than assume.
    pub fn expected_decode_tps(&self) -> f64 {
        if let Some(measured) = self.observed_decode_tps.filter(|t| *t > 0.0) {
            return measured;
        }
        match self.placement {
            Placement::Resident => 40.0,
            Placement::Marginal => 20.0,
            Placement::Cloud => 30.0,
            Placement::Spilled | Placement::Unknown => 5.0,
        }
    }
}

/// Ordering weight for placement. Higher is better.
fn placement_rank(p: Placement) -> u8 {
    match p {
        Placement::Resident => 3,
        Placement::Marginal => 2,
        Placement::Cloud => 1,
        Placement::Spilled | Placement::Unknown => 0,
    }
}

fn is_resident(p: Placement) -> bool {
    matches!(p, Placement::Resident | Placement::Marginal)
}


/// Load every installed model's profile.
///
/// `/api/show` is one round-trip per model, so calls are issued concurrently.
pub async fn load_profiles(
    client: &OllamaClient,
    budget: VramBudget,
) -> Result<Vec<ModelProfile>> {
    let listing = client.list().await?;

    let futures = listing.models.into_iter().map(|entry| {
        let client = client.clone();
        async move { (entry.clone(), client.show(&entry.name).await.ok()) }
    });
    let results = futures_util::future::join_all(futures).await;

    Ok(results
        .into_iter()
        .map(|(entry, show)| build_profile(entry, show, budget))
        .collect())
}

fn build_profile(
    entry: ModelEntry,
    show: Option<ShowResponse>,
    budget: VramBudget,
) -> ModelProfile {
    let details = entry.details.clone().unwrap_or_default();
    let is_cloud = entry.name.ends_with(":cloud") || entry.size == 0;

    let (capabilities, max_context, kv_per_token, architecture) = match &show {
        Some(s) => (
            Capabilities::from(s),
            s.max_context(),
            s.kv_bytes_per_token(),
            s.architecture(),
        ),
        None => (Capabilities { completion: true, ..Default::default() }, None, None, None),
    };

    let (placement, recommended_context) = if is_cloud {
        (Placement::Cloud, max_context)
    } else if entry.size > 0 {
        estimate_placement(entry.size, kv_per_token, max_context, budget)
    } else {
        (Placement::Unknown, None)
    };

    ModelProfile {
        name: entry.name,
        family: details.family,
        parameter_size: details.parameter_size,
        quantization: details.quantization_level,
        weights_bytes: entry.size,
        capabilities,
        max_context,
        kv_bytes_per_token: kv_per_token,
        architecture,
        placement,
        recommended_context,
        observed_decode_tps: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1024 * 1024 * 1024;

    fn budget_8gb() -> VramBudget {
        VramBudget { total_bytes: 8 * GB, reserved_bytes: 2 * GB }
    }

    #[test]
    fn oversized_model_is_reported_as_spilled() {
        // qwen3.8 27B Q4_K_M: 17 GB of weights against 6 GB available.
        let (placement, ctx) = estimate_placement(17 * GB, Some(65_536), Some(262_144), budget_8gb());
        assert_eq!(placement, Placement::Spilled);
        assert_eq!(ctx, None, "no context makes a spilled model fit");
    }

    #[test]
    fn small_model_is_resident_with_large_context() {
        // qwen2.5-coder:7b — 4.7 GB, measured at 57 tok/s fully on GPU.
        let (placement, ctx) = estimate_placement(4_700_000_000, Some(128 * 1024), Some(32_768), budget_8gb());
        assert_eq!(placement, Placement::Resident);
        assert!(ctx.unwrap() >= 4_096);
    }

    #[test]
    fn near_capacity_model_is_marginal() {
        let (placement, _) = estimate_placement(5 * GB, Some(65_536), Some(32_768), budget_8gb());
        assert_eq!(placement, Placement::Marginal);
    }

    #[test]
    fn recommended_context_never_exceeds_trained_maximum() {
        let (_, ctx) = estimate_placement(1 * GB, Some(1_024), Some(8_192), budget_8gb());
        assert!(ctx.unwrap() <= 8_192);
    }

    fn profile(name: &str, bytes: u64, placement: Placement, tools: bool, thinking: bool) -> ModelProfile {
        ModelProfile {
            name: name.into(),
            family: "test".into(),
            parameter_size: "7B".into(),
            quantization: "Q4_K_M".into(),
            weights_bytes: bytes,
            capabilities: Capabilities {
                completion: true,
                tools,
                thinking,
                ..Default::default()
            },
            max_context: Some(32_768),
            kv_bytes_per_token: Some(65_536),
            architecture: Some("test".into()),
            placement,
            recommended_context: Some(8_192),
            observed_decode_tps: None,
        }
    }

    fn seeing(name: &str, bytes: u64, placement: Placement) -> ModelProfile {
        let mut p = profile(name, bytes, placement, true, true);
        p.capabilities.vision = true;
        p
    }

    #[test]
    fn vision_takes_the_smallest_model_that_is_best_placed() {
        // Bigger buys nothing for reading a screenshot and costs a great deal
        // when it does not fit.
        let models = vec![
            seeing("vl-7b", 6_000_000_000, Placement::Resident),
            seeing("vl-8b", 6_100_000_000, Placement::Resident),
            seeing("vl-27b", 17_700_000_000, Placement::Spilled),
        ];
        let policy = RoutingPolicy::derive(&models);
        assert_eq!(policy.get(TaskClass::Vision), Some("vl-7b"));
    }

    #[test]
    fn a_spilled_giant_is_not_chosen_to_read_an_image() {
        // The measured failure: with a 17.7 GB model loaded, every vision model
        // reads as spilled, and ranking by size then picked the giant — 249 s
        // to describe one screenshot.
        let models = vec![
            seeing("vl-7b", 6_000_000_000, Placement::Spilled),
            seeing("vl-27b", 17_700_000_000, Placement::Spilled),
        ];
        let policy = RoutingPolicy::derive(&models);
        assert_eq!(policy.get(TaskClass::Vision), Some("vl-7b"));
    }

    #[test]
    fn placement_still_outranks_size_for_vision() {
        // Smaller is preferred among equals, not in spite of a worse fit: a
        // resident 8 GB model beats a spilled 6 GB one.
        let models = vec![
            seeing("vl-small-spilled", 6_000_000_000, Placement::Spilled),
            seeing("vl-large-resident", 8_000_000_000, Placement::Resident),
        ];
        let policy = RoutingPolicy::derive(&models);
        assert_eq!(policy.get(TaskClass::Vision), Some("vl-large-resident"));
    }


    #[test]
    fn routing_sends_classification_to_the_cheapest_model() {
        let profiles = vec![
            profile("tiny:0.5b", 400_000_000, Placement::Resident, false, false),
            profile("coder:7b", 4_700_000_000, Placement::Resident, true, false),
            profile("big:27b", 17 * GB, Placement::Spilled, true, true),
        ];
        let policy = RoutingPolicy::derive(&profiles);

        assert_eq!(policy.get(TaskClass::Classify), Some("tiny:0.5b"));
        assert_eq!(policy.get(TaskClass::Coding), Some("coder:7b"));
        // The spilled 27B never wins a slot while a resident model can do it.
        assert_eq!(policy.get(TaskClass::Reasoning), Some("coder:7b"));
    }

    #[test]
    fn placement_outranks_size_when_choosing_a_workhorse() {
        // Taken from a real installation. Sorting by size alone picks
        // llama3.1 (4.6 GB, Marginal) over qwen2.5-coder:7b (4.4 GB,
        // Resident) — slower, less headroom, and worse at the actual task.
        let profiles = vec![
            profile("qwen2.5-coder:7b", 4_400_000_000, Placement::Resident, true, false),
            profile("llama3.1:latest", 4_600_000_000, Placement::Marginal, true, false),
            profile("qwen3.8:latest", 17 * GB, Placement::Spilled, true, true),
        ];
        let policy = RoutingPolicy::derive(&profiles);
        assert_eq!(policy.get(TaskClass::Coding), Some("qwen2.5-coder:7b"));
    }

    #[test]
    fn a_tiny_thinking_model_does_not_win_reasoning() {
        // Also from a real installation: qwen3.5:0.8b is resident and has the
        // thinking capability, but routing planning to a 0.8B model over a 7B
        // one is backwards. Thinking is a tiebreak, not a trump card.
        let profiles = vec![
            profile("qwen3.5:0.8b", 1_000_000_000, Placement::Resident, true, true),
            profile("qwen2.5-coder:7b", 4_400_000_000, Placement::Resident, true, false),
        ];
        let policy = RoutingPolicy::derive(&profiles);
        assert_eq!(policy.get(TaskClass::Reasoning), Some("qwen2.5-coder:7b"));
    }

    #[test]
    fn a_comparable_thinking_model_does_win_reasoning() {
        // The tiebreak still applies when the thinker is in the same class.
        let profiles = vec![
            profile("thinker:7b", 4_500_000_000, Placement::Resident, true, true),
            profile("coder:7b", 4_400_000_000, Placement::Resident, true, false),
        ];
        let policy = RoutingPolicy::derive(&profiles);
        assert_eq!(policy.get(TaskClass::Reasoning), Some("thinker:7b"));
    }

    #[test]
    fn cloud_models_are_excluded_from_local_routing() {
        let mut cloud = profile("glm:cloud", 0, Placement::Cloud, true, true);
        cloud.name = "glm-5:cloud".into();
        let profiles = vec![cloud, profile("coder:7b", 4_700_000_000, Placement::Resident, true, false)];
        let policy = RoutingPolicy::derive(&profiles);
        assert_eq!(policy.get(TaskClass::Coding), Some("coder:7b"));
    }
}
