//! Integration tests against a live Ollama daemon.
//!
//! These exercise the exact path the UI's `list_models` command takes, which
//! unit tests cannot cover: real GGUF metadata, real capability lists, and the
//! placement estimate applied to models that actually exist on this machine.
//!
//! Ignored by default so a checkout without Ollama still passes `cargo test`.
//! Run them with:
//!
//! ```text
//! cargo test --test live_ollama -- --ignored --nocapture
//! ```

use cayadev_ai_hub_lib::ollama::{
    load_profiles, OllamaClient, Placement, RoutingPolicy, TaskClass, Timeouts, VramBudget,
    DEFAULT_HOST,
};

const GB: u64 = 1024 * 1024 * 1024;

fn client() -> OllamaClient {
    OllamaClient::new(DEFAULT_HOST, Timeouts::default())
}

#[tokio::test]
#[ignore = "requires a running Ollama daemon"]
async fn daemon_is_reachable() {
    let version = client().version().await.expect("Ollama should be reachable");
    assert!(!version.is_empty());
    println!("Ollama version: {version}");
}

#[tokio::test]
#[ignore = "requires a running Ollama daemon"]
async fn profiles_load_with_usable_metadata() {
    let budget = VramBudget { total_bytes: 8 * GB, reserved_bytes: 2 * GB };
    let profiles = load_profiles(&client(), budget).await.expect("profiles should load");

    assert!(!profiles.is_empty(), "no models installed to test against");

    for p in &profiles {
        println!(
            "{:<28} {:>7.1} GB  {:?}  ctx={:?}  rec={:?}  tools={}",
            p.name,
            p.weights_bytes as f64 / GB as f64,
            p.placement,
            p.max_context,
            p.recommended_context,
            p.capabilities.tools,
        );

        // A local model that reported a size must land in a definite bucket.
        if !p.is_cloud() && p.weights_bytes > 0 {
            assert_ne!(
                p.placement,
                Placement::Unknown,
                "{} has a size but no placement",
                p.name
            );
        }
        // A recommendation must never exceed what the weights were trained for.
        if let (Some(rec), Some(max)) = (p.recommended_context, p.max_context) {
            assert!(rec <= max, "{}: recommended {rec} exceeds max {max}", p.name);
        }
        // Spilled models get no context recommendation — none would help.
        if p.placement == Placement::Spilled {
            assert!(p.recommended_context.is_none(), "{}", p.name);
        }
    }
}

#[tokio::test]
#[ignore = "requires a running Ollama daemon"]
async fn routing_never_sends_work_to_a_spilled_model_when_a_resident_one_exists() {
    let budget = VramBudget { total_bytes: 8 * GB, reserved_bytes: 2 * GB };
    let profiles = load_profiles(&client(), budget).await.expect("profiles should load");
    let policy = RoutingPolicy::derive(&profiles);

    let resident_exists = profiles
        .iter()
        .any(|p| !p.is_cloud() && p.capabilities.tools && p.placement == Placement::Resident);

    println!("routing: {:#?}", policy.assignments);

    if !resident_exists {
        eprintln!("no resident tool-capable model installed; nothing to assert");
        return;
    }

    for class in [TaskClass::Coding, TaskClass::Edit, TaskClass::Classify] {
        let Some(chosen) = policy.get(class) else { continue };
        let profile = profiles
            .iter()
            .find(|p| p.name == chosen)
            .unwrap_or_else(|| panic!("{class:?} routed to unknown model {chosen}"));

        assert_ne!(
            profile.placement,
            Placement::Spilled,
            "{class:?} routed to spilled model {chosen} while a resident one exists"
        );
    }
}
