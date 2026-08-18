//! Web tools against the real internet.
//!
//! The search parser reads DuckDuckGo's HTML, which is scraped rather than a
//! supported API — so unit tests against a fixed sample prove the parser works
//! on *last week's* markup and nothing more. These run against the live
//! endpoint, which is the only thing that catches it changing.
//!
//! ```text
//! cargo test --test live_web -- --ignored --nocapture
//! ```

use cayadev_ai_hub_lib::tools::web::WebTools;
use serde_json::{json, Map, Value};

fn args(v: Value) -> Map<String, Value> {
    serde_json::from_value(v).expect("object")
}

#[tokio::test]
#[ignore = "requires internet access"]
async fn search_returns_usable_results() {
    let tools = WebTools::new();
    let out = tools
        .execute("web_search", &args(json!({"query": "rust programming language", "limit": 5})))
        .await
        .expect("search should succeed");

    println!("\n{}", out.chars().take(900).collect::<String>());

    assert!(out.contains("UNTRUSTED CONTENT"), "results were not labelled");
    assert!(out.contains("http"), "no links came back — the parser may be stale");
    // A result block is "N. title / url / snippet"; two of them means parsing
    // survived whatever the live markup currently is.
    assert!(out.contains("1. ") && out.contains("2. "), "fewer than two results parsed");
}

#[tokio::test]
#[ignore = "requires internet access"]
async fn a_page_becomes_readable_text() {
    let tools = WebTools::new();
    let out = tools
        .execute("web_read", &args(json!({"url": "https://example.com"})))
        .await
        .expect("fetch should succeed");

    println!("\n{out}");

    assert!(out.contains("UNTRUSTED CONTENT"));
    assert!(out.contains("Example Domain"), "page text missing: {out}");
    // Markup must not survive into what the model reads.
    assert!(!out.contains("<html"), "raw HTML leaked");
    assert!(!out.contains("<div"), "raw HTML leaked");
}

#[tokio::test]
#[ignore = "requires internet access"]
async fn a_dead_host_fails_with_a_readable_message() {
    let tools = WebTools::new();
    let err = tools
        .execute("web_read", &args(json!({"url": "https://this-host-does-not-exist.invalid"})))
        .await
        .expect_err("should fail");

    println!("\n{err}");
    // The model needs to understand it and move on, not retry blindly.
    assert!(err.contains("cannot fetch"), "unhelpful error: {err}");
}
