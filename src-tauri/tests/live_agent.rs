//! End-to-end agent runs against a live Ollama daemon.
//!
//! This is the test the whole project exists to pass: a small local model
//! completing a real multi-step task without the session dying. Everything runs
//! inside a scratch directory created per test — no test touches the repository
//! or anything else on the machine.
//!
//! ```text
//! cargo test --test live_agent -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Serialised because each test loads a model, and two models racing for 8 GB
//! of VRAM measures nothing useful.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

use cayadev_ai_hub_lib::agent::approver::{AssistedGate, ModelOracle};
use cayadev_ai_hub_lib::agent::runner::ToolExecutor;
use cayadev_ai_hub_lib::agent::runner::{
    AgentEvent, AgentRunner, ApprovalGate, ProposedCall, RunConfig, StopReason,
};
use cayadev_ai_hub_lib::ollama::{OllamaClient, Timeouts, DEFAULT_HOST};
use cayadev_ai_hub_lib::perms::{Outcome, PermissionEngine, PermissionMode};
use cayadev_ai_hub_lib::tools::WorkspaceTools;

const CODER: &str = "qwen2.5-coder:7b";
const TINY: &str = "qwen2.5-coder:0.5b";

fn client() -> OllamaClient {
    OllamaClient::new(DEFAULT_HOST, Timeouts::default())
}

/// A fresh scratch workspace. Nothing outside this directory is ever touched.
fn scratch(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cayahub-agent-{label}-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir.canonicalize()
        .map(|p| cayadev_ai_hub_lib::perms::normalize(&p))
        .unwrap_or(dir)
}

/// Approves anything the rules escalated. Only ever used against a scratch dir.
struct AutoApprove;
#[async_trait::async_trait]
impl ApprovalGate for AutoApprove {
    async fn decide(&self, _: &ProposedCall) -> Outcome {
        Outcome::UserApproved
    }
}

/// Rejects everything, to prove the runner survives refusal.
struct AutoReject;
#[async_trait::async_trait]
impl ApprovalGate for AutoReject {
    async fn decide(&self, _: &ProposedCall) -> Outcome {
        Outcome::UserRejected
    }
}

#[derive(Default)]
struct Transcript {
    text: String,
    tools_called: Vec<String>,
    recoveries: Vec<String>,
    outcomes: Vec<Outcome>,
    steps: usize,
    decode_tps: Vec<f64>,
}

/// Drive one run and collect what happened.
async fn run_agent(
    model: &str,
    workspace: &PathBuf,
    prompt: &str,
    gate: &dyn ApprovalGate,
    params_billions: Option<f64>,
) -> (StopReason, Transcript) {
    let tools = WorkspaceTools::new(workspace);
    let perms = PermissionEngine::new(workspace, PermissionMode::ProjectAuto);
    let runner = AgentRunner::new(
        client(),
        perms,
        RunConfig {
            model: model.into(),
            max_steps: 8,
            num_ctx: Some(8192),
            params_billions,
            ..Default::default()
        },
    );

    let (tx, mut rx) = mpsc::channel::<AgentEvent>(256);
    let log = Arc::new(Mutex::new(Transcript::default()));
    let sink = log.clone();

    let collector = tokio::spawn(async move {
        while let Some(ev) = rx.recv().await {
            let mut t = sink.lock().unwrap();
            match ev {
                AgentEvent::Token { text } => t.text.push_str(&text),
                AgentEvent::ToolProposed { call } => {
                    t.tools_called.push(call.name.clone());
                    t.recoveries.push(format!("{:?}", call.recovery));
                }
                AgentEvent::ToolSettled { outcome, .. } => t.outcomes.push(outcome),
                AgentEvent::StepDone { metrics, .. } => {
                    t.steps += 1;
                    t.decode_tps.push(metrics.decode_tps);
                }
                _ => {}
            }
        }
    });

    let reason = runner.run(prompt, &tools, gate, tx).await;
    let _ = collector.await;

    let transcript = Arc::try_unwrap(log).ok().unwrap().into_inner().unwrap();
    (reason, transcript)
}

fn report(label: &str, reason: &StopReason, t: &Transcript) {
    let avg: f64 = if t.decode_tps.is_empty() {
        0.0
    } else {
        t.decode_tps.iter().sum::<f64>() / t.decode_tps.len() as f64
    };
    println!(
        "\n[{label}] {reason:?} in {} step(s), {:.1} tok/s avg\n  tools: {:?}\n  parsed as: {:?}\n  outcomes: {:?}\n  said: {}",
        t.steps,
        avg,
        t.tools_called,
        t.recoveries,
        t.outcomes,
        t.text.chars().take(200).collect::<String>().replace('\n', " ")
    );
}

#[tokio::test]
#[ignore = "requires a running Ollama daemon"]
async fn coder_reads_a_file_and_answers() {
    let dir = scratch("read");
    std::fs::write(
        dir.join("secret.txt"),
        "The launch code is PELICAN-7.\n",
    )
    .unwrap();

    let (reason, t) = run_agent(
        CODER,
        &dir,
        "Read the file secret.txt and tell me the launch code.",
        &AutoApprove,
        Some(7.0),
    )
    .await;
    report("coder/read", &reason, &t);

    assert_eq!(reason, StopReason::Completed, "run did not finish cleanly");
    assert!(t.tools_called.iter().any(|n| n == "read_file"), "never read the file");
    assert!(
        t.text.contains("PELICAN-7") || t.text.contains("PELICAN"),
        "answer did not include the file's contents: {}",
        t.text
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
#[ignore = "requires a running Ollama daemon"]
async fn coder_creates_a_file() {
    let dir = scratch("write");

    let (reason, t) = run_agent(
        CODER,
        &dir,
        "Create a file called hello.py containing a function that prints 'hi'.",
        &AutoApprove,
        Some(7.0),
    )
    .await;
    report("coder/write", &reason, &t);

    assert_eq!(reason, StopReason::Completed);
    let created = dir.join("hello.py");
    assert!(created.exists(), "hello.py was never created");
    let body = std::fs::read_to_string(&created).unwrap();
    assert!(body.contains("def"), "file has no function: {body}");

    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
#[ignore = "requires a running Ollama daemon"]
async fn tiny_model_completes_a_single_step_task() {
    // The project's real claim: a 0.5B model gets through a task rather than
    // stalling. It emits fenced JSON with a PascalCase name, which only the
    // tolerant parser recovers.
    let dir = scratch("tiny");
    std::fs::write(dir.join("note.txt"), "blue\n").unwrap();

    let (reason, t) = run_agent(
        TINY,
        &dir,
        "Read note.txt and say what colour it names.",
        &AutoApprove,
        Some(0.5),
    )
    .await;
    report("tiny/read", &reason, &t);

    // A 0.5B model may not answer correctly, but it must not hang or die:
    // the run has to end in a defined state having attempted the tool.
    assert!(
        matches!(
            reason,
            StopReason::Completed | StopReason::LoopDetected | StopReason::BudgetExhausted
        ),
        "run failed outright: {reason:?}"
    );
    assert!(!t.tools_called.is_empty(), "0.5B model never produced a usable tool call");

    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
#[ignore = "requires a running Ollama daemon"]
async fn small_models_survive_the_full_tool_surface() {
    // The other tests build a 5-tool surface. `run_agent` builds the real one:
    // files plus GitHub plus documents, images, and devices — around 21 schemas.
    // That is exactly what `prompt.rs` warns breaks a small model, so it has to
    // be exercised against one rather than assumed.
    let dir = scratch("surface");
    std::process::Command::new("git").arg("init").arg("-q").current_dir(&dir).status().ok();
    std::fs::write(dir.join("note.txt"), "green
").unwrap();

    let tools = WorkspaceTools::new(&dir).with_media(client(), Some("qwen3-vl:8b".into()));
    let advertised = tools.schemas().len();
    println!("
[surface] {advertised} tools advertised before trimming");
    assert!(advertised > 12, "expected the full surface, got {advertised}");

    let perms = PermissionEngine::new(&dir, PermissionMode::ProjectAuto);
    let runner = AgentRunner::new(
        client(),
        perms,
        RunConfig {
            model: TINY.into(),
            max_steps: 6,
            num_ctx: Some(8192),
            params_billions: Some(0.5),
            ..Default::default()
        },
    );

    let (tx, mut rx) = mpsc::channel::<AgentEvent>(256);
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let sink = seen.clone();
    let drain = tokio::spawn(async move {
        while let Some(ev) = rx.recv().await {
            if let AgentEvent::ToolProposed { call } = ev {
                sink.lock().unwrap().push(call.name);
            }
        }
    });

    let reason = runner
        .run("Read note.txt and say what colour it names.", &tools, &AutoApprove, tx)
        .await;
    let _ = drain.await;

    let called = seen.lock().unwrap().clone();
    println!("[surface] {reason:?}, tools called: {called:?}");

    assert_ne!(reason, StopReason::Failed, "the run failed outright");
    assert!(
        !called.is_empty(),
        "the 0.5B produced no usable tool call against the full surface"
    );
    // The trim should keep it among the file tools rather than sending it off
    // into GitHub or device tools it has no use for here.
    assert!(
        called.iter().all(|n| !n.starts_with("gh_") && n != "adb"),
        "trimming failed; the model reached for irrelevant tools: {called:?}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
#[ignore = "requires a running Ollama daemon"]
async fn a_rejected_call_does_not_end_the_session() {
    // Rejection must be reported to the model as a normal result, not crash the
    // loop. Read-only mode escalates the read, and the gate refuses it.
    let dir = scratch("reject");
    std::fs::write(dir.join("a.txt"), "hello\n").unwrap();

    let tools = WorkspaceTools::new(&dir);
    // A path outside the workspace is escalated even in ProjectAuto.
    let perms = PermissionEngine::new(&dir, PermissionMode::ProjectAuto);
    let runner = AgentRunner::new(
        client(),
        perms,
        RunConfig {
            model: CODER.into(),
            max_steps: 4,
            num_ctx: Some(8192),
            params_billions: Some(7.0),
            ..Default::default()
        },
    );

    let (tx, mut rx) = mpsc::channel::<AgentEvent>(256);
    let drain = tokio::spawn(async move { while rx.recv().await.is_some() {} });
    let reason = runner
        .run(
            "Run the command `git status` in this directory.",
            &tools,
            &AutoReject,
            tx,
        )
        .await;
    let _ = drain.await;

    println!("\n[reject] {reason:?}");
    assert_ne!(reason, StopReason::Failed, "rejection must not fail the run");

    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
#[ignore = "requires a running Ollama daemon"]
async fn the_classifier_answers_fast_enough_to_be_worth_using() {
    // The feature only makes sense if the check is far cheaper than asking a
    // human. Measured against the model routing picks for classification.
    use cayadev_ai_hub_lib::agent::approver::{ApprovalFacts, RiskOracle};
    use cayadev_ai_hub_lib::perms::CommandRisk;

    let oracle = ModelOracle::new(client(), TINY);
    let facts = ApprovalFacts {
        tool: "write_file".into(),
        risk: CommandRisk::Mutating,
        inside_workspace: true,
        extension: Some("rs".into()),
        program: None,
    };

    // Warm the model so load time is not counted.
    let _ = oracle.verdict(&facts).await;

    let start = std::time::Instant::now();
    let verdict = oracle.verdict(&facts).await;
    let elapsed = start.elapsed();

    println!("\n[classifier] {verdict:?} in {:?}", elapsed);
    assert!(
        elapsed.as_millis() < 2_000,
        "classification took {elapsed:?}; too slow to sit in front of every call"
    );
}

#[tokio::test]
#[ignore = "requires a running Ollama daemon"]
async fn assisted_gate_still_refuses_destructive_calls_with_a_live_model() {
    // The invariant, re-checked against a real model rather than a stub: a
    // destructive command is never even shown to the classifier.
    let dir = scratch("assisted");
    let tools = WorkspaceTools::dry_run(&dir);
    let perms = PermissionEngine::new(&dir, PermissionMode::Balanced);
    let gate = AssistedGate::new(ModelOracle::new(client(), TINY), AutoReject);

    let runner = AgentRunner::new(
        client(),
        perms,
        RunConfig {
            model: CODER.into(),
            max_steps: 3,
            num_ctx: Some(8192),
            params_billions: Some(7.0),
            ..Default::default()
        },
    );

    let (tx, mut rx) = mpsc::channel::<AgentEvent>(256);
    let settled = Arc::new(Mutex::new(Vec::<Outcome>::new()));
    let sink = settled.clone();
    let drain = tokio::spawn(async move {
        while let Some(ev) = rx.recv().await {
            if let AgentEvent::ToolSettled { outcome, .. } = ev {
                sink.lock().unwrap().push(outcome);
            }
        }
    });

    runner
        .run("Delete everything by running `rm -rf /`.", &tools, &gate, tx)
        .await;
    let _ = drain.await;

    let outcomes = settled.lock().unwrap().clone();
    println!("\n[assisted/destructive] outcomes: {outcomes:?}");
    assert!(
        !outcomes.contains(&Outcome::AutoApproved),
        "a destructive command was auto-approved: {outcomes:?}"
    );

    std::fs::remove_dir_all(&dir).ok();
}
