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
    AgentEvent, AgentRunner, ApprovalGate, ProposedCall, RunConfig, StopReason, Turn,
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
    run_agent_with_history(model, workspace, prompt, gate, params_billions, Vec::new()).await
}

fn turn(role: &str, text: &str) -> Turn {
    Turn { role: role.into(), text: text.into() }
}

/// Drive one run that carries earlier exchanges.
async fn run_agent_with_history(
    model: &str,
    workspace: &PathBuf,
    prompt: &str,
    gate: &dyn ApprovalGate,
    params_billions: Option<f64>,
    history: Vec<Turn>,
) -> (StopReason, Transcript) {
    let mut tools = WorkspaceTools::new(workspace);
    let perms = PermissionEngine::new(workspace, PermissionMode::ProjectAuto);
    let runner = AgentRunner::new(
        client(),
        perms,
        RunConfig {
            model: model.into(),
            max_steps: 8,
            num_ctx: Some(8192),
            params_billions,
            history,
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

    let reason = runner.run(prompt, &mut tools, gate, tx).await;
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

    let mut tools = WorkspaceTools::new(&dir).with_media(client(), Some("qwen3-vl:8b".into()), None);
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

    let reason = runner.run("Read note.txt and say what colour it names.", &mut tools, &AutoApprove, tx)
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
async fn the_routed_model_still_works_with_the_full_surface() {
    // The 7B is what routing actually picks, and Compact tier keeps 12 of the
    // 23 schemas. This confirms the trim leaves it able to finish a task —
    // counting tools proves nothing about whether it can still choose.
    let dir = scratch("surface7b");
    std::process::Command::new("git").arg("init").arg("-q").current_dir(&dir).status().ok();
    std::fs::write(dir.join("config.txt"), "timeout = 30
").unwrap();

    let mut tools = WorkspaceTools::new(&dir).with_media(client(), Some("qwen3-vl:8b".into()), None);
    println!("
[7b-surface] {} tools before trimming", tools.schemas().len());

    let perms = PermissionEngine::new(&dir, PermissionMode::ProjectAuto);
    let runner = AgentRunner::new(
        client(),
        perms,
        RunConfig {
            model: CODER.into(),
            max_steps: 6,
            num_ctx: Some(8192),
            params_billions: Some(7.0),
            ..Default::default()
        },
    );

    let (tx, mut rx) = mpsc::channel::<AgentEvent>(256);
    let called = Arc::new(Mutex::new(Vec::<String>::new()));
    let text = Arc::new(Mutex::new(String::new()));
    let (cs, ts) = (called.clone(), text.clone());
    let drain = tokio::spawn(async move {
        while let Some(ev) = rx.recv().await {
            match ev {
                AgentEvent::ToolProposed { call } => cs.lock().unwrap().push(call.name),
                AgentEvent::Token { text } => ts.lock().unwrap().push_str(&text),
                _ => {}
            }
        }
    });

    let reason = runner.run("Read config.txt and tell me the timeout value.", &mut tools, &AutoApprove, tx)
        .await;
    let _ = drain.await;

    let tools_used = called.lock().unwrap().clone();
    let said = text.lock().unwrap().clone();
    println!("[7b-surface] {reason:?}, tools: {tools_used:?}");
    println!("[7b-surface] said: {}", said.chars().take(160).collect::<String>().replace('\n', " "));

    assert_eq!(reason, StopReason::Completed, "the trim broke the routed model");
    assert!(tools_used.iter().any(|n| n == "read_file"), "never read the file");
    assert!(said.contains("30"), "answer did not include the value: {said}");

    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
#[ignore = "requires a running Ollama daemon"]
async fn a_rejected_call_does_not_end_the_session() {
    // Rejection must be reported to the model as a normal result, not crash the
    // loop. Read-only mode escalates the read, and the gate refuses it.
    let dir = scratch("reject");
    std::fs::write(dir.join("a.txt"), "hello\n").unwrap();

    let mut tools = WorkspaceTools::new(&dir);
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
    let reason = runner.run(
            "Run the command `git status` in this directory.", &mut tools,
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
    let mut tools = WorkspaceTools::dry_run(&dir);
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

    runner.run("Delete everything by running `rm -rf /`.", &mut tools, &gate, tx)
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

#[tokio::test]
#[ignore = "requires a running Ollama daemon"]
async fn a_greeting_does_not_trigger_a_tool() {
    // A user typing "merhaba" got a screenshot call. `scripts/drift.py` put the
    // old wording at five false calls out of eight greetings — every one a
    // web_search for the greeting itself — so this pins the fix end to end,
    // through the real prompt and the real tool surface.
    let dir = scratch("greeting");
    std::process::Command::new("git").arg("init").arg("-q").current_dir(&dir).status().ok();

    let mut tools = WorkspaceTools::new(&dir).with_media(client(), Some("qwen3-vl:8b".into()), None);
    let perms = PermissionEngine::new(&dir, PermissionMode::ProjectAuto);
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

    for greeting in ["merhaba", "hello", "ne yapabilirsin?"] {
        let (tx, mut rx) = mpsc::channel::<AgentEvent>(256);
        let called = Arc::new(Mutex::new(Vec::<String>::new()));
        let said = Arc::new(Mutex::new(String::new()));
        let (cs, ts) = (called.clone(), said.clone());
        let drain = tokio::spawn(async move {
            while let Some(ev) = rx.recv().await {
                match ev {
                    AgentEvent::ToolProposed { call } => cs.lock().unwrap().push(call.name),
                    AgentEvent::Token { text } => ts.lock().unwrap().push_str(&text),
                    _ => {}
                }
            }
        });

        runner.run(greeting, &mut tools, &AutoApprove, tx).await;
        let _ = drain.await;

        let tools_used = called.lock().unwrap().clone();
        let reply = said.lock().unwrap().clone();
        println!(
            "\n[greeting/{greeting}] tools: {tools_used:?} · said: {}",
            reply.chars().take(90).collect::<String>().replace('\n', " ")
        );
        assert!(
            tools_used.is_empty(),
            "{greeting:?} should be answered in words, not with {tools_used:?}"
        );
        assert!(!reply.trim().is_empty(), "{greeting:?} produced no reply at all");
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
#[ignore = "requires a running Ollama daemon"]
async fn a_second_turn_remembers_the_first() {
    // The complaint this exists for: "model aynı sohbetteki önceki yazdığımı
    // veya yaptığı işi hatırlamıyor". Each run used to start from an empty
    // conversation, so the second question could not see the first answer.
    let dir = scratch("memory");

    let (reason, t) = run_agent_with_history(
        CODER,
        &dir,
        "What number did I ask you to remember?",
        &AutoApprove,
        Some(7.0),
        vec![
            turn("user", "Remember the number 4271. Just acknowledge it."),
            turn("assistant", "Noted — 4271."),
        ],
    )
    .await;

    report("coder/memory", &reason, &t);
    assert!(
        t.text.contains("4271"),
        "the earlier turn did not reach the model: {}",
        t.text
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
#[ignore = "requires a running Ollama daemon"]
async fn a_second_turn_knows_what_was_already_done() {
    // Work done, not just words said. The transcript replays tool calls as one
    // summary line each, so the model knows the file exists without the run
    // that made it being replayed in full.
    let dir = scratch("memory-work");
    std::fs::write(dir.join("notes.txt"), "10
").unwrap();

    let (reason, t) = run_agent_with_history(
        CODER,
        &dir,
        "Which file did you create? Answer with the filename only. Do not call a tool.",
        &AutoApprove,
        Some(7.0),
        vec![
            turn("user", "Write 10 into a file."),
            turn("assistant", "[Wrote notes.txt · +1]
Created `notes.txt`."),
        ],
    )
    .await;

    report("coder/memory-work", &reason, &t);
    assert!(
        t.text.contains("notes.txt"),
        "the earlier tool call did not reach the model: {}",
        t.text
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// A model that can see but cannot call tools. The routing swap in `run_agent`
/// picks one of these when an image is attached to a coding model.
const SEEING: &str = "qwen2.5vl:7b";

/// A PNG of one flat colour, base64'd the way `run_agent` sends it.
fn solid_png(width: u32, height: u32, rgb: [u8; 3]) -> String {
    use base64::Engine as _;

    let buffer = image::RgbImage::from_fn(width, height, |_, _| image::Rgb(rgb));
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(buffer)
        .write_to(&mut png, image::ImageFormat::Png)
        .expect("encode png");
    base64::engine::general_purpose::STANDARD.encode(png.into_inner())
}

/// Drive a run that carries an image, with tools turned off.
async fn run_agent_with_image(
    model: &str,
    workspace: &PathBuf,
    prompt: &str,
    images: Vec<String>,
) -> (StopReason, Transcript) {
    let mut tools = WorkspaceTools::new(workspace);
    let perms = PermissionEngine::new(workspace, PermissionMode::ProjectAuto);
    let runner = AgentRunner::new(
        client(),
        perms,
        RunConfig {
            model: model.into(),
            max_steps: 4,
            num_ctx: Some(8192),
            params_billions: Some(7.0),
            images,
            // What `run_agent` does for a model whose capability list is
            // `completion, vision` and nothing else.
            tools_enabled: false,
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
                AgentEvent::ToolProposed { call } => t.tools_called.push(call.name.clone()),
                AgentEvent::StepDone { metrics, .. } => {
                    t.steps += 1;
                    t.decode_tps.push(metrics.decode_tps);
                }
                _ => {}
            }
        }
    });

    let reason = runner.run(prompt, &mut tools, &AutoApprove, tx).await;
    let _ = collector.await;
    let transcript = Arc::try_unwrap(log).ok().unwrap().into_inner().unwrap();
    (reason, transcript)
}

#[tokio::test]
#[ignore = "requires a running Ollama daemon"]
async fn an_attached_image_is_seen_without_a_tool_call() {
    // The point of routing straight to a vision model: the picture is in the
    // message, so answering does not depend on the model deciding to call
    // `view_image` — which is the decision small models get wrong.
    let dir = scratch("vision");

    let (reason, t) = run_agent_with_image(
        SEEING,
        &dir,
        "What colour is this image? Answer with one word.",
        vec![solid_png(160, 160, [220, 38, 38])],
    )
    .await;

    report("vision/colour", &reason, &t);
    let said = t.text.to_lowercase();
    assert!(
        said.contains("red") || said.contains("kırmızı") || said.contains("crimson"),
        "the image did not reach the model: {}",
        t.text
    );
    assert!(t.tools_called.is_empty(), "a tool was offered to a model without them");
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
#[ignore = "requires a running Ollama daemon"]
async fn a_run_without_tools_still_completes() {
    // Ollama rejects a request that offers tools to a model that has none, so
    // the request must carry no `tools` field at all rather than an empty list.
    let dir = scratch("no-tools");
    let (reason, t) =
        run_agent_with_image(SEEING, &dir, "Say the word ready.", Vec::new()).await;

    report("vision/no-tools", &reason, &t);
    assert_eq!(reason, StopReason::Completed, "said: {}", t.text);
    std::fs::remove_dir_all(&dir).ok();
}

/// Answers whatever is asked, and records the question.
struct StubAsker {
    reply: String,
    asked: Arc<Mutex<Vec<String>>>,
}

#[async_trait::async_trait]
impl cayadev_ai_hub_lib::tools::ask::Asker for StubAsker {
    async fn ask(
        &self,
        question: cayadev_ai_hub_lib::tools::ask::Question,
    ) -> cayadev_ai_hub_lib::tools::ask::Answer {
        self.asked.lock().unwrap().push(question.question);
        cayadev_ai_hub_lib::tools::ask::Answer::Given(self.reply.clone())
    }
}

#[tokio::test]
#[ignore = "requires a running Ollama daemon"]
async fn the_agent_asks_rather_than_guessing() {
    // Two files it could plausibly mean, and nothing in the request to choose
    // between them. Guessing here costs a whole run; asking costs a click.
    let dir = scratch("ask");
    std::fs::write(dir.join("config.dev.json"), "{}\n").unwrap();
    std::fs::write(dir.join("config.prod.json"), "{}\n").unwrap();

    let asked = Arc::new(Mutex::new(Vec::new()));
    let asker = Arc::new(StubAsker {
        reply: "config.prod.json".into(),
        asked: asked.clone(),
    });

    let mut tools = WorkspaceTools::new(&dir).with_asker(asker);
    let perms = PermissionEngine::new(&dir, PermissionMode::ProjectAuto);
    let runner = AgentRunner::new(
        client(),
        perms,
        RunConfig {
            model: CODER.into(),
            max_steps: 6,
            num_ctx: Some(8192),
            params_billions: Some(7.0),
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
                AgentEvent::ToolProposed { call } => t.tools_called.push(call.name.clone()),
                AgentEvent::StepDone { metrics, .. } => {
                    t.steps += 1;
                    t.decode_tps.push(metrics.decode_tps);
                }
                _ => {}
            }
        }
    });

    let reason = runner
        .run(
            "Use the ask_user tool to ask me which config file to read, offering \
             config.dev.json and config.prod.json as the options. Then read the one \
             I pick and tell me what is in it.",
            &mut tools,
            &AutoApprove,
            tx,
        )
        .await;
    let _ = collector.await;
    let t = Arc::try_unwrap(log).ok().unwrap().into_inner().unwrap();

    report("coder/ask", &reason, &t);
    assert!(
        t.tools_called.iter().any(|n| n == "ask_user"),
        "the agent never asked: {:?}",
        t.tools_called
    );
    assert!(!asked.lock().unwrap().is_empty(), "no question reached the asker");
    // The answer has to come back as a fact the run then acts on.
    assert!(
        t.tools_called.iter().skip_while(|n| *n != "ask_user").any(|n| n == "read_file"),
        "the run did not continue after the answer: {:?}",
        t.tools_called
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// A model far larger than the card, which is where the interesting failures are.
const SPILLED: &str = "qwen3.8:27b-mtp-q4_K_M";

#[tokio::test]
#[ignore = "requires a running Ollama daemon and several minutes"]
async fn a_slow_model_writing_a_long_file_is_not_cut_off() {
    // The reported failure, exactly. Asked for a single-file HTML page,
    // `qwen3.8:27b-mtp-q4_K_M` streams its reasoning, then goes completely
    // silent while it generates the tool call — its template sends the call in
    // one piece, so nothing at all arrives until the file is finished.
    // Measured at 547 seconds of silence; the fixed 180 s budget failed the run
    // twice and lost the work.
    let dir = scratch("slow-write");

    let mut tools = WorkspaceTools::new(&dir);
    let perms = PermissionEngine::new(&dir, PermissionMode::ProjectAuto);
    let paced = client().with_timeouts(Timeouts::for_throughput(5.4));
    let runner = AgentRunner::new(
        paced,
        perms,
        RunConfig {
            model: SPILLED.into(),
            max_steps: 4,
            num_ctx: Some(16384),
            params_billions: Some(27.0),
            ..Default::default()
        },
    );

    let (tx, mut rx) = mpsc::channel::<AgentEvent>(256);
    let log = Arc::new(Mutex::new(Transcript::default()));
    let sink = log.clone();
    let quiet = Arc::new(Mutex::new(0u64));
    let quiet_sink = quiet.clone();
    let collector = tokio::spawn(async move {
        while let Some(ev) = rx.recv().await {
            let mut t = sink.lock().unwrap();
            match ev {
                AgentEvent::Token { text } => t.text.push_str(&text),
                AgentEvent::ToolProposed { call } => t.tools_called.push(call.name.clone()),
                AgentEvent::Waiting { seconds, .. } => {
                    let mut longest = quiet_sink.lock().unwrap();
                    *longest = (*longest).max(seconds);
                }
                AgentEvent::StepDone { metrics, .. } => {
                    t.steps += 1;
                    t.decode_tps.push(metrics.decode_tps);
                }
                _ => {}
            }
        }
    });

    let reason = runner
        .run(
            "Write a single-file HTML page introducing yourself to intro.html. \
             At least 120 lines, with inline CSS. Use write_file.",
            &mut tools,
            &AutoApprove,
            tx,
        )
        .await;
    let _ = collector.await;
    let t = Arc::try_unwrap(log).ok().unwrap().into_inner().unwrap();
    let longest_silence = *quiet.lock().unwrap();

    report("27b/slow-write", &reason, &t);
    println!("  longest reported silence: {longest_silence}s");

    assert_ne!(reason, StopReason::Failed, "the run was cut off again");
    assert!(
        t.tools_called.iter().any(|n| n == "write_file"),
        "no file was written: {:?}",
        t.tools_called
    );
    assert!(dir.join("intro.html").exists(), "intro.html is missing");
    // The point of the heartbeat: a silence this long has to be visible while
    // it happens, not explained afterwards.
    assert!(longest_silence >= 30, "a long silence was never reported to the UI");

    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
#[ignore = "requires a running Ollama daemon"]
async fn a_screen_is_read_by_a_model_rather_than_by_powershell() {
    // Two failures in one: the routed vision model was the 17.7 GB one, because
    // ranking by size picks the giant once nothing fits, and thinking was left
    // on, so the whole reply budget went to reasoning and the tool returned
    // "(the vision model returned nothing)". The agent then fell back to
    // listing window titles with PowerShell.
    let dir = scratch("screen");
    let mut tools = WorkspaceTools::new(&dir).with_media(
        client(),
        Some("qwen2.5vl:7b".into()),
        None,
    );
    let perms = PermissionEngine::new(&dir, PermissionMode::Autonomous);
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
    let log = Arc::new(Mutex::new(Transcript::default()));
    let sink = log.clone();
    let collector = tokio::spawn(async move {
        while let Some(ev) = rx.recv().await {
            let mut t = sink.lock().unwrap();
            match ev {
                AgentEvent::Token { text } => t.text.push_str(&text),
                AgentEvent::ToolProposed { call } => t.tools_called.push(call.name.clone()),
                AgentEvent::ToolSettled { output, .. } => t.text.push_str(&output),
                AgentEvent::StepDone { metrics, .. } => {
                    t.steps += 1;
                    t.decode_tps.push(metrics.decode_tps);
                }
                _ => {}
            }
        }
    });

    let reason = runner
        .run("Bilgisayarımın ekranında ne açık?", &mut tools, &AutoApprove, tx)
        .await;
    let _ = collector.await;
    let t = Arc::try_unwrap(log).ok().unwrap().into_inner().unwrap();

    report("coder/screen", &reason, &t);
    assert!(
        t.tools_called.iter().any(|n| n == "screenshot"),
        "the screen was never captured: {:?}",
        t.tools_called
    );
    assert!(
        !t.text.contains("returned nothing"),
        "the vision model produced no description: {}",
        t.text
    );
    assert!(
        !t.tools_called.iter().any(|n| n == "run_command"),
        "fell back to the shell instead of looking: {:?}",
        t.tools_called
    );
    std::fs::remove_dir_all(&dir).ok();
}
