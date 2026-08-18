# Roadmap

Honest status. Anything marked **Done** has code and passing tests in the repo;
everything else is not built yet, however well specified it looks.

## Done

### Ollama integration — `src-tauri/src/ollama/`
- Wire types tolerant of missing fields across daemon versions
- Streaming chat over newline-delimited JSON, with chunk reassembly
- **Idle-based timeouts** rather than total-request timeouts, split into a
  first-token budget (covers model load + prefill) and an inter-token budget
- Model registry: capability detection, GGUF metadata parsing, true context
  limits, KV-cache cost per token
- **VRAM fit estimation** that accounts for hybrid attention
  (`full_attention_interval`), so it does not overestimate the cache by 4x on
  SSM-hybrid models like `qwen3.8`
- **Task-based routing policy** derived automatically from installed models,
  ranking placement above raw size — a resident 4.4 GB model beats a marginal
  4.6 GB one, and a small thinking model does not outrank a much larger
  workhorse for planning
- Tauri command layer (`list_models`, `routing_policy`, `ollama_version`), so
  the UI never calls the daemon cross-origin
- Live integration tests against a real daemon (`cargo test --test live_ollama
  -- --ignored`), which is how both routing bugs above were found

### Agent reliability — `src-tauri/src/agent/`
- **Tolerant tool-call extraction**: native field, `<tool_call>` tags, fenced
  blocks, `<function=name>` tags, bare JSON in prose, Python-style calls
- JSON repair: single quotes, unquoted keys, trailing commas, Python literals,
  and truncated output with unbalanced braces
- Guards against false positives — unknown tool names and config snippets in
  prose are not treated as calls
- Tiered system prompts sized to what a model can actually follow, with a
  byte-stable prefix so the daemon reuses its KV cache
- Tool-schema compaction for small models that preserves names and required
  fields

### Permissions — `src-tauri/src/perms/`
- Four modes: read-only, project-auto, balanced, unrestricted
- Component-wise workspace containment — not fooled by `..` traversal, shared
  string prefixes (`/work/project` vs `/work/project-backup`), Windows case
  differences, or `\\?\` verbatim prefixes
- Command risk classification with **shell payload inspection**: `bash -c "git
  status && rm -rf /"` classifies by its worst segment, including `$(...)` and
  backtick substitutions
- Credential paths and catastrophic commands refused in *every* mode, including
  unrestricted

### Agent loop — `src-tauri/src/agent/runner.rs`
- Runs to completion against real models: 54-61 tok/s on `qwen2.5-coder:7b`
- Schema validation before execution, naming the missing argument
- Loop detection on the call signature, which is what stops a 0.5B model
  cycling read_file -> edit_file -> run_command until the budget runs out
- Step budget, cancellation, streamed events

### Tools — `src-tauri/src/tools/`
- read_file, write_file, edit_file, list_dir, run_command
- Paths resolved and normalised *before* the permission engine sees them
- Output capped so a large file cannot consume a small model's whole context
- `edit_file` refuses an ambiguous match rather than replacing the first of
  several occurrences

### Model-assisted approval — `src-tauri/src/agent/approver.rs`
- Can only turn Ask into Allow; never sees Deny, Destructive, Catastrophic, or
  anything leaving the workspace
- Input is computed facts only, each sanitised to a short alphanumeric token
- Output must be the single word ALLOW, matched exactly
- 188 ms per decision on the routed classifier
- Injection tests assert a hostile payload produces byte-identical input

### Desktop UI
- Session sidebar grouped by age, persisted locally; a session takes its title
  from the first instruction and carries the workspace forward
- Tool calls collapsed to one scannable line, expandable on demand, with
  refused and rejected calls struck through
- A "rescued" flag when the tolerant parser had to recover a malformed call
- Status line: elapsed, tokens, live decode rate, active permission mode, Stop
- Streaming chat, approve/reject bar, permission-mode selector with a typed
  confirmation for Unrestricted
- Model list with placement, routing table, GPU re-read

### GPU probing — `src-tauri/src/ollama/gpu.rs`
- Budget read from `nvidia-smi` rather than assumed, fixing a wrong routing
  decision caused by the old fixed reserve

### Checkpoints — `src-tauri/src/checkpoint/`
- Prior contents captured before every write and edit, content-addressed by
  SHA-256 so repeated edits to unchanged files cost one object
- Files over 8 MB journalled as "not captured" rather than copied
- The store is refused to the agent in every mode and gitignores itself

### GitHub — `src-tauri/src/tools/github.rs`
- Status, issues, PRs, releases, Actions runs, checks, diffs, commit, push
- `gh_open_pr` and `gh_create_release` collapse multi-command flows into one
  call, because six tool calls is six model turns a small model will not survive
- Publishing is `Action::Publish`, confirmed in every mode including
  Unrestricted
- Only advertised inside a git repository

### Documents, images, devices — `src-tauri/src/tools/media.rs`
- PDF, Word, Excel, PowerPoint via a Python sidecar; text and CSV in Rust
- `view_image` and `screenshot` delegate to the routed vision model and return
  a description, so a non-multimodal coding model can still act on an image
- ADB: devices, screen, UI hierarchy, logcat
- Tools are advertised only when their dependency is actually present

### Interface
- Turkish and English, Turkish default, switchable in the sidebar

### Measurement
- Reproducible benchmark harness (`scripts/bench.py`) that pins `num_ctx` and
  records the placement the daemon actually chose
- Results published in [`BENCHMARKS.md`](BENCHMARKS.md)

## Known gaps

- **GPU probing is NVIDIA-only.** AMD and Intel hosts fall back to the fixed
  assumption. `refresh_vram_budget` returns an error there, and the UI should
  offer a manual figure.
- **Document extraction needs Python** with pymupdf, python-docx, and openpyxl.
  Bundling an interpreter would roughly double the release, so it is a declared
  prerequisite and the tools hide themselves when it is absent.
- **Windows-only testing.** Nothing is deliberately platform-specific, but the
  path and command tests have only been exercised on Windows.
- **The UI has not been driven by hand end to end, and its appearance is
  unverified.** The Rust layer is covered by live tests; the frontend
  typechecks, its command and event names are cross-checked against Rust, and
  its rendered structure was inspected in a browser. Nobody has clicked through
  a full session, and no one has looked at the pixels.

## Next

In dependency order.

1. **Undo from the UI** — checkpoints are captured and restorable from Rust,
   but no button surfaces them yet
2. **Search tool** — content and filename search, which the agent currently
   works around with `run_command`
3. **Streamed command output** — long-running commands return only when they
   finish
4. **Audio and video** — transcription and analysis are still unbuilt

## Later

Specified but not designed in detail yet.

- **Model-assisted approval** — a small fast model pre-classifying reversible
  calls, able to turn "ask" into "allow" but never to authorise anything the
  rules called destructive
- **GitHub integration** — issues, PRs, reviews, releases, Actions status, with
  the multi-step flows collapsed into single tool calls so a weak model can
  complete them
- **Attachments** — PDF, Word, Excel, CSV via a Python sidecar
- **Vision** — screenshots and images routed to vision-capable models
- **Audio and video** — transcription and analysis
- **ADB** — device screenshots, UI inspection, input
- **Computer use** — desktop screenshots and control

## Non-goals

- **Making a spilled model fast.** A 27B model on an 8 GB card is bound by
  memory bandwidth. The app routes around that; it cannot remove it.
- **Bundling models.** Ollama manages models; this is a client.
- **A cloud service.** Everything runs locally except explicitly cloud-tagged
  Ollama models.
