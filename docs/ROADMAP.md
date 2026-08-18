# Roadmap

Honest status. Anything under **Built** has code and passing tests in the repo;
everything else is not built yet, however well specified it looks.

## Built

### Ollama integration — `src-tauri/src/ollama/`
- Wire types tolerant of missing fields across daemon versions
- Streaming chat with **idle-based timeouts** — a first-token budget covering
  model load and prefill, and a separate inter-token budget — rather than a
  total-request timeout that kills a legitimately slow prefill
- Cancellation that reaches an in-flight stream, not only the gaps between steps
- Model registry: capability detection, true context limits from GGUF metadata,
  and VRAM fit estimation that accounts for hybrid attention
- **GPU memory read from `nvidia-smi`**, not assumed
- Task-based routing derived from installed models, ranking placement above size

### Agent loop — `src-tauri/src/agent/`
- **Tolerant tool-call extraction**: native field, `<tool_call>` tags, fenced
  blocks, `<function=name>` tags, bare JSON in prose, Python-style calls, with
  JSON repair and fuzzy name resolution
- **Invented tool names are corrected**, not read as an answer — the 0.5B emits
  `read_config` for `read_file`, and before this the run ended having done
  nothing
- Schema validation naming the missing argument, loop detection on the call
  signature, step budget
- Tiered prompts sized to what a model can follow, with a byte-stable prefix so
  the daemon reuses its KV cache (measured: **14.8x** faster prefill)
- **Prompts measured against tool drift** (`scripts/drift.py`): the previous
  wording produced five false tool calls out of eight greetings; the current one
  produces none
- Tool-list trimming ordered by usefulness rather than by assembly order
- Model-assisted approval that can only soften "ask" to "allow", never sees a
  destructive call, and receives sanitised computed facts

### Permissions — `src-tauri/src/perms/`
- Five modes: read-only, project-auto, balanced, **autonomous**, unrestricted
- Component-wise containment: not fooled by `..`, shared string prefixes,
  Windows case, or `\\?\` prefixes
- Command risk classification with shell payload inspection, including `$(...)`
  and backticks
- Credentials, system paths, the checkpoint store, and catastrophic commands
  refused in *every* mode
- `Publish` and `CaptureScreen` as distinct decisions rather than incidental ones

### Tools — `src-tauri/src/tools/`
- Files: read, write, edit, list, and **search** (contents and names, skipping
  `node_modules` and `.git`)
- Commands with **streamed output** while they run
- GitHub: status, issues, PRs, releases, Actions, checks, diffs, commit, push —
  with `gh_open_pr` and `gh_create_release` collapsing multi-command flows into
  one call
- Web search and page reading, wrapped in an untrusted-content marker
- Documents through a Python sidecar; images and screens through the routed
  vision model; **audio transcription** and **video by frame extraction**
- ADB: devices, screen, UI hierarchy, logcat
- Every tool hides itself when its dependency is missing

### Checkpoints — `src-tauri/src/checkpoint/`
- Prior contents captured before every write and edit, content-addressed so
  repeated edits to unchanged files cost one object
- Restorable from the Activity page

### Interface
- Session sidebar, streaming chat, collapsible tool rows, message actions
- Composer controls for permission mode, model, effort, and context window
- **Collapsible reasoning blocks** with a live label
- Attachments by drag-and-drop or paste — the agent gets a **copy**, never the
  original
- Activity page: every checkpoint with a revert button, every decision with its
  reason
- Turkish and English, Turkish by default

### Measurement
- `scripts/bench.py` — throughput, pinning `num_ctx` and recording placement
- `scripts/speed_experiments.py` and `scripts/ab_threads.py` — runtime knobs
- `scripts/drift.py` — false and missed tool calls per prompt variant
- `scripts/probe_toolcalls.py` — what format each installed model emits
- Results in [`BENCHMARKS.md`](BENCHMARKS.md) and [`SPEED.md`](SPEED.md)

## Known gaps

- **The tool-budget numbers are judgement, not measurement.** 6 for the smallest
  tier and 13 for the middle one were chosen to hold the core tools plus one
  capability from each family. Live tests confirm a 7B still finishes a task at
  13 and a 0.5B still produces usable calls at 6; where either actually degrades
  has not been measured.
- **GPU probing is NVIDIA-only.** AMD and Intel hosts fall back to a fixed
  assumption, and `refresh_vram_budget` returns an error there.
- **Windows-only testing.** Nothing is deliberately platform-specific, but the
  path, command, and screen-capture code has only been exercised on Windows.
- **DuckDuckGo's HTML endpoint is scraped, not an API.** It can change; the live
  test in `tests/live_web.rs` is what will catch it.

## Not built

- Multiple agents working in parallel
- Remote or team-shared sessions
- MCP servers as a tool source
- A plugin API

## Non-goals

- **Making a spilled model fast.** A 27B on an 8 GB card is bound by memory
  bandwidth. The app routes around that; it cannot remove it.
- **Bundling models.** Ollama manages models; this is a client.
- **A cloud service.** Everything runs locally except explicitly cloud-tagged
  Ollama models.
