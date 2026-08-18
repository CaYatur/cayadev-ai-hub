# Roadmap

Honest status. Anything under **Built** has code and passing tests in the repo;
everything else is not built yet, however well specified it looks.

## Built

### Ollama integration — `src-tauri/src/ollama/`
- Wire types tolerant of missing fields across daemon versions
- Streaming chat with **idle-based timeouts** — a first-token budget covering
  model load and prefill, and a separate inter-token budget — rather than a
  total-request timeout that kills a legitimately slow prefill
- **Budgets bounded by what the model may produce, not by a guess.** Some
  templates send a whole tool call in one piece: `qwen3.8:27b` streams its
  reasoning, then goes silent for **547 seconds** while it writes a file. A
  fixed budget read that as a stall, and so did a larger fixed budget. The reply
  is now capped with an explicit `num_predict`, which makes the longest legal
  silence a calculation rather than a constant: the cap divided by the rate this
  model actually decodes at — measured after the first step, estimated from
  placement before it
- A **daemon liveness check** during long silences. Ollama cannot be asked
  whether it is generating, but it can be asked whether the model is loaded, and
  during a request it must be; if it has gone, the run fails at once instead of
  waiting out the budget
- A **heartbeat during silence**, so a model loading or writing says so instead
  of looking hung
- Cancellation that reaches an in-flight stream, not only the gaps between steps
- Model registry: capability detection, true context limits from GGUF metadata,
  and VRAM fit estimation that accounts for hybrid attention
- **GPU memory read from `nvidia-smi`**, not assumed
- Task-based routing derived from installed models, ranking placement above size
  — except for vision, which takes the *smallest* well-placed model: reading a
  screenshot is a bounded job, and ranking by size picked the 17.7 GB model once
  nothing fit, at 249 s a picture against 16 s

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
- **Conversation memory**: earlier turns replayed between the system prompt and
  the instruction, each tool call folded to one summary line, so a follow-up
  knows what was said *and* what was done without carrying the tool output
- **Trimming before sending, from the middle**, pinning the system prompt, the
  current instruction and the recent steps — Ollama truncates from the front,
  which is where the task is, and its template then answers
  `500 no user query found in messages`. Cuts well under the threshold so
  several steps pass before the next trim and the KV prefix survives them
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
- `Publish` and `CaptureScreen` as distinct decisions rather than incidental
  ones; `AskUser` allowed in every mode, since refusing it means guessing

### Tools — `src-tauri/src/tools/`
- Files: read, write, edit, list, and **search** (contents and names, skipping
  `node_modules` and `.git`)
- Commands with **streamed output** while they run
- GitHub: status, issues, PRs, releases, Actions, checks, diffs, commit, push —
  with `gh_open_pr` and `gh_create_release` collapsing multi-command flows into
  one call
- Web search and page reading, wrapped in an untrusted-content marker
- Documents through a Python sidecar; **audio transcription** and **video by
  frame extraction**
- **Attached images ride in the message**, on a model that can see them — the
  run is handed to one if the chosen model cannot, and carries no tool field at
  all for the vision models that cannot call tools
- A model that can see reads its own screenshots; routing elsewhere is only for
  models that cannot see at all. Reasoning is turned off for the describing
  call, because a thinking model spent its whole reply budget on it and returned
  nothing
- Screen capture of **every monitor**, scaled to 1920px before it is sent, and
  ADB: devices, screen, UI hierarchy, logcat
- **`ask_user`**: a question to the user mid-run, with choices or a written
  answer, riding the approval channel's shape but its own map
- Every tool hides itself when its dependency is missing
- Every child process is spawned with `CREATE_NO_WINDOW`, so nothing flashes a
  console on Windows

### Checkpoints — `src-tauri/src/checkpoint/`
- Prior contents captured before every write and edit, content-addressed so
  repeated edits to unchanged files cost one object
- Restorable from the Activity page
- Stored **outside the project**, one directory per workspace under the OS data
  directory. In `.cayahub/` the agent found the app's own journal, read it, and
  spent steps reasoning about it — and it showed up in searches and git status

### Interface
- Session sidebar, streaming chat, collapsible tool rows, message actions
- Composer controls for permission mode, model, effort, and context window
- **Collapsible reasoning blocks** with a live label, following their own text
- **Markdown replies**: headings, lists, tables, syntax-coloured code, and links
  that open — a URL in the browser, a file path in its application, except for
  extensions that would run something, which are revealed in the file manager
- Attachments by drag-and-drop or paste — the agent gets a **copy**, never the
  original — shown with a preview on the turn that sent them, and gone from the
  composer once sent
- **Live tok/s and context occupancy**, timed from the first token so a cold
  model load is not averaged into the rate
- Following the newest output, and stopping the moment the user scrolls away
- A question panel for `ask_user`: choices, a written answer, or Skip
- An instruction box that grows upward as the instruction does, to about twelve
  lines, then scrolls
- Activity page: every checkpoint with a revert button, every decision with its
  reason
- Turkish and English, Turkish by default

### Measurement
- `scripts/bench.py` — throughput, pinning `num_ctx` and recording placement
- `scripts/speed_experiments.py` and `scripts/ab_threads.py` — runtime knobs
- `scripts/drift.py` — false and missed tool calls per prompt variant
- `scripts/probe_toolcalls.py` — what format each installed model emits
- `scripts/demo/` — the app over an example transcript, for the README pictures
- Results in [`BENCHMARKS.md`](BENCHMARKS.md) and [`SPEED.md`](SPEED.md)

## Known gaps

- **The smallest tool budget is judgement, not measurement.** Six for the
  smallest tier was chosen to hold the core tools; a 0.5B still produces usable
  calls there, but where it actually degrades has not been measured. The middle
  tier's sixteen *is* measured — `scripts/drift.py` offers exactly that surface
  to qwen2.5-coder:7b and it makes no false calls on eight conversational inputs
  and misses none of eight tasks.
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
