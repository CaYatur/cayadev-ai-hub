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

### Measurement
- Reproducible benchmark harness (`scripts/bench.py`) that pins `num_ctx` and
  records the placement the daemon actually chose
- Results published in [`BENCHMARKS.md`](BENCHMARKS.md)

## Known gaps

- **The VRAM budget is assumed, not probed.** It defaults to 8 GB total with
  2 GB reserved. On the development machine that marks `qwen3-vl:8b` (5.7 GB)
  as spilled, but it was measured running fully on the GPU at 21 tok/s — the
  reserve is too pessimistic by a few hundred megabytes. A real probe of free
  VRAM is the fix; until then `set_vram_budget` lets the UI correct it.
- Because of that, vision currently routes to the only model the estimate calls
  resident, which is smaller than ideal for image work. Recalibrating the budget
  should change that decision on its own.

## Next

In dependency order.

1. **Agent runner** — the loop tying model, tools, and permissions together;
   step budgets, retry-with-repair on malformed calls, cancellation
2. **Tool surface** — file read/write/edit, directory listing, search, command
   execution with streamed output
3. **Tauri command layer** — expose the core to the frontend, stream events
4. **Desktop UI** — chat with streaming, tool-call cards with approve/deny,
   model picker showing real context and placement, permission-mode banner
5. **i18n** — Turkish and English, Turkish default
6. **Checkpoints** — content-addressed undo of agent file changes, stored in the
   project directory, deduplicated so history stays small

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
