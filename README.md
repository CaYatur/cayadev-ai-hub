<div align="center">

# CaYaDev AI Hub

**Capable coding agents on local Ollama models — including the small ones.**

[![License: MIT](https://img.shields.io/badge/License-MIT-dc2626.svg)](LICENSE)
[![Tauri](https://img.shields.io/badge/Tauri-2-dc2626.svg)](https://tauri.app)
[![Ollama](https://img.shields.io/badge/Ollama-0.32%2B-dc2626.svg)](https://ollama.com)

</div>

---

## The problem

Point a general-purpose agent tool at a local model and one of two things
usually happens: the session dies before any work starts, or it crawls. Both
have specific, fixable causes, and neither is really "the model is too small".

| Symptom | Actual cause |
| --- | --- |
| Session ends immediately | The daemon's default context silently truncated a long system prompt, so the model never saw the tool definitions |
| "No tool calls" on a tool-capable model | The model emitted `<tool_call>` tags or a fenced JSON block, and a strict parser saw prose |
| Hangs, then dies | A *total* request timeout fired during a legitimately slow prefill |
| Painfully slow | A model too large for available VRAM spilled to system RAM |

CaYaDev AI Hub is built around those four causes.

## What makes it fast

The honest version, measured on the development machine — full table in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md):

| Model | Decode | TTFT | Placement |
| --- | ---: | ---: | --- |
| `qwen3.8` 27B Q4_K_M | 5.4 tok/s | 1.5 s | 80% CPU / 20% GPU |
| `qwen2.5-coder:7b` | **42.3 tok/s** | **0.40 s** | 100% GPU |
| `qwen2.5-coder:0.5b` | 316.3 tok/s | 0.19 s | 100% GPU |

On an 8 GB card, a 7B model that fits in VRAM decodes **~8x faster** than a 27B
model that does not (42 tok/s reproduced across three warm runs; it reached 57
in favourable conditions, but the conservative figure is the one quoted). No sampling flag closes that gap — we measured context
sizes across a 16x range on the 27B and every result landed within noise of 5
tok/s.

So the hub's core feature is **task-based routing**: planning goes to the
strongest model that will actually fit, edits go to the fast resident coder, and
the constant yes/no decisions go to a 0.5B model that answers in milliseconds.

> **What this does not do:** make a 27B model on an 8 GB card fast. Nothing can —
> that is a memory-bandwidth limit, not a software one. If you have a 24 GB card,
> the 27B fits entirely in VRAM and the routing table should say so.

## Features

**Reliability**
- Tolerant tool-call parsing — recovers calls from `<tool_call>` tags, fenced
  blocks, `<function=name>` tags, bare JSON, and Python-style syntax, with
  repair for single quotes, unquoted keys, trailing commas, and truncated output
- Idle-based timeouts, not total-request timeouts, so a slow prefill is never
  mistaken for a hang
- Real context accounting — shows the model's *trained* maximum from GGUF
  metadata alongside what the daemon actually allocated

**Speed**
- Task-based model routing across reasoning / coding / edit / classify / vision
- VRAM fit estimation that accounts for hybrid attention architectures
- Stable prompt prefixes so the daemon reuses its KV cache instead of
  re-prefilling every turn
- `keep_alive` management so big models are not reloaded between turns

**Safety**
- Four permission modes, from read-only to unrestricted
- Component-wise workspace containment that is not fooled by `..` traversal or
  shared string prefixes
- Shell payload inspection — `bash -c "git status && rm -rf /"` is classified by
  its worst segment, not its first
- Credential paths and catastrophic commands refused in *every* mode, including
  unrestricted
- Full audit log of what was auto-approved, what was asked, and what was refused

## Status

Early, and specific about it. **The agent runs.** The Rust core — client,
registry and routing, tolerant parsing, permission engine, agent loop, tool
surface, and model-assisted approval — is covered by 109 unit tests and 9 live
integration tests that drive real models end to end.

Measured on the development machine, in a scratch directory:

| Live test | Result |
| --- | --- |
| `qwen2.5-coder:7b` reads a file and answers | 2 steps, 54 tok/s |
| `qwen2.5-coder:7b` writes a working Python file | 2 steps, 61 tok/s |
| `qwen2.5-coder:0.5b` attempts the same task | 294 tok/s, loop detected and stopped cleanly |
| Approval classifier verdict | 188 ms |
| `rm -rf /` with the classifier enabled | refused |

**Every tool call `qwen2.5-coder:7b` made arrived as bare JSON in the message
body with an empty `tool_calls` field.** Without tolerant parsing this agent
would do nothing at all — which is the whole thesis, confirmed end to end.

**There is no chat UI yet** — the app lists models but you drive the agent from
tests. That, and the wider tool surface, are next; see
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## Requirements

- [Ollama](https://ollama.com) 0.32 or newer, with at least one model pulled
- Node.js 20+ and Rust 1.80+ to build from source

## Build

```bash
npm install
npm run tauri dev
```

Run the Rust test suite — the permission and parser tests are the ones worth
reading:

```bash
cd src-tauri && cargo test
```

Reproduce the benchmark table on your own hardware:

```bash
npm run bench
```

## Design

The interface follows the [cayadev.com](https://cayadev.com) palette; tokens are
lifted verbatim from the site's stylesheets into
[`src/styles/tokens.css`](src/styles/tokens.css) rather than re-picked by eye.

## License

MIT — see [LICENSE](LICENSE).
