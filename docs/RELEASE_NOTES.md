# v0.1.0 — first release

A local-first agent runtime for Ollama, built around one finding: on consumer
hardware the dominant cost is *which model runs a step*, not how any single
model is tuned.

## What it is for

Pointing a general-purpose agent tool at a local model usually ends one of two
ways — the session dies before any work starts, or it crawls. Every cause was
measured on an 8 GB laptop GPU and fixed against a real model.

| Symptom | What it actually was |
| --- | --- |
| Session ends immediately | The model named a tool that does not exist, and the runner read that as an answer |
| "No tool calls" on a tool-capable model | `qwen2.5-coder:7b` returns bare JSON in the message body with an empty `tool_calls` field — *every single time* |
| Says hello, then takes a screenshot | Five false tool calls out of eight greetings, measured; every one a `web_search` for the greeting |
| Hangs, then dies | A total request timeout firing during a legitimately slow prefill |
| Painfully slow | A 27B model spilling to system RAM |

## Measured

| Model | Decode | TTFT | Placement |
| --- | ---: | ---: | --- |
| `qwen3.8` 27B Q4_K_M | 5.4 tok/s | 1.5 s | 80% CPU / 20% GPU |
| `qwen2.5-coder:7b` | 42.3 tok/s | 0.40 s | 100% GPU |
| `qwen2.5-coder:0.5b` | 316.3 tok/s | 0.19 s | 100% GPU |

- Fitting in VRAM is worth **~8x**. Context size across a 16x range changed
  nothing on the spilled model.
- A byte-identical prompt prefix prefills **14.8x** faster (481 ms → 32 ms).
- `num_thread` moved the needle 2.7% across interleaved runs — noise, and
  deliberately not shipped as a setting.

Full tables in [`docs/BENCHMARKS.md`](BENCHMARKS.md) and
[`docs/SPEED.md`](SPEED.md). Reproduce with `npm run bench`.

## In this release

**Reliability.** Tolerant tool-call parsing across six output formats with JSON
repair; invented tool names corrected instead of ending the run; loop detection;
schema validation; idle-based timeouts; a Stop button that stops mid-stream.

**Speed.** Task-based routing, GPU memory read from the driver, byte-stable
prompt prefixes, and a tool list trimmed to what the running model can choose
between.

**Safety.** Five permission modes including an autonomous one that trades
prompts for an audit trail. Component-wise workspace containment, shell payload
inspection, credentials and catastrophic commands refused in every mode,
publishing to GitHub confirmed even when unrestricted, and every agent edit
checkpointed and revertible.

**Reach.** Files and search, commands with streamed output, GitHub, web search
and page reading, documents, images, screenshots, audio transcription, video,
and Android over ADB. Turkish and English.

## Requirements

[Ollama](https://ollama.com) 0.32+ with at least one model pulled. Optional
extras — `gh`, Python packages, `ffmpeg`, `adb` — each hide their own tools when
absent rather than failing mid-task. See the README.

## Honest limits

- Tested on Windows with an NVIDIA GPU. Nothing is deliberately
  platform-specific, but nothing else has been exercised either.
- GPU probing is NVIDIA-only; other hosts fall back to a fixed assumption.
- The tool-budget numbers are judgement rather than measurement.
- Web search scrapes DuckDuckGo's HTML endpoint, which can change.
