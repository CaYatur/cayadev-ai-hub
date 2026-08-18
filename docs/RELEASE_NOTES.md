# v0.2.2 — bound the wait by what the model may produce

A fix for long generations on slow or spilled models, replacing fixed timeouts
with a strict bound on generation length and a budget calculated from real throughput.

## Why the first fix did not hold

In v0.2.1 the budget was raised to a higher fixed number. But in real use, a spilled
27B model on a 65k context window stalled again at 623 seconds: without an explicit
`num_predict`, nothing bounded what the model was allowed to generate. At 5.7 tok/s,
filling a 65k window would mean over 3 hours of legitimate silence. No static number
can cover that.

## Calculated budget and explicit response cap

- **Response cap (`num_predict`)**: Requests now explicitly cap output tokens (default 8192,
  scaled down on smaller context windows). 8192 tokens provides ample headroom for long
  single-file code generation (~2500 tokens) while guaranteeing generation is bounded.
  If the cap is ever hit, the app reports it clearly instead of failing silently.
- **Dynamic budget**: The wait budget is now computed as `cap / model_throughput` plus
  margin, rather than a guessed constant.
- **Fail-fast liveness check**: During long silences, the daemon is periodically checked
  via `ps` (every 10s after 30s of quiet). If the model has been evicted from memory or
  the daemon crashed, the run fails immediately rather than waiting out the full budget.
- **Accurate retries**: Retries no longer drop conversation context when the model was
  already actively generating.

## Install

Windows: `.msi` or `.exe` below. Needs [Ollama](https://ollama.com) running with
at least one model pulled.

---

# v0.2.1 — a slow model is not a broken one

Four fixes from real use, three of them the same mistake in different places:
treating "this is taking a long time" as "this has failed".

## A big model writing a file was being killed

Asked for a single-file HTML page, `qwen3.8:27b` failed twice and lost the work
both times. Measuring the stream said why: the model streams its reasoning
normally, then sends **nothing at all for 547 seconds**, and finally delivers the
whole tool call in one chunk. Its template does not stream a tool call — it
buffers it. On a model running from system RAM at 5 tok/s, writing a long file
is nine minutes of complete silence, and the fixed 180-second budget read that
as a stall.

The budget is now sized to the model, and the unit is work rather than time:
enough silence to produce four thousand tokens at whatever rate this model
actually manages. A resident 7B keeps its three minutes — three minutes of
nothing from a 42 tok/s model really is broken. A spilled 27B gets a quarter of
an hour.

That rate is measured, not assumed. Every step reports one; the fastest from a
run is kept on the model and carried across a refresh, so from the second run
onwards the budget is against what this machine really does.

And waiting in silence no longer looks like hanging. The wait reports itself
every ten seconds, and the status line says which it is — the model loading, or
the model writing something it will not send until it is finished.

**Verified on the case that failed:** three steps, 199 lines written, 431
seconds, with a 340-second silence shown while it happened.

## "The vision model returned nothing"

Asked what was on screen, the agent captured it, got nothing back, and fell
back to listing window titles with PowerShell. Two causes, both measured:

- Vision was routed to the **largest** model that could see. Once a big model is
  loaded nothing else fits, so every vision model reads as spilled and "largest"
  picked the 17.7 GB one — **249 seconds** to read one screenshot against **16**
  for a 6 GB one. Vision now takes the *smallest* well-placed model; size buys
  nothing for reading a picture.
- The describing call left reasoning on, so a thinking model spent its whole
  reply budget reasoning and returned empty content. Reasoning is now off, the
  budget is large enough to answer, and if a model reasons anyway its reasoning
  is returned rather than thrown away.

A model that can see now reads its own screenshots — routing elsewhere is only
for models that cannot see at all.

## Screenshots covered one monitor

`PrimaryScreen` on a two-monitor desk returns a confident description of the
wrong screen, which is worse than an error. Capture now spans the whole virtual
desktop, including displays left of or above the primary one, and is scaled to
1920px before being sent — two 2560-wide screens is megabytes of base64 that no
model reads at full resolution.

## The instruction box was a two-line slot

It now grows upward as the instruction does, to about twelve lines, then stops
and scrolls.

## Also

The README has screenshots, rendered by `scripts/demo/` from the real components
over an example transcript — so a picture of the app cannot drift from the app.

## Install

Windows: `.msi` or `.exe` below. Needs [Ollama](https://ollama.com) running with
at least one model pulled.

---

# v0.2.0 — the things that broke in real use

v0.1.0 was measured against benchmarks. This one is measured against a week of
somebody actually using it. Every item below is a bug that showed up in normal
use, and every fix is verified against a real model rather than argued for.

## Replies were being cut off

Ollama trims a request that exceeds the window from the **front** — which is
where the system prompt and the task are — and its template then answers
`500 {"error":"no user query found in messages"}`. Long sessions died mid-reply.

The conversation is now trimmed before it is sent, from the middle, pinning the
system prompt, the current instruction and the recent steps. It cuts well under
the threshold rather than to it, so several steps pass before the next trim and
the daemon keeps reusing its KV cache. A stall gets one retry at half the window
instead of ending the run.

## The model could not remember the conversation

Every run started from an empty conversation. Asked *"Az önce ben sana ne
yazmıştım?"*, the model answered with that question — genuinely the only message
it had.

Earlier turns are now replayed between the system prompt and the instruction, so
the system prefix stays byte-stable and the cache still hits. Reasoning is left
out; tool calls are folded to one summary line each, so the model knows it
already wrote `test.txt` without carrying the four thousand characters a read
costs. Declined and failed calls are marked, so it does not retry what was
refused, and an attached image leaves a marker so a follow-up knows a picture
was there.

Verified on `qwen2.5-coder:7b`: a second turn recalls a number from the first,
and names a file the previous turn created.

## "I don't have access to your computer's screen"

Asked *"bilgisayarımın ekranında şuanda ne açık?"*, `gpt-oss:20b` reasoned its
way to that answer with `screenshot` sitting in its tool list. Its own reasoning
quoted the rule back: call a tool only for something "specific in this project",
and a screen is not a file in a folder.

The rule now names what else counts — a file here, the screen, a web page, a
connected phone — and adds one line: never claim you cannot see something you
have a tool for. The wording it replaces was measured, so the replacement was
measured the same way. `scripts/drift.py` gained a third category for things
that need a tool but are not in the project; the new wording holds **zero false
calls out of eight** conversational inputs on the 7B, where the old one fails the
screen question on gpt-oss:20b.

## An attached image took two hops and a guess

Attaching a picture put a line in the message saying where the file was, and the
coding model had to decide to call `view_image`, which went to a vision model and
came back. The first hop is a decision, and deciding is what small models get
wrong.

The image now goes in the message, on a model that can see it — the run is handed
to one if the chosen model cannot. Some vision models cannot call tools at all
(`qwen2.5vl:7b` advertises `completion` and `vision` and nothing else, and Ollama
refuses a request that offers tools to it), so those runs carry no tools field at
all. Measured: a red square, one step, no tool call, answer "Red" at 66 tok/s.

## A console window flashed on every message

A Windows GUI app has no console, so one was created for every child process. All
spawning now goes through one place that sets `CREATE_NO_WINDOW`, so a new call
site cannot forget.

## The agent read the app's own notes

Undo history lived in `.cayahub/` inside the project. The agent found it, read
the journal, and spent steps reasoning about it — and it showed up in searches
and in git status. It now lives under the OS data directory, one directory per
workspace, named and hashed so two projects called `app` stay separate.

## New

- **`ask_user`** — the agent asks instead of guessing. Choices or a written
  answer, and Skip is reported as a skip so it proceeds on its own judgement
  rather than reading silence as agreement. Verified: given two plausible config
  files and nothing to choose between them, the 7B asks, takes the answer, and
  reads the one picked.
- **Markdown replies** — headings, lists, tables, syntax-coloured code against
  the app's own palette, and links that open: a URL in the browser, a file path
  in its application. Paths in backticks are clickable; `array.map` is not.
- **Live tok/s and context occupancy**, timed from the first token so a cold
  model load is not averaged into the rate.
- **Following the output properly.** The old effect fired on the entry count,
  which does not change while a reply streams into the last entry, and used
  smooth scrolling, which restarts on every token and never arrives. The
  reasoning box scrolls inside itself and needed the same treatment separately.
- **Attachments belong to their message** — shown with a preview on the turn
  that sent them, gone from the composer once sent, and not silently re-sent
  with the next question. Sizes read "4.2 KB" rather than "0 MB".
- **Stop reaches a parked run** — one waiting on an approval or a question, not
  only one mid-token.
- A stall now says which kind it was: nothing at all means the weights are still
  loading or do not fit; a gap mid-reply means they fit badly and are being
  paged.

## Tested

206 unit tests, and 21 live tests against `qwen2.5-coder:7b`, `qwen2.5vl:7b`,
`qwen2.5-coder:0.5b` and `gpt-oss:20b` on an 8 GB card.

## Install

Windows: `.msi` or `.exe` below. Needs [Ollama](https://ollama.com) running with
at least one model pulled.

---

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
