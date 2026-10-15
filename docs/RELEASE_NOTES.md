# v0.2.9 - long runs no longer lose the question

On a small context window with an MCP server connected, a run could stop with
`Ollama returned 500: {"error":"no user query found in messages"}`. The
request had outgrown the window, and Ollama cut it from the front, taking the
user's instruction with it.

## What changed

- Tool schemas sent to the model now count against the window when the
  conversation is trimmed. Before, they were left out, and with many MCP
  tools on an 8K window they took a large share of it.
- When dropping older steps is not enough, the longest messages are shortened,
  each with a note saying how much was left out. The system prompt and the
  current instruction are never shortened.
- This applies to every model, MCP server and tool.

## Install

Windows: `.msi` or `.exe` below. Needs [Ollama](https://ollama.com) running with
at least one model pulled. MCP servers started through `npx` need Node.js.
# v0.2.8 - crashes are caught and logged

The app could vanish mid-use with no message. The release build aborted on any
panic, and nothing was ever written to disk, so there was no trace to follow.

## The cause that was found

With more than 40 MCP tools connected, the Full prompt tier sized a list by an
unlimited budget and the allocation overflowed ("capacity overflow"). That run
now works, and a test covers it.

## A crash no longer closes the app

- A panic now fails only the current run, with an error message in the chat.
- Panics, render errors and unhandled promise rejections are written to
  `crash.log` in `%LOCALAPPDATA%\com.cayadev.aihub\logs` (rotated at 1 MB).
- A render error shows a recovery screen with "try again" and "reload"
  instead of a blank window.

## Smaller fixes

- Turkish characters and emoji are no longer cut in half when text is
  shortened, which could break a saved conversation.
- Tool summaries no longer assume a model's arguments are strings.

## Install

Windows: `.msi` or `.exe` below. Needs [Ollama](https://ollama.com) running with
at least one model pulled. MCP servers started through `npx` need Node.js.
# v0.2.7 — MCP results no longer flood the model

An MCP tool can return far more than a model can usefully read. A music
visualizer's "now playing" tool answered with the cover art inlined as a
292 KB base64 data URI; a small local model received all of it, lost the
question, and answered about something unrelated. Nothing in the app limited
what an MCP result could put into the conversation.

## Binary data becomes a short note

Every MCP result and error is now cleaned before the model sees it, whatever
server or tool it came from.

- Data URIs and long base64 runs, in plain text or anywhere inside nested
  JSON, are replaced with a note such as `[image omitted, image/png, 292 KB]`.
  Hashes, short ids, prose and file paths are left as they are.
- Image, audio and binary resource blocks become the same kind of note.

## Size limits that follow the model

- Arrays keep their first 25 items, objects their first 60 keys, strings
  their first 2,000 characters, and nesting stops at depth 12; each cut says
  how much was left out.
- The whole result is capped relative to the model's context window, between
  2,000 and 24,000 characters (about 5,900 for an 8K context).

Built-in tools are unchanged, so file reads stay exact for edits.

## Install

Windows: `.msi` or `.exe` below. Needs [Ollama](https://ollama.com) running with
at least one model pulled. MCP servers started through `npx` need Node.js.

# v0.2.6 — the agent can use MCP tools

Everything the agent could do was built into the app: files, commands, GitHub,
the web, documents, the screen. A tool that lived anywhere else - a database, a
ticket tracker, a company API - needed a code change. The Model Context Protocol
is the standard way to offer such tools, and the app had no way to reach it:
no MCP client, and nowhere to add a server.

## An MCP page

A new **MCP** tab lists your servers. **Add server** opens a form, and two
presets (**Everything** for a quick test, **Filesystem** starting at your
Documents folder) fill it in for you.

- **Test connection** starts the server once and lists the tools it offers,
  before anything is saved.
- **Agent can use this server** (*Ajan bu sunucuyu kullanabilir*) attaches or
  detaches a whole server; a tick box per tool switches single tools off.
- Servers start the first time the agent needs one, not when the app opens, and
  every server process is closed when the app exits.
- A server that fails to start is reported in the chat; the run continues with
  the tools that did load.

Settings are saved as `mcp.json` in the app's config folder.

## Two transports

| Transport | For | Settings |
| --- | --- | --- |
| stdio | A local command | command, arguments, environment, working directory |
| Streamable HTTP | A remote server | URL, headers, bearer token |

`npx` and other `.cmd` launchers work as commands on Windows. The older
HTTP+SSE transport is not supported. Every connect, list and call has a timeout
(per server, 60 s by default), and a call that finds its connection dropped
reconnects once before reporting an error.

The client is the official Rust SDK, `rmcp`.

## Permissions

An MCP tool does whatever its server says it does, so the rules lean on what
the server declares and on what you decided:

- Tools the server marks **read-only** run without asking in Project auto and
  Balanced.
- Every other tool asks first, unless the server is set to **Run tools without
  asking** (*Araçları sormadan çalıştır*).
- Autonomous and Unrestricted run them like any other tool.

## Tools for models that cannot call them

A model without native tool calling used to get no tools at all. It now gets
all of them, built-in and MCP, described in the system prompt with one strict
format:

```json
{"name": "<tool name>", "arguments": {...}}
```

Results come back as ordinary messages, which every template understands.
The existing parser already rescues fenced blocks, trailing prose and broken
JSON, so a near miss still becomes a call. Models with native tool calling are
unchanged, and a vision-only model asked about an attached image still gets no
tools.

MCP tools get their own room in the trimmed tool list - 4, 10 or 40 depending on
model size - so attaching a server does not push out the built-in tools, and a
small model is not handed forty schemas.

## Arguments are checked before the call

Every call, built-in or MCP, is checked against the tool's input schema:
required arguments, types, allowed values and unknown keys. Harmless slips are
repaired in place - `"5"` for 5, `"true"` for true, one value where a list was
expected. Anything else goes back to the model as one line it can act on, with
the signature it should have used:

```text
Error: invalid arguments for `mcp__files__search`: `depth` must be integer, got a string ("deep").
Call it again with corrected arguments. Expected: mcp__files__search(path: string*, depth: integer).
```

## Install

Windows: `.msi` or `.exe` below. Needs [Ollama](https://ollama.com) running with
at least one model pulled. MCP servers started through `npx` need Node.js.

# v0.2.5 — decide when a thinking model may reason

Reasoning models like DeepSeek-R1 and QwQ can spend hundreds of tokens thinking
before answering. Previously, whether the model was allowed to reason was tied
strictly to the Effort setting: Thorough turned thinking on where supported,
while Fast and Balanced forced it off.

If you wanted Balanced's 80-step leash but needed the model to think through a
complex refactor, or wanted Thorough's 200-step ceiling without paying the
wall-clock penalty of chain-of-thought tokens on every turn, there was no way to
separate them.

## Explicit thinking toggle in the effort menu

The Effort popover now inspects whether the active model advertises thinking
capabilities (`capabilities.thinking`).

- When the model **supports thinking**, a toggle appears in the Effort menu
  below the step presets: **Thinking support** (*Düşünme desteği*).
- Clicking the toggle lets you turn reasoning on or off independently of the
  step ceiling.
- Switching presets (Fast, Balanced, Thorough) sets sensible defaults (Fast and
  Balanced disable thinking; Thorough enables it), which you can then toggle at
  will.
- When the model **does not support thinking**, the option is hidden so the
  interface stays clean.

## Honest request flags to Ollama

When thinking is enabled on a capable model, the runner sends `"think": true`.
When thinking is turned off, the runner now explicitly sends `"think": false`,
instructing Ollama to suppress chain-of-thought tokens rather than relying on
the model's built-in prompt template default. Non-thinking models omit the field
entirely.

## Install

Windows: `.msi` or `.exe` below. Needs [Ollama](https://ollama.com) running with
at least one model pulled.

# v0.2.4 — a run that is working is not stopped

A 20B repairing a Python file ran for fifteen minutes and stopped with
`budget exhausted`, the file half-fixed. The worst outcome available: the wait
was paid and the work was lost. Three separate limits were waiting to end that
run, and only one of them was visible.

## The ceiling was sized for a model going in circles

Balanced allowed 24 tool steps. That number was chosen against a 0.5B calling
the same tool over and over, where ending in seconds is a mercy. A repair is
read, edit, run, read the traceback, edit again — thirty to eighty steps before
anything is finished.

| Effort | Was | Now |
| --- | ---: | ---: |
| Fast | 8 | 24 |
| Balanced | 24 | 80 |
| Thorough | 48 | 200 |

The ceiling is there to bound a run nobody is watching. It is not a judgement
about whether the work is done, so it now sits far above what real work costs.

## Raising it alone would have moved the wall, not removed it

Loop detection counted identical calls **cumulatively across the whole run**,
three tolerated. In the run that prompted this, `python ringtone_generator.py`
had already failed twice. On a longer leash the fourth run of that command would
have ended the session around step 30 with `loop detected` — the same failure
wearing a different name.

The guard tells the model *the result will not change*, and that is only true
while nothing has changed. So the memory now resets whenever the run changes
something:

- Re-running the same failing command after an edit is **repair**, and the count
  starts again.
- The same call four times with **nothing changed in between** still stops the
  run.
- A change cannot excuse its own repetition: writing the identical file over and
  over keeps its own count, so it still trips.

The limit this leaves is written down rather than hidden. Two calls that both
change something can alternate indefinitely — edit, run, edit, run — and that is
also exactly what a real repair looks like, so the guard cannot tell them apart
and does not try. The step ceiling is what bounds it.

### A trap inside the fix

The first version asked the permission engine whether the call was risky.
`Evaluation::allow` carries `ReadOnly` whatever it allowed, so an auto-approved
write inside the workspace — Autonomous mode, the ordinary case — reported that
nothing had changed. The signal is now the action itself, and a test asserts
both halves in the same breath: the evaluation says `ReadOnly`, the action says
it changed something.

## One stall no longer costs the whole run

A stall was tolerated once per run. On an 80-step run that means a single
transient stall at step 70 discards an hour of finished work. Retries are now
counted consecutively: two stalls in a row is a real failure, a stall at step
five and another at step seventy are unrelated events.

## The limit is visible, and says so in Turkish

The status line carries the step count against the ceiling — `adım 31/80` —
amber once the run is against it. A wall you can watch approach is not the same
thing as one that lands without warning.

And the stop reasons are translated. `Durdu: budget exhausted` was raw English
in a Turkish interface, and said nothing about what to do; it now reads *Durdu:
adım tavanına ulaşıldı. Kaldığı yerden sürdürmek için "devam et" yazın* — and
typing that does continue, because earlier turns are replayed with each tool
call folded to one line.

## Install

Windows: `.msi` or `.exe` below. Needs [Ollama](https://ollama.com) running with
at least one model pulled.

---

# v0.2.3 — a working model is no longer cut off, and it shows its work

v0.2.2 bounded the wait by what the model may produce, which was the right idea
and still killed a run at the thirty-minute mark. Two arithmetic errors, both
now fixed and both covered by tests that name the runs they came from.

## The budget was sized from the fastest step

For a timeout that is backwards. It has to be the pessimistic figure, because
sizing from a fast step and then meeting a slow one is exactly how a working run
gets killed. The same model reported **6.4 tok/s on its short steps and 4.0
while writing a long file** — 6.4 gives a thirty-minute budget, and the file
needed longer.

It is now the slowest step of the run, ignoring steps too short to mean
anything: a handful of tokens around a tool call reports a rate dominated by
overhead, not by decoding.

## And the cap was below the honest bound

2400 s. A 27B decoding at 4 tok/s cannot finish the reply it is *allowed* to
produce in forty minutes, so the cap fired before the model could possibly have
been done. A cap under the real bound is the worst of both — the user waits
*and* loses the work.

The margin over the estimated rate goes from 1.4x to 3x, and the cap now covers
a model managing one token a second:

| Model | Silence tolerated |
| --- | ---: |
| 7B, wholly in VRAM (42 tok/s) | 10 min |
| 27B, short steps (6.4 tok/s) | 64 min |
| 27B, writing a long file (4.0 tok/s) | 102 min |
| 27B under memory pressure (2.0 tok/s) | 120 min |

The two errors are not symmetric. Being generous costs a wait that Stop ends at
any moment; being tight costs an hour of finished work, and that had happened
three times. Detecting a genuinely dead run is the liveness check's job — the
daemon is asked every minute whether it still holds the model — not the
timeout's.

## The wait now shows progress

A twenty-minute silence looked identical to a hang. Nothing can be streamed —
the template holds the whole tool call back until it is finished — but both
terms of an estimate are known: how long it has been quiet, and how fast this
model decodes. The status line now reads

> ● writing — this model sends nothing until it is done (12m 4s) · **~2.9k written** ▬▬▬▬▭▭▭

with the bar against the reply allowance, and the hover text saying it is an
estimate and how it is calculated.

## A price tag on a model that does not fit

Choosing a model too large for the card now shows what it costs, measured
through the app's own agent code on the same job:

| Model | Job | Per line |
| --- | ---: | ---: |
| `qwen2.5-coder:7b` | 13 s | 0.22 s |
| `qwen3.8:27b` | 356 s | 2.25 s |

Ten times per line written, twenty-seven times the wall clock. The selection
still stands — it is a price tag, not a warning.

## Why there is no speedup in this release

A long investigation into making a too-large model faster is written up in
[`SPEED_RESEARCH.md`](docs/SPEED_RESEARCH.md), including the parts that failed.
The short version: **no runtime change tested beats Ollama on either model.**
Speculative decoding with the model's own MTP head accepts 96% of its drafts and
still nets nothing, because llama.cpp's placement for hybrid SSM models starts
1.5x behind. On the 27B the bound is physical — 16.8 GB read per token out of
DDR5.

One reported result was wrong and is corrected there: an n-gram speedup on the
7B did not reproduce under a proper two-round comparison. The measurement
harnesses are kept in `scripts/` so the question can be re-asked on a later
llama.cpp.

## Install

Windows: `.msi` or `.exe` below. Needs [Ollama](https://ollama.com) running with
at least one model pulled.

---

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
