# Making local models faster

Every knob below was measured on the reference machine (RTX 5070 Laptop, 8 GB
VRAM). Reproduce with `python scripts/speed_experiments.py` and
`python scripts/ab_threads.py`.

The short version: **there is no runtime setting that meaningfully speeds up a
model too large for your VRAM.** The wins that exist are structural — which
model runs, whether it stays loaded, and whether the prompt prefix changes
between turns. They are listed here largest first.

## What works

### 1. Route to a model that fits — worth ~8x

| Model | Placement | Decode |
| --- | --- | ---: |
| `qwen3.8` 27B Q4_K_M | 80% CPU / 20% GPU | 5.4–8.6 tok/s |
| `qwen2.5-coder:7b` | 100% GPU | 42–57 tok/s |

This dwarfs everything else on this page combined. It is why `RoutingPolicy`
ranks placement above model size, and why the app reports placement per model
instead of just listing names.

### 2. Keep the system prompt byte-identical — ~3.4x faster prefill per step

An agent re-sends the whole conversation every step. If the prefix is unchanged,
Ollama reuses its KV cache:

| Turn | Prefill time | Wall |
| --- | ---: | ---: |
| Cold prefix (1517 tokens) | 463 ms | 0.74 s |
| Same prefix, different question | 32 ms | 0.22 s |
| Prefix changed by one sentence | — | 0.23 s |

**Ollama still reports the full `prompt_eval_count` when it reuses the cache**,
so counting tokens says "no reuse" while the clock says otherwise. Measure this
by time, not token count — an easy way to conclude the wrong thing.

This is why [`prompt.rs`](../src-tauri/src/agent/prompt.rs) keeps volatile
content — timestamps, step counters, file lists — out of the system prompt, and
why a test asserts the prompt is byte-identical across calls.

### 3. Keep the model loaded — saves ~30 s per avoided reload

A cold 27B load costs about 30 seconds. `RunConfig::keep_alive` defaults to 15
minutes so a multi-step task never pays it twice.

### 4. Close other GPU applications — larger than any tuning knob

Observed over one working session on the reference machine: VRAM used by *other*
applications moved from **1787 MiB to 5696 MiB**. On an 8151 MiB card that is
the difference between a 6 GB model fitting and not fitting, and it changes
throughput more than every option below combined.

It also explains why the 27B measured 4.5–5.9 tok/s early in a session and
8.4–9.1 tok/s later: different amounts of it were on the GPU. Any single reading
of a spilled model is worth ±30%.

The app reads the budget from the driver at startup and offers a re-read, so
placement estimates track reality instead of a fixed assumption.

## What does not work

Measured, found wanting, and deliberately not shipped as settings.

| Knob | Effect | Verdict |
| --- | ---: | --- |
| `num_thread` (8 / 16 / 32 on the spilled 27B) | +2.7% median | Noise. No setting. |
| `flash_attn` on the 7B | +3.8% | Real and honoured per request, but small. On by default, not exposed. |
| `num_batch` 512 → 2048 | −2% | Nothing beyond 512. |
| `num_ctx` on a spilled model | 0% across a 16x range | See [BENCHMARKS.md](BENCHMARKS.md). |

`num_batch` 128 → 512 *is* worth +69% on prefill (1754 → 2973 tok/s), but 512 is
already Ollama's default — the finding is "do not lower it", not "raise it".

## What we checked and could not use

- **Smaller quantisations of `qwen3.8`.** Only `q4_K_M` and `q8_0` are published;
  there is no q3 or q2 to drop to.
- **The MTP build.** `qwen3.8:27b-mtp-q4_K_M` resolves to the same digest as
  `qwen3.8:latest`, so the default already carries the multi-token-prediction
  head and there is no separate variant to switch to.

## Where the remaining wins are

Not runtime flags — loop design:

- **Fewer output tokens per step.** Decode is the slow half. Prompts that ask for
  terse tool calls and no narration finish sooner than any flag makes them.
- **Fewer steps.** Loop detection and schema validation exist partly for this:
  a step spent on a malformed call is a whole round-trip wasted.
- **Smaller tool schemas.** `compact_tools` trims descriptions for small models,
  which both shortens prefill and keeps the model inside its effective
  instruction window.
