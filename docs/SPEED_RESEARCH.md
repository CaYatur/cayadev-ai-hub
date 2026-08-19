# Making a 27B faster on an 8 GB card — research log

[`SPEED.md`](SPEED.md) records the settled advice. This is the working log for
the open question behind it: whether a model far too large for the graphics card
can be made meaningfully faster, rather than only routed around.

Reference machine: RTX 5070 Laptop (8 GB), Ryzen 9 8940HX (16 cores), 28.7 GB
DDR5-5200 dual channel, PCIe 5.0 x8. Model: `qwen3.8:27b-mtp-q4_K_M`, 16.8 GB of
weights, 18.3 GB loaded at 8k context.

## Where the time actually goes

Every token of a dense model reads every weight. 16.8 GB at the ~55 GB/s a
dual-channel DDR5-5200 laptop achieves in practice is about 0.3 s per token —
roughly 3.3 tok/s for the part in system RAM. Measured throughput of 4–8 tok/s
with a quarter of the model on the card is consistent with that.

So the model is **memory-bandwidth bound, not compute bound**. That rules out
most tuning: threads, batch sizes and attention kernels all make the arithmetic
faster, and the arithmetic is not what is slow. It also says exactly which
levers can work:

1. **Read fewer bytes per token** — a smaller model or a smaller quantisation.
   This is routing, and it is already what the app does.
2. **Read them faster** — fit more in VRAM. Bounded by an 8 GB card.
3. **Produce more tokens per read** — speculative decoding. One weight-read
   verifies several tokens at once. This is the only lever that attacks the
   bound directly, and it is the subject of this log.

## Speculative decoding is available, and the model already carries the head

`qwen3.8:27b-mtp-q4_K_M` reports `qwen35.nextn_predict_layers = 1`: it ships
with a multi-token-prediction head, which is a draft model built into the
weights. No second model, no extra memory, no vocabulary mismatch.

- **Ollama cannot use it.** The strings `draft-mtp` and `spec-draft-n-max` are
  present in `ollama.exe`, so its embedded llama.cpp has the code, but Ollama
  builds the argument list itself and exposes no flag, no environment variable
  and no runner subprocess to inject one. Checked on 0.32.14.
- **llama.cpp can**, via `--spec-type draft-mtp --spec-draft-n-max N`, from
  b10488. It loads Ollama's own GGUF blob directly, so nothing is re-downloaded.

## Measured

Same session, back to back, memory verified free before each run. Decode tok/s
at 8k context, q8_0 KV, flash attention on. `toolcall` is JSON in the shape the
agent actually emits.

| Configuration | prose | code | toolcall | draft accepted |
| --- | ---: | ---: | ---: | ---: |
| **Ollama (what the app uses today)** | **4.20** | **6.19** | **7.62** | — |
| llama.cpp, 20/65 layers on GPU | 2.55 | 2.60 | 2.56 | — |
| llama.cpp + MTP, n=3, 20 layers | 4.09 | 6.21 | 6.64 | 0.96 |
| llama.cpp + MTP, n=3, 24 layers | 4.67 | 5.94 | 6.94 | 0.95 |

**MTP works, and works well** — 96% of drafted tokens are accepted, and it lifts
llama.cpp's own throughput by 1.6–2.6x. But llama.cpp's baseline is 2.4x *below*
Ollama's, so the two roughly cancel and the honest result against the path the
app uses today is a tie, not a speedup.

## Why llama.cpp starts so far behind

Its own load log says so:

```
resolve_fused_ops: layer 0 is assigned to device CPU but fused Gated Delta Net
  (chunked) is assigned to device CUDA0 (usually due to missing support)
```

Gated Delta Net is this architecture's linear-attention component;
`full_attention_interval = 4` means three of every four layers use it. With 20
of 65 layers on the card, the other 45 have their SSM operation shipped to the
GPU and back — 45 round trips over PCIe per token, on a model that was already
bandwidth bound.

Two ways out were tried, and neither closes the gap.

| Attempt | prose | code | toolcall | against 20-layer baseline |
| --- | ---: | ---: | ---: | --- |
| `-ngl 0` — no split at all | 1.93 | — | — | **worse**; the 20 GPU layers earn their keep despite the round trips |
| `--no-op-offload` — keep the split, stop shipping host ops | 2.82 | 2.75 | 2.80 | +11% / +6% / +9% |

`--no-op-offload` removes the warning entirely and is a real gain, but a small
one: the round trips were not the main cost. Pure CPU is worse still, so the
card is helping even while llama.cpp uses it badly.

### Verdict for the 27B

**There is no net speedup available on this model through llama.cpp today.**
Three independent attempts converge on it: speculation works beautifully (96%
acceptance, 1.6–2.6x over llama.cpp's own baseline) but that baseline starts
1.5x behind Ollama, and the two cancel. The blocker is llama.cpp's placement for
hybrid SSM models being slower than Ollama's, not speculation failing.

Recorded rather than hidden, because it answers the question directly: for
`qwen3.8:27b` on an 8 GB card, 9 tok/s cannot be doubled by changing runtime or
settings. It is bounded by reading 16.8 GB per token from DDR5.

What *would* change it, none of which is a setting:

- **A quantisation that fits.** ~7 GB of weights would put the whole model on
  the card and is worth the measured 8x. No published quant of this model is
  that small, and re-quantising from q4 loses more than it gains.
- **llama.cpp fixing hybrid-SSM placement.** Then MTP's 96% acceptance lands on
  a baseline that is not handicapped, and the 2x is real. Worth re-testing on
  each release; the measurement harness is in `scripts/`.
- **More memory bandwidth.** A desktop card with 16 GB+ ends the problem
  outright.

## What the numbers mean for the work the agent does

Throughput is a proxy. The question worth answering is how long a real job
takes, so the same instruction — "write a single-file HTML page introducing
yourself, inline CSS, at least 120 lines" — was run through `AgentRunner`
itself, same tools, same permission engine, only the model changed.

| Model | Job | Rate | Written |
| --- | ---: | ---: | --- |
| `qwen2.5-coder:7b` | **13 s** | 64.1 tok/s | 59 lines |
| `qwen3.8:27b` | **431 s** | 4.0 tok/s | 199 lines |

The pages differ in length, so the fair comparison is per line of output:
**2.17 s/line against 0.22 s/line — about ten times.** The 27B is not ten times
better at writing an HTML page.

That is the actual answer to "how do I make this faster", and it needs no new
runtime: the app already routes coding work to the 7B, and the measurement says
routing is worth more than every runtime trick attempted here combined. The
27B's value is judgement on hard problems, not throughput on ordinary ones.

## Where speculation *does* pay, and what is still open

On the 7B — which fits entirely in VRAM, so none of the placement problems
apply — llama.cpp measures 53.5 tok/s on prose and 39.5 on code with the whole
model on the card. Whether a vocabulary-matched drafter (`qwen2.5-coder:0.5b`,
0.4 GB, measured at 316 tok/s) lifts that further is the obvious next
experiment; `scripts/` carries the harness for it. It was not completed here:
the two models plus both KV caches sit close to the 8 GB limit and the runs
became unreliable before a trustworthy number came out. Anything reported from
those runs would have been a guess.

## Measurement discipline

Two mistakes early on, both worth recording because either invalidates a whole
session:

- **A leftover `llama-server` holding 14.5 GB.** With a 16.8 GB model on a
  28.7 GB machine, one stale process turns every later reading into a
  measurement of disk paging. `clean_bench.py` now kills everything, unloads the
  daemon, and asserts free memory before each configuration.
- **`-ngl 99` on a card that cannot hold the model.** Windows allows VRAM
  oversubscription and pages it over PCIe, which measured **0.69 tok/s** —
  six times *slower* than leaving the weights in RAM. Never offload more than
  fits; `llama-fit-params` computes the number.

Readings on this model move ±30% between sessions depending on what else holds
VRAM, so every number above is against a control taken minutes apart on the same
machine state, never against a figure from an earlier day.
