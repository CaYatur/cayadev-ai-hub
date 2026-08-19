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

| Model | Job | Rate | Written | Per line |
| --- | ---: | ---: | --- | ---: |
| `qwen2.5-coder:7b` | **13 s** | 64.1 tok/s | 59 lines | 0.22 s |
| `qwen3.8:27b` | **356 s** | 4.1 tok/s | 158 lines | 2.25 s |

Twenty-seven times the wall clock, and the pages differ in length, so the fair
comparison is per line written: **0.22 s against 2.25 s — ten times.** The 27B
is not ten times better at writing an HTML page.

The same run measured on a separate occasion at 431 s for 199 lines — 2.17 s per
line. Two sittings, the same ratio.

That is the actual answer to "how do I make this faster", and it needs no new
runtime: the app already routes coding work to the 7B, and the measurement says
routing is worth more than every runtime trick attempted here combined. The
27B's value is judgement on hard problems, not throughput on ordinary ones.

## The 7B: no speedup either, and a correction

A first pass measured n-gram speculation on `qwen2.5-coder:7b` at 66.6 tok/s
against a 39.6 baseline and it was written up here as a 1.4–1.7x. **That does
not reproduce.** Re-measured properly — two rounds, interleaved, and against
Ollama rather than against llama.cpp's own baseline:

| Configuration | code (r1/r2) | toolcall (r1/r2) |
| --- | ---: | ---: |
| **Ollama** | **60.9 / 50.9** | **41.6 / 41.2** |
| llama.cpp, whole model on GPU | 42.8 / 46.8 | 39.1 / 39.7 |
| llama.cpp + `ngram-cache`, n=5 | 47.0 / 45.6 | 40.9 / 40.2 |

n-gram against llama.cpp's own baseline is inside the round-to-round spread —
no gain. Against Ollama, which is what the app uses, it is *slower*. The 66.6
reading came from the same batch that reported an impossible 18.4 tok/s for the
baseline on the same prompt; that batch was contaminated and both numbers should
have been discarded together.

The failure mode is worth naming: **a single reading that agrees with what you
hoped for does not get checked as hard as one that does not.** The 18.4 looked
wrong so it was re-run; the 66.6 looked right so it was published. Interleaved
rounds are now the protocol precisely because they make that impossible — a
configuration is measured twice, alternating with its control, and disagreement
between rounds is a result about the measurement rather than about the model.

### What does still hold from that work

- A 0.5B drafter is **worse than no drafter on prose** (31.7 against 48.6):
  every rejected draft is a wasted verification pass, and prose rejects most of
  them. Measured 31% acceptance on prose against 65% on code.
- Speculation is **not bit-lossless** in llama.cpp even at temperature 0. The
  same tool-call prompt produced a different page title with n-gram than
  without, while running the same configuration twice was byte-identical. The
  control is what makes that meaningful. Probably numerical — verifying a batch
  does not take the same floating-point path as decoding one token at a time,
  and a near-tie flips. `scripts/spec_lossless.py` keeps the check.

## Verdict for the whole investigation

**No runtime change tested here beats Ollama on either model.** Not speculative
decoding with the model's own MTP head, not a separate drafter, not n-gram, not
tensor placement, not KV quantisation, not thread or batch tuning. On the 27B
the bound is physical — 16.8 GB read per token out of DDR5. On the 7B, Ollama is
already at the top of what was measured.

The one thing that does move the number by an order of magnitude is choosing a
model that fits in the card, which the app already does and now puts a price on.

## Measurement discipline## Measurement discipline

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
- **Publishing a single reading that agreed with the hypothesis.** The 7B n-gram
  result was written up as a 1.4–1.7x from one batch, and did not survive a
  second look. The reading that looked wrong in that same batch was re-run; the
  one that looked right was not. `scripts/spec_confirm.py` now measures each
  configuration twice, interleaved with its control, so a number has to agree
  with itself before it means anything.

Readings on this model move ±30% between sessions depending on what else holds
VRAM, so every number above is against a control taken minutes apart on the same
machine state, never against a figure from an earlier day.
