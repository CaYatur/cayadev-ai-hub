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

That is a placement bug, not a property of speculation, and it is the thing
standing between a measured 96% draft acceptance and a real speedup. Two ways
out are under test: `--no-op-offload`, which stops host tensors having their
operations offloaded, and `-ngl 0`, which removes the split entirely.

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
