# Benchmarks

Every number here was measured on the development machine with
[`scripts/bench.py`](../scripts/bench.py). Nothing is extrapolated or claimed.
Reproduce with `npm run bench`.

## Reference machine

| Component | Spec |
| --- | --- |
| GPU | NVIDIA RTX 5070 Laptop — **8 GB VRAM** (~6.3 GB free at idle) |
| CPU | AMD Ryzen 9 8940HX, 16C / 32T |
| RAM | 28.7 GB |
| Ollama | 0.32.14 |
| OS | Windows 11 |

## Results

Decode throughput, `think: false`, `num_predict: 160`, measured on a warm model
so load time is excluded.

| Configuration | Decode | TTFT | Footprint | Placement |
| --- | ---: | ---: | ---: | --- |
| `qwen3.8` (27B Q4_K_M) @ ctx 65536 *(daemon default)* | 5.51 tok/s | 0.99 s | 20 GB | 80% CPU / 20% GPU |
| `qwen3.8` @ ctx 16384 | 4.48 tok/s | 1.57 s | 18 GB | 80% CPU / 20% GPU |
| `qwen3.8` @ ctx 8192 | 5.42 tok/s | 1.49 s | 18 GB | 80% CPU / 20% GPU |
| `qwen3.8` @ ctx 4096 | 5.07 tok/s | 0.95 s | 18 GB | 80% CPU / 20% GPU |
| `qwen3.8` @ ctx 8192 + flash_attn | 5.93 tok/s | 0.95 s | 18 GB | 80% CPU / 20% GPU |
| **`qwen2.5-coder:7b` @ ctx 8192** | **42.3–45.9 tok/s** | **0.40 s** | 4.8 GB | 100% GPU |
| `qwen2.5-coder:7b` @ ctx 32768 | 57.32 tok/s | 0.40 s | 6.0 GB | 100% GPU |
| `qwen2.5-coder:7b` @ ctx 16384 | 57.10 tok/s | 0.43 s | 6.0 GB | 100% GPU |
| `qwen3-vl:8b` @ ctx 8192 | 21.33 tok/s | 0.46 s | 6.3 GB | 100% GPU |
| `qwen2.5-coder:0.5b` @ ctx 8192 | 316.31 tok/s | 0.19 s | 532 MB | 100% GPU |

## What the numbers mean

**Shrinking the context window does not speed up a spilled model.** The four
`qwen3.8` context rows span a 16x range of `num_ctx` and land within noise of
each other (4.5–5.5 tok/s). This surprises people, and it is architecture
specific: `qwen3.8` reports `full_attention_interval: 4`, so roughly a quarter
of its 65 layers carry a KV cache and the rest hold SSM state. The KV cache was
never the thing filling VRAM — the 17 GB of weights was. Advice to "lower your
context for speed" is wrong for this model, and the app does not give it.

**Placement is the whole story.** A 27B Q4_K_M model needs ~17 GB of weights and
the card has 8 GB, so ~80% of every token's work happens on the CPU at system
RAM bandwidth. No sampling flag, cache quantisation, or context setting changes
that. Flash attention bought 0.5 tok/s — real, and nowhere near enough to
matter.

**Fitting in VRAM is worth roughly 8x.** `qwen2.5-coder:7b` is 4.7 GB, lands
entirely on the GPU, and reproduces at 42–46 tok/s against the 27B's 5.4.
Time-to-first-token drops from ~1.0 s to ~0.4 s at the same time.

The 7B row spans 42–57 tok/s across runs. Three consecutive warm runs at ctx
8192 gave 45.9 / 42.3 / 42.3, so **42 tok/s is the figure to plan around** and
57 is a favourable outlier — most likely clock and thermal state on a laptop
GPU. The claim in the README uses the reproduced number, not the best one.

**This is why the hub routes by task instead of picking one model.** A 0.5B
model classifying a tool call at 316 tok/s costs milliseconds; spending a 27B
turn on the same decision costs seconds. The policy lives in
[`registry.rs`](../src-tauri/src/ollama/registry.rs) (`RoutingPolicy::derive`)
and is derived from whichever models you actually have installed.

## Honest limits

- These are single-machine numbers. An RTX 4090 (24 GB) fits `qwen3.8` entirely
  in VRAM and would invert the table's conclusion for that model.
- Software cannot make a memory-bandwidth-bound model fast. What the hub can do
  is avoid sending work to a spilled model when a resident one would do, keep
  the resident model loaded, and stop re-prefilling a prompt that has not
  changed.
- `qwen3.8:27b-mtp-q4_K_M` resolves to the same digest as `qwen3.8:latest` —
  the default build already carries the MTP head, so there is no separate
  speculative-decoding variant to switch to.
- Laptop GPU throughput is not stable run-to-run. The 7B measured 30.6 tok/s
  once (first run after a model swap), 42–46 across three later warm runs, and
  57 twice at larger contexts. Any single reading here is worth ±25%; the
  conclusion survives because the gap to the 27B is nearly an order of
  magnitude, not because any one number is precise.
