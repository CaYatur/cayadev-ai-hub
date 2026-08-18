"""Measure which runtime knobs actually change local-model speed.

Every experiment reports a before/after pair on the same model in the same
session, so a result is a delta rather than an absolute worth arguing about.
Nothing here is assumed — options that turn out to do nothing are reported as
doing nothing, and the app does not ship a setting for them.

Run: python scripts/speed_experiments.py
"""
import json
import statistics
import sys
import time
import urllib.request

HOST = "http://localhost:11434"
CODER = "qwen2.5-coder:7b"
BIG = "qwen3.8:latest"

# ~2000 tokens of stable prefix, the shape an agent's system prompt takes.
BULK = ("You are a coding assistant working in a local project. "
        "Read before you write. Prefer the smallest change that solves the problem. ") * 60


def chat(model, messages, options=None, keep_alive="10m"):
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": options or {},
        "keep_alive": keep_alive,
    }
    req = urllib.request.Request(
        f"{HOST}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        out = json.loads(r.read())
    out["_wall"] = time.time() - t0
    return out


def rates(out):
    ec, ed = out.get("eval_count", 0), out.get("eval_duration", 1) / 1e9
    pc, pd = out.get("prompt_eval_count", 0), out.get("prompt_eval_duration", 1) / 1e9
    return {
        "decode_tps": round(ec / ed, 2) if ed else 0,
        "prefill_tps": round(pc / pd, 2) if pd else 0,
        "prompt_tokens": pc,
        "wall": round(out["_wall"], 2),
    }


def median_decode(model, options, runs=3):
    """Median of `runs` warm measurements, to survive laptop clock wobble."""
    chat(model, [{"role": "user", "content": "hi"}], options)  # warm
    values = []
    for _ in range(runs):
        out = chat(
            model,
            [{"role": "user", "content": "Write a Python function to reverse a list. Code only."}],
            {**options, "num_predict": 120, "temperature": 0.2},
        )
        values.append(rates(out)["decode_tps"])
    return round(statistics.median(values), 2), values


# ---------------------------------------------------------------------------


def experiment_prefix_reuse(model):
    """Does a byte-identical prefix skip re-prefill on the next turn?

    This is the one that matters most for agent loops: every step re-sends the
    whole conversation, so if the daemon does not reuse its KV cache the system
    prompt is re-read from scratch on every single step.
    """
    print(f"\n=== prefix reuse :: {model} ===")
    system = {"role": "system", "content": BULK}

    first = chat(model, [system, {"role": "user", "content": "Say A."}],
                 {"num_ctx": 8192, "num_predict": 8})
    r1 = rates(first)
    print(f"  turn 1 (cold prefix): prompt_tokens={r1['prompt_tokens']} "
          f"prefill={r1['prefill_tps']} tok/s wall={r1['wall']}s")

    # Same system prompt, byte for byte; only the user turn differs.
    second = chat(model, [system, {"role": "user", "content": "Say B."}],
                  {"num_ctx": 8192, "num_predict": 8})
    r2 = rates(second)
    print(f"  turn 2 (same prefix): prompt_tokens={r2['prompt_tokens']} "
          f"prefill={r2['prefill_tps']} tok/s wall={r2['wall']}s")

    # And a deliberately different prefix, as the control.
    third = chat(model,
                 [{"role": "system", "content": BULK + " Extra."},
                  {"role": "user", "content": "Say C."}],
                 {"num_ctx": 8192, "num_predict": 8})
    r3 = rates(third)
    print(f"  turn 3 (changed prefix): prompt_tokens={r3['prompt_tokens']} "
          f"prefill={r3['prefill_tps']} tok/s wall={r3['wall']}s")

    reused = r2["prompt_tokens"] < r1["prompt_tokens"] * 0.5
    print(f"  --> prefix reuse: {'YES' if reused else 'NO'} "
          f"({r1['prompt_tokens']} -> {r2['prompt_tokens']} tokens)")
    return reused


def experiment_flash_attn(model):
    """Is `flash_attn` honoured per request, or is the earlier row just noise?"""
    print(f"\n=== flash_attn per-request :: {model} ===")
    off, off_all = median_decode(model, {"num_ctx": 8192, "flash_attn": False})
    on, on_all = median_decode(model, {"num_ctx": 8192, "flash_attn": True})
    print(f"  off: {off} tok/s {off_all}")
    print(f"  on:  {on} tok/s {on_all}")
    delta = round((on - off) / off * 100, 1) if off else 0
    print(f"  --> delta {delta:+}%")
    return delta


def experiment_num_batch(model):
    """Does prefill batch size change time-to-first-token?"""
    print(f"\n=== num_batch :: {model} ===")
    system = {"role": "system", "content": BULK}
    results = {}
    for batch in (128, 512, 2048):
        # Vary the prompt so no run benefits from the previous one's cache.
        out = chat(model,
                   [system, {"role": "user", "content": f"Say {batch}."}],
                   {"num_ctx": 8192, "num_predict": 8, "num_batch": batch})
        r = rates(out)
        results[batch] = r["prefill_tps"]
        print(f"  num_batch={batch:<5} prefill={r['prefill_tps']} tok/s "
              f"(prompt_tokens={r['prompt_tokens']})")
    return results


def experiment_num_thread(model):
    """CPU thread count. Only expected to matter for a model that spilled."""
    print(f"\n=== num_thread :: {model} ===")
    results = {}
    for threads in (8, 16, 32):
        tps, _ = median_decode(model, {"num_ctx": 8192, "num_thread": threads}, runs=2)
        results[threads] = tps
        print(f"  num_thread={threads:<3} {tps} tok/s")
    return results


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("all", "prefix"):
        experiment_prefix_reuse(CODER)
    if which in ("all", "flash"):
        experiment_flash_attn(CODER)
    if which in ("all", "batch"):
        experiment_num_batch(CODER)
    if which in ("all", "thread"):
        # Threads only plausibly matter where the CPU is doing the work.
        experiment_num_thread(BIG)
