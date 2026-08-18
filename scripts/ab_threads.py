"""Interleaved A/B on the options that looked promising in the sweep.

A laptop GPU's clocks drift over a run, so measuring all of A then all of B
attributes drift to the option under test. Every comparison here alternates
A/B/A/B and reports medians, which cancels a monotonic drift.

Run: python scripts/ab_threads.py [model]
"""
import json
import statistics
import sys
import time
import urllib.request

HOST = "http://localhost:11434"
PROMPT = ("Write a Python function that reverses a linked list iteratively. "
          "Include a short docstring. Output only code.")


def chat(model, options, messages=None):
    body = {
        "model": model,
        "messages": messages or [{"role": "user", "content": PROMPT}],
        "stream": False,
        "think": False,
        "options": {"num_predict": 120, "temperature": 0.2, **options},
        "keep_alive": "20m",
    }
    req = urllib.request.Request(
        f"{HOST}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    out["_wall"] = time.time() - t0
    return out


def decode_tps(out):
    ec, ed = out.get("eval_count", 0), out.get("eval_duration", 1) / 1e9
    return round(ec / ed, 2) if ed else 0.0


def ab(model, label_a, opts_a, label_b, opts_b, rounds=4):
    """Alternate A and B, reporting medians."""
    print(f"\n=== {model}: {label_a} vs {label_b} ===", flush=True)
    a_vals, b_vals = [], []

    # Warm each configuration once. Switching num_thread can reload the model,
    # and a reload inside a measured run would dominate the result.
    chat(model, opts_a)
    chat(model, opts_b)

    for i in range(rounds):
        a = decode_tps(chat(model, opts_a))
        b = decode_tps(chat(model, opts_b))
        a_vals.append(a)
        b_vals.append(b)
        print(f"  round {i + 1}: {label_a}={a:<7} {label_b}={b}", flush=True)

    ma, mb = statistics.median(a_vals), statistics.median(b_vals)
    delta = round((mb - ma) / ma * 100, 1) if ma else 0
    print(f"  --> median {label_a}={ma}  {label_b}={mb}  delta {delta:+}%", flush=True)
    return ma, mb, delta


def prefix_reuse_by_time(model):
    """Prefix reuse, measured by prefill *time* rather than token count.

    Ollama reports the full `prompt_eval_count` even when it reuses the cache,
    so counting tokens says 'no reuse' while the clock says otherwise. Time is
    the honest signal.
    """
    print(f"\n=== {model}: prefix reuse (timing) ===", flush=True)
    bulk = ("You are a coding assistant working in a local project. "
            "Read before you write. Prefer the smallest change. ") * 60
    system = {"role": "system", "content": bulk}

    def prefill_seconds(user):
        out = chat(model, {"num_ctx": 8192, "num_predict": 4},
                   [system, {"role": "user", "content": user}])
        return out.get("prompt_eval_duration", 0) / 1e9, out.get("prompt_eval_count", 0)

    cold, n1 = prefill_seconds("Say A.")
    warm, n2 = prefill_seconds("Say B.")
    # A different system prompt cannot reuse the cache; this is the control.
    other = {"role": "system", "content": bulk + " Extra sentence."}
    out = chat(model, {"num_ctx": 8192, "num_predict": 4},
               [other, {"role": "user", "content": "Say C."}])
    changed = out.get("prompt_eval_duration", 0) / 1e9

    print(f"  cold prefix:    {cold * 1000:.0f} ms ({n1} tokens)", flush=True)
    print(f"  same prefix:    {warm * 1000:.0f} ms ({n2} tokens)", flush=True)
    print(f"  changed prefix: {changed * 1000:.0f} ms", flush=True)
    speedup = round(cold / warm, 1) if warm else 0
    print(f"  --> reusing an identical prefix is {speedup}x faster to prefill", flush=True)
    return speedup


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.8:latest"

    # The headline question: does pinning threads help a model that spilled?
    ab(model, "default", {"num_ctx": 8192}, "threads=16",
       {"num_ctx": 8192, "num_thread": 16})

    ab(model, "threads=16", {"num_ctx": 8192, "num_thread": 16},
       "threads=16+flash", {"num_ctx": 8192, "num_thread": 16, "flash_attn": True})

    prefix_reuse_by_time("qwen2.5-coder:7b")
