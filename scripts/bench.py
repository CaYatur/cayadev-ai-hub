"""Throughput matrix for local Ollama models.

Every configuration pins num_ctx explicitly and records the context/placement
Ollama actually chose (via `ollama ps`), so before/after rows stay comparable.
`think` is recorded per row because it changes token count enough to swamp the
effect being measured.
"""
import json
import re
import subprocess
import sys
import time
import urllib.request

HOST = "http://localhost:11434"
PROMPT = (
    "Write a Python function that reverses a linked list iteratively. "
    "Include a short docstring. Output only code."
)
RESULTS = []


def ps_state():
    """Read SIZE / PROCESSOR / CONTEXT that Ollama actually allocated."""
    try:
        out = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:  # noqa: BLE001
        return {}
    for line in out.splitlines()[1:]:
        if not line.strip():
            continue
        m = re.match(r"(\S+)\s+(\S+)\s+([\d.]+\s*\w+)\s+(\S+)\s+(\d+)", line)
        if m:
            return {"size": m.group(3), "processor": m.group(4), "ctx": int(m.group(5))}
    return {}


def run(model, num_ctx, think=False, extra=None, label=""):
    opts = {"num_ctx": num_ctx, "num_predict": 160, "temperature": 0.2}
    opts.update(extra or {})
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "think": think,
        "options": opts,
        "keep_alive": "10m",
    }
    req = urllib.request.Request(
        f"{HOST}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1200) as resp:
            out = json.loads(resp.read())
    except Exception as exc:  # noqa: BLE001
        row = {"label": label, "model": model, "num_ctx": num_ctx, "error": str(exc)[:140]}
        print(json.dumps(row), flush=True)
        RESULTS.append(row)
        return
    wall = time.time() - t0

    ec, ed = out.get("eval_count", 0), out.get("eval_duration", 1) / 1e9
    pc, pd = out.get("prompt_eval_count", 0), out.get("prompt_eval_duration", 1) / 1e9
    load = out.get("load_duration", 0) / 1e9
    row = {
        "label": label,
        "model": model,
        "num_ctx_req": num_ctx,
        "think": think,
        "decode_tps": round(ec / ed, 2) if ed else 0,
        "prefill_tps": round(pc / pd, 2) if pd else 0,
        "decode_toks": ec,
        "load_s": round(load, 2),
        "ttft_s": round(load + pd, 2),
        "wall_s": round(wall, 2),
        **ps_state(),
    }
    print(json.dumps(row), flush=True)
    RESULTS.append(row)


def warm(model, num_ctx, extra=None):
    """Load into VRAM at this exact config so load time is excluded from the measured run."""
    run(model, num_ctx, extra=extra, label="__warm__")
    RESULTS.pop()


if __name__ == "__main__":
    BIG = "qwen3.8:latest"
    SMALL = "qwen2.5-coder:7b"
    VL = "qwen3-vl:8b"
    TINY = "qwen2.5-coder:0.5b"

    plan = [
        # Big model: does shrinking num_ctx move layers back onto the GPU?
        (BIG, 65536, {}, "big/ctx65536-default"),
        (BIG, 16384, {}, "big/ctx16384"),
        (BIG, 8192, {}, "big/ctx8192"),
        (BIG, 4096, {}, "big/ctx4096"),
        (BIG, 8192, {"flash_attn": True}, "big/ctx8192+flash"),
        # Models that fit entirely in 8GB VRAM.
        (SMALL, 8192, {}, "small7b/ctx8192"),
        (SMALL, 16384, {}, "small7b/ctx16384"),
        (SMALL, 32768, {}, "small7b/ctx32768"),
        (VL, 8192, {}, "vl8b/ctx8192"),
        (TINY, 8192, {}, "tiny0.5b/ctx8192"),
    ]

    for model, ctx, extra, label in plan:
        print(f"\n--- {label} ---", flush=True)
        warm(model, ctx, extra)
        run(model, ctx, extra=extra, label=label)

    with open("bench_results.json", "w", encoding="utf-8") as fh:
        json.dump(RESULTS, fh, indent=2)
    print("\n=== SUMMARY (decode tok/s) ===", flush=True)
    for r in RESULTS:
        if "error" in r:
            print(f"  {r['label']:26s} ERROR {r['error'][:60]}")
        else:
            print(
                f"  {r['label']:26s} {r['decode_tps']:7.2f} tok/s  "
                f"ttft {r['ttft_s']:6.2f}s  ctx={r.get('ctx')}  {r.get('processor','')}"
            )
