"""Decode throughput: Ollama against llama.cpp, with speculative decoding.

Answers one question — can a model too large for the graphics card be made
faster — and keeps answering it, because llama.cpp's placement for hybrid SSM
models is the thing standing in the way and that is expected to improve. See
`docs/SPEED_RESEARCH.md` for what it measured and why the answer was no.

Needs a llama.cpp build with CUDA (`llama-server.exe`) and reads Ollama's own
GGUF blobs, so nothing is downloaded twice. Set APP and TARGET below.

    python scripts/spec_bench.py ollama llama_ngl20 mtp3_ngl20

The protocol matters as much as the configurations. A stale server holding
14.5 GB turned every reading into a measurement of disk paging, so each run
kills everything, unloads the daemon, and asserts how much memory is free.
"""
import ctypes
import json
import os
import subprocess
import sys
import time
import urllib.request

APP = r"C:\Users\cagan\AppData\Local\Temp\claude\C--Users-cagan-Desktop-CaYaAIAgentHub\ec8c0c7c-5842-4529-900f-d28c3b55a6d3\scratchpad\llamacpp\app"
BLOBS = r"C:\Users\cagan\.ollama\models\blobs"
TARGET = os.path.join(BLOBS, "sha256-f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d")
DRAFT = os.path.join(BLOBS, "sha256-afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5")
MODEL = "qwen3.8:27b-mtp-q4_K_M"
PORT = 8099
LLAMA = f"http://127.0.0.1:{PORT}"
OLLAMA = "http://localhost:11434"

# The model must fit in RAM with room to work, or the reading measures paging.
MIN_FREE_GB = 18.0

PROMPTS = [
    ("prose", "Write three paragraphs about why local language models are "
              "slower than hosted ones. Plain prose, no lists."),
    ("code", "Write a complete Python class implementing a fixed-size ring "
             "buffer with push, pop, peek, len and an iterator. Include "
             "docstrings and a short usage example."),
    ("toolcall", "You have a tool `write_file(path, content)`. Emit only a JSON "
                 "tool call, no prose, that writes a 40-line HTML page with "
                 "inline CSS to index.html. Format: "
                 '{"name":"write_file","arguments":{"path":...,"content":...}}'),
]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def free_gb():
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 1e9


def quiesce(label):
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                   capture_output=True)
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"{OLLAMA}/api/generate",
            data=json.dumps({"model": MODEL, "keep_alive": 0, "prompt": "x",
                             "options": {"num_predict": 1}}).encode(),
            headers={"Content-Type": "application/json"}), timeout=600).read()
    except Exception:
        pass
    for _ in range(20):
        time.sleep(2)
        if free_gb() >= MIN_FREE_GB:
            break
    got = free_gb()
    print(f"  [{label}] free RAM before run: {got:.1f} GB", flush=True)
    if got < MIN_FREE_GB:
        print(f"  [{label}] WARNING: only {got:.1f} GB free; reading will "
              f"measure paging, not the model", flush=True)
    return got


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ollama_run(n_predict=128):
    quiesce("ollama")
    out = {}
    for label, prompt in PROMPTS:
        post(f"{OLLAMA}/api/generate", {"model": MODEL, "prompt": prompt,
             "stream": False, "think": False, "keep_alive": "10m",
             "options": {"num_ctx": 8192, "num_predict": 8, "temperature": 0.0}})
        d = post(f"{OLLAMA}/api/generate", {"model": MODEL, "prompt": prompt,
                 "stream": False, "think": False, "keep_alive": "10m",
                 "options": {"num_ctx": 8192, "num_predict": n_predict,
                             "temperature": 0.0}})
        n = d.get("eval_count") or 0
        dur = (d.get("eval_duration") or 0) / 1e9
        out[label] = round(n / dur, 2) if dur else None
        print(f"  ollama {label}: {out[label]} tok/s", flush=True)
    return out


def llama_run(name, extra, n_predict=128):
    quiesce(name)
    log = os.path.join(os.path.dirname(__file__), f"srv-{name}.log")
    cmd = [os.path.join(APP, "llama-server.exe"),
           "--host", "127.0.0.1", "--port", str(PORT), "-m", TARGET,
           "-c", "8192", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
           "--no-webui"] + extra
    with open(log, "wb") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    try:
        start = time.time()
        while time.time() - start < 900:
            if proc.poll() is not None:
                return {"error": "server exited"}
            try:
                if urllib.request.urlopen(f"{LLAMA}/health", timeout=3).status == 200:
                    break
            except Exception:
                time.sleep(2)
        else:
            return {"error": "never ready"}

        out = {}
        for label, prompt in PROMPTS:
            post(f"{LLAMA}/completion", {"prompt": prompt, "n_predict": 8,
                                         "temperature": 0.0, "cache_prompt": False})
            d = post(f"{LLAMA}/completion", {"prompt": prompt, "n_predict": n_predict,
                                             "temperature": 0.0, "cache_prompt": False})
            t = d.get("timings", {})
            out[label] = round(t.get("predicted_per_second") or 0, 2)
            print(f"  {name} {label}: {out[label]} tok/s", flush=True)
        out["accept"] = acceptance(log)
        return out
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(4)


def acceptance(log):
    """Draft acceptance is only in the server log, not the HTTP response."""
    best = None
    try:
        with open(log, "r", errors="ignore") as f:
            for line in f:
                if "draft acceptance" in line:
                    try:
                        best = float(line.split("draft acceptance =")[1].split("(")[0])
                    except (IndexError, ValueError):
                        pass
    except OSError:
        pass
    return best


CONFIGS = {
    "llama_ngl20": ["-ngl", "20"],
    "llama_ngl24": ["-ngl", "24"],
    "llama_ngl28": ["-ngl", "28"],
    "llama_nommap": ["-ngl", "20", "--no-mmap"],
    "mtp2_ngl20": ["-ngl", "20", "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
    "mtp3_ngl20": ["-ngl", "20", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"],
    "mtp2_ngl24": ["-ngl", "24", "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
    "mtp3_ngl24": ["-ngl", "24", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"],
    "mtp3_nommap": ["-ngl", "20", "--no-mmap", "--spec-type", "draft-mtp",
                    "--spec-draft-n-max", "3"],
    "ngram4": ["-ngl", "20", "--spec-type", "ngram-cache", "--spec-draft-n-max", "4"],
    "mtp3_ngram": ["-ngl", "20", "--spec-type", "draft-mtp,ngram-cache",
                   "--spec-draft-n-max", "3"],
    "draft08b": ["-ngl", "20", "--spec-type", "draft-simple", "-md", DRAFT,
                 "--spec-draft-n-max", "4"],
    # A different split entirely. `-ngl N` cuts the model by layer, so a layer
    # is either wholly on the card or wholly in RAM. This cuts it by *tensor
    # kind* instead: the feed-forward matrices are about 9.7 GB of the 16.8 and
    # carry most of the arithmetic, while attention and the SSM state are small
    # and latency-sensitive. Putting the small half on the card and streaming
    # the big half is the opposite trade to the usual one, and nobody appears to
    # have measured it on a hybrid model this size.
    "ffn_cpu": ["-ngl", "99", "-ot", "ffn.*=CPU"],
    "ffn_cpu_mtp3": ["-ngl", "99", "-ot", "ffn.*=CPU", "--spec-type", "draft-mtp",
                     "--spec-draft-n-max", "3"],
    # Pin the weights so Windows cannot evict them under memory pressure. The
    # machine has 28.7 GB and the model wants 18.3, so eviction is one browser
    # window away.
    # The split itself is the problem. llama.cpp warns that layers assigned to
    # the CPU still have their fused Gated Delta Net — this architecture's SSM
    # component — running on CUDA0, so every one of the 45 CPU layers makes a
    # round trip across PCIe for every token. Keeping the whole model on one
    # device removes 45 round trips per token, which may beat having a quarter
    # of it on a fast card.
    # `--no-op-offload` is the direct answer to that warning: it stops tensors
    # that live on the host from having their operations shipped to the card.
    # The 20 layers genuinely on the GPU stay there; the other 45 stop making a
    # round trip per token.
    "ngl20_noop": ["-ngl", "20", "--no-op-offload"],
    "ngl20_noop_mtp3": ["-ngl", "20", "--no-op-offload", "--spec-type", "draft-mtp",
                        "--spec-draft-n-max", "3"],
    "ngl24_noop_mtp3": ["-ngl", "24", "--no-op-offload", "--spec-type", "draft-mtp",
                        "--spec-draft-n-max", "3"],
    "cpu_only": ["-ngl", "0"],
    "cpu_only_mtp3": ["-ngl", "0", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"],
    "cpu_only_mtp4": ["-ngl", "0", "--spec-type", "draft-mtp", "--spec-draft-n-max", "4"],
    "mtp3_mlock": ["-ngl", "20", "--mlock", "--spec-type", "draft-mtp",
                   "--spec-draft-n-max", "3"],
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or ["ollama", "llama_ngl20", "mtp3_ngl20"]
    table = {}
    for name in wanted:
        print(f"--- {name} ---", flush=True)
        table[name] = ollama_run() if name == "ollama" else llama_run(name, CONFIGS[name])
        print(json.dumps({name: table[name]}), flush=True)

    print("\n=== decode tok/s ===", flush=True)
    for name, r in table.items():
        if r.get("error"):
            print(f"  {name:14s} FAILED: {r['error']}")
            continue
        acc = f"  accept={r['accept']:.2f}" if r.get("accept") else ""
        print(f"  {name:14s} prose={r.get('prose')} code={r.get('code')} "
              f"toolcall={r.get('toolcall')}{acc}", flush=True)
