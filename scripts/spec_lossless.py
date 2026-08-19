"""Does speculation change the answer, or only the speed?

At temperature 0 it is supposed to be exactly lossless: the drafter proposes,
the target verifies, and any token the target would not have produced is
rejected. Measured on `qwen2.5-coder:7b`, that does not hold — n-gram
speculation produced a different tool call from the same prompt, while running
the same configuration twice produced identical bytes. Run the control before
believing any difference:

    python scripts/spec_lossless.py ngram_n5     # speculation against base
    python scripts/spec_lossless.py base         # base against itself

A difference only means something if the second command shows none. See
`docs/SPEED_RESEARCH.md`.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coder7b_bench import CONFIGS, TARGET  # noqa: E402
from clean_bench import APP, LLAMA, PORT  # noqa: E402

PROMPTS = [
    ("code", "Write a complete Python class implementing a fixed-size ring "
             "buffer with push, pop, peek, len and an iterator. Include "
             "docstrings and a short usage example."),
    ("toolcall", "You have a tool `write_file(path, content)`. Emit only a JSON "
                 "tool call, no prose, that writes a 40-line HTML page with "
                 "inline CSS to index.html. Format: "
                 '{"name":"write_file","arguments":{"path":...,"content":...}}'),
]


def post(body, timeout=1800):
    req = urllib.request.Request(f"{LLAMA}/completion",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def texts(name):
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    time.sleep(4)
    log = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"v-{name}.log")
    cmd = [os.path.join(APP, "llama-server.exe"), "--host", "127.0.0.1",
           "--port", str(PORT), "-m", TARGET, "-c", "8192", "-fa", "on",
           "-ctk", "q8_0", "-ctv", "q8_0", "--no-webui"] + CONFIGS[name]
    with open(log, "wb") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    try:
        start = time.time()
        while time.time() - start < 600:
            if proc.poll() is not None:
                raise RuntimeError("server exited")
            try:
                if urllib.request.urlopen(f"{LLAMA}/health", timeout=3).status == 200:
                    break
            except Exception:
                time.sleep(2)

        out = {}
        for label, prompt in PROMPTS:
            d = post({"prompt": prompt, "n_predict": 200, "temperature": 0.0,
                      "top_k": 1, "seed": 1234, "cache_prompt": False})
            out[label] = d.get("content") or ""
        return out
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(4)


if __name__ == "__main__":
    a = texts("base")
    b = texts(sys.argv[1] if len(sys.argv) > 1 else "ngram_n5")

    print("\n=== identical output? ===", flush=True)
    for label in a:
        ha = hashlib.sha256(a[label].encode()).hexdigest()[:12]
        hb = hashlib.sha256(b[label].encode()).hexdigest()[:12]
        same = "YES" if a[label] == b[label] else "NO"
        print(f"  {label:10s} base {ha}  spec {hb}  identical: {same}", flush=True)
        if a[label] != b[label]:
            for i, (x, y) in enumerate(zip(a[label], b[label])):
                if x != y:
                    print(f"    first difference at char {i}:", flush=True)
                    print(f"      base: ...{a[label][max(0,i-40):i+40]!r}", flush=True)
                    print(f"      spec: ...{b[label][max(0,i-40):i+40]!r}", flush=True)
                    break
            else:
                print(f"    one is a prefix of the other "
                      f"({len(a[label])} vs {len(b[label])} chars)", flush=True)
