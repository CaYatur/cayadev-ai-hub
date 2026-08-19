"""Is the 7B speedup real, and real against the right baseline?

Two things this settles that the first run did not:

  * The reported 1.68x was llama.cpp against llama.cpp. What the app uses is
    Ollama, and on the 27B that distinction turned a claimed 2x into a tie. The
    same check has to be made here before the number stands.
  * It was one reading. Each configuration is measured twice, interleaved, so a
    drift in machine state shows up as disagreement rather than as a result.

Ollama's leg is deliberately simple — no unloading of models it does not need,
which is what made the first attempt fail silently.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

APP = r"C:\Users\cagan\AppData\Local\Temp\claude\C--Users-cagan-Desktop-CaYaAIAgentHub\ec8c0c7c-5842-4529-900f-d28c3b55a6d3\scratchpad\llamacpp\app"
TARGET = (r"C:\Users\cagan\.ollama\models\blobs"
          r"\sha256-60e05f2100071479f596b964f89f510f057ce397ea22f2833a0cfe029bfc2463")
MODEL = "qwen2.5-coder:7b"
PORT = 8099
LLAMA = f"http://127.0.0.1:{PORT}"
OLLAMA = "http://localhost:11434"

PROMPTS = [
    ("code", "Write a complete Python class implementing a fixed-size ring "
             "buffer with push, pop, peek, len and an iterator. Include "
             "docstrings and a short usage example."),
    ("toolcall", "You have a tool `write_file(path, content)`. Emit only a JSON "
                 "tool call, no prose, that writes a 40-line HTML page with "
                 "inline CSS to index.html. Format: "
                 '{"name":"write_file","arguments":{"path":...,"content":...}}'),
]

CONFIGS = {
    "llama_base": ["-ngl", "99"],
    "llama_ngram": ["-ngl", "99", "--spec-type", "ngram-cache", "--spec-draft-n-max", "5"],
}


def post(url, body, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def kill_servers():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    time.sleep(4)


def ollama_leg():
    """Only touches the 7B. Unloading models it does not need is what hung the
    first attempt — a keep_alive:0 request *loads* the model to unload it."""
    kill_servers()
    out = {}
    for label, prompt in PROMPTS:
        post(f"{OLLAMA}/api/generate",
             {"model": MODEL, "prompt": prompt, "stream": False, "keep_alive": "10m",
              "options": {"num_ctx": 8192, "num_predict": 16, "temperature": 0.0}})
        d = post(f"{OLLAMA}/api/generate",
                 {"model": MODEL, "prompt": prompt, "stream": False, "keep_alive": "10m",
                  "options": {"num_ctx": 8192, "num_predict": 256, "temperature": 0.0}})
        n, dur = d.get("eval_count") or 0, (d.get("eval_duration") or 0) / 1e9
        out[label] = round(n / dur, 2) if dur else None
    return out


def llama_leg(name):
    kill_servers()
    # Ollama must not hold the card while llama.cpp measures.
    try:
        post(f"{OLLAMA}/api/generate",
             {"model": MODEL, "keep_alive": 0, "prompt": "x",
              "options": {"num_predict": 1}}, timeout=300)
    except Exception:
        pass
    time.sleep(5)

    log = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"cf-{name}.log")
    cmd = [os.path.join(APP, "llama-server.exe"), "--host", "127.0.0.1",
           "--port", str(PORT), "-m", TARGET, "-c", "8192", "-fa", "on",
           "-ctk", "q8_0", "-ctv", "q8_0", "--no-webui"] + CONFIGS[name]
    with open(log, "wb") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    try:
        start = time.time()
        while time.time() - start < 400:
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
            post(f"{LLAMA}/completion", {"prompt": prompt, "n_predict": 16,
                                         "temperature": 0.0, "cache_prompt": False})
            d = post(f"{LLAMA}/completion", {"prompt": prompt, "n_predict": 256,
                                             "temperature": 0.0, "cache_prompt": False})
            out[label] = round(d.get("timings", {}).get("predicted_per_second") or 0, 2)
        return out
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(4)


if __name__ == "__main__":
    legs = ["ollama", "llama_base", "llama_ngram"]
    rounds = {}
    for r in (1, 2):
        for name in legs:
            got = ollama_leg() if name == "ollama" else llama_leg(name)
            rounds.setdefault(name, []).append(got)
            print(f"  round {r} {name}: {json.dumps(got)}", flush=True)

    print("\n=== qwen2.5-coder:7b, two rounds interleaved ===", flush=True)
    for name in legs:
        cells = []
        for label, _ in PROMPTS:
            vals = [r.get(label) for r in rounds[name] if isinstance(r, dict) and r.get(label)]
            cells.append(f"{label}=" + "/".join(f"{v:.1f}" for v in vals) if vals else f"{label}=?")
        print(f"  {name:12s} {'  '.join(cells)}", flush=True)
