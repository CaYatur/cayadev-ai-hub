"""Probe how each installed model actually emits tool calls.

The tolerant parser in `src-tauri/src/agent/toolcall.rs` exists because models
disagree about the output format. This script records what each one really does,
so the parser's strategies are driven by observed behaviour rather than guesses.

Run: python scripts/probe_toolcalls.py
"""
import json
import re
import sys
import urllib.request

HOST = "http://localhost:11434"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path"}},
                "required": ["path"],
            },
        },
    }
]

PROMPT = "Read the file src/main.rs and tell me what it contains."


def classify(text, native):
    """Mirror the Rust parser's strategy order, for reporting only."""
    if native:
        return "native"
    if re.search(r"<tool_call>|<tool-call>|<function_call>", text):
        return "xml_tag"
    if re.search(r"<function\s*=", text):
        return "function_tag"
    if re.search(r"```(?:json|tool_code|tool)?\s*\n", text):
        return "code_fence"
    if re.search(r'\{[^{}]*"(?:name|tool)"\s*:', text):
        return "loose_json"
    if re.search(r"(?m)^\s*read_file\s*\(", text):
        return "python_call"
    return "NONE — no call detected"


def probe(model, with_tools=True):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "think": False,
        "options": {"num_ctx": 8192, "num_predict": 200, "temperature": 0.1},
        "keep_alive": "5m",
    }
    if with_tools:
        body["tools"] = TOOLS

    req = urllib.request.Request(
        f"{HOST}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            out = json.loads(resp.read())
    except Exception as exc:  # noqa: BLE001
        return {"model": model, "error": str(exc)[:160]}

    msg = out.get("message", {}) or {}
    content = msg.get("content", "") or ""
    native = msg.get("tool_calls") or []

    return {
        "model": model,
        "tools_sent": with_tools,
        "format": classify(content, native),
        "native_count": len(native),
        "content_preview": content[:180].replace("\n", "\\n"),
    }


if __name__ == "__main__":
    models = sys.argv[1:] or [
        "qwen2.5-coder:7b",
        "qwen2.5-coder:0.5b",
        "llama3.2:1b",
        "qwen3-vl:8b",
    ]
    print("=== TOOL CALL FORMAT PROBE ===\n")
    for m in models:
        result = probe(m)
        print(json.dumps(result, ensure_ascii=False), flush=True)
