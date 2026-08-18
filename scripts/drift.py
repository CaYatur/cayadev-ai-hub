"""Measure how often a model reaches for a tool it should not.

A user typing "merhaba" got a `screenshot` call. That is the failure this
measures: not whether the model can call tools, but whether it knows when not
to. Two rates, and a prompt change has to improve one without wrecking the
other:

  false calls  — conversational input that produced a tool call
  missed calls — task input that produced none

The tool list is the one `run_agent` actually builds, because the whole point
is that offering twenty-odd tools makes a small model want to use them.

Run: python scripts/drift.py [model] [params_billions]
"""
import json
import sys
import urllib.request

HOST = "http://localhost:11434"

# Things a person says that need no tool at all.
CHAT = [
    "merhaba",
    "hello",
    "nasılsın?",
    "who are you?",
    "ne yapabilirsin?",
    "thanks, that worked",
    "what is a linked list?",
    "explain recursion in one sentence",
]

# Things that genuinely need one.
TASKS = [
    "Read config.txt and tell me the timeout.",
    "List the files in src.",
    "Create notes.md with a single heading.",
    "What is in the file README.md?",
]

# The surface `run_agent` advertises, trimmed the way Compact tier trims it.
TOOLS = [
    ("read_file", "Read a file's contents.", ["path"]),
    ("edit_file", "Replace one exact string in a file.", ["path", "old", "new"]),
    ("write_file", "Create a file or replace its entire contents.", ["path", "content"]),
    ("list_dir", "List the entries of a directory.", ["path"]),
    ("run_command", "Run a shell command in the workspace.", ["command"]),
    ("web_search", "Search the web.", ["query"]),
    ("web_read", "Read a web page and return its text.", ["url"]),
    ("read_document", "Extract text from a PDF, Word, or Excel file.", ["path"]),
    ("gh_status", "Repository, branch, and uncommitted files.", []),
    ("gh_diff", "Uncommitted changes, or a pull request's diff.", []),
    ("gh_open_pr", "Branch, commit, push, and open a pull request.", ["title", "body"]),
    ("view_image", "Look at an image and answer a question about it.", ["path"]),
]


def schemas():
    out = []
    for name, description, required in TOOLS:
        props = {r: {"type": "string"} for r in required}
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        })
    return out


# Prompt variants under test. `current` is what ships today.
PROMPTS = {
    "current": (
        "You are a coding assistant working in a local project.\n\n"
        "Rules:\n"
        "- Use tools to read files before editing them. Never guess a file's contents.\n"
        "- Call one tool at a time and wait for the result.\n"
        "- Keep replies short. No preamble, no summary of what you are about to do.\n"
        "- When the task is done, say so in one sentence.\n"
    ),
    "answer_first": (
        "You are a coding assistant working in a local project.\n\n"
        "Tools are for acting on the project. Most messages are not that.\n"
        "- If the user is greeting you, thanking you, or asking a general question, "
        "just reply. Do not call a tool.\n"
        "- Call a tool only when the user asked you to look at or change something "
        "specific in this project.\n"
        "- Read a file before editing it. Never guess its contents.\n"
        "- One tool at a time. Wait for the result.\n"
        "- Keep replies short. No preamble.\n"
    ),
    "answer_first_examples": (
        "You are a coding assistant working in a local project.\n\n"
        "Tools are for acting on the project. Most messages are not that.\n"
        "- If the user is greeting you, thanking you, or asking a general question, "
        "just reply. Do not call a tool.\n"
        '- "hello" or "what can you do?" -> answer in words, no tool.\n'
        '- "read config.txt" or "list src" -> call the tool.\n'
        "- Read a file before editing it. Never guess its contents.\n"
        "- One tool at a time. Wait for the result.\n"
        "- Keep replies short. No preamble.\n"
    ),
}


def called_a_tool(model, system, message):
    """True when the turn produced a tool call, by any route the app accepts."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ],
        "tools": schemas(),
        "stream": False,
        "think": False,
        "options": {"num_ctx": 8192, "num_predict": 200, "temperature": 0.2},
        "keep_alive": "15m",
    }
    req = urllib.request.Request(
        f"{HOST}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            out = json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        return None, f"error: {exc}"

    msg = out.get("message", {}) or {}
    native = msg.get("tool_calls") or []
    content = msg.get("content", "") or ""

    # Mirror the tolerant parser: a call in the body counts as a call.
    names = "|".join(name for name, _, _ in TOOLS)
    import re

    recovered = bool(
        re.search(r"<tool_call>|<function\s*=", content)
        or re.search(rf'"(?:name|tool)"\s*:\s*"(?:{names})"', content)
        or re.search(rf"(?m)^\s*(?:{names})\s*\(", content)
    )
    return bool(native) or recovered, content[:90].replace("\n", " ")


def evaluate(model, label, system):
    false_calls = []
    for message in CHAT:
        called, preview = called_a_tool(model, system, message)
        if called is None:
            print(f"  ! {preview}")
            continue
        if called:
            false_calls.append((message, preview))

    missed = []
    for message in TASKS:
        called, preview = called_a_tool(model, system, message)
        if called is False:
            missed.append((message, preview))

    print(f"\n[{label}] false calls {len(false_calls)}/{len(CHAT)} "
          f"· missed calls {len(missed)}/{len(TASKS)}")
    for message, preview in false_calls:
        print(f"    false: {message!r} -> {preview}")
    for message, preview in missed:
        print(f"    missed: {message!r} -> {preview}")
    return len(false_calls), len(missed)


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-coder:7b"
    print(f"=== tool drift :: {model} :: {len(TOOLS)} tools ===", flush=True)

    results = {}
    for label, system in PROMPTS.items():
        results[label] = evaluate(model, label, system)

    print("\n=== summary (lower is better) ===")
    for label, (false_calls, missed) in results.items():
        print(f"  {label:24s} false {false_calls}/{len(CHAT)}  missed {missed}/{len(TASKS)}")
