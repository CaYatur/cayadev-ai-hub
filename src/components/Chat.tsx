import { useEffect, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import {
  api,
  subscribe,
  type AgentEvent,
  type ModelProfile,
  type PermissionMode,
  type ProposedCall,
  type Recovery,
} from "../lib/agent";

/** One entry in the transcript. Tool calls are entries too, not annotations. */
type Entry =
  | { kind: "user"; text: string }
  | { kind: "assistant"; text: string }
  | { kind: "tool"; call: ProposedCall; output?: string; settled?: string }
  | { kind: "notice"; text: string; tone: "info" | "error" };

const MODE_LABELS: Record<PermissionMode, { name: string; hint: string }> = {
  read_only: { name: "Read only", hint: "Reads run; writes and commands are confirmed." },
  project_auto: { name: "Project auto", hint: "Reads and writes in the folder run; commands are confirmed." },
  balanced: { name: "Balanced", hint: "Also runs commands the rules classify as read-only." },
  unrestricted: { name: "Unrestricted", hint: "Runs almost everything without asking. Risky." },
};

/** Only shown when it is not `native` — that is when the parser did the work. */
const RECOVERY_NOTE: Partial<Record<Recovery, string>> = {
  xml_tag: "recovered from <tool_call> tags",
  code_fence: "recovered from a code block",
  loose_json: "recovered from bare JSON",
  function_tag: "recovered from a <function=> tag",
  python_call: "recovered from Python-style syntax",
};

export default function Chat({ models }: { models: ModelProfile[] }) {
  const [workspace, setWorkspace] = useState<string | null>(null);
  const [mode, setMode] = useState<PermissionMode>("project_auto");
  const [assisted, setAssisted] = useState(true);
  const [model, setModel] = useState<string>("");
  const [prompt, setPrompt] = useState("");
  const [entries, setEntries] = useState<Entry[]>([]);
  const [pending, setPending] = useState<ProposedCall | null>(null);
  const [running, setRunning] = useState(false);
  const [rate, setRate] = useState<number | null>(null);

  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries, pending]);

  useEffect(() => {
    let off: (() => void) | undefined;

    subscribe(
      (event: AgentEvent) => {
        setEntries((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];

          switch (event.kind) {
            case "token":
              // Append to the open assistant turn rather than starting a new one.
              if (last?.kind === "assistant") {
                next[next.length - 1] = { ...last, text: last.text + event.text };
              } else {
                next.push({ kind: "assistant", text: event.text });
              }
              break;
            case "tool_proposed":
              next.push({ kind: "tool", call: event.call });
              break;
            case "tool_settled":
              for (let i = next.length - 1; i >= 0; i--) {
                const entry = next[i];
                if (entry.kind === "tool" && entry.call.id === event.id) {
                  next[i] = { ...entry, output: event.output, settled: event.outcome };
                  break;
                }
              }
              break;
            case "error":
              next.push({ kind: "notice", text: event.message, tone: "error" });
              break;
            case "done":
              next.push({ kind: "notice", text: `Finished: ${event.reason}`, tone: "info" });
              break;
          }
          return next;
        });

        if (event.kind === "step_done") setRate(event.metrics.decode_tps);
      },
      (call) => setPending(call),
    ).then((fn) => {
      off = fn;
    });

    return () => off?.();
  }, []);

  async function pickFolder() {
    const chosen = await open({ directory: true, multiple: false });
    if (typeof chosen !== "string") return;
    try {
      setWorkspace(await api.setWorkspace(chosen));
    } catch (e) {
      setEntries((p) => [...p, { kind: "notice", text: String(e), tone: "error" }]);
    }
  }

  async function changeMode(next: PermissionMode) {
    // Unrestricted is the one mode that can damage things outside the project,
    // so entering it takes a deliberate confirmation.
    if (next === "unrestricted") {
      const ok = window.confirm(
        "Unrestricted mode runs commands and writes files without asking, " +
          "including outside this folder.\n\nContinue?",
      );
      if (!ok) return;
    }
    setMode(next);
    await api.setMode(next);
  }

  async function send() {
    if (!prompt.trim() || running) return;
    if (!workspace) {
      setEntries((p) => [
        ...p,
        { kind: "notice", text: "Choose a project folder first.", tone: "error" },
      ]);
      return;
    }
    const text = prompt;
    setPrompt("");
    setEntries((p) => [...p, { kind: "user", text }]);
    setRunning(true);
    try {
      await api.run(text, model || undefined);
    } catch (e) {
      setEntries((p) => [...p, { kind: "notice", text: String(e), tone: "error" }]);
    } finally {
      setRunning(false);
      setPending(null);
    }
  }

  async function answer(approved: boolean) {
    if (!pending) return;
    await api.resolveApproval(pending.id, approved);
    setPending(null);
  }

  const usable = models.filter((m) => m.capabilities.tools);

  return (
    <div className="chat">
      <div className="chat-bar">
        <button className="btn" onClick={pickFolder}>
          {workspace ? "Change folder" : "Choose folder"}
        </button>
        <span className="chat-path" title={workspace ?? ""}>
          {workspace ?? "no folder selected"}
        </span>

        <select
          className="select"
          value={mode}
          onChange={(e) => changeMode(e.target.value as PermissionMode)}
          title={MODE_LABELS[mode].hint}
        >
          {Object.entries(MODE_LABELS).map(([value, { name }]) => (
            <option key={value} value={value}>
              {name}
            </option>
          ))}
        </select>

        <select className="select" value={model} onChange={(e) => setModel(e.target.value)}>
          <option value="">Auto (routed)</option>
          {usable.map((m) => (
            <option key={m.name} value={m.name}>
              {m.name}
            </option>
          ))}
        </select>

        <label className="toggle" title="A small model settles routine confirmations. It can never approve destructive actions.">
          <input
            type="checkbox"
            checked={assisted}
            onChange={async (e) => {
              setAssisted(e.target.checked);
              await api.setAssisted(e.target.checked);
            }}
          />
          Assisted approval
        </label>

        {rate !== null && <span className="rate">{rate.toFixed(1)} tok/s</span>}
      </div>

      {mode === "unrestricted" && (
        <div className="danger-banner">
          Unrestricted mode is on. The agent runs commands and writes files without asking.
        </div>
      )}

      <div className="transcript">
        {entries.length === 0 && (
          <div className="empty">
            <p>Pick a folder, then describe what you want done.</p>
            <p className="muted">
              Routing picks a model that fits in VRAM unless you choose one.
            </p>
          </div>
        )}

        {entries.map((entry, i) => {
          if (entry.kind === "user") {
            return (
              <div key={i} className="msg msg-user">
                {entry.text}
              </div>
            );
          }
          if (entry.kind === "assistant") {
            return (
              <div key={i} className="msg msg-assistant">
                {entry.text}
              </div>
            );
          }
          if (entry.kind === "notice") {
            return (
              <div key={i} className={`notice notice-${entry.tone}`}>
                {entry.text}
              </div>
            );
          }
          const note = RECOVERY_NOTE[entry.call.recovery];
          return (
            <div key={i} className={`toolcard tool-${entry.settled ?? "waiting"}`}>
              <div className="toolcard-head">
                <code>{entry.call.name}</code>
                {entry.settled && <span className="tool-outcome">{entry.settled.replace("_", " ")}</span>}
              </div>
              <pre className="toolcard-args">
                {JSON.stringify(entry.call.arguments, null, 2)}
              </pre>
              {note && <div className="recovery">{note}</div>}
              {entry.output && (
                <pre className="toolcard-output">{entry.output.slice(0, 1500)}</pre>
              )}
            </div>
          );
        })}
        <div ref={bottom} />
      </div>

      {pending && (
        <div className="approval">
          <div className="approval-body">
            <div className="approval-title">
              Approve <code>{pending.name}</code>?
            </div>
            <div className="approval-reason">{pending.evaluation.reason}</div>
            <pre className="approval-args">
              {JSON.stringify(pending.arguments, null, 2)}
            </pre>
          </div>
          <div className="approval-actions">
            <button className="btn btn-ghost" onClick={() => answer(false)}>
              Reject
            </button>
            <button className="btn btn-primary" onClick={() => answer(true)}>
              Approve
            </button>
          </div>
        </div>
      )}

      <div className="composer">
        <textarea
          className="input"
          rows={3}
          placeholder="What should the agent do?"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          disabled={running}
        />
        <button className="btn btn-primary" onClick={send} disabled={running || !prompt.trim()}>
          {running ? "Running…" : "Send"}
        </button>
      </div>
    </div>
  );
}
