import { useEffect, useMemo, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import type { ModelProfile, PermissionMode, ProposedCall } from "../lib/agent";
import {
  formatElapsed,
  formatTokens,
  summarize,
  RECOVERY_NOTE,
  type Entry,
  type Session,
} from "../lib/sessions";

const MODE_LABELS: Record<PermissionMode, { name: string; hint: string }> = {
  read_only: { name: "Read only", hint: "Reads run. Writes and commands are confirmed." },
  project_auto: { name: "Project auto", hint: "Reads and writes in the folder run. Commands are confirmed." },
  balanced: { name: "Balanced", hint: "Also runs commands classified as read-only." },
  unrestricted: { name: "Unrestricted", hint: "Runs almost everything unattended. Risky." },
};

/** A tool call, collapsed to one scannable line until the user opens it. */
function ToolRow({ entry }: { entry: Extract<Entry, { kind: "tool" }> }) {
  const [open, setOpen] = useState(false);
  const note = RECOVERY_NOTE[entry.call.recovery];
  const state = entry.settled ?? "waiting";

  return (
    <div className={`toolrow toolrow-${state}`}>
      <button className="toolrow-head" onClick={() => setOpen((v) => !v)}>
        <span className={open ? "chev chev-open" : "chev"}>›</span>
        <span className="toolrow-summary">{summarize(entry.call, entry.output)}</span>
        {note && <span className="toolrow-flag" title={note}>rescued</span>}
        {state === "waiting" && <span className="toolrow-wait">waiting</span>}
        {state === "denied" && <span className="toolrow-denied">refused</span>}
        {state === "user_rejected" && <span className="toolrow-denied">rejected</span>}
      </button>

      {open && (
        <div className="toolrow-body">
          <div className="toolrow-label">arguments</div>
          <pre className="code-block">{JSON.stringify(entry.call.arguments, null, 2)}</pre>
          {note && <div className="toolrow-note">{note}</div>}
          {entry.output && (
            <>
              <div className="toolrow-label">result</div>
              <pre className="code-block">{entry.output.slice(0, 4000)}</pre>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default function Chat({
  session,
  models,
  running,
  pending,
  mode,
  assisted,
  onSend,
  onAnswer,
  onWorkspace,
  onCancel,
  onMode,
  onAssisted,
}: {
  session: Session;
  models: ModelProfile[];
  running: boolean;
  pending: ProposedCall | null;
  /// Owned by `App` so it survives unmounting this view. Local state here
  /// would reset the label to the default while the engine kept the old mode
  /// — and a stale "Project auto" label over a live Unrestricted engine is
  /// the one divergence that can cost the user something.
  mode: PermissionMode;
  assisted: boolean;
  onSend: (text: string, model?: string) => void;
  onAnswer: (approved: boolean) => void;
  onWorkspace: (path: string) => void;
  onCancel: () => void;
  onMode: (mode: PermissionMode) => void;
  onAssisted: (enabled: boolean) => void;
}) {
  const [prompt, setPrompt] = useState("");
  const [model, setModel] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [session.entries.length, pending]);

  const usable = useMemo(() => models.filter((m) => m.capabilities.tools), [models]);

  async function pickFolder() {
    const chosen = await open({ directory: true, multiple: false });
    if (typeof chosen === "string") onWorkspace(chosen);
  }

  function changeMode(next: PermissionMode) {
    // The one mode that can damage things outside the project takes a
    // deliberate confirmation, not a single click in a dropdown.
    if (next === "unrestricted") {
      const ok = window.confirm(
        "Unrestricted mode runs commands and writes files without asking, " +
          "including outside this folder.\n\nContinue?",
      );
      if (!ok) return;
    }
    onMode(next);
  }

  function send() {
    if (!prompt.trim() || running) return;
    onSend(prompt, model || undefined);
    setPrompt("");
  }

  const folderName = session.workspace?.split(/[\\/]/).filter(Boolean).pop();

  return (
    <div className="chat">
      <header className="chat-head">
        <div className="chat-title" title={session.title}>
          {session.title}
        </div>
        <button className="chip chip-folder" onClick={pickFolder} title={session.workspace ?? "No folder selected"}>
          {folderName ?? "choose folder"}
        </button>
        <button
          className={settingsOpen ? "icon-btn icon-btn-on" : "icon-btn"}
          onClick={() => setSettingsOpen((v) => !v)}
          title="Session settings"
        >
          ⋮
        </button>
      </header>

      {settingsOpen && (
        <div className="settings-strip">
          <label className="field">
            <span>Permissions</span>
            <select
              className="select"
              value={mode}
              onChange={(e) => changeMode(e.target.value as PermissionMode)}
            >
              {Object.entries(MODE_LABELS).map(([value, { name }]) => (
                <option key={value} value={value}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Model</span>
            <select className="select" value={model} onChange={(e) => setModel(e.target.value)}>
              <option value="">Auto (routed)</option>
              {usable.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name} · {m.placement}
                </option>
              ))}
            </select>
          </label>
          <label className="field field-inline" title="A small model settles routine confirmations. It can never approve a destructive action.">
            <input
              type="checkbox"
              checked={assisted}
              onChange={(e) => onAssisted(e.target.checked)}
            />
            <span>Assisted approval</span>
          </label>
          <p className="settings-hint">{MODE_LABELS[mode].hint}</p>
        </div>
      )}

      {mode === "unrestricted" && (
        <div className="danger-banner">
          Unrestricted mode — the agent runs commands and writes files without asking.
        </div>
      )}

      <div className="transcript">
        {session.entries.length === 0 && (
          <div className="empty">
            <div className="empty-mark">▮</div>
            <h2>What should the agent do?</h2>
            <p>
              {session.workspace
                ? "Describe a task. Routing picks a model that fits in VRAM."
                : "Choose a project folder to begin."}
            </p>
          </div>
        )}

        {session.entries.map((entry, i) => {
          if (entry.kind === "user") {
            return (
              <div key={i} className="turn turn-user">
                {entry.text}
              </div>
            );
          }
          if (entry.kind === "assistant") {
            return entry.text.trim() ? (
              <div key={i} className="turn turn-assistant">
                {entry.text}
              </div>
            ) : null;
          }
          if (entry.kind === "notice") {
            return (
              <div key={i} className={`notice notice-${entry.tone}`}>
                {entry.text}
              </div>
            );
          }
          return <ToolRow key={i} entry={entry} />;
        })}

        {running && !pending && (
          <div className="working">
            <span className="pulse" /> working…
          </div>
        )}
        <div ref={bottom} />
      </div>

      {pending && (
        <div className="approval">
          <div className="approval-body">
            <div className="approval-title">
              Approve <code>{pending.name}</code>?
            </div>
            <div className="approval-reason">{pending.evaluation.reason}</div>
            <pre className="code-block">{JSON.stringify(pending.arguments, null, 2)}</pre>
          </div>
          <div className="approval-actions">
            <button className="btn" onClick={() => onAnswer(false)}>
              Reject
            </button>
            <button className="btn btn-primary" onClick={() => onAnswer(true)}>
              Approve
            </button>
          </div>
        </div>
      )}

      <div className="statusline">
        <span className="status-mark">✳</span>
        <span>{formatElapsed(session.elapsedMs)}</span>
        <span className="sep">·</span>
        <span>{formatTokens(session.tokens)} tokens</span>
        {session.lastRate !== null && (
          <>
            <span className="sep">·</span>
            <span className="status-rate">{session.lastRate.toFixed(1)} tok/s</span>
          </>
        )}
        <span className="sep">·</span>
        <span className="status-mode">{MODE_LABELS[mode].name}</span>
        {running && (
          <button className="status-cancel" onClick={onCancel}>
            Stop
          </button>
        )}
      </div>

      <div className="composer">
        <textarea
          className="input"
          rows={2}
          placeholder={session.workspace ? "Describe a task…" : "Choose a folder first"}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          disabled={running || !session.workspace}
        />
        <button
          className="send"
          onClick={send}
          disabled={running || !prompt.trim() || !session.workspace}
          title="Send (Enter)"
        >
          ↵
        </button>
      </div>
    </div>
  );
}
