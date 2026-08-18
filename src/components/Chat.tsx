import { useEffect, useMemo, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import {
  api,
  type ContextOptions,
  type Effort,
  type ModelProfile,
  type PermissionMode,
  type ProposedCall,
} from "../lib/agent";
import {
  formatAgo,
  formatElapsed,
  formatTokens,
  summarize,
  RECOVERY_NOTE,
  type Entry,
  type Session,
} from "../lib/sessions";
import { Popover, PopoverDivider, PopoverItem, PopoverLabel } from "./Popover";

const MODES: { value: PermissionMode; name: string; hint: string }[] = [
  { value: "read_only", name: "Read only", hint: "Reads run. Writes and commands are confirmed." },
  { value: "project_auto", name: "Project auto", hint: "Reads and writes in the folder run automatically." },
  { value: "balanced", name: "Balanced", hint: "Also runs commands classified as read-only." },
  { value: "unrestricted", name: "Unrestricted", hint: "Runs almost everything unattended." },
];

const EFFORTS: { value: Effort; name: string; hint: string }[] = [
  { value: "fast", name: "Fast", hint: "No thinking, 8 steps. For edits and lookups." },
  { value: "balanced", name: "Balanced", hint: "No thinking, 24 steps." },
  { value: "thorough", name: "Thorough", hint: "Thinking where supported, 48 steps." },
];

const MODE_NAME = (m: PermissionMode) => MODES.find((x) => x.value === m)?.name ?? m;
const EFFORT_NAME = (e: Effort) => EFFORTS.find((x) => x.value === e)?.name ?? e;

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
        {note && (
          <span className="toolrow-flag" title={note}>
            rescued
          </span>
        )}
        {state === "waiting" && <span className="toolrow-wait">waiting</span>}
        {(state === "denied" || state === "user_rejected") && (
          <span className="toolrow-denied">{state === "denied" ? "refused" : "rejected"}</span>
        )}
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

/** A user turn, with the actions that apply to it revealed on hover. */
function UserTurn({
  entry,
  index,
  busy,
  onCopy,
  onRetry,
  onRemove,
  onBranch,
}: {
  entry: Extract<Entry, { kind: "user" }>;
  index: number;
  busy: boolean;
  onCopy: (text: string) => void;
  onRetry: (index: number) => void;
  onRemove: (index: number) => void;
  onBranch: (index: number) => void;
}) {
  const [copied, setCopied] = useState(false);

  function copy() {
    onCopy(entry.text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  return (
    <div className="userwrap">
      <div className="bubble">{entry.text}</div>
      <div className="msg-actions">
        <button className="msg-action" title="Copy" onClick={copy}>
          {copied ? "✓" : "⧉"}
        </button>
        <button
          className="msg-action"
          title="Run this again, discarding what followed"
          onClick={() => onRetry(index)}
          disabled={busy}
        >
          ↺
        </button>
        <button
          className="msg-action"
          title="Continue from here in a new session"
          onClick={() => onBranch(index)}
        >
          ⑂
        </button>
        <button
          className="msg-action msg-action-danger"
          title="Delete this message and everything after it"
          onClick={() => onRemove(index)}
          disabled={busy}
        >
          ✕
        </button>
        <span className="msg-time">{formatAgo(entry.at)}</span>
      </div>
    </div>
  );
}

/** Context occupancy: what the window holds now against its limit. */
function ContextMeter({
  used,
  limit,
  trainedMax,
}: {
  used: number;
  limit: number | null;
  trainedMax: number | null;
}) {
  if (!limit) return <span className="ctx-empty">context —</span>;
  const pct = Math.min(100, Math.round((used / limit) * 100));
  const tone = pct >= 90 ? "ctx-hot" : pct >= 70 ? "ctx-warm" : "ctx-ok";

  return (
    <span
      className={`ctx ${tone}`}
      title={`${used} of ${limit} tokens in the window${
        trainedMax ? `; trained for ${trainedMax}` : ""
      }`}
    >
      <span className="ctx-bar">
        <span className="ctx-fill" style={{ width: `${pct}%` }} />
      </span>
      <span className="ctx-text">
        {formatTokens(used)}/{formatTokens(limit)}
        <span className="ctx-pct"> ({pct}%)</span>
      </span>
    </span>
  );
}

export default function Chat({
  session,
  models,
  routing,
  running,
  pending,
  mode,
  assisted,
  model,
  effort,
  numCtx,
  onSend,
  onAnswer,
  onWorkspace,
  onCancel,
  onMode,
  onAssisted,
  onModel,
  onEffort,
  onNumCtx,
  onRetry,
  onRemove,
  onCopy,
  onBranch,
}: {
  session: Session;
  models: ModelProfile[];
  /// Task -> model, so "Auto" can show which model would actually be used.
  routing: Record<string, string> | null;
  running: boolean;
  pending: ProposedCall | null;
  mode: PermissionMode;
  assisted: boolean;
  model: string;
  effort: Effort;
  numCtx: number | null;
  onSend: (text: string) => void;
  onAnswer: (approved: boolean) => void;
  onWorkspace: (path: string) => void;
  onCancel: () => void;
  onMode: (mode: PermissionMode) => void;
  onAssisted: (enabled: boolean) => void;
  onModel: (model: string) => void;
  onEffort: (effort: Effort) => void;
  onNumCtx: (n: number | null) => void;
  onRetry: (index: number) => void;
  onRemove: (index: number) => void;
  onCopy: (text: string) => void;
  onBranch: (index: number) => void;
}) {
  const [prompt, setPrompt] = useState("");
  const [menu, setMenu] = useState<"mode" | "model" | "effort" | "ctx" | null>(null);
  const [ctxOptions, setCtxOptions] = useState<ContextOptions | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [session.entries.length, pending]);

  const usable = useMemo(() => models.filter((m) => m.capabilities.tools), [models]);

  // With no explicit pick, routing decides — so resolve what it would choose
  // rather than showing nothing. A context meter that reads "—" by default is
  // the same as not having one.
  const effectiveModel = model || routing?.coding || "";
  const activeModel = useMemo(
    () => models.find((m) => m.name === effectiveModel) ?? null,
    [models, effectiveModel],
  );

  useEffect(() => {
    let stale = false;
    if (!effectiveModel) {
      setCtxOptions(null);
      return;
    }
    api
      .contextOptions(effectiveModel)
      .then((o) => {
        if (!stale) setCtxOptions(o);
      })
      .catch(() => {});
    return () => {
      stale = true;
    };
  }, [effectiveModel]);

  const limit = numCtx ?? ctxOptions?.recommended ?? activeModel?.recommended_context ?? null;
  const trainedMax = ctxOptions?.trained_max ?? activeModel?.max_context ?? null;

  async function pickFolder() {
    const chosen = await open({ directory: true, multiple: false });
    if (typeof chosen === "string") onWorkspace(chosen);
  }

  function changeMode(next: PermissionMode) {
    // The one mode that can damage things outside the project takes a
    // deliberate confirmation, not a single click in a menu.
    if (next === "unrestricted") {
      const ok = window.confirm(
        "Unrestricted mode runs commands and writes files without asking, " +
          "including outside this folder.\n\nContinue?",
      );
      if (!ok) return;
    }
    onMode(next);
    setMenu(null);
  }

  function send() {
    if (!prompt.trim() || running) return;
    onSend(prompt);
    setPrompt("");
  }

  const folderName = session.workspace?.split(/[\\/]/).filter(Boolean).pop();

  return (
    <div className="chat">
      <header className="chat-head">
        <div className="chat-title" title={session.title}>
          {session.title}
        </div>
        <button
          className="chip chip-folder"
          onClick={pickFolder}
          title={session.workspace ?? "No folder selected"}
        >
          {folderName ?? "choose folder"}
        </button>
      </header>

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
              <UserTurn
                key={i}
                entry={entry}
                index={i}
                busy={running}
                onCopy={onCopy}
                onRetry={onRetry}
                onRemove={onRemove}
                onBranch={onBranch}
              />
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

      <div className="composer">
        <div className="composer-box">
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

        <div className="controlbar">
          <div className="control">
            <button
              className={menu === "mode" ? "ctl ctl-on" : "ctl"}
              onClick={() => setMenu(menu === "mode" ? null : "mode")}
            >
              {MODE_NAME(mode)}
              <span className="caret">⌃</span>
            </button>
            <Popover open={menu === "mode"} onClose={() => setMenu(null)} width={290}>
              <PopoverLabel>Permissions</PopoverLabel>
              {MODES.map((m, i) => (
                <PopoverItem
                  key={m.value}
                  label={m.name}
                  hint={m.hint}
                  selected={mode === m.value}
                  shortcut={String(i + 1)}
                  danger={m.value === "unrestricted"}
                  onClick={() => changeMode(m.value)}
                />
              ))}
              <PopoverDivider />
              <PopoverItem
                label="Assisted approval"
                hint="A small model settles routine confirmations. Never destructive ones."
                selected={assisted}
                onClick={() => onAssisted(!assisted)}
              />
            </Popover>
          </div>

          <div className="control">
            <button
              className={menu === "ctx" ? "ctl ctl-quiet ctl-on" : "ctl ctl-quiet"}
              onClick={() => setMenu(menu === "ctx" ? null : "ctx")}
              disabled={!ctxOptions}
              title={ctxOptions ? "Context window" : "No model resolved yet"}
            >
              <ContextMeter used={session.contextUsed} limit={limit} trainedMax={trainedMax} />
            </button>
            <Popover open={menu === "ctx"} onClose={() => setMenu(null)} width={270}>
              <PopoverLabel>Context window</PopoverLabel>
              <div className="popover-note">
                {trainedMax
                  ? `Trained for ${formatTokens(trainedMax)} tokens. A larger window reserves more VRAM and can push the model out of it.`
                  : "No trained maximum reported."}
              </div>
              <PopoverDivider />
              <PopoverItem
                label="Recommended"
                hint={
                  ctxOptions?.recommended
                    ? `${formatTokens(ctxOptions.recommended)} — the largest that keeps the model resident`
                    : "Not computed"
                }
                selected={numCtx === null}
                onClick={() => {
                  onNumCtx(null);
                  setMenu(null);
                }}
              />
              {ctxOptions?.choices.map((c) => (
                <PopoverItem
                  key={c}
                  label={formatTokens(c)}
                  selected={numCtx === c}
                  onClick={() => {
                    onNumCtx(c);
                    setMenu(null);
                  }}
                />
              ))}
            </Popover>
          </div>

          <div className="ctl-spacer" />

          <div className="control">
            <button
              className={menu === "model" ? "ctl ctl-on" : "ctl"}
              onClick={() => setMenu(menu === "model" ? null : "model")}
            >
              {model || `Auto${routing?.coding ? ` · ${routing.coding}` : ""}`}
              <span className="caret">⌃</span>
            </button>
            <Popover
              open={menu === "model"}
              onClose={() => setMenu(null)}
              align="right"
              width={330}
            >
              <PopoverLabel>Models</PopoverLabel>
              <PopoverItem
                label="Auto"
                hint="Routing picks by task and what fits in VRAM"
                selected={model === ""}
                onClick={() => {
                  onModel("");
                  setMenu(null);
                }}
              />
              <PopoverDivider />
              {usable.map((m, i) => (
                <PopoverItem
                  key={m.name}
                  label={m.name}
                  hint={
                    m.parameter_size
                      ? `${m.parameter_size} · ${m.placement.replace("_", " ")}`
                      : m.placement
                  }
                  badge={m.placement === "spilled" ? "slow" : undefined}
                  selected={model === m.name}
                  shortcut={i < 9 ? String(i + 1) : undefined}
                  onClick={() => {
                    onModel(m.name);
                    setMenu(null);
                  }}
                />
              ))}
            </Popover>
          </div>

          <div className="control">
            <button
              className={menu === "effort" ? "ctl ctl-on" : "ctl"}
              onClick={() => setMenu(menu === "effort" ? null : "effort")}
            >
              {EFFORT_NAME(effort)}
            </button>
            <Popover
              open={menu === "effort"}
              onClose={() => setMenu(null)}
              align="right"
              width={290}
            >
              <PopoverLabel>Effort</PopoverLabel>
              <div className="popover-note">
                Thinking and step budget. Both change how long a task takes on a
                local model.
              </div>
              <PopoverDivider />
              {EFFORTS.map((e, i) => (
                <PopoverItem
                  key={e.value}
                  label={e.name}
                  hint={e.hint}
                  selected={effort === e.value}
                  shortcut={String(i + 1)}
                  onClick={() => {
                    onEffort(e.value);
                    setMenu(null);
                  }}
                />
              ))}
            </Popover>
          </div>
        </div>

        <div className="statusline">
          <span className="status-mark">✳</span>
          <span>{formatElapsed(session.elapsedMs)}</span>
          <span className="sep">·</span>
          <span>{formatTokens(session.tokens)} used</span>
          {session.lastRate !== null && (
            <>
              <span className="sep">·</span>
              <span className="status-rate">{session.lastRate.toFixed(1)} tok/s</span>
            </>
          )}
          {running && (
            <button className="status-cancel" onClick={onCancel}>
              Stop
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
