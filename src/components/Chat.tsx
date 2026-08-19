import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { open } from "@tauri-apps/plugin-dialog";
import {
  api,
  formatBytes,
  type Attachment,
  type ContextOptions,
  type Effort,
  type ModelProfile,
  type PermissionMode,
  type ProposedCall,
  type Question,
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
import { Markdown } from "./Markdown";
import { Popover, PopoverDivider, PopoverItem, PopoverLabel } from "./Popover";

const MODES: PermissionMode[] = [
  "read_only",
  "project_auto",
  "balanced",
  "autonomous",
  "unrestricted",
];
const EFFORTS: Effort[] = ["fast", "balanced", "thorough"];

/// How tall the instruction box may grow, in pixels.
///
/// Roughly twelve lines. Past that the transcript starts disappearing behind
/// the composer, which costs more than the extra lines are worth.
const INPUT_MAX_HEIGHT = 260;

/** A tool call, collapsed to one scannable line until the user opens it. */
function ToolRow({ entry }: { entry: Extract<Entry, { kind: "tool" }> }) {
  const { t } = useTranslation();
  const note = RECOVERY_NOTE[entry.call.recovery];
  const state = entry.settled ?? "waiting";
  // A command that is still running and already producing output opens itself:
  // streaming it and then hiding it would defeat the point.
  const streaming = !entry.settled && Boolean(entry.output);
  const [manual, setManual] = useState<boolean | null>(null);
  const open = manual ?? streaming;
  const setOpen = (next: boolean | ((v: boolean) => boolean)) =>
    setManual(typeof next === "function" ? next(open) : next);

  return (
    <div className={`toolrow toolrow-${state}`}>
      <button className="toolrow-head" onClick={() => setOpen((v) => !v)}>
        <span className={open ? "chev chev-open" : "chev"}>›</span>
        <span className="toolrow-summary">{summarize(entry.call, entry.output)}</span>
        {note && (
          <span className="toolrow-flag" title={note}>
            {t("chat.rescued")}
          </span>
        )}
        {state === "waiting" && <span className="toolrow-wait">{t("chat.waiting")}</span>}
        {(state === "denied" || state === "user_rejected") && (
          <span className="toolrow-denied">
            {state === "denied" ? t("chat.refused") : t("chat.rejected")}
          </span>
        )}
      </button>

      {open && (
        <div className="toolrow-body">
          <div className="toolrow-label">{t("chat.arguments")}</div>
          <pre className="code-block">{JSON.stringify(entry.call.arguments, null, 2)}</pre>
          {note && <div className="toolrow-note">{note}</div>}
          {entry.output && (
            <>
              <div className="toolrow-label">{t("chat.result")}</div>
              <pre className="code-block">{entry.output.slice(0, 4000)}</pre>
            </>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * A model's reasoning, folded away by default.
 *
 * On a thinking-capable model this is routinely longer than the answer, so it
 * collapses to one line and expands on click. The label animates while tokens
 * are still arriving, which is the only signal that anything is happening
 * during a long silent reasoning pass.
 */
function ThinkingBlock({ text, live }: { text: string; live: boolean }) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const body = useRef<HTMLDivElement>(null);
  const follow = useRef(true);

  // The reasoning box scrolls inside itself, so following the transcript is not
  // enough — without this the text streams on out of sight below the fold.
  useEffect(() => {
    if (!open || !follow.current) return;
    const el = body.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [text, open]);

  return (
    <div className={open ? "thinking thinking-open" : "thinking"}>
      <button className="thinking-head" onClick={() => setOpen((v) => !v)}>
        <span className={open ? "chev chev-open" : "chev"}>›</span>
        <span className={live ? "thinking-label thinking-live" : "thinking-label"}>
          {live ? t("chat.thinkingLive") : t("chat.thinking")}
        </span>
        <span className="thinking-size">{text.trim().split(/\s+/).length}</span>
      </button>
      {open && (
        <div
          className="thinking-body"
          ref={body}
          // Reading back through the reasoning while it is still being written
          // must not be yanked away; scrolling up stops the follow until the
          // user returns to the bottom.
          onScroll={(e) => {
            const el = e.currentTarget;
            follow.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
          }}
        >
          {text}
        </div>
      )}
    </div>
  );
}

/**
 * The agent's own question, put to the user.
 *
 * Options and a written answer are offered together rather than as separate
 * modes: the model's list is a guess at what the user might say, and the guess
 * is often nearly right. Skip is always there — a question the user does not
 * want to answer must not be able to hold a run open.
 */
function QuestionPanel({
  question,
  onReply,
}: {
  question: Question;
  onReply: (answer: string | null) => void;
}) {
  const { t } = useTranslation();
  const [picked, setPicked] = useState<string[]>([]);
  const [written, setWritten] = useState("");

  function toggle(option: string) {
    if (!question.multiple) {
      onReply(option);
      return;
    }
    setPicked((prev) =>
      prev.includes(option) ? prev.filter((p) => p !== option) : [...prev, option],
    );
  }

  const answer = picked.length > 0 ? picked.join(", ") : written.trim();

  return (
    <div className="approval approval-question">
      <div className="approval-body">
        <div className="approval-title">{t("ask.title")}</div>
        <div className="ask-question">{question.question}</div>
        {question.options.length > 0 && (
          <div className="ask-options">
            {question.options.map((option) => (
              <button
                key={option}
                className={picked.includes(option) ? "ask-option ask-option-on" : "ask-option"}
                onClick={() => toggle(option)}
              >
                {question.multiple && (
                  <span className="ask-check">{picked.includes(option) ? "✓" : ""}</span>
                )}
                {option}
              </button>
            ))}
          </div>
        )}
        <input
          className="ask-input"
          placeholder={
            question.options.length > 0 ? t("ask.orWrite") : t("ask.writePlaceholder")
          }
          value={written}
          onChange={(e) => setWritten(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && answer) onReply(answer);
          }}
          autoFocus
        />
      </div>
      <div className="approval-actions">
        <button className="btn" onClick={() => onReply(null)}>
          {t("ask.skip")}
        </button>
        <button className="btn btn-primary" disabled={!answer} onClick={() => onReply(answer)}>
          {t("ask.send")}
        </button>
      </div>
    </div>
  );
}

/**
 * One attached file, shown as what it is.
 *
 * An image shows the image. Everything else shows its name and size — which is
 * the whole point of the size being right: "notes.txt · 4.2 KB" says something,
 * "notes.txt · 0 MB" says the app is broken.
 */
function AttachmentCard({ attachment }: { attachment: Attachment }) {
  const [big, setBig] = useState(false);

  if (attachment.preview) {
    return (
      <>
        <button
          className="file-thumb"
          onClick={() => setBig(true)}
          title={`${attachment.name} · ${formatBytes(attachment.bytes)}`}
        >
          <img src={attachment.preview} alt={attachment.name} />
        </button>
        {big && (
          // The preview is a thumbnail, so this is as large as it gets. Full
          // resolution would mean holding every screenshot in the transcript.
          <div className="lightbox" onClick={() => setBig(false)}>
            <img src={attachment.preview} alt={attachment.name} />
          </div>
        )}
      </>
    );
  }

  return (
    <span className="file-card" title={attachment.path}>
      <span className="file-card-name">{attachment.name}</span>
      <span className="file-card-size">{formatBytes(attachment.bytes)}</span>
    </span>
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
  const { t, i18n } = useTranslation();
  const [copied, setCopied] = useState(false);

  function copy() {
    onCopy(entry.text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  return (
    <div className="userwrap">
      {entry.attachments && entry.attachments.length > 0 && (
        <div className="msg-files">
          {entry.attachments.map((a) => (
            <AttachmentCard key={a.path} attachment={a} />
          ))}
        </div>
      )}
      <div className="bubble">{entry.text}</div>
      <div className="msg-actions">
        <button className="msg-action" title={t("chat.copy")} onClick={copy}>
          {copied ? "✓" : "⧉"}
        </button>
        <button
          className="msg-action"
          title={t("chat.retry")}
          onClick={() => onRetry(index)}
          disabled={busy}
        >
          ↺
        </button>
        <button
          className="msg-action"
          title={t("chat.branch")}
          onClick={() => onBranch(index)}
        >
          ⑂
        </button>
        <button
          className="msg-action msg-action-danger"
          title={t("chat.delete")}
          onClick={() => onRemove(index)}
          disabled={busy}
        >
          ✕
        </button>
        <span className="msg-time">{formatAgo(entry.at, i18n.language)}</span>
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
  const { t } = useTranslation();
  if (!limit) return <span className="ctx-empty">{t("context.none")}</span>;
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
  question,
  mode,
  assisted,
  model,
  effort,
  numCtx,
  attachments,
  dropping,
  liveTokens,
  runTokens,
  waiting,
  runStartedAt,
  onPaste,
  onDetach,
  onSend,
  onAnswer,
  onReply,
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
  question: Question | null;
  mode: PermissionMode;
  assisted: boolean;
  model: string;
  effort: Effort;
  numCtx: number | null;
  attachments: Attachment[];
  dropping: boolean;
  /// Counted client-side while tokens stream, so the status line moves
  /// during a step instead of only between them.
  liveTokens: number;
  runTokens: number;
  waiting: { seconds: number; firstToken: boolean } | null;
  runStartedAt: number | null;
  onPaste: (file: File) => void;
  onDetach: (path: string) => void;
  onSend: (text: string) => void;
  onAnswer: (approved: boolean) => void;
  onReply: (answer: string | null) => void;
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
  const { t } = useTranslation();
  const [prompt, setPrompt] = useState("");
  const [menu, setMenu] = useState<"mode" | "model" | "effort" | "ctx" | null>(null);
  const [ctxOptions, setCtxOptions] = useState<ContextOptions | null>(null);
  const [tick, setTick] = useState(0);
  const transcript = useRef<HTMLDivElement>(null);
  const [atBottom, setAtBottom] = useState(true);
  const input = useRef<HTMLTextAreaElement>(null);

  /// Grow the box upward as the instruction gets longer, then stop.
  ///
  /// A two-line window on a fifteen-line instruction means writing through a
  /// slot. Growing without limit is the other failure — the transcript
  /// disappears behind the composer — so it stops at [`INPUT_MAX_HEIGHT`] and
  /// scrolls from there.
  useEffect(() => {
    const el = input.current;
    if (!el) return;
    // Reset first: `scrollHeight` never shrinks below the height already set.
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, INPUT_MAX_HEIGHT)}px`;
  }, [prompt]);

  // A clock that only advances between steps is not a clock.
  useEffect(() => {
    if (runStartedAt === null) return;
    const timer = window.setInterval(() => setTick((n) => n + 1), 500);
    return () => window.clearInterval(timer);
  }, [runStartedAt]);

  const elapsedMs =
    session.elapsedMs + (runStartedAt !== null ? Date.now() - runStartedAt : 0);
  const totalTokens = session.tokens + liveTokens;
  void tick;

  /// Live decode rate, timed from the first token rather than from the button.
  ///
  /// Counting from when the run started would fold a thirty-second model load
  /// into the average and report six tok/s for a model doing forty.
  const firstTokenAt = useRef<number | null>(null);
  const tokensAtStart = useRef(0);
  if (runStartedAt === null) {
    firstTokenAt.current = null;
    tokensAtStart.current = 0;
  } else if (firstTokenAt.current === null && runTokens > 0) {
    firstTokenAt.current = Date.now();
    tokensAtStart.current = runTokens;
  }
  const sinceFirst = firstTokenAt.current ? (Date.now() - firstTokenAt.current) / 1000 : 0;
  const liveRate =
    sinceFirst > 0.6 ? (runTokens - tokensAtStart.current) / sinceFirst : null;
  // While a run is in flight the live figure is the true one; between runs the
  // daemon's own count from the last step is more accurate than our estimate.
  const rate = liveRate ?? session.lastRate;

  // The last entry grows token by token without the entry count changing, so
  // its length is what tells this effect that there is new text on screen.
  const last = session.entries[session.entries.length - 1];
  const tailLength =
    last === undefined
      ? 0
      : last.kind === "tool"
        ? (last.output?.length ?? 0)
        : last.text.length;

  /// Follow the newest output, unless the user has scrolled away from it.
  useEffect(() => {
    if (!atBottom) return;
    const el = transcript.current;
    if (!el) return;
    // After paint, and without smoothing: a smooth scroll restarted by the next
    // token never arrives, which is what made this feel unreliable.
    const id = requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight;
    });
    return () => cancelAnimationFrame(id);
  }, [session.entries.length, tailLength, pending, running, atBottom]);

  // Cloud models report no capability metadata, so filtering on `tools` would
  // hide them entirely. They are listed and labelled instead — whether tool
  // calling works is between the user and the remote model.
  const usable = useMemo(
    () => models.filter((m) => m.capabilities.tools || m.placement === "cloud"),
    [models],
  );

  /// Whether the chosen model is one that spills, while a resident one exists.
  ///
  /// Not a warning about the model — it is the user's machine and their choice.
  /// It is a price tag. Measured on the reference machine, the same instruction
  /// through the same agent code took 13 s on a resident 7B and 431 s on a
  /// spilled 27B: about ten times per line of output. That is worth knowing
  /// before waiting, and nothing else in the interface says it at the moment of
  /// choosing.
  const slowChoice = useMemo(() => {
    const chosen = models.find((m) => m.name === model);
    if (!chosen || chosen.placement !== "spilled") return false;
    return models.some((m) => m.placement === "resident" && m.capabilities.tools);
  }, [models, model]);

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
      const ok = window.confirm(t("modes.unrestrictedWarning"));
      if (!ok) return;
    }
    onMode(next);
    setMenu(null);
  }

  function send() {
    // The box stays editable during a run, so this is where sending is refused.
    if (!prompt.trim() || running) return;
    onSend(prompt);
    setPrompt("");
  }

  const folderName = session.workspace?.split(/[\\/]/).filter(Boolean).pop();

  return (
    <div className={dropping ? "chat chat-dropping" : "chat"}>
      <header className="chat-head">
        <div className="chat-title" title={session.title}>
          {session.title}
        </div>
        <button
          className="chip chip-folder"
          onClick={pickFolder}
          title={session.workspace ?? t("chat.noFolder")}
        >
          {folderName ?? t("chat.chooseFolder")}
        </button>
      </header>

      {mode === "unrestricted" && (
        <div className="danger-banner">
          {t("modes.unrestrictedBanner")}
        </div>
      )}

      <div
        className="transcript"
        ref={transcript}
        onScroll={(e) => {
          const el = e.currentTarget;
          // A threshold rather than an exact match: sub-pixel layout means the
          // numbers rarely land on zero even when the view is at the bottom.
          const near = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
          setAtBottom((was) => (was === near ? was : near));
        }}
      >
        {session.entries.length === 0 && (
          <div className="empty">
            <div className="empty-mark">▮</div>
            <h2>{t("chat.emptyTitle")}</h2>
            <p>
              {session.workspace ? t("chat.emptyWithFolder") : t("chat.emptyNoFolder")}
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
          if (entry.kind === "thinking") {
            // Only the last entry can still be receiving tokens.
            const live = running && i === session.entries.length - 1;
            return <ThinkingBlock key={i} text={entry.text} live={live} />;
          }
          if (entry.kind === "assistant") {
            return entry.text.trim() ? (
              <div key={i} className="turn turn-assistant">
                <Markdown text={entry.text} />
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
            <span className="pulse" />{" "}
            {waiting
              ? t(waiting.firstToken ? "chat.loading" : "chat.generating", {
                  time: formatElapsed(waiting.seconds * 1000),
                })
              : t("chat.working")}
          </div>
        )}
        {!atBottom && (
          <button
            className="jump-latest"
            onClick={() => {
              const el = transcript.current;
              if (el) el.scrollTop = el.scrollHeight;
              setAtBottom(true);
            }}
          >
            {t("chat.jumpLatest")} ↓
          </button>
        )}
      </div>

      {question && <QuestionPanel question={question} onReply={onReply} />}

      {pending && (
        <div className="approval">
          <div className="approval-body">
            <div className="approval-title">
              {t("approval.title", { tool: pending.name })}
            </div>
            <div className="approval-reason">{pending.evaluation.reason}</div>
            <pre className="code-block">{JSON.stringify(pending.arguments, null, 2)}</pre>
          </div>
          <div className="approval-actions">
            <button className="btn" onClick={() => onAnswer(false)}>
              {t("approval.reject")}
            </button>
            <button className="btn btn-primary" onClick={() => onAnswer(true)}>
              {t("approval.approve")}
            </button>
          </div>
        </div>
      )}

      <div className="composer">
        {attachments.length > 0 && (
          <div className="attachments">
            {attachments.map((a) => (
              <span
                key={a.path}
                className={a.preview ? "attachment attachment-image" : "attachment"}
                title={`${a.path} · ${a.tool}`}
              >
                {a.preview && <img className="attachment-thumb" src={a.preview} alt="" />}
                <span className="attachment-name">{a.name}</span>
                <span className="attachment-size">{formatBytes(a.bytes)}</span>
                <button
                  className="attachment-remove"
                  onClick={() => onDetach(a.path)}
                  title={t("chat.detach")}
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        )}
        <div className="composer-box">
          <textarea
            className="input"
            ref={input}
            rows={2}
            placeholder={
              running
                ? t("chat.placeholderBusy")
                : session.workspace
                  ? t("chat.placeholder")
                  : t("chat.placeholderNoFolder")
            }
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            onPaste={(e) => {
              // A pasted image arrives as a clipboard file with no path, so it
              // is staged by bytes rather than by name.
              const files = Array.from(e.clipboardData.files);
              if (files.length === 0) return;
              e.preventDefault();
              files.forEach(onPaste);
            }}
            // Deliberately not disabled while a run is in flight: composing the
            // next instruction is the natural thing to do while waiting, and a
            // dead text box for a minute is the wrong answer. Only sending is
            // blocked.
            disabled={!session.workspace}
          />
          <button
            className="send"
            onClick={send}
            disabled={running || !prompt.trim() || !session.workspace}
            title={t("chat.send")}
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
              {t(`modes.${mode}`)}
              <span className="caret">⌃</span>
            </button>
            <Popover open={menu === "mode"} onClose={() => setMenu(null)} width={290}>
              <PopoverLabel>{t("modes.label")}</PopoverLabel>
              {MODES.map((m, i) => (
                <PopoverItem
                  key={m}
                  label={t(`modes.${m}`)}
                  hint={t(`modes.${m}_hint`)}
                  selected={mode === m}
                  shortcut={String(i + 1)}
                  danger={m === "unrestricted"}
                  badge={m === "autonomous" ? t("modes.recommended") : undefined}
                  onClick={() => changeMode(m)}
                />
              ))}
              <PopoverDivider />
              <PopoverItem
                label={t("modes.assisted")}
                hint={t("modes.assistedHint")}
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
              title={ctxOptions ? t("context.label") : t("context.pickModel")}
            >
              <ContextMeter
                // The daemon reports the window's true occupancy only when a
                // step finishes. Adding what has streamed since keeps the meter
                // moving during the step, which is the only time it matters.
                used={session.contextUsed + (running ? liveTokens : 0)}
                limit={limit}
                trainedMax={trainedMax}
              />
            </button>
            <Popover open={menu === "ctx"} onClose={() => setMenu(null)} width={270}>
              <PopoverLabel>{t("context.label")}</PopoverLabel>
              <div className="popover-note">
                {trainedMax
                  ? t("context.trained", { max: formatTokens(trainedMax) })
                  : t("context.noMax")}
              </div>
              <PopoverDivider />
              <PopoverItem
                label={t("context.recommended")}
                hint={
                  ctxOptions?.recommended
                    ? t("context.recommendedHint", { size: formatTokens(ctxOptions.recommended) })
                    : t("context.notComputed")
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
              {model || `${t("modelsView.auto")}${routing?.coding ? ` · ${routing.coding}` : ""}`}
              {slowChoice && (
                <span className="ctl-cost" title={t("modelsView.spilledCostHint")}>
                  {t("modelsView.spilledCost")}
                </span>
              )}
              <span className="caret">⌃</span>
            </button>
            <Popover
              open={menu === "model"}
              onClose={() => setMenu(null)}
              align="right"
              width={330}
            >
              <PopoverLabel>{t("modelsView.label")}</PopoverLabel>
              <PopoverItem
                label={t("modelsView.auto")}
                hint={t("modelsView.autoHint")}
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
                  badge={m.placement === "spilled" ? t("modelsView.slow") : undefined}
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
              {t(`effort.${effort}`)}
            </button>
            <Popover
              open={menu === "effort"}
              onClose={() => setMenu(null)}
              align="right"
              width={290}
            >
              <PopoverLabel>{t("effort.label")}</PopoverLabel>
              <div className="popover-note">{t("effort.note")}</div>
              <PopoverDivider />
              {EFFORTS.map((e, i) => (
                <PopoverItem
                  key={e}
                  label={t(`effort.${e}`)}
                  hint={t(`effort.${e}_hint`)}
                  selected={effort === e}
                  shortcut={String(i + 1)}
                  onClick={() => {
                    onEffort(e);
                    setMenu(null);
                  }}
                />
              ))}
            </Popover>
          </div>
        </div>

        <div className="statusline">
          <span className="status-mark" aria-hidden="true">◷</span>
          <span>{formatElapsed(elapsedMs)}</span>
          <span className="sep">·</span>
          <span>{formatTokens(totalTokens)} {t("chat.used")}</span>
          {rate !== null && (
            <>
              <span className="sep">·</span>
              <span className={liveRate !== null ? "status-rate status-rate-live" : "status-rate"}>
                {rate.toFixed(1)} tok/s
              </span>
            </>
          )}
          {running && (
            <button className="status-cancel" onClick={onCancel}>
              {t("chat.stop")}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
