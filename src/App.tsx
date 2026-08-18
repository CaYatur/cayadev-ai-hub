import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import Activity from "./components/Activity";
import Chat from "./components/Chat";
import Models from "./components/Models";
import Sidebar from "./components/Sidebar";
import {
  api,
  onFileDrop,
  subscribe,
  type Attachment,
  type AgentEvent,
  type Effort,
  type ModelProfile,
  type PermissionMode,
  type ProposedCall,
} from "./lib/agent";
import {
  applyMetrics,
  load,
  newSession,
  save,
  titleFrom,
  type Entry,
  type Session,
} from "./lib/sessions";
import "./styles/app.css";

type View = "chat" | "models" | "activity";

export default function App() {
  const { t } = useTranslation();
  const [sessions, setSessions] = useState<Session[]>(() => {
    const stored = load();
    return stored.length > 0 ? stored : [newSession()];
  });
  const [activeId, setActiveId] = useState<string | null>(null);
  const [view, setView] = useState<View>("chat");
  const [models, setModels] = useState<ModelProfile[]>([]);
  const [routing, setRouting] = useState<Record<string, string> | null>(null);
  const [version, setVersion] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [pending, setPending] = useState<ProposedCall | null>(null);

  // Run settings live here, not in `Chat`. `Chat` unmounts when the Models view
  // is shown, and local state there would silently revert to defaults while the
  // engine kept the old value — which for permission mode means a safe-looking
  // label over an unrestricted engine.
  const [mode, setModeState] = useState<PermissionMode>("project_auto");
  const [assisted, setAssistedState] = useState(true);
  const [model, setModel] = useState("");
  const [effort, setEffort] = useState<Effort>("balanced");
  const [numCtx, setNumCtx] = useState<number | null>(null);
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [dropping, setDropping] = useState(false);
  // Metrics only arrive when a step finishes, so a long step left the
  // status line frozen. These are counted client-side as tokens stream.
  const [liveTokens, setLiveTokens] = useState(0);
  const [runStartedAt, setRunStartedAt] = useState<number | null>(null);

  // The event stream is global, so handlers need the current session id without
  // resubscribing on every switch.
  const activeRef = useRef<string | null>(null);
  useEffect(() => {
    activeRef.current = activeId ?? sessions[0]?.id ?? null;
  }, [activeId, sessions]);

  useEffect(() => {
    if (!activeId && sessions.length > 0) setActiveId(sessions[0].id);
  }, [activeId, sessions]);

  useEffect(() => {
    save(sessions);
  }, [sessions]);

  const active = sessions.find((s) => s.id === activeId) ?? sessions[0];

  const patch = useCallback((id: string, fn: (s: Session) => Session) => {
    setSessions((prev) => prev.map((s) => (s.id === id ? fn(s) : s)));
  }, []);

  const append = useCallback(
    (id: string, entry: Entry) => {
      patch(id, (s) => ({ ...s, entries: [...s.entries, entry] }));
    },
    [patch],
  );

  const loadModels = useCallback(async () => {
    try {
      setVersion(await api.version());
      const list = await api.listModels();
      setModels([...list].sort((a, b) => a.weights_bytes - b.weights_bytes));
      setRouting((await api.routing()).assignments);
    } catch {
      setVersion(null);
    }
  }, []);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  useEffect(() => {
    let off: (() => void) | undefined;
    let cancelled = false;

    subscribe(
      (event: AgentEvent) => {
        const id = activeRef.current;
        if (!id) return;

        switch (event.kind) {
          case "token":
            // Rough, and deliberately so: an exact count needs the model's
            // tokenizer and a round-trip. ~3.6 chars/token is close enough for
            // a number that changes several times a second.
            setLiveTokens((n) => n + Math.max(1, Math.round(event.text.length / 3.6)));
            patch(id, (s) => {
              const entries = [...s.entries];
              const last = entries[entries.length - 1];
              // Append to the open assistant turn rather than starting a new one.
              if (last?.kind === "assistant") {
                entries[entries.length - 1] = { ...last, text: last.text + event.text };
              } else {
                entries.push({ kind: "assistant", text: event.text });
              }
              return { ...s, entries };
            });
            break;

          case "thinking":
            patch(id, (s) => {
              const entries = [...s.entries];
              const last = entries[entries.length - 1];
              if (last?.kind === "thinking") {
                entries[entries.length - 1] = { ...last, text: last.text + event.text };
              } else {
                entries.push({ kind: "thinking", text: event.text });
              }
              return { ...s, entries };
            });
            break;

          case "tool_proposed":
            append(id, { kind: "tool", call: event.call });
            break;

          case "tool_progress":
            patch(id, (s) => {
              const entries = [...s.entries];
              // Attach to the call still running, which is the last unsettled one.
              for (let i = entries.length - 1; i >= 0; i--) {
                const e = entries[i];
                if (e.kind === "tool" && !e.settled) {
                  entries[i] = { ...e, output: (e.output ?? "") + event.text + "\n" };
                  break;
                }
              }
              return { ...s, entries };
            });
            break;

          case "tool_settled":
            patch(id, (s) => {
              const entries = [...s.entries];
              for (let i = entries.length - 1; i >= 0; i--) {
                const e = entries[i];
                if (e.kind === "tool" && e.call.id === event.id) {
                  entries[i] = { ...e, output: event.output, settled: event.outcome };
                  break;
                }
              }
              return { ...s, entries };
            });
            break;

          case "step_done":
            patch(id, (s) => applyMetrics(s, event.metrics));
            break;

          case "error":
            append(id, { kind: "notice", text: event.message, tone: "error" });
            break;

          case "done":
            // `StopReason` serialises snake_case, so compare in that case.
            if (event.reason !== "completed") {
              append(id, {
                kind: "notice",
                text: t("chat.stopped", { reason: event.reason.replace(/_/g, " ") }),
                tone: "info",
              });
            }
            break;
        }
      },
      (call) => setPending(call),
    ).then((fn) => {
      // The subscription resolves asynchronously, so a cleanup that ran first
      // would leave this listener attached forever — and every event would then
      // be handled twice, duplicating text and tool rows.
      if (cancelled) fn();
      else off = fn;
    });

    return () => {
      cancelled = true;
      off?.();
    };
  }, [append, patch, t]);

  const attach = useCallback(
    async (paths: string[]) => {
      const id = activeRef.current;
      try {
        const added = await api.attachFiles(paths);
        // De-duplicate: dropping the same file twice should not add it twice.
        setAttachments((prev) => {
          const seen = new Set(prev.map((a) => a.path));
          return [...prev, ...added.filter((a) => !seen.has(a.path))];
        });
      } catch (e) {
        if (id) append(id, { kind: "notice", text: String(e), tone: "error" });
      }
    },
    [append],
  );

  /// Stage an image pasted from the clipboard, which has no path of its own.
  const attachPasted = useCallback(
    async (file: File) => {
      const id = activeRef.current;
      try {
        const bytes = Array.from(new Uint8Array(await file.arrayBuffer()));
        const extension = file.type.split("/")[1] || "png";
        const added = await api.attachBytes(file.name || `pasted.${extension}`, bytes);
        setAttachments((prev) => [...prev, added]);
      } catch (e) {
        if (id) append(id, { kind: "notice", text: String(e), tone: "error" });
      }
    },
    [append],
  );

  const detach = useCallback(async (path: string) => {
    setAttachments((prev) => prev.filter((a) => a.path !== path));
  }, []);

  useEffect(() => {
    let off: (() => void) | undefined;
    let cancelled = false;
    onFileDrop(attach, setDropping).then((fn) => {
      if (cancelled) fn();
      else off = fn;
    });
    return () => {
      cancelled = true;
      off?.();
    };
  }, [attach]);

  const setWorkspace = useCallback(
    async (path: string) => {
      if (!active) return;
      try {
        const resolved = await api.setWorkspace(path);
        patch(active.id, (s) => ({ ...s, workspace: resolved }));
      } catch (e) {
        append(active.id, { kind: "notice", text: String(e), tone: "error" });
      }
    },
    [active, patch, append],
  );

  const changeMode = useCallback(async (next: PermissionMode) => {
    setModeState(next);
    await api.setMode(next);
  }, []);

  const changeAssisted = useCallback(async (enabled: boolean) => {
    setAssistedState(enabled);
    await api.setAssisted(enabled);
  }, []);

  const run = useCallback(
    async (id: string, text: string) => {
      if (!active) return;
      // Sessions each remember a folder but the engine holds one; re-assert it
      // so a run always uses the folder the header is showing.
      if (active.workspace) {
        try {
          await api.setWorkspace(active.workspace);
        } catch (e) {
          append(id, { kind: "notice", text: String(e), tone: "error" });
          return;
        }
      }
      setRunning(true);
      setRunStartedAt(Date.now());
      setLiveTokens(0);
      try {
        await api.run(text, model || undefined, numCtx ?? undefined, effort);
      } catch (e) {
        append(id, { kind: "notice", text: String(e), tone: "error" });
      } finally {
        setRunning(false);
        setRunStartedAt(null);
        setLiveTokens(0);
        setPending(null);
      }
    },
    [active, append, model, numCtx, effort],
  );

  const send = useCallback(
    async (text: string) => {
      if (!active) return;
      const id = active.id;
      // Tell the model where the attached files are, in the message rather
      // than the system prompt — the prompt has to stay byte-stable for the
      // daemon to reuse its KV cache.
      const withFiles = attachments.length
        ? `${text}\n\nAttached files:\n${attachments
            .map((a) => `- ${a.path} (open with ${a.tool})`)
            .join("\n")}`
        : text;

      patch(id, (s) => ({
        ...s,
        title: s.entries.length === 0 ? titleFrom(text) : s.title,
        entries: [...s.entries, { kind: "user", text, at: Date.now() }],
      }));
      await run(id, withFiles);
    },
    [active, patch, run, attachments],
  );

  /// Re-run a user turn, discarding everything that followed it.
  const retry = useCallback(
    async (index: number) => {
      if (!active || running) return;
      const id = active.id;
      const entry = active.entries[index];
      if (entry?.kind !== "user") return;

      patch(id, (s) => ({ ...s, entries: s.entries.slice(0, index + 1) }));
      await run(id, entry.text);
    },
    [active, running, patch, run],
  );

  /// Drop a user turn and everything it produced.
  const removeFrom = useCallback(
    (index: number) => {
      if (!active) return;
      patch(active.id, (s) => ({ ...s, entries: s.entries.filter((_, i) => i < index) }));
    },
    [active, patch],
  );

  /// Copy a turn to the clipboard.
  const copy = useCallback((text: string) => {
    navigator.clipboard?.writeText(text).catch(() => {});
  }, []);

  /// Continue from a turn in a new session, leaving this one intact.
  const branch = useCallback(
    (index: number) => {
      if (!active) return;
      const created = newSession(active.workspace);
      created.entries = active.entries.slice(0, index + 1);
      const first = created.entries.find((e) => e.kind === "user");
      created.title = first && first.kind === "user" ? titleFrom(first.text) : "Branch";
      setSessions((prev) => [created, ...prev]);
      setActiveId(created.id);
    },
    [active],
  );

  const answer = useCallback(
    async (approved: boolean) => {
      if (!pending) return;
      await api.resolveApproval(pending.id, approved);
      setPending(null);
    },
    [pending],
  );

  const cancel = useCallback(async () => {
    await api.cancel();
    setPending(null);
  }, []);

  const startSession = useCallback(() => {
    // Carry the folder over — a new session in the same project is the norm.
    const created = newSession(active?.workspace ?? null);
    setSessions((prev) => [created, ...prev]);
    setActiveId(created.id);
    setView("chat");
  }, [active]);

  const deleteSession = useCallback((id: string) => {
    setSessions((prev) => {
      const next = prev.filter((s) => s.id !== id);
      return next.length > 0 ? next : [newSession()];
    });
    setActiveId((current) => (current === id ? null : current));
  }, []);

  return (
    <div className="shell">
      <Sidebar
        sessions={sessions}
        activeId={active?.id ?? null}
        onSelect={(id) => {
          setActiveId(id);
          setView("chat");
        }}
        onNew={startSession}
        onDelete={deleteSession}
        version={version}
      />

      <div className="main">
        <div className="viewbar">
          <button
            className={view === "chat" ? "viewtab viewtab-on" : "viewtab"}
            onClick={() => setView("chat")}
          >
            {t("app.agent")}
          </button>
          <button
            className={view === "models" ? "viewtab viewtab-on" : "viewtab"}
            onClick={() => setView("models")}
          >
            {t("app.models")}
          </button>
          <button
            className={view === "activity" ? "viewtab viewtab-on" : "viewtab"}
            onClick={() => setView("activity")}
          >
            {t("app.activity")}
          </button>
        </div>

        {view === "chat" && active ? (
          <Chat
            session={active}
            models={models}
            routing={routing}
            running={running}
            pending={pending}
            mode={mode}
            assisted={assisted}
            model={model}
            effort={effort}
            numCtx={numCtx}
            onSend={send}
            onAnswer={answer}
            onWorkspace={setWorkspace}
            onCancel={cancel}
            onMode={changeMode}
            onAssisted={changeAssisted}
            onModel={setModel}
            onEffort={setEffort}
            onNumCtx={setNumCtx}
            attachments={attachments}
            liveTokens={liveTokens}
            runStartedAt={runStartedAt}
            onPaste={attachPasted}
            dropping={dropping}
            onDetach={detach}
            onRetry={retry}
            onRemove={removeFrom}
            onCopy={copy}
            onBranch={branch}
          />
        ) : view === "activity" && active ? (
          <main className="content">
            <Activity
              session={active}
              onRestored={(paths) =>
                append(active.id, {
                  kind: "notice",
                  text: t("activity.restored", { files: paths.join(", ") }),
                  tone: "info",
                })
              }
            />
          </main>
        ) : (
          <main className="content">
            <Models
              models={models}
              routing={routing}
              onRefreshVram={async () => {
                await api.refreshVram();
                await loadModels();
              }}
            />
          </main>
        )}
      </div>
    </div>
  );
}
