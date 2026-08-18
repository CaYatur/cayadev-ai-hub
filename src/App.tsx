import { useCallback, useEffect, useRef, useState } from "react";
import Chat from "./components/Chat";
import Models from "./components/Models";
import Sidebar from "./components/Sidebar";
import {
  api,
  subscribe,
  type AgentEvent,
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

type View = "chat" | "models";

export default function App() {
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
  // Permission state lives here, not in `Chat`, so switching to the Models
  // view and back cannot show a mode the engine is not actually in.
  const [mode, setModeState] = useState<PermissionMode>("project_auto");
  const [assisted, setAssistedState] = useState(true);

  // The event stream is global, so handlers need the current session id without
  // being torn down and resubscribed on every switch.
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

    subscribe(
      (event: AgentEvent) => {
        const id = activeRef.current;
        if (!id) return;

        switch (event.kind) {
          case "token":
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

          case "tool_proposed":
            append(id, { kind: "tool", call: event.call });
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
            if (event.reason !== "Completed") {
              append(id, { kind: "notice", text: `Stopped: ${event.reason}`, tone: "info" });
            }
            break;
        }
      },
      (call) => setPending(call),
    ).then((fn) => {
      off = fn;
    });

    return () => off?.();
  }, [append, patch]);

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

  const send = useCallback(
    async (text: string, model?: string) => {
      if (!active) return;
      const id = active.id;

      // Sessions each remember a folder, but the engine holds only one. Re-assert
      // it here so a run always uses the folder the header is showing.
      if (active.workspace) {
        try {
          await api.setWorkspace(active.workspace);
        } catch (e) {
          append(id, { kind: "notice", text: String(e), tone: "error" });
          return;
        }
      }

      patch(id, (s) => ({
        ...s,
        title: s.entries.length === 0 ? titleFrom(text) : s.title,
        entries: [...s.entries, { kind: "user", text }],
      }));
      setRunning(true);
      try {
        await api.run(text, model);
      } catch (e) {
        append(id, { kind: "notice", text: String(e), tone: "error" });
      } finally {
        setRunning(false);
        setPending(null);
      }
    },
    [active, patch, append],
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

  const deleteSession = useCallback(
    (id: string) => {
      setSessions((prev) => {
        const next = prev.filter((s) => s.id !== id);
        return next.length > 0 ? next : [newSession()];
      });
      setActiveId((current) => (current === id ? null : current));
    },
    [],
  );

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
            Agent
          </button>
          <button
            className={view === "models" ? "viewtab viewtab-on" : "viewtab"}
            onClick={() => setView("models")}
          >
            Models
          </button>
        </div>

        {view === "chat" && active ? (
          <Chat
            session={active}
            models={models}
            running={running}
            pending={pending}
            mode={mode}
            assisted={assisted}
            onSend={send}
            onAnswer={answer}
            onWorkspace={setWorkspace}
            onCancel={cancel}
            onMode={changeMode}
            onAssisted={changeAssisted}
          />
        ) : (
          <main className="content">
            <Models models={models} routing={routing} onRefreshVram={async () => {
              await api.refreshVram();
              await loadModels();
            }} />
          </main>
        )}
      </div>
    </div>
  );
}
