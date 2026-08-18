import { useCallback, useEffect, useState } from "react";
import Chat from "./components/Chat";
import Models from "./components/Models";
import { api, type ModelProfile } from "./lib/agent";
import "./styles/app.css";

type Tab = "chat" | "models";

export default function App() {
  const [tab, setTab] = useState<Tab>("chat");
  const [models, setModels] = useState<ModelProfile[]>([]);
  const [routing, setRouting] = useState<Record<string, string> | null>(null);
  const [version, setVersion] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setVersion(await api.version());
      const list = await api.listModels();
      setModels([...list].sort((a, b) => a.weights_bytes - b.weights_bytes));
      // Depends on list_models having populated state, so it runs after.
      setRouting((await api.routing()).assignments);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const refreshVram = useCallback(async () => {
    try {
      await api.refreshVram();
      await load();
    } catch (e) {
      setError(String(e));
    }
  }, [load]);

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          CaYa<span>Dev</span> AI Hub
        </div>
        <nav className="tabs">
          <button
            className={tab === "chat" ? "tab tab-active" : "tab"}
            onClick={() => setTab("chat")}
          >
            Agent
          </button>
          <button
            className={tab === "models" ? "tab tab-active" : "tab"}
            onClick={() => setTab("models")}
          >
            Models
          </button>
        </nav>
        <div className="topbar-meta">
          {version ? `Ollama ${version}` : "connecting…"}
        </div>
      </header>

      {error && (
        <div className="state state-error">
          Could not reach Ollama. <span className="muted">{error}</span>
        </div>
      )}

      {tab === "chat" ? (
        <Chat models={models} />
      ) : (
        <main className="content">
          <Models models={models} routing={routing} onRefreshVram={refreshVram} />
        </main>
      )}
    </div>
  );
}
