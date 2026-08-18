import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./styles/app.css";

type Placement = "resident" | "marginal" | "spilled" | "cloud" | "unknown";

/** Mirrors `ModelProfile` in src-tauri/src/ollama/registry.rs. */
type ModelProfile = {
  name: string;
  family: string;
  parameter_size: string;
  quantization: string;
  weights_bytes: number;
  capabilities: {
    completion: boolean;
    tools: boolean;
    vision: boolean;
    thinking: boolean;
    embedding: boolean;
  };
  max_context: number | null;
  recommended_context: number | null;
  placement: Placement;
};

type Routing = { assignments: Record<string, string>; prefer_resident: boolean };

const PLACEMENT_COPY: Record<Placement, { label: string; hint: string }> = {
  resident: { label: "Fits in VRAM", hint: "Full speed — tens of tokens/sec." },
  marginal: { label: "Tight fit", hint: "Fits only when little else uses the GPU." },
  spilled: {
    label: "Spills to RAM",
    hint: "Memory-bandwidth bound. Expect single-digit tokens/sec.",
  },
  cloud: { label: "Cloud", hint: "Runs remotely; local memory does not apply." },
  unknown: { label: "Unknown", hint: "Not enough metadata to estimate." },
};

const TASK_LABELS: Record<string, string> = {
  reasoning: "Planning & hard debugging",
  coding: "Writing code",
  edit: "Mechanical edits",
  classify: "Approvals & routing",
  vision: "Images",
  summarize: "Summaries & titles",
};

function formatBytes(n: number): string {
  if (!n) return "—";
  const gb = n / 1024 ** 3;
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(n / 1024 ** 2).toFixed(0)} MB`;
}

function capabilityList(c: ModelProfile["capabilities"]): string[] {
  return Object.entries(c)
    .filter(([, on]) => on)
    .map(([name]) => name);
}

export default function App() {
  const [models, setModels] = useState<ModelProfile[]>([]);
  const [routing, setRouting] = useState<Routing | null>(null);
  const [version, setVersion] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const v = await invoke<string>("ollama_version");
        const list = await invoke<ModelProfile[]>("list_models");
        // Depends on list_models having populated state, so it runs after.
        const policy = await invoke<Routing>("routing_policy");
        if (cancelled) return;
        setVersion(v);
        setModels([...list].sort((a, b) => a.weights_bytes - b.weights_bytes));
        setRouting(policy);
      } catch (e) {
        if (!cancelled) setError(String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          CaYa<span>Dev</span> AI Hub
        </div>
        <div className="topbar-meta">
          {version ? `Ollama ${version} · localhost:11434` : "connecting…"}
        </div>
      </header>

      <main className="content">
        <section className="intro">
          <h1>Installed models</h1>
          <p>
            Placement is estimated against available VRAM, accounting for hybrid
            attention. A model that spills to system RAM is bound by memory
            bandwidth — routing work to a resident model is worth roughly 10x on
            this hardware.
          </p>
        </section>

        {loading && <div className="state">Reading model metadata…</div>}

        {error && (
          <div className="state state-error">
            Could not reach Ollama.
            <br />
            <span className="muted">{error}</span>
          </div>
        )}

        {routing && (
          <section className="routing">
            <h2>Routing</h2>
            <div className="routing-rows">
              {Object.entries(routing.assignments).map(([task, model]) => (
                <div key={task} className="routing-row">
                  <span className="routing-task">{TASK_LABELS[task] ?? task}</span>
                  <span className="routing-model">{model}</span>
                </div>
              ))}
            </div>
          </section>
        )}

        {!loading && !error && (
          <div className="grid">
            {models.map((m) => (
              <article key={m.name} className={`card card-${m.placement}`}>
                <div className="card-head">
                  <h2>{m.name}</h2>
                  <span className={`chip chip-${m.placement}`}>
                    {PLACEMENT_COPY[m.placement].label}
                  </span>
                </div>
                <p className="hint">{PLACEMENT_COPY[m.placement].hint}</p>
                <dl className="facts">
                  <div>
                    <dt>Size</dt>
                    <dd>{formatBytes(m.weights_bytes)}</dd>
                  </div>
                  <div>
                    <dt>Params</dt>
                    <dd>{m.parameter_size || "—"}</dd>
                  </div>
                  <div>
                    <dt>Max context</dt>
                    <dd>{m.max_context?.toLocaleString() ?? "—"}</dd>
                  </div>
                  <div>
                    <dt>Suggested</dt>
                    <dd>{m.recommended_context?.toLocaleString() ?? "—"}</dd>
                  </div>
                </dl>
                <div className="caps">
                  {capabilityList(m.capabilities).map((c) => (
                    <span key={c} className="cap">
                      {c}
                    </span>
                  ))}
                </div>
              </article>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
