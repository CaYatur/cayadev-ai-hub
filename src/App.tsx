import { useEffect, useState } from "react";
import "./styles/app.css";

/** A model as reported by /api/tags, enriched with what /api/show tells us. */
type Model = {
  name: string;
  sizeBytes: number;
  params: string;
  quant: string;
  capabilities: string[];
  maxContext: number | null;
  placement: Placement;
};

type Placement = "resident" | "marginal" | "spilled" | "cloud" | "unknown";

const HOST = "http://localhost:11434";

/** Mirrors `estimate_placement` in src-tauri/src/ollama/registry.rs. */
const OVERHEAD = 600 * 1024 * 1024;

function estimatePlacement(weights: number, availableVram: number): Placement {
  if (weights === 0) return "cloud";
  const withOverhead = weights + OVERHEAD;
  if (withOverhead >= availableVram) return "spilled";
  if (withOverhead > availableVram * 0.85) return "marginal";
  return "resident";
}

const PLACEMENT_COPY: Record<Placement, { label: string; hint: string }> = {
  resident: { label: "Fits in VRAM", hint: "Full speed — expect tens of tokens/sec." },
  marginal: { label: "Tight fit", hint: "Fits only when little else uses the GPU." },
  spilled: { label: "Spills to RAM", hint: "Memory-bandwidth bound. Expect single-digit tokens/sec." },
  cloud: { label: "Cloud", hint: "Runs on Ollama's servers; local memory does not apply." },
  unknown: { label: "Unknown", hint: "Not enough metadata to estimate." },
};

function formatBytes(n: number): string {
  if (n === 0) return "—";
  const gb = n / 1024 ** 3;
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(n / 1024 ** 2).toFixed(0)} MB`;
}

export default function App() {
  const [models, setModels] = useState<Model[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // Not probed from the GPU yet — the Rust side will supply the real budget.
  const availableVram = 6 * 1024 ** 3;

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const tags = await fetch(`${HOST}/api/tags`).then((r) => r.json());
        const entries = await Promise.all(
          (tags.models ?? []).map(async (m: any) => {
            let show: any = {};
            try {
              show = await fetch(`${HOST}/api/show`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model: m.name }),
              }).then((r) => r.json());
            } catch {
              // A model whose metadata will not load still deserves a row.
            }
            const arch = show?.model_info?.["general.architecture"];
            return {
              name: m.name,
              sizeBytes: m.size ?? 0,
              params: m.details?.parameter_size ?? "—",
              quant: m.details?.quantization_level ?? "—",
              capabilities: show?.capabilities ?? [],
              maxContext: arch ? show?.model_info?.[`${arch}.context_length`] ?? null : null,
              placement: estimatePlacement(m.size ?? 0, availableVram),
            } as Model;
          }),
        );
        if (cancelled) return;
        // Fastest-first: the routing story is easier to read this way.
        entries.sort((a, b) => a.sizeBytes - b.sizeBytes);
        setModels(entries);
      } catch (e) {
        if (!cancelled) setError(String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [availableVram]);

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          CaYa<span>Dev</span> AI Hub
        </div>
        <div className="topbar-meta">Ollama · localhost:11434</div>
      </header>

      <main className="content">
        <section className="intro">
          <h1>Installed models</h1>
          <p>
            Placement is estimated against available VRAM. A model that spills to
            system RAM is bound by memory bandwidth — routing work to a resident
            model is worth roughly 10x on this hardware.
          </p>
        </section>

        {loading && <div className="state">Reading model metadata…</div>}

        {error && (
          <div className="state state-error">
            Could not reach Ollama at {HOST}.
            <br />
            <span className="muted">{error}</span>
          </div>
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
                    <dd>{formatBytes(m.sizeBytes)}</dd>
                  </div>
                  <div>
                    <dt>Params</dt>
                    <dd>{m.params}</dd>
                  </div>
                  <div>
                    <dt>Quant</dt>
                    <dd>{m.quant}</dd>
                  </div>
                  <div>
                    <dt>Max context</dt>
                    <dd>{m.maxContext ? m.maxContext.toLocaleString() : "—"}</dd>
                  </div>
                </dl>
                {m.capabilities.length > 0 && (
                  <div className="caps">
                    {m.capabilities.map((c) => (
                      <span key={c} className="cap">
                        {c}
                      </span>
                    ))}
                  </div>
                )}
              </article>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
