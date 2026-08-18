import { formatBytes, type ModelProfile, type Placement } from "../lib/agent";

const PLACEMENT_COPY: Record<Placement, { label: string; hint: string }> = {
  resident: { label: "Fits in VRAM", hint: "Full speed — tens of tokens/sec." },
  marginal: { label: "Tight fit", hint: "Fits only while little else uses the GPU." },
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

function capabilityList(c: ModelProfile["capabilities"]): string[] {
  return Object.entries(c)
    .filter(([, on]) => on)
    .map(([name]) => name);
}

export default function Models({
  models,
  routing,
  onRefreshVram,
}: {
  models: ModelProfile[];
  routing: Record<string, string> | null;
  onRefreshVram: () => void;
}) {
  return (
    <>
      <section className="intro">
        <h1>Installed models</h1>
        <p>
          Placement is read from the GPU driver, not assumed. A model that spills
          to system RAM is bound by memory bandwidth — routing work to one that
          fits is worth roughly 8x on this hardware.
        </p>
        <button className="btn" onClick={onRefreshVram}>
          Re-read GPU memory
        </button>
      </section>

      {routing && (
        <section className="routing">
          <h2>Routing</h2>
          <div className="routing-rows">
            {Object.entries(routing).map(([task, model]) => (
              <div key={task} className="routing-row">
                <span className="routing-task">{TASK_LABELS[task] ?? task}</span>
                <span className="routing-model">{model}</span>
              </div>
            ))}
          </div>
        </section>
      )}

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
    </>
  );
}
