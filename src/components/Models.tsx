import { useTranslation } from "react-i18next";
import { formatBytes, type ModelProfile, type Placement } from "../lib/agent";

const PLACEMENTS: Placement[] = ["resident", "marginal", "spilled", "cloud", "unknown"];

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
  const { t } = useTranslation();

  // A placement the backend adds but the catalogue has not caught up with
  // should still render, so fall back to the raw value rather than blank.
  const label = (p: Placement) =>
    PLACEMENTS.includes(p) ? t(`modelsView.${p}`) : p;
  const hint = (p: Placement) =>
    PLACEMENTS.includes(p) ? t(`modelsView.${p}_hint`) : "";

  return (
    <>
      <section className="intro">
        <h1>{t("modelsView.title")}</h1>
        <p>{t("modelsView.intro")}</p>
        <button className="btn" onClick={onRefreshVram}>
          {t("modelsView.refresh")}
        </button>
      </section>

      {routing && (
        <section className="routing">
          <h2>{t("modelsView.routing")}</h2>
          <div className="routing-rows">
            {Object.entries(routing).map(([task, model]) => (
              <div key={task} className="routing-row">
                <span className="routing-task">{t(`tasks.${task}`, task)}</span>
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
              <span className={`chip chip-${m.placement}`}>{label(m.placement)}</span>
            </div>
            <p className="hint">{hint(m.placement)}</p>
            <dl className="facts">
              <div>
                <dt>{t("modelsView.size")}</dt>
                <dd>{formatBytes(m.weights_bytes)}</dd>
              </div>
              <div>
                <dt>{t("modelsView.params")}</dt>
                <dd>{m.parameter_size || "—"}</dd>
              </div>
              <div>
                <dt>{t("modelsView.maxContext")}</dt>
                <dd>{m.max_context?.toLocaleString() ?? "—"}</dd>
              </div>
              <div>
                <dt>{t("modelsView.suggested")}</dt>
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
