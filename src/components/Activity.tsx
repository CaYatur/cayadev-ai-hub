import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, type Checkpoint } from "../lib/agent";
import type { Entry, Session } from "../lib/sessions";
import { summarize } from "../lib/sessions";

/**
 * What the agent did, and how to take it back.
 *
 * Autonomous mode trades confirmation prompts for this panel: nothing stops to
 * ask, but every decision is listed with the reason the engine gave, and every
 * file change can be reverted. A mode that only removed the prompts would be
 * Unrestricted under a friendlier name.
 */
export default function Activity({
  session,
  onRestored,
}: {
  session: Session;
  onRestored: (paths: string[]) => void;
}) {
  const { t } = useTranslation();
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [sizeBytes, setSizeBytes] = useState<number | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setCheckpoints(await api.checkpointList());
      setSizeBytes(await api.checkpointSize());
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    load();
  }, [load, session.entries.length]);

  async function restore(id: string) {
    setBusy(id);
    try {
      const paths = await api.checkpointRestore(id);
      onRestored(paths);
      await load();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(null);
    }
  }

  // Every settled tool call in this session, newest first.
  const decisions = session.entries
    .map((e, i) => ({ entry: e, i }))
    .filter((x): x is { entry: Extract<Entry, { kind: "tool" }>; i: number } =>
      x.entry.kind === "tool")
    .reverse();

  return (
    <>
      <section className="intro">
        <h1>{t("activity.title")}</h1>
        <p>{t("activity.intro")}</p>
      </section>

      {error && <div className="state state-error">{error}</div>}

      <section className="routing">
        <h2>
          {t("activity.undo")}
          {sizeBytes !== null && (
            <span className="activity-size">
              {t("activity.storeSize", { size: formatBytes(sizeBytes) })}
            </span>
          )}
        </h2>

        {checkpoints.length === 0 ? (
          <p className="hint">{t("activity.noCheckpoints")}</p>
        ) : (
          <div className="ckpt-list">
            {checkpoints.map((c) => (
              <div key={c.id} className="ckpt">
                <div className="ckpt-main">
                  <span className="ckpt-label">{c.label}</span>
                  <span className="ckpt-files">
                    {c.files.map((f) => f.path).join(", ") || t("activity.noFiles")}
                  </span>
                </div>
                <span className="ckpt-time">{new Date(c.timestamp).toLocaleTimeString()}</span>
                <button
                  className="btn"
                  disabled={busy === c.id || c.files.length === 0}
                  onClick={() => restore(c.id)}
                >
                  {busy === c.id ? t("activity.restoring") : t("activity.restore")}
                </button>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="routing">
        <h2>{t("activity.decisions")}</h2>
        {decisions.length === 0 ? (
          <p className="hint">{t("activity.noDecisions")}</p>
        ) : (
          <div className="decision-list">
            {decisions.map(({ entry, i }) => (
              <div key={i} className={`decision decision-${entry.settled ?? "waiting"}`}>
                <span className="decision-outcome">
                  {t(`outcome.${entry.settled ?? "waiting"}`)}
                </span>
                <span className="decision-summary">{summarize(entry.call, entry.output)}</span>
                <span className="decision-reason">{entry.call.evaluation.reason}</span>
              </div>
            ))}
          </div>
        )}
      </section>
    </>
  );
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 ** 2).toFixed(1)} MB`;
}
