import { useTranslation } from "react-i18next";
import { LANGUAGES, setLanguage, type LanguageCode } from "../i18n";
import { groupByAge, type Session } from "../lib/sessions";

export default function Sidebar({
  sessions,
  activeId,
  onSelect,
  onNew,
  onDelete,
  version,
}: {
  sessions: Session[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  version: string | null;
}) {
  const { t, i18n } = useTranslation();
  const groups = groupByAge(sessions);

  return (
    <aside className="sidebar">
      <div className="sidebar-top">
        <div className="brand">
          CaYa<span>Dev</span> {t("app.brandSuffix")}
        </div>
      </div>

      <button className="new-session" onClick={onNew}>
        <span className="plus">+</span> {t("app.newSession")}
      </button>

      <nav className="session-list">
        {sessions.length === 0 && <p className="sidebar-empty">{t("app.noSessions")}</p>}
        {groups.map(([key, list]) => (
          <div key={key} className="session-group">
            <div className="session-group-label">{t(`app.${key}`)}</div>
            {list.map((s) => (
              <div
                key={s.id}
                className={s.id === activeId ? "session-item session-active" : "session-item"}
                onClick={() => onSelect(s.id)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => e.key === "Enter" && onSelect(s.id)}
              >
                <span className="session-dot" />
                <span className="session-title" title={s.title}>
                  {s.title}
                </span>
                <button
                  className="session-delete"
                  title={t("app.deleteSession")}
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(s.id);
                  }}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        ))}
      </nav>

      <div className="sidebar-foot">
        <span className={version ? "dot dot-ok" : "dot dot-off"} />
        <span className="foot-text">
          {version ? t("app.online", { version }) : t("app.offline")}
        </span>
        <select
          className="lang-select"
          value={i18n.language}
          onChange={(e) => setLanguage(e.target.value as LanguageCode)}
          title={t("language.label")}
        >
          {LANGUAGES.map((l) => (
            <option key={l.code} value={l.code}>
              {l.label}
            </option>
          ))}
        </select>
      </div>
    </aside>
  );
}
