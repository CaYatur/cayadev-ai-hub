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
  const groups = groupByAge(sessions);

  return (
    <aside className="sidebar">
      <div className="sidebar-top">
        <div className="brand">
          CaYa<span>Dev</span> AI Hub
        </div>
      </div>

      <button className="new-session" onClick={onNew}>
        <span className="plus">+</span> New session
      </button>

      <nav className="session-list">
        {sessions.length === 0 && (
          <p className="sidebar-empty">No sessions yet.</p>
        )}
        {groups.map(([label, list]) => (
          <div key={label} className="session-group">
            <div className="session-group-label">{label}</div>
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
                  title="Delete session"
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
        {version ? `Ollama ${version}` : "Ollama offline"}
      </div>
    </aside>
  );
}
