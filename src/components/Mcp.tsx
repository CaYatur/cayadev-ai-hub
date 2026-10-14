import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { documentDir, homeDir } from "@tauri-apps/api/path";
import {
  blankServer,
  formatPairs,
  joinArgs,
  mcpApi,
  parsePairs,
  splitArgs,
  type McpConfig,
  type McpServer,
  type McpStatus,
  type McpTool,
} from "../lib/mcp";

/** Ready-made servers, so a first test does not start from a blank form. */
const PRESETS: { key: string; server: () => Promise<McpServer> }[] = [
  {
    key: "everything",
    server: async () => ({
      ...blankServer(),
      name: "Everything",
      transport: {
        kind: "stdio",
        command: "npx",
        args: ["-y", "@modelcontextprotocol/server-everything", "stdio"],
        env: {},
      },
    }),
  },
  {
    key: "filesystem",
    server: async () => ({
      ...blankServer(),
      name: "Filesystem",
      transport: {
        kind: "stdio",
        command: "npx",
        // The user's Documents folder, resolved on this machine. It is only a
        // starting point: the argument stays editable before saving.
        args: ["-y", "@modelcontextprotocol/server-filesystem", await userFolder()],
        env: {},
      },
    }),
  },
];

async function userFolder(): Promise<string> {
  const trim = (p: string) => p.replace(/[\\/]+$/, "");
  try {
    return trim(await documentDir());
  } catch {
    try {
      return trim(await homeDir());
    } catch {
      return ".";
    }
  }
}

type Draft = {
  /** Index in the saved list, or `null` for a new server. */
  index: number | null;
  server: McpServer;
  args: string;
  env: string;
  headers: string;
};

function toDraft(server: McpServer, index: number | null): Draft {
  const t = server.transport;
  return {
    index,
    server,
    args: t.kind === "stdio" ? joinArgs(t.args) : "",
    env: t.kind === "stdio" ? formatPairs(t.env, "=") : "",
    headers: t.kind === "http" ? formatPairs(t.headers, ":") : "",
  };
}

function fromDraft(d: Draft): McpServer {
  const s = d.server;
  if (s.transport.kind === "stdio") {
    return {
      ...s,
      transport: {
        kind: "stdio",
        command: s.transport.command.trim(),
        args: splitArgs(d.args),
        env: parsePairs(d.env, "="),
        cwd: s.transport.cwd?.trim() || null,
      },
    };
  }
  return {
    ...s,
    transport: {
      kind: "http",
      url: s.transport.url.trim(),
      headers: parsePairs(d.headers, ":"),
      bearer: s.transport.bearer?.trim() || null,
    },
  };
}

export default function Mcp() {
  const { t } = useTranslation();
  const [config, setConfig] = useState<McpConfig>({ servers: [] });
  const [status, setStatus] = useState<McpStatus[]>([]);
  const [draft, setDraft] = useState<Draft | null>(null);
  /** Tools found by the last test, per server id (or "draft"). */
  const [tools, setTools] = useState<Record<string, McpTool[]>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setConfig(await mcpApi.getConfig());
      setStatus(await mcpApi.status());
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const persist = async (servers: McpServer[]) => {
    setError(null);
    try {
      const saved = await mcpApi.saveConfig({ servers });
      setConfig(saved);
      setStatus(await mcpApi.status());
      return saved;
    } catch (e) {
      setError(String(e));
      return null;
    }
  };

  const test = async (server: McpServer, key: string) => {
    setBusy(key);
    setError(null);
    setNotice(null);
    try {
      const found = await mcpApi.testServer(server);
      setTools((prev) => ({ ...prev, [key]: found }));
      setNotice(t("mcpView.testOk", { count: found.length }));
    } catch (e) {
      setTools((prev) => ({ ...prev, [key]: [] }));
      setError(t("mcpView.testFailed", { error: String(e) }));
    } finally {
      setBusy(null);
    }
  };

  const saveDraft = async () => {
    if (!draft) return;
    const server = fromDraft(draft);
    const servers = [...config.servers];
    if (draft.index === null) servers.push(server);
    else servers[draft.index] = server;
    const saved = await persist(servers);
    if (saved) {
      setDraft(null);
      setNotice(t("mcpView.saved"));
    }
  };

  const update = (index: number, change: Partial<McpServer>) => {
    const servers = config.servers.map((s, i) => (i === index ? { ...s, ...change } : s));
    persist(servers);
  };

  const remove = (index: number) => {
    const s = config.servers[index];
    if (!window.confirm(t("mcpView.confirmDelete", { name: s.name }))) return;
    persist(config.servers.filter((_, i) => i !== index));
  };

  const toggleTool = (index: number, tool: string, on: boolean) => {
    const s = config.servers[index];
    const disabled = on
      ? s.disabled_tools.filter((x) => x !== tool)
      : [...new Set([...s.disabled_tools, tool])];
    update(index, { disabled_tools: disabled });
  };

  const setServer = (change: Partial<McpServer>) =>
    draft && setDraft({ ...draft, server: { ...draft.server, ...change } });

  return (
    <>
      <section className="intro">
        <h1>{t("mcpView.title")}</h1>
        <p>{t("mcpView.intro")}</p>
        <div className="mcp-actions">
          <button className="btn btn-primary" onClick={() => setDraft(toDraft(blankServer(), null))}>
            {t("mcpView.add")}
          </button>
          {PRESETS.map((p) => (
            <button key={p.key} className="btn" onClick={async () => setDraft(toDraft(await p.server(), null))}>
              {t(`mcpView.preset.${p.key}`)}
            </button>
          ))}
        </div>
      </section>

      {error && <div className="mcp-msg mcp-msg-error">{error}</div>}
      {notice && !error && <div className="mcp-msg">{notice}</div>}

      {draft && (
        <section className="mcp-editor card">
          <h2>{draft.index === null ? t("mcpView.newServer") : t("mcpView.editServer")}</h2>
          <div className="mcp-form">
            <label className="field">
              {t("mcpView.name")}
              <input
                className="mcp-input"
                value={draft.server.name}
                placeholder="Filesystem"
                onChange={(e) => setServer({ name: e.target.value })}
              />
            </label>
            <label className="field">
              {t("mcpView.transport")}
              <select
                className="select"
                value={draft.server.transport.kind}
                onChange={(e) =>
                  setServer({
                    transport:
                      e.target.value === "http"
                        ? { kind: "http", url: "http://127.0.0.1:3001/mcp", headers: {} }
                        : { kind: "stdio", command: "npx", args: [], env: {} },
                  })
                }
              >
                <option value="stdio">{t("mcpView.stdio")}</option>
                <option value="http">{t("mcpView.http")}</option>
              </select>
            </label>

            {draft.server.transport.kind === "stdio" ? (
              <>
                <label className="field">
                  {t("mcpView.command")}
                  <input
                    className="mcp-input mono"
                    value={draft.server.transport.command}
                    placeholder="npx"
                    onChange={(e) =>
                      draft.server.transport.kind === "stdio" &&
                      setServer({ transport: { ...draft.server.transport, command: e.target.value } })
                    }
                  />
                </label>
                <label className="field">
                  {t("mcpView.args")}
                  <input
                    className="mcp-input mono"
                    value={draft.args}
                    placeholder="-y @modelcontextprotocol/server-filesystem C:\Projects"
                    onChange={(e) => setDraft({ ...draft, args: e.target.value })}
                  />
                </label>
                <label className="field">
                  {t("mcpView.env")}
                  <textarea
                    className="mcp-input mono"
                    rows={3}
                    value={draft.env}
                    placeholder="API_KEY=..."
                    onChange={(e) => setDraft({ ...draft, env: e.target.value })}
                  />
                </label>
                <label className="field">
                  {t("mcpView.cwd")}
                  <input
                    className="mcp-input mono"
                    value={draft.server.transport.cwd ?? ""}
                    onChange={(e) =>
                      draft.server.transport.kind === "stdio" &&
                      setServer({ transport: { ...draft.server.transport, cwd: e.target.value } })
                    }
                  />
                </label>
              </>
            ) : (
              <>
                <label className="field">
                  {t("mcpView.url")}
                  <input
                    className="mcp-input mono"
                    value={draft.server.transport.url}
                    onChange={(e) =>
                      draft.server.transport.kind === "http" &&
                      setServer({ transport: { ...draft.server.transport, url: e.target.value } })
                    }
                  />
                </label>
                <label className="field">
                  {t("mcpView.bearer")}
                  <input
                    className="mcp-input mono"
                    type="password"
                    value={draft.server.transport.bearer ?? ""}
                    onChange={(e) =>
                      draft.server.transport.kind === "http" &&
                      setServer({ transport: { ...draft.server.transport, bearer: e.target.value } })
                    }
                  />
                </label>
                <label className="field">
                  {t("mcpView.headers")}
                  <textarea
                    className="mcp-input mono"
                    rows={3}
                    value={draft.headers}
                    placeholder="X-Api-Key: ..."
                    onChange={(e) => setDraft({ ...draft, headers: e.target.value })}
                  />
                </label>
              </>
            )}

            <label className="field">
              {t("mcpView.timeout")}
              <input
                className="mcp-input"
                type="number"
                min={5}
                max={600}
                value={draft.server.timeout_secs}
                onChange={(e) => setServer({ timeout_secs: Number(e.target.value) || 60 })}
              />
            </label>
            <label className="field field-inline">
              <input
                type="checkbox"
                checked={draft.server.auto_approve}
                onChange={(e) => setServer({ auto_approve: e.target.checked })}
              />
              {t("mcpView.autoApprove")}
            </label>
          </div>
          <p className="hint">{t("mcpView.transportHint")}</p>

          <div className="mcp-actions">
            <button
              className="btn"
              disabled={busy !== null}
              onClick={() => test(fromDraft(draft), "draft")}
            >
              {busy === "draft" ? t("mcpView.testing") : t("mcpView.test")}
            </button>
            <button className="btn btn-primary" onClick={saveDraft}>
              {t("mcpView.save")}
            </button>
            <button className="btn" onClick={() => setDraft(null)}>
              {t("mcpView.cancel")}
            </button>
          </div>
          {tools.draft && <ToolList tools={tools.draft} />}
        </section>
      )}

      {config.servers.length === 0 && !draft && <p className="hint">{t("mcpView.empty")}</p>}

      <div className="mcp-list">
        {config.servers.map((s, i) => {
          const st = status.find((x) => x.id === s.id);
          const found = tools[s.id];
          return (
            <article key={s.id} className="card mcp-card">
              <div className="card-head">
                <h2>{s.name}</h2>
                <span className={st?.connected ? "chip chip-resident" : st?.error ? "chip chip-spilled" : "chip chip-unknown"}>
                  {st?.connected
                    ? t("mcpView.connected", { count: st.tool_count })
                    : st?.error
                      ? t("mcpView.failed")
                      : t("mcpView.idle")}
                </span>
              </div>
              <p className="hint mono">
                {s.transport.kind === "stdio"
                  ? `${s.transport.command} ${joinArgs(s.transport.args)}`
                  : s.transport.url}
              </p>
              {st?.error && <p className="mcp-msg mcp-msg-error">{st.error}</p>}
              <div className="mcp-actions">
                <label className="field field-inline">
                  <input
                    type="checkbox"
                    checked={s.enabled}
                    onChange={(e) => update(i, { enabled: e.target.checked })}
                  />
                  {t("mcpView.enabled")}
                </label>
                <label className="field field-inline">
                  <input
                    type="checkbox"
                    checked={s.auto_approve}
                    onChange={(e) => update(i, { auto_approve: e.target.checked })}
                  />
                  {t("mcpView.autoApprove")}
                </label>
              </div>
              <div className="mcp-actions">
                <button className="btn" disabled={busy !== null} onClick={() => test(s, s.id)}>
                  {busy === s.id ? t("mcpView.testing") : t("mcpView.test")}
                </button>
                <button className="btn" onClick={() => setDraft(toDraft(s, i))}>
                  {t("mcpView.edit")}
                </button>
                <button className="btn" onClick={() => remove(i)}>
                  {t("mcpView.delete")}
                </button>
              </div>
              {found && (
                <ToolList
                  tools={found}
                  isOn={(name) => !s.disabled_tools.includes(name)}
                  onToggle={(name, on) => toggleTool(i, name, on)}
                />
              )}
            </article>
          );
        })}
      </div>
    </>
  );
}

function ToolList({
  tools,
  isOn,
  onToggle,
}: {
  tools: McpTool[];
  isOn?: (name: string) => boolean;
  onToggle?: (name: string, on: boolean) => void;
}) {
  const { t } = useTranslation();
  if (tools.length === 0) return <p className="hint">{t("mcpView.noTools")}</p>;
  return (
    <div className="mcp-tools">
      <div className="session-group-label">
        {onToggle ? t("mcpView.toolsForAgent") : t("mcpView.toolsFound", { count: tools.length })}
      </div>
      {tools.map((tool) => (
        <label key={tool.name} className="mcp-tool" title={tool.description}>
          {onToggle && (
            <input
              type="checkbox"
              checked={isOn ? isOn(tool.name) : true}
              onChange={(e) => onToggle(tool.name, e.target.checked)}
            />
          )}
          <span className="mono">{tool.name}</span>
          {tool.read_only && <span className="cap">{t("mcpView.readOnly")}</span>}
          <span className="mcp-tool-desc">{tool.description}</span>
        </label>
      ))}
    </div>
  );
}
