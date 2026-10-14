import { invoke } from "@tauri-apps/api/core";

/** How the app reaches an MCP server. Mirrors `mcp::config::Transport`. */
export type McpTransport =
  | {
      kind: "stdio";
      command: string;
      args: string[];
      env: Record<string, string>;
      cwd?: string | null;
    }
  | {
      kind: "http";
      url: string;
      headers: Record<string, string>;
      bearer?: string | null;
    };

export type McpServer = {
  /** Slug used in tool names. Filled in by the backend when empty. */
  id: string;
  name: string;
  /** Whether the agent may use this server. */
  enabled: boolean;
  transport: McpTransport;
  /** Tools switched off for the agent, by their original name. */
  disabled_tools: string[];
  /** Run this server's tools without asking. */
  auto_approve: boolean;
  timeout_secs: number;
};

export type McpConfig = { servers: McpServer[] };

export type McpTool = {
  server_id: string;
  server_name: string;
  name: string;
  /** The name the model sees: `mcp__<server>__<tool>`. */
  exposed: string;
  description: string;
  input_schema: Record<string, unknown>;
  read_only: boolean;
  enabled: boolean;
};

export type McpStatus = {
  id: string;
  connected: boolean;
  tool_count: number;
  error: string | null;
};

export const mcpApi = {
  getConfig: () => invoke<McpConfig>("mcp_get_config"),
  saveConfig: (config: McpConfig) => invoke<McpConfig>("mcp_save_config", { config }),
  testServer: (server: McpServer) => invoke<McpTool[]>("mcp_test_server", { server }),
  status: () => invoke<McpStatus[]>("mcp_status"),
};

export function blankServer(): McpServer {
  return {
    id: "",
    name: "",
    enabled: true,
    transport: { kind: "stdio", command: "npx", args: [], env: {} },
    disabled_tools: [],
    auto_approve: false,
    timeout_secs: 60,
  };
}

/** `KEY=value` (or `Key: value`) lines to a map; blank lines are skipped. */
export function parsePairs(text: string, sep: "=" | ":"): Record<string, string> {
  const out: Record<string, string> = {};
  for (const line of text.split(/\r?\n/)) {
    const i = line.indexOf(sep);
    if (i <= 0) continue;
    const key = line.slice(0, i).trim();
    if (key) out[key] = line.slice(i + 1).trim();
  }
  return out;
}

export function formatPairs(map: Record<string, string>, sep: "=" | ":"): string {
  return Object.entries(map)
    .map(([k, v]) => (sep === "=" ? `${k}=${v}` : `${k}: ${v}`))
    .join("\n");
}

/**
 * Split an argument line the way a shell would, respecting quotes, so
 * `-y @scope/pkg "C:\My Folder"` becomes three arguments.
 */
export function splitArgs(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let quote: string | null = null;
  let has = false;
  for (const c of line) {
    if (quote) {
      if (c === quote) quote = null;
      else cur += c;
    } else if (c === '"' || c === "'") {
      quote = c;
      has = true;
    } else if (/\s/.test(c)) {
      if (cur || has) out.push(cur);
      cur = "";
      has = false;
    } else {
      cur += c;
    }
  }
  if (cur || has) out.push(cur);
  return out;
}

export function joinArgs(args: string[]): string {
  return args.map((a) => (a === "" || /\s/.test(a) ? `"${a}"` : a)).join(" ");
}
