import type { Metrics, Outcome, ProposedCall, Recovery } from "./agent";

/// One line in a conversation. Tool calls are entries, not annotations on a
/// message, because the user needs to scan and expand them independently.
export type Entry =
  | { kind: "user"; text: string; at?: number }
  | { kind: "assistant"; text: string }
  | {
      kind: "tool";
      call: ProposedCall;
      output?: string;
      settled?: Outcome;
    }
  | { kind: "notice"; text: string; tone: "info" | "error" };

export type Session = {
  id: string;
  title: string;
  workspace: string | null;
  entries: Entry[];
  createdAt: number;
  /// Running totals shown in the status line.
  tokens: number;
  elapsedMs: number;
  lastRate: number | null;
  /// Tokens occupying the model's window right now, from the last step's
  /// prompt evaluation. Distinct from `tokens`, which is cumulative usage.
  contextUsed: number;
};

const STORAGE_KEY = "cayahub.sessions.v1";
/// Beyond this the sidebar stops being scannable and storage starts to matter.
const MAX_SESSIONS = 50;

export function newSession(workspace: string | null = null): Session {
  return {
    id: crypto.randomUUID(),
    title: "New session",
    workspace,
    entries: [],
    createdAt: Date.now(),
    tokens: 0,
    elapsedMs: 0,
    lastRate: null,
    contextUsed: 0,
  };
}

/// Name a session from its first instruction, the way the user would refer to it.
export function titleFrom(text: string): string {
  const firstLine = text.trim().split("\n")[0].trim();
  if (!firstLine) return "New session";
  return firstLine.length > 48 ? `${firstLine.slice(0, 48)}…` : firstLine;
}

export function load(): Session[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    // Corrupt storage should cost the history, not the app.
    return [];
  }
}

export function save(sessions: Session[]): void {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(sessions.slice(0, MAX_SESSIONS)),
    );
  } catch {
    // Quota exceeded — the transcript is not worth interrupting a run for.
  }
}

/// Human-readable elapsed time, in the compact form a status line wants.
export function formatElapsed(ms: number): string {
  const total = Math.floor(ms / 1000);
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  if (minutes < 60) return `${minutes}m ${seconds}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

/// Relative time in the short form a message footer wants.
export function formatAgo(at?: number): string {
  if (!at) return "";
  const seconds = Math.floor((Date.now() - at) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

export function formatTokens(n: number): string {
  if (n < 1000) return String(n);
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)}k`;
  return `${(n / 1_000_000).toFixed(2)}M`;
}

/// Group sessions the way a person thinks about them.
export function groupByAge(sessions: Session[]): [string, Session[]][] {
  const day = 24 * 60 * 60 * 1000;
  const now = Date.now();
  const buckets: Record<string, Session[]> = { Today: [], Yesterday: [], Earlier: [] };

  for (const s of sessions) {
    const age = now - s.createdAt;
    if (age < day) buckets.Today.push(s);
    else if (age < 2 * day) buckets.Yesterday.push(s);
    else buckets.Earlier.push(s);
  }
  return Object.entries(buckets).filter(([, list]) => list.length > 0);
}

/// One-line summary for a collapsed tool row.
export function summarize(call: ProposedCall, output?: string): string {
  const args = call.arguments as Record<string, string | undefined>;
  switch (call.name) {
    case "read_file":
      return `Read ${args.path ?? "a file"}`;
    case "write_file": {
      const lines = (args.content ?? "").split("\n").length;
      return `Wrote ${args.path ?? "a file"} · +${lines}`;
    }
    case "edit_file":
      return `Edited ${args.path ?? "a file"}`;
    case "list_dir":
      return `Listed ${args.path ?? "a directory"}`;
    case "run_command": {
      const failed = output?.startsWith("exit ") && !output.startsWith("exit 0");
      return `${failed ? "Failed" : "Ran"}: ${args.command ?? ""}`;
    }
    default:
      return call.name;
  }
}

/// Shown only when the parser had to rescue a malformed call — that is a fact
/// about the model worth surfacing, not an implementation detail.
export const RECOVERY_NOTE: Partial<Record<Recovery, string>> = {
  xml_tag: "rescued from <tool_call> tags",
  code_fence: "rescued from a code block",
  loose_json: "rescued from bare JSON",
  function_tag: "rescued from a <function=> tag",
  python_call: "rescued from Python-style syntax",
};

export function applyMetrics(session: Session, metrics: Metrics): Session {
  return {
    ...session,
    tokens: session.tokens + metrics.prompt_tokens + metrics.output_tokens,
    elapsedMs: session.elapsedMs + metrics.total_seconds * 1000,
    lastRate: metrics.decode_tps,
    // What the window actually holds after this step. Ollama reports the full
    // prompt length even when it reuses the KV prefix, which is exactly the
    // number an occupancy meter wants.
    contextUsed: metrics.prompt_tokens + metrics.output_tokens,
  };
}
