import type { Metrics, Outcome, ProposedCall, Recovery } from "./agent";

/// One line in a conversation. Tool calls are entries, not annotations on a
/// message, because the user needs to scan and expand them independently.
export type Entry =
  | { kind: "user"; text: string; at?: number }
  | { kind: "assistant"; text: string }
  /// Reasoning from a thinking-capable model. Kept separate from the
  /// answer so it can be folded away — it is usually much longer than
  /// the reply and reading it is opt-in.
  | { kind: "thinking"; text: string }
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
export function formatAgo(at?: number, locale?: string): string {
  if (!at) return "";
  const rtf = new Intl.RelativeTimeFormat(locale ?? "en", { numeric: "auto" });
  const seconds = Math.round((at - Date.now()) / 1000);
  if (Math.abs(seconds) < 60) return rtf.format(seconds, "second");
  const minutes = Math.round(seconds / 60);
  if (Math.abs(minutes) < 60) return rtf.format(minutes, "minute");
  const hours = Math.round(minutes / 60);
  if (Math.abs(hours) < 24) return rtf.format(hours, "hour");
  return rtf.format(Math.round(hours / 24), "day");
}

export function formatTokens(n: number): string {
  if (n < 1000) return String(n);
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)}k`;
  return `${(n / 1_000_000).toFixed(2)}M`;
}

/// Group sessions the way a person thinks about them.
///
/// Returns translation keys rather than words, so the sidebar reads in the
/// user's language.
export function groupByAge(sessions: Session[]): [string, Session[]][] {
  const day = 24 * 60 * 60 * 1000;
  const now = Date.now();
  const buckets: Record<string, Session[]> = { today: [], yesterday: [], earlier: [] };

  for (const s of sessions) {
    const age = now - s.createdAt;
    if (age < day) buckets.today.push(s);
    else if (age < 2 * day) buckets.yesterday.push(s);
    else buckets.earlier.push(s);
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

/// One earlier exchange as the engine wants it.
export type Turn = { role: "user" | "assistant"; text: string };

/// Turns replayed at most. Beyond this the oldest are dropped here rather than
/// shipped and then trimmed again by the engine.
const MAX_HISTORY_TURNS = 16;
/// Characters of history replayed at most, counted from the newest backwards.
const MAX_HISTORY_CHARS = 8000;
/// A single turn longer than this is shortened from the middle — the start says
/// what it was about and the end says how it finished.
const MAX_TURN_CHARS = 1500;

function clamp(text: string): string {
  if (text.length <= MAX_TURN_CHARS) return text;
  const half = Math.floor(MAX_TURN_CHARS / 2);
  return `${text.slice(0, half)}
…
${text.slice(-half)}`;
}

/// Flatten a transcript into the exchanges worth replaying to the model.
///
/// Not a copy of the transcript. Reasoning is left out — replaying
/// chain-of-thought burns window and confuses the templates that emit it. Tool
/// output is left out too, but the *fact* of each call is folded into the
/// assistant's turn as one summary line, because "önceki yaptığı işi
/// hatırlamıyor" is about the work, not just the words: the model has to know
/// it already wrote `test.txt` without carrying the four thousand characters a
/// `read_file` returned.
export function historyFrom(entries: Entry[]): Turn[] {
  const turns: Turn[] = [];

  const intoAssistant = (text: string) => {
    const last = turns[turns.length - 1];
    if (last?.role === "assistant") last.text = `${last.text}
${text}`.trim();
    else turns.push({ role: "assistant", text });
  };

  for (const entry of entries) {
    switch (entry.kind) {
      case "user":
        turns.push({ role: "user", text: entry.text });
        break;
      case "assistant":
        if (entry.text.trim()) intoAssistant(entry.text.trim());
        break;
      case "tool":
        // Rejected and failed calls matter as much as successful ones: without
        // them the model retries what the user already refused.
        if (entry.settled === "user_rejected") {
          intoAssistant(`[tried ${summarize(entry.call, entry.output)} — you declined]`);
        } else if (entry.settled === "failed") {
          intoAssistant(`[tried ${summarize(entry.call, entry.output)} — it failed]`);
        } else {
          intoAssistant(`[${summarize(entry.call, entry.output)}]`);
        }
        break;
      default:
        // Reasoning and app notices are ours, not the conversation's.
        break;
    }
  }

  const kept: Turn[] = [];
  let budget = MAX_HISTORY_CHARS;
  for (let i = turns.length - 1; i >= 0 && kept.length < MAX_HISTORY_TURNS; i--) {
    const text = clamp(turns[i].text);
    if (!text) continue;
    if (budget - text.length < 0 && kept.length > 0) break;
    budget -= text.length;
    kept.unshift({ role: turns[i].role, text });
  }
  // Starting on the assistant's half of an exchange reads as the model talking
  // to itself; drop it so history begins with something the user said.
  while (kept.length > 0 && kept[0].role === "assistant") kept.shift();
  return kept;
}

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
