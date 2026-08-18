import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import type { Turn } from "./sessions";

export type Placement = "resident" | "marginal" | "spilled" | "cloud" | "unknown";

export type ModelProfile = {
  name: string;
  parameter_size: string;
  quantization: string;
  weights_bytes: number;
  capabilities: {
    completion: boolean;
    tools: boolean;
    vision: boolean;
    thinking: boolean;
    embedding: boolean;
  };
  max_context: number | null;
  recommended_context: number | null;
  placement: Placement;
};

export type PermissionMode =
  | "read_only"
  | "project_auto"
  | "balanced"
  | "autonomous"
  | "unrestricted";

/** A restorable point, one per tool call that changed files. */
export type Checkpoint = {
  id: string;
  timestamp: string;
  label: string;
  files: { path: string; kind: string; before: string | null }[];
};

/** Maps to real levers: thinking on/off and how many tool steps are allowed. */
export type Effort = "fast" | "balanced" | "thorough";

export type ContextOptions = {
  recommended: number | null;
  trained_max: number | null;
  choices: number[];
};

export type Decision = "allow" | "ask" | "deny";
export type Outcome =
  | "auto_approved"
  | "user_approved"
  | "user_rejected"
  | "denied"
  | "failed";

/** How the parser recovered a tool call. Anything but `native` means rescue. */
export type Recovery =
  | "native"
  | "xml_tag"
  | "code_fence"
  | "loose_json"
  | "function_tag"
  | "python_call";

export type ProposedCall = {
  id: string;
  step: number;
  name: string;
  arguments: Record<string, unknown>;
  recovery: Recovery;
  evaluation: {
    decision: Decision;
    reason: string;
    escapes_workspace: boolean;
    risk: "read_only" | "mutating" | "destructive" | "catastrophic";
  };
};

export type Metrics = {
  decode_tps: number;
  prefill_tps: number;
  prompt_tokens: number;
  output_tokens: number;
  ttft_seconds: number;
  total_seconds: number;
};

export type AgentEvent =
  | { kind: "token"; text: string }
  | { kind: "thinking"; text: string }
  | { kind: "tool_proposed"; call: ProposedCall }
  | { kind: "tool_settled"; id: string; outcome: Outcome; output: string }
  | { kind: "tool_progress"; text: string }
  | { kind: "step_done"; step: number; metrics: Metrics }
  | { kind: "done"; reason: string }
  | { kind: "error"; message: string };

export const api = {
  version: () => invoke<string>("ollama_version"),
  listModels: () => invoke<ModelProfile[]>("list_models"),
  routing: () => invoke<{ assignments: Record<string, string> }>("routing_policy"),
  refreshVram: () => invoke<{ total_bytes: number; reserved_bytes: number }>("refresh_vram_budget"),
  setWorkspace: (path: string) => invoke<string>("set_workspace", { path }),
  setMode: (mode: PermissionMode) => invoke<void>("set_permission_mode", { mode }),
  setAssisted: (enabled: boolean) => invoke<void>("set_assisted_approval", { enabled }),
  resolveApproval: (id: string, approved: boolean) =>
    invoke<boolean>("resolve_approval", { id, approved }),
  /** Answer the agent's question. `null` skips it. */
  answerQuestion: (id: string, answer: string | null) =>
    invoke<boolean>("answer_question", { id, answer }),
  cancel: () => invoke<void>("cancel_run"),
  run: (
    prompt: string,
    model?: string,
    numCtx?: number,
    effort?: Effort,
    history?: Turn[],
    attachments?: string[],
  ) =>
    invoke<string>("run_agent", { prompt, model, numCtx, effort, history, attachments }),
  contextOptions: (model: string) => invoke<ContextOptions>("context_options", { model }),
  checkpointList: () => invoke<Checkpoint[]>("checkpoint_list"),
  checkpointRestore: (id: string) => invoke<string[]>("checkpoint_restore", { id }),
  checkpointSize: () => invoke<number>("checkpoint_size"),
  attachFiles: (paths: string[]) => invoke<Attachment[]>("attach_files", { paths }),
  clearAttachments: () => invoke<void>("clear_attachments"),
  attachBytes: (name: string, data: number[]) =>
    invoke<Attachment>("attach_bytes", { name, data }),
};

/**
 * A question the agent asked, rather than one the permission engine asked.
 *
 * Approval is "may I do this?" about something already decided. This is
 * "which?" or "what?" about something not yet decided — the choice that is the
 * user's to make, asked instead of guessed.
 */
export type Question = {
  id: string;
  question: string;
  /** Choices to offer. Empty means the answer is written, not picked. */
  options: string[];
  multiple: boolean;
};

/** Subscribe to run events, approval questions, and the agent's own questions. */
export async function subscribe(
  onEvent: (e: AgentEvent) => void,
  onApproval: (c: ProposedCall) => void,
  onQuestion: (q: Question) => void,
): Promise<UnlistenFn> {
  const offEvent = await listen<AgentEvent>("agent://event", (e) => onEvent(e.payload));
  const offApproval = await listen<ProposedCall>("agent://approval", (e) =>
    onApproval(e.payload),
  );
  const offQuestion = await listen<Question>("agent://question", (e) =>
    onQuestion(e.payload),
  );
  return () => {
    offEvent();
    offApproval();
    offQuestion();
  };
}

/** A file the user handed to the agent. */
export type Attachment = {
  path: string;
  name: string;
  bytes: number;
  /** Which tool will open it, so the UI can say so. */
  tool: string;
  /**
   * A small `data:` image for attachments that have something to show.
   *
   * Carried in the payload rather than read back off disk, so the transcript
   * keeps showing it without the webview needing filesystem access.
   */
  preview: string | null;
};

/**
 * Files dropped onto the window.
 *
 * The webview's own drag events never fire for OS file drops in a Tauri
 * window — the native side intercepts them — so this listens to Tauri's
 * event instead. `hover` and `cancel` drive the drop-target highlight.
 */
export async function onFileDrop(
  onDrop: (paths: string[]) => void,
  onHover: (over: boolean) => void,
): Promise<UnlistenFn> {
  return getCurrentWebview().onDragDropEvent((event) => {
    if (event.payload.type === "over") onHover(true);
    else if (event.payload.type === "leave") onHover(false);
    else if (event.payload.type === "drop") {
      onHover(false);
      onDrop(event.payload.paths);
    }
  });
}

/// Size in the largest unit that leaves a number worth reading.
///
/// This used to stop at MB with no decimals, because it was written for model
/// weights. Reused for attachments it reported every screenshot and text file
/// as "0 MB".
export function formatBytes(n: number): string {
  if (!n) return "—";
  if (n < 1024) return `${n} B`;
  const kb = n / 1024;
  if (kb < 1024) return `${kb < 10 ? kb.toFixed(1) : Math.round(kb)} KB`;
  const mb = kb / 1024;
  if (mb < 1024) return `${mb < 10 ? mb.toFixed(1) : Math.round(mb)} MB`;
  return `${(mb / 1024).toFixed(1)} GB`;
}
