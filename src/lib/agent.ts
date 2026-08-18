import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

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

export type PermissionMode = "read_only" | "project_auto" | "balanced" | "unrestricted";

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
  cancel: () => invoke<void>("cancel_run"),
  run: (prompt: string, model?: string, numCtx?: number, effort?: Effort) =>
    invoke<string>("run_agent", { prompt, model, numCtx, effort }),
  contextOptions: (model: string) => invoke<ContextOptions>("context_options", { model }),
};

/** Subscribe to run events and approval questions. */
export async function subscribe(
  onEvent: (e: AgentEvent) => void,
  onApproval: (c: ProposedCall) => void,
): Promise<UnlistenFn> {
  const offEvent = await listen<AgentEvent>("agent://event", (e) => onEvent(e.payload));
  const offApproval = await listen<ProposedCall>("agent://approval", (e) =>
    onApproval(e.payload),
  );
  return () => {
    offEvent();
    offApproval();
  };
}

export function formatBytes(n: number): string {
  if (!n) return "—";
  const gb = n / 1024 ** 3;
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(n / 1024 ** 2).toFixed(0)} MB`;
}
