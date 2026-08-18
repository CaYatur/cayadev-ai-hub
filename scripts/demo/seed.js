// Fixture for the README screenshots.
//
// Written before the app boots, so `scripts/demo/` renders the real components
// over a conversation that has already happened. Nothing here is invented for
// effect: the model names, sizes, placements and throughput figures are the
// measured ones from `docs/BENCHMARKS.md` on the development machine, and the
// transcript is a task the agent actually does.
//
// Two things are stubbed because a browser is not a Tauri window: the command
// bridge, so the model list and version render, and `localStorage`, which the
// app would have written itself after a real session.

const WORKSPACE = "C:\\Users\\cagan\\Desktop\\CaYaAIAgentHub";

// -------------------------------------------------------------- the bridge

const MODELS = [
  ["qwen2.5-coder:0.5b", "0.5B", 400_000_000, "resident", 32768],
  ["qwen2.5-coder:7b", "7B", 4_700_000_000, "resident", 32768],
  ["qwen2.5vl:7b", "7B", 6_000_000_000, "marginal", 16384],
  ["qwen3-vl:8b", "8B", 6_100_000_000, "marginal", 16384],
  ["gpt-oss:20b", "20B", 13_800_000_000, "spilled", 8192],
  ["qwen3.8:27b-mtp-q4_K_M", "27B", 17_700_000_000, "spilled", 8192],
];

const CAPS = {
  "qwen2.5vl:7b": { vision: true, tools: false },
  "qwen3-vl:8b": { vision: true, thinking: true },
  "gpt-oss:20b": { thinking: true },
  "qwen3.8:27b-mtp-q4_K_M": { vision: true, thinking: true },
};

const RESPONSES = {
  ollama_version: "0.32.14",
  list_models: MODELS.map(([name, size, bytes, placement, ctx]) => ({
    name,
    parameter_size: size,
    quantization: "Q4_K_M",
    weights_bytes: bytes,
    capabilities: {
      completion: true,
      tools: true,
      vision: false,
      thinking: false,
      embedding: false,
      ...(CAPS[name] ?? {}),
    },
    max_context: 262144,
    recommended_context: ctx,
    placement,
  })),
  routing_policy: {
    assignments: {
      coding: "qwen2.5-coder:7b",
      edit: "qwen2.5-coder:7b",
      reasoning: "qwen2.5-coder:7b",
      classify: "qwen2.5-coder:0.5b",
      vision: "qwen2.5vl:7b",
    },
  },
  context_options: { recommended: 32768, trained_max: 262144, choices: [4096, 8192, 16384, 32768] },
  checkpoint_list: [
    {
      id: "ck-2",
      timestamp: new Date(Date.now() - 4 * 60 * 1000).toISOString(),
      label: "edit_file src-tauri/src/agent/runner.rs",
      files: [{ path: "src-tauri/src/agent/runner.rs", kind: "modified", before: "a1b2c3" }],
    },
    {
      id: "ck-1",
      timestamp: new Date(Date.now() - 9 * 60 * 1000).toISOString(),
      label: "write_file docs/CONTEXT.md",
      files: [{ path: "docs/CONTEXT.md", kind: "created", before: null }],
    },
  ],
  checkpoint_size: 1_240_000,
  set_workspace: WORKSPACE,
};

window.__TAURI_INTERNALS__ = {
  transformCallback: (cb) => cb,
  invoke: (cmd) => Promise.resolve(RESPONSES[cmd] ?? null),
  metadata: { currentWindow: { label: "main" }, currentWebview: { label: "main" } },
};

// ---------------------------------------------------------- the transcript

function call(name, args, id, risk) {
  return {
    id,
    step: 1,
    name,
    arguments: args,
    recovery: "native",
    evaluation: {
      decision: "allow",
      reason: "Inside the project folder.",
      escapes_workspace: false,
      risk: risk ?? "read_only",
    },
  };
}

const REPLY = [
  "İki yerde `num_ctx` ayarlanıyor ve ikisi aynı fikirde değil.",
  "",
  "## Ne oluyor",
  "",
  "`commands.rs` modelin önerdiği pencereyi kullanıyor, ama `RunConfig` hâlâ",
  "8192 döndürüyor. Açık bir seçim yapılmadığında ikincisi kazanıyor.",
  "",
  "```rust",
  "let context = num_ctx",
  "    .or_else(|| recommended(profile))",
  "    .unwrap_or(8192);",
  "```",
  "",
  "| Model | Yerleşim | Önerilen |",
  "| --- | --- | ---: |",
  "| qwen2.5-coder:7b | Resident | 32768 |",
  "| qwen3.8:27b | Spilled | 8192 |",
  "",
  "Düzeltmesi tek satır: `RunConfig` içindeki varsayılanı kaldırmak yeterli.",
].join("\n");

const CHAT = {
  id: "demo-chat",
  title: "Bağlam penceresi neden 8192'de kalıyor?",
  workspace: WORKSPACE,
  createdAt: Date.now() - 5 * 60 * 1000,
  tokens: 14200,
  elapsedMs: 41000,
  lastRate: 42.3,
  contextUsed: 9840,
  entries: [
    {
      kind: "user",
      text: "Bağlam penceresi neden hep 8192'de kalıyor? Modeli değiştirsem de değişmiyor.",
      at: Date.now() - 5 * 60 * 1000,
    },
    {
      kind: "thinking",
      text:
        "The window is chosen in run_agent, and RunConfig also carries a default. " +
        "One of them is winning over the other. Read both rather than guess which.",
    },
    {
      kind: "tool",
      call: call("search", { query: "num_ctx" }, "t1"),
      settled: "auto_approved",
      output:
        "src-tauri/src/commands.rs:227  let context = num_ctx\n" +
        "src-tauri/src/agent/runner.rs:118  pub num_ctx: Option<u32>,\n" +
        "src-tauri/src/agent/runner.rs:141  num_ctx: None,",
    },
    {
      kind: "tool",
      call: call("read_file", { path: "src-tauri/src/commands.rs" }, "t2"),
      settled: "auto_approved",
      output: "…4210 bytes…",
    },
    { kind: "assistant", text: REPLY },
  ],
};

const VISION = {
  id: "demo-vision",
  title: "Bu ekranda ne açık?",
  workspace: WORKSPACE,
  createdAt: Date.now() - 60 * 60 * 1000,
  tokens: 3100,
  elapsedMs: 16000,
  lastRate: 57.1,
  contextUsed: 2400,
  entries: [
    {
      kind: "user",
      text: "Bilgisayarımın ekranında şu anda ne açık?",
      at: Date.now() - 60 * 60 * 1000,
    },
    {
      kind: "tool",
      call: call("screenshot", { question: "Ekranda hangi pencereler açık?" }, "s1", "read_only"),
      settled: "user_approved",
      output:
        "Ekranda üç pencere görünüyor: bir kod düzenleyici (Rust kaynak dosyası " +
        "açık), bir terminal (cargo test çıktısı, 206 test geçmiş) ve bir tarayıcı.",
    },
    {
      kind: "assistant",
      text:
        "Ekranında üç pencere açık:\n\n" +
        "- **Kod düzenleyici** — bir Rust kaynak dosyası açık\n" +
        "- **Terminal** — `cargo test` çıktısı, 206 test geçmiş\n" +
        "- **Tarayıcı**\n",
    },
  ],
};

// Newest first, and `?session=vision` puts the other one on top instead — the
// app opens whichever session is first.
const wanted = new URLSearchParams(location.search);
const order = wanted.get("session") === "vision" ? [VISION, CHAT] : [CHAT, VISION];
localStorage.setItem("cayahub.sessions.v1", JSON.stringify(order));
localStorage.setItem("i18nextLng", "tr");

// `?view=Modeller` opens that tab by clicking it, rather than by reaching into
// the app's state — a screenshot of a view nobody can navigate to is not a
// screenshot of the app.
const view = wanted.get("view");
if (view) {
  const open = () => {
    const tab = [...document.querySelectorAll("button, a")].find(
      (el) => el.textContent.trim() === view,
    );
    if (tab) tab.click();
    else setTimeout(open, 120);
  };
  window.addEventListener("DOMContentLoaded", () => setTimeout(open, 400));
}
