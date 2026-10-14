import { Component, type ErrorInfo, type ReactNode } from "react";
import { invoke } from "@tauri-apps/api/core";
import i18next from "i18next";

/// Send an error to the backend's crash.log. Never throws.
export function reportError(message: string): void {
  invoke("report_frontend_error", { message }).catch(() => {});
}

function describe(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message}\n${error.stack ?? ""}`;
  try {
    return String(error);
  } catch {
    return "unprintable error";
  }
}

let installed = false;

/// Errors outside React's render (event handlers, timers, promises) are
/// recorded too; they do not blank the window, but they explain one that did.
export function installGlobalErrorLogging(): void {
  if (installed) return;
  installed = true;
  window.addEventListener("error", (e) => {
    reportError(`window.onerror: ${describe(e.error ?? e.message)}`);
  });
  window.addEventListener("unhandledrejection", (e) => {
    reportError(`unhandledrejection: ${describe(e.reason)}`);
  });
}

type State = { error: Error | null; logPath: string | null };

/**
 * Without this, one component that throws while rendering unmounts the whole
 * tree and the window goes blank, which looks exactly like the app crashing.
 * The transcript lives in localStorage, so recovering is a reload away.
 */
export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null, logPath: null };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    reportError(`render: ${describe(error)}\ncomponent stack:${info.componentStack ?? ""}`);
    invoke<string>("crash_log_path")
      .then((logPath) => this.setState({ logPath }))
      .catch(() => {});
  }

  render() {
    const { error, logPath } = this.state;
    if (!error) return this.props.children;
    const t = (key: string) => i18next.t(key);
    return (
      <div className="crash-screen" role="alert">
        <h2>{t("crash.title")}</h2>
        <p>{t("crash.body")}</p>
        <pre className="code-block">{error.message}</pre>
        {logPath && (
          <p className="crash-path">
            {t("crash.logAt")} <code>{logPath}</code>
          </p>
        )}
        <div className="crash-actions">
          <button onClick={() => this.setState({ error: null })}>{t("crash.retry")}</button>
          <button onClick={() => window.location.reload()}>{t("crash.reload")}</button>
        </div>
      </div>
    );
  }
}
