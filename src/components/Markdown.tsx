import { isValidElement, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { openUrl, revealItemInDir, openPath } from "@tauri-apps/plugin-opener";

/// Extensions that run something when opened.
///
/// A model can write any link it likes, and the text it is summarising may
/// itself have come off a web page. Handing one of these to the shell is the
/// one click in this app that could execute code, so those reveal the file in
/// the file manager instead — the user still gets there, nothing runs.
const EXECUTABLE = new Set([
  "exe", "msi", "bat", "cmd", "com", "scr", "ps1", "psm1", "vbs", "vbe", "js",
  "jse", "wsf", "wsh", "hta", "jar", "lnk", "reg", "dll", "sh", "bash", "zsh",
  "app", "pkg", "dmg", "deb", "rpm", "run", "bin", "appimage",
]);

/// Anything that reads as a path rather than prose.
///
/// Deliberately narrow. Inline code holds identifiers far more often than
/// paths, and turning `useState` into a broken link is worse than leaving a
/// real path as plain text.
const PATH_SHAPED =
  /^(?:[A-Za-z]:[\\/]|\.{0,2}[\\/])?(?:[\w .@+-]+[\\/])*[\w .@+-]+\.[A-Za-z]\w{0,9}$/;

/// Extensions worth turning into a link when the string has no separator.
///
/// Shape alone is not enough: `array.map`, `Object.keys` and `str.trim` all
/// look exactly like `notes.txt` to the pattern above, and a bare `README.md`
/// is far commoner in prose than a bare method call is rare. So a string with
/// no slash in it has to end in something that is actually a file.
const FILE_EXTENSIONS = new Set([
  "rs", "ts", "tsx", "js", "jsx", "py", "go", "java", "kt", "rb", "php", "c",
  "h", "cpp", "hpp", "cs", "swift", "sh", "bash", "ps1", "sql", "css", "scss",
  "html", "xml", "json", "yaml", "yml", "toml", "ini", "cfg", "conf", "env",
  "md", "mdx", "txt", "csv", "tsv", "log", "lock", "gitignore", "dockerfile",
  "png", "jpg", "jpeg", "gif", "webp", "svg", "bmp", "ico", "pdf", "docx",
  "xlsx", "pptx", "zip", "tar", "gz", "mp3", "wav", "mp4", "mov",
]);

function looksLikeAPath(text: string): boolean {
  if (text.length > 200 || !PATH_SHAPED.test(text)) return false;
  if (/[\\/]/.test(text)) return true;
  return FILE_EXTENSIONS.has(extensionOf(text));
}

function extensionOf(path: string): string {
  const dot = path.lastIndexOf(".");
  return dot === -1 ? "" : path.slice(dot + 1).toLowerCase();
}

/// Open a local path, or reveal it when opening it would run something.
async function reveal(path: string): Promise<void> {
  try {
    if (EXECUTABLE.has(extensionOf(path))) await revealItemInDir(path);
    else await openPath(path);
  } catch {
    // A path the model invented, or a file since deleted. Nothing to report —
    // the click simply does nothing.
  }
}

function isWebUrl(href: string): boolean {
  return /^(https?|mailto):/i.test(href);
}

/// A link that leaves the app, by whichever route suits what it points at.
function Link({ href, children }: { href?: string; children?: ReactNode }) {
  if (!href) return <>{children}</>;
  const web = isWebUrl(href);
  return (
    <a
      className={web ? "md-link" : "md-link md-link-file"}
      href={href}
      title={href}
      onClick={(e) => {
        // Navigating the webview would replace the app with the page.
        e.preventDefault();
        if (web) openUrl(href).catch(() => {});
        else reveal(href.replace(/^file:\/\/\/?/, ""));
      }}
    >
      {children}
    </a>
  );
}

/// Read the language tag off the `<code>` element inside a fenced block.
function languageOf(children: ReactNode): string | null {
  if (!isValidElement<{ className?: string }>(children)) return null;
  return /language-([\w+-]+)/.exec(children.props.className ?? "")?.[1] ?? null;
}

/// A fenced block, with the language shown and one-click copy.
function CodeBlock({ children }: { children?: ReactNode }) {
  const [copied, setCopied] = useState(false);
  const language = languageOf(children);

  function copy(e: React.MouseEvent<HTMLButtonElement>) {
    const text = e.currentTarget.parentElement?.querySelector("code")?.textContent ?? "";
    navigator.clipboard?.writeText(text).catch(() => {});
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  return (
    <div className="md-code">
      {language && <span className="md-code-lang">{language}</span>}
      <button className="md-code-copy" onClick={copy} title="Copy">
        {copied ? "✓" : "⧉"}
      </button>
      <pre>{children}</pre>
    </div>
  );
}

/// Inline code, clickable when it is plainly a file path.
///
/// Models write paths in backticks constantly — "I edited `src/main.rs`" — and
/// that is the reference the user most wants to follow.
function InlineCode({ children }: { children?: ReactNode }) {
  const text = typeof children === "string" ? children : null;
  if (text && looksLikeAPath(text)) {
    return (
      <button className="md-inline-path" onClick={() => reveal(text)} title={text}>
        {text}
      </button>
    );
  }
  return <code className="md-inline">{children}</code>;
}

/// Assistant output, rendered as the markdown it is written in.
///
/// Models emit markdown whether or not anything asks them to: headings, fenced
/// code, tables, links. Showing the raw characters made every reply harder to
/// read than the plain text it would have been.
///
/// Streaming means this renders half-finished input on every token — an
/// unclosed fence, a table with one row. `react-markdown` recovers from all of
/// those, which is the reason for taking the dependency over a hand-rolled
/// pass: a renderer that throws on incomplete input is unusable here.
export function Markdown({ text }: { text: string }) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        // `ignoreMissing` so an unknown language tag renders as plain code
        // rather than failing the whole block.
        rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }]]}
        components={{
          a: Link,
          pre: ({ children }) => <CodeBlock>{children}</CodeBlock>,
          code: ({ className, children }) =>
            className ? (
              <code className={className}>{children}</code>
            ) : (
              <InlineCode>{children}</InlineCode>
            ),
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
