//! Make any MCP tool result safe to hand to a small model.
//!
//! A server can return anything: a data URI for album art inside a JSON text
//! block, a base64 PDF as a resource, ten thousand log lines, an array of
//! every preset with a thumbnail each. One such result fills a 4B model's
//! window, and what comes back next is not an answer but a hallucination about
//! something else entirely. Nothing here knows any server or tool by name: the
//! same rules apply to every result.
//!
//! In order:
//! 1. Binary payloads become short placeholders: `data:` URIs anywhere in
//!    text, base64-looking runs, and image/audio/blob content blocks.
//! 2. JSON (in text blocks or `structuredContent`) is walked: long arrays and
//!    objects keep their head and say how much was left out, very long strings
//!    are clipped, very deep nesting is cut.
//! 3. The whole result is capped to a share of the model's context.

use regex::Regex;
use serde_json::{Map, Value};
use std::sync::LazyLock;

/// Items kept from an array before the rest are summarised.
const MAX_ARRAY_ITEMS: usize = 25;
/// Keys kept from one object before the rest are summarised.
const MAX_OBJECT_KEYS: usize = 60;
/// Characters kept from a single JSON string value.
const MAX_STRING_CHARS: usize = 2_000;
/// Nesting below this depth is replaced with a marker.
const MAX_DEPTH: usize = 12;
/// Base64 runs shorter than this are left alone (ids, hashes, tokens).
const MIN_BLOB_CHARS: usize = 120;

/// Bounds of the overall cap, in characters.
const MIN_CAP: usize = 2_000;
const MAX_CAP: usize = 24_000;

static DATA_URI: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"data:([a-zA-Z0-9.+-]+/[a-zA-Z0-9.+-]+)?(?:;[a-zA-Z0-9=._-]+)*;base64,[A-Za-z0-9+/=_-]*")
        .expect("static regex")
});

static BLOB: LazyLock<Regex> = LazyLock::new(|| {
    // Escaped newlines (`\n`) appear inside base64 that was JSON-encoded with
    // line wrapping, so they are allowed inside a run.
    Regex::new(r"[A-Za-z0-9+/_-]{40,}(?:(?:\\n|\s)?[A-Za-z0-9+/_-]{40,})*={0,2}").expect("static regex")
});

/// Characters of tool result a model with `num_ctx` tokens of context gets.
///
/// About a fifth of the window (at ~3.6 characters per token), so a result
/// never crowds out the task, the system prompt and the model's own reply.
pub fn cap_for_context(num_ctx: Option<u32>) -> usize {
    let ctx = num_ctx.unwrap_or(8192) as usize;
    (ctx * 36 / 10 / 5).clamp(MIN_CAP, MAX_CAP)
}

fn size_label(encoded_len: usize) -> String {
    let bytes = encoded_len * 3 / 4;
    if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{} KB", bytes / 1024)
    } else {
        format!("{bytes} bytes")
    }
}

fn kind_of(mime: Option<&str>) -> &'static str {
    match mime.and_then(|m| m.split('/').next()) {
        Some("image") => "image",
        Some("audio") => "audio",
        Some("video") => "video",
        Some("font") => "font",
        _ => "binary data",
    }
}

/// Placeholder for an omitted payload, e.g. `[image omitted, 12 KB]`.
pub fn placeholder(mime: Option<&str>, encoded_len: usize) -> String {
    match mime {
        Some(m) if !m.is_empty() => format!("[{} omitted, {m}, {}]", kind_of(Some(m)), size_label(encoded_len)),
        _ => format!("[{} omitted, {}]", kind_of(None), size_label(encoded_len)),
    }
}

/// A run that is base64 rather than a long word, path or hex hash.
fn looks_like_base64(run: &str) -> bool {
    let compact: String = run.chars().filter(|c| !c.is_whitespace()).collect::<String>().replace("\\n", "");
    if compact.len() < MIN_BLOB_CHARS {
        return false;
    }
    let upper = compact.chars().filter(|c| c.is_ascii_uppercase()).count();
    let lower = compact.chars().filter(|c| c.is_ascii_lowercase()).count();
    let digit = compact.chars().filter(|c| c.is_ascii_digit()).count();
    // Hex (a sha256, a git object) has no uppercase-and-lowercase mix; a
    // long identifier or path rarely has digits throughout. Real base64 has
    // all three in roughly even shares.
    let n = compact.len() as f64;
    upper as f64 / n > 0.1 && lower as f64 / n > 0.1 && digit as f64 / n > 0.02
}

/// Replace binary payloads in free text.
pub fn clean_text(text: &str) -> String {
    let without_uris = DATA_URI.replace_all(text, |c: &regex::Captures| {
        let whole = c.get(0).map_or("", |m| m.as_str());
        let payload = whole.split_once("base64,").map_or(0, |(_, p)| p.len());
        placeholder(c.get(1).map(|m| m.as_str()), payload)
    });
    BLOB.replace_all(&without_uris, |c: &regex::Captures| {
        let run = c.get(0).map_or("", |m| m.as_str());
        if looks_like_base64(run) { placeholder(None, run.len()) } else { run.to_string() }
    })
    .into_owned()
}

fn clip(s: &str, limit: usize) -> String {
    if s.chars().count() <= limit {
        return s.to_string();
    }
    let total = s.chars().count();
    let head: String = s.chars().take(limit).collect();
    format!("{head}… [{} more characters omitted]", total - limit)
}

/// Sanitize a JSON value in place.
pub fn clean_value(value: &mut Value) {
    walk(value, 0);
}

fn walk(value: &mut Value, depth: usize) {
    if depth >= MAX_DEPTH && (value.is_array() || value.is_object()) {
        *value = Value::String("[nested data omitted]".into());
        return;
    }
    match value {
        Value::String(s) => {
            let cleaned = clean_text(s);
            *s = clip(&cleaned, MAX_STRING_CHARS);
        }
        Value::Array(items) => {
            let total = items.len();
            if total > MAX_ARRAY_ITEMS {
                items.truncate(MAX_ARRAY_ITEMS);
                for item in items.iter_mut() {
                    walk(item, depth + 1);
                }
                items.push(Value::String(format!("[{} more items omitted, {total} in total]", total - MAX_ARRAY_ITEMS)));
            } else {
                for item in items.iter_mut() {
                    walk(item, depth + 1);
                }
            }
        }
        Value::Object(map) => {
            let total = map.len();
            if total > MAX_OBJECT_KEYS {
                let kept: Map<String, Value> = std::mem::take(map).into_iter().take(MAX_OBJECT_KEYS).collect();
                *map = kept;
                map.insert("_omitted".into(), Value::String(format!("{} more keys omitted", total - MAX_OBJECT_KEYS)));
            }
            for v in map.values_mut() {
                walk(v, depth + 1);
            }
        }
        _ => {}
    }
}

/// Sanitize one piece of text that may itself be JSON.
///
/// Many servers return a JSON document inside a text block; walking it is
/// better than regexes over its serialisation, because a long array gets
/// summarised rather than cut mid-item.
pub fn clean_text_or_json(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        if let Ok(mut v) = serde_json::from_str::<Value>(trimmed) {
            clean_value(&mut v);
            return serde_json::to_string(&v).unwrap_or_else(|_| clean_text(text));
        }
    }
    clean_text(text)
}

/// Cap a finished result at `cap` characters, saying what was dropped.
pub fn cap(text: &str, cap: usize) -> String {
    let total = text.chars().count();
    if total <= cap {
        return text.to_string();
    }
    let head: String = text.chars().take(cap).collect();
    format!("{head}\n\n[truncated: {cap} of {total} characters shown. Ask for less, e.g. with a filter or a limit, if you need the rest.]")
}

/// Flatten and sanitize a `tools/call` result for the model.
///
/// `Err` carries the text of a result the server flagged with `isError`.
pub fn render(result: &Value, max_chars: usize) -> Result<String, String> {
    let mut parts: Vec<String> = Vec::new();
    for block in result.get("content").and_then(Value::as_array).into_iter().flatten() {
        let ty = block.get("type").and_then(Value::as_str);
        let mime = block.get("mimeType").and_then(Value::as_str);
        let data_len = |b: &Value| b.get("data").and_then(Value::as_str).map_or(0, str::len);
        match ty {
            Some("text") => parts.push(clean_text_or_json(block.get("text").and_then(Value::as_str).unwrap_or(""))),
            // Images are never forwarded to the model, even a vision-capable
            // one: tool results go back as text, and a small model is better
            // served by knowing an image exists than by its bytes.
            Some("image") | Some("audio") => parts.push(placeholder(
                mime.or(Some(if ty == Some("image") { "image/*" } else { "audio/*" })),
                data_len(block),
            )),
            Some("resource") => {
                let r = block.get("resource").cloned().unwrap_or(Value::Null);
                let uri = r.get("uri").and_then(Value::as_str).unwrap_or("?");
                let rmime = r.get("mimeType").and_then(Value::as_str);
                if let Some(text) = r.get("text").and_then(Value::as_str) {
                    parts.push(format!("[resource {uri}]\n{}", clean_text_or_json(text)));
                } else if let Some(blob) = r.get("blob").and_then(Value::as_str) {
                    parts.push(format!("[resource {uri}] {}", placeholder(rmime, blob.len())));
                } else {
                    parts.push(format!("[resource {uri}]"));
                }
            }
            Some("resource_link") => parts.push(format!(
                "[link {}]",
                block.get("uri").and_then(Value::as_str).unwrap_or("?")
            )),
            _ => {
                let mut b = block.clone();
                clean_value(&mut b);
                parts.push(b.to_string());
            }
        }
    }
    if parts.iter().all(|p| p.trim().is_empty()) {
        if let Some(s) = result.get("structuredContent").filter(|v| !v.is_null()) {
            let mut s = s.clone();
            clean_value(&mut s);
            parts = vec![s.to_string()];
        }
    }
    let mut text = parts.join("\n");
    if text.trim().is_empty() {
        text = "(no output)".into();
    }
    let text = cap(&text, max_chars);
    if result.get("isError").and_then(Value::as_bool).unwrap_or(false) {
        Err(text)
    } else {
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A realistic base64 payload: mixed case, digits, `+` and `/`.
    fn b64(len: usize) -> String {
        let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        (0..len).map(|i| alphabet[(i * 7 + i / 3) % 64] as char).collect()
    }

    /// The report that prompted this: album art as a data URI inside a JSON
    /// text block. The model must still see title and artist.
    #[test]
    fn now_playing_artwork_is_replaced_and_metadata_kept() {
        let inner = json!({"ok": true, "nowPlaying": {
            "title": "Save Your Tears", "artist": "The Weeknd",
            "artwork": format!("data:image/png;base64,{}", b64(400_000))
        }});
        let result = json!({"content": [{"type": "text", "text": inner.to_string()}]});
        let out = render(&result, cap_for_context(Some(8192))).unwrap();
        assert!(out.contains("Save Your Tears") && out.contains("The Weeknd"), "{out}");
        assert!(out.contains("[image omitted, image/png, 292 KB]"), "{out}");
        assert!(out.len() < 300, "{} chars: {out}", out.len());
    }

    #[test]
    fn data_uris_in_plain_text_and_without_a_mime_are_replaced() {
        let text = format!("before data:;base64,{} after data:audio/mpeg;base64,{} end", b64(2000), b64(8000));
        let out = clean_text(&text);
        assert!(out.starts_with("before [binary data omitted, 1 KB] after [audio omitted, audio/mpeg, 5 KB] end"), "{out}");
    }

    #[test]
    fn bare_base64_anywhere_in_nested_json_is_replaced() {
        let mut v = json!({"a": [{"b": {"thumb": b64(5000), "id": "a1B2c3"}}], "sha": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"});
        clean_value(&mut v);
        assert_eq!(v["a"][0]["b"]["thumb"], json!("[binary data omitted, 3 KB]"));
        assert_eq!(v["a"][0]["b"]["id"], json!("a1B2c3"), "short ids stay");
        assert!(v["sha"].as_str().unwrap().starts_with("e3b0c442"), "a hex hash is not base64");
    }

    #[test]
    fn ordinary_long_text_is_not_mistaken_for_base64() {
        let prose = "The quick brown fox jumps over the lazy dog. ".repeat(50);
        assert_eq!(clean_text(&prose), prose);
        let path = "C:/Users/cagan/Desktop/SomeVeryLongFolderName/AnotherLongFolderName/AndAnotherOne/file.txt";
        assert_eq!(clean_text(path), path);
    }

    #[test]
    fn image_audio_and_blob_blocks_become_placeholders() {
        let result = json!({"content": [
            {"type": "text", "text": "Here is the image:"},
            {"type": "image", "mimeType": "image/png", "data": b64(4096)},
            {"type": "audio", "mimeType": "audio/wav", "data": b64(40)},
            {"type": "resource", "resource": {"uri": "file:///a.pdf", "mimeType": "application/pdf", "blob": b64(20000)}},
            {"type": "resource", "resource": {"uri": "file:///b.txt", "text": "hello"}}
        ]});
        let out = render(&result, 10_000).unwrap();
        assert!(out.contains("[image omitted, image/png, 3 KB]"), "{out}");
        assert!(out.contains("[audio omitted, audio/wav, 30 bytes]"), "{out}");
        assert!(out.contains("[resource file:///a.pdf] [binary data omitted, application/pdf, 14 KB]"), "{out}");
        assert!(out.contains("[resource file:///b.txt]\nhello"), "{out}");
        assert!(!out.contains(&b64(100)));
    }

    #[test]
    fn huge_arrays_objects_and_strings_are_summarised() {
        let items: Vec<Value> = (0..500).map(|i| json!({"id": i, "name": format!("preset {i}")})).collect();
        let many_keys: Map<String, Value> = (0..200).map(|i| (format!("k{i:03}"), json!(i))).collect();
        let doc = json!({"presets": items, "settings": many_keys, "log": "x".repeat(10_000)});
        let result = json!({"content": [{"type": "text", "text": doc.to_string()}]});
        let out = render(&result, 24_000).unwrap();
        let v: Value = serde_json::from_str(&out).expect("still valid JSON");
        let presets = v["presets"].as_array().unwrap();
        assert_eq!(presets.len(), MAX_ARRAY_ITEMS + 1);
        assert_eq!(presets.last().unwrap(), &json!("[475 more items omitted, 500 in total]"));
        assert_eq!(v["settings"]["_omitted"], json!("140 more keys omitted"));
        assert!(v["log"].as_str().unwrap().ends_with("[8000 more characters omitted]"));
    }

    #[test]
    fn deep_nesting_is_cut() {
        let mut v = json!("leaf");
        for _ in 0..30 {
            v = json!({"n": v});
        }
        clean_value(&mut v);
        assert!(v.to_string().contains("[nested data omitted]"));
    }

    #[test]
    fn structured_content_is_sanitized_too() {
        let result = json!({"content": [], "structuredContent": {"art": format!("data:image/jpeg;base64,{}", b64(3000)), "n": 1}});
        let out = render(&result, 5_000).unwrap();
        assert_eq!(out, r#"{"art":"[image omitted, image/jpeg, 2 KB]","n":1}"#);
    }

    #[test]
    fn the_total_is_capped_with_a_note_and_errors_stay_errors() {
        let result = json!({"content": [{"type": "text", "text": "line\n".repeat(5_000)}], "isError": true});
        let err = render(&result, 2_000).unwrap_err();
        assert!(err.contains("[truncated: 2000 of 25000 characters shown"), "{}", &err[err.len() - 120..]);
        assert!(err.chars().count() < 2_200);
    }

    #[test]
    fn the_cap_follows_the_context_window() {
        assert_eq!(cap_for_context(Some(2048)), MIN_CAP);
        assert_eq!(cap_for_context(Some(8192)), 5_898);
        assert!(cap_for_context(Some(32768)) > cap_for_context(Some(8192)));
        assert_eq!(cap_for_context(Some(1_000_000)), MAX_CAP);
    }
}
