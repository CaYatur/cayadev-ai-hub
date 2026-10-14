//! Tool-argument validation against a JSON Schema, tuned for small models.
//!
//! Not a full JSON Schema implementation: it covers what tool schemas use in
//! practice (`type`, `required`, `properties`, `items`, `enum`,
//! `additionalProperties: false`) and fails open on anything else. The point
//! is not to police the model but to turn a bad call into one short message it
//! can act on, before the call reaches a server that answers with a stack
//! trace.
//!
//! Harmless type slips are repaired instead of refused. A 2B model writes
//! `"count": "5"` or `"recursive": "true"` constantly; refusing those costs a
//! whole step for no gain, so they are coerced in place.

use serde_json::{Map, Value};

/// Validate (and lightly repair) `args` against `schema`.
///
/// On failure the message names the argument and what was expected, so the
/// model can fix exactly that.
pub fn validate(schema: &Value, args: &mut Map<String, Value>) -> Result<(), String> {
    let mut obj = Value::Object(std::mem::take(args));
    let result = check(schema, &mut obj, "");
    if let Value::Object(m) = obj {
        *args = m;
    }
    result
}

fn types_of(schema: &Value) -> Vec<&str> {
    match schema.get("type") {
        Some(Value::String(t)) => vec![t.as_str()],
        Some(Value::Array(ts)) => ts.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

fn label(path: &str) -> String {
    if path.is_empty() { "arguments".into() } else { format!("`{path}`") }
}

fn check(schema: &Value, value: &mut Value, path: &str) -> Result<(), String> {
    let types = types_of(schema);
    if !types.is_empty() && !types.iter().any(|t| matches_type(t, value)) {
        // Try each declared type's coercion, keeping the first that fits.
        let fixed = types.iter().find_map(|t| coerce(t, value));
        match fixed {
            Some(v) => *value = v,
            None => {
                return Err(format!(
                    "{} must be {}, got {}",
                    label(path),
                    types.join(" or "),
                    describe(value)
                ))
            }
        }
    }

    if let Some(Value::Array(allowed)) = schema.get("enum") {
        if !allowed.contains(value) {
            let list: Vec<String> = allowed.iter().map(|v| v.to_string()).collect();
            return Err(format!("{} must be one of {}", label(path), list.join(", ")));
        }
    }

    match value {
        Value::Object(map) => check_object(schema, map, path),
        Value::Array(items) => {
            if let Some(item_schema) = schema.get("items").filter(|s| s.is_object()) {
                for (i, item) in items.iter_mut().enumerate() {
                    check(item_schema, item, &format!("{path}[{i}]"))?;
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn check_object(schema: &Value, map: &mut Map<String, Value>, path: &str) -> Result<(), String> {
    let props = schema.get("properties").and_then(Value::as_object);
    let join = |k: &str| if path.is_empty() { k.to_string() } else { format!("{path}.{k}") };

    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        for key in required.iter().filter_map(Value::as_str) {
            let missing = match map.get(key) {
                None | Some(Value::Null) => true,
                Some(Value::String(s)) => s.is_empty(),
                _ => false,
            };
            if missing {
                return Err(format!("missing required argument `{}`", join(key)));
            }
        }
    }

    // A null for an optional argument means "not given"; drop it rather than
    // fail a type check the model did not mean to make.
    map.retain(|_, v| !v.is_null());

    if let Some(props) = props {
        if schema.get("additionalProperties") == Some(&Value::Bool(false)) {
            if let Some(extra) = map.keys().find(|k| !props.contains_key(*k)) {
                let allowed: Vec<&str> = props.keys().map(String::as_str).collect();
                return Err(format!(
                    "unknown argument `{}`; allowed: {}",
                    join(extra),
                    allowed.join(", ")
                ));
            }
        }
        for (key, prop_schema) in props {
            if let Some(v) = map.get_mut(key) {
                check(prop_schema, v, &join(key))?;
            }
        }
    }
    Ok(())
}

fn matches_type(t: &str, v: &Value) -> bool {
    match t {
        "string" => v.is_string(),
        "integer" => v.as_i64().is_some() || v.as_u64().is_some()
            || v.as_f64().is_some_and(|f| f.fract() == 0.0),
        "number" => v.is_number(),
        "boolean" => v.is_boolean(),
        "array" => v.is_array(),
        "object" => v.is_object(),
        "null" => v.is_null(),
        // Unknown type keyword: fail open.
        _ => true,
    }
}

/// Repair one common slip, or `None` when the value really is wrong.
fn coerce(t: &str, v: &Value) -> Option<Value> {
    match (t, v) {
        ("integer", Value::String(s)) => s.trim().parse::<i64>().ok().map(Value::from),
        ("integer", Value::Number(n)) => n.as_f64().filter(|f| f.fract() == 0.0).map(|f| Value::from(f as i64)),
        ("number", Value::String(s)) => s.trim().parse::<f64>().ok().and_then(|f| serde_json::Number::from_f64(f)).map(Value::Number),
        ("boolean", Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "yes" | "1" => Some(Value::Bool(true)),
            "false" | "no" | "0" => Some(Value::Bool(false)),
            _ => None,
        },
        ("boolean", Value::Number(n)) => match n.as_i64() {
            Some(0) => Some(Value::Bool(false)),
            Some(1) => Some(Value::Bool(true)),
            _ => None,
        },
        ("string", Value::Number(n)) => Some(Value::String(n.to_string())),
        ("string", Value::Bool(b)) => Some(Value::String(b.to_string())),
        // A JSON document passed as a string, or a single value where a list
        // was wanted.
        ("array", Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Array(a)) => Some(Value::Array(a)),
            _ => Some(Value::Array(vec![v.clone()])),
        },
        ("array", other) if !other.is_null() && !other.is_object() => Some(Value::Array(vec![other.clone()])),
        ("object", Value::String(s)) => match crate::agent::toolcall::repair_json(s) {
            Some(Value::Object(o)) => Some(Value::Object(o)),
            _ => None,
        },
        _ => None,
    }
}

fn describe(v: &Value) -> String {
    let kind = match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    };
    let mut shown = v.to_string();
    if shown.chars().count() > 40 {
        shown = shown.chars().take(40).collect::<String>() + "...";
    }
    format!("{kind} ({shown})")
}

/// One-line signature such as `path: string*, depth: integer`, used both in
/// error feedback and in the text-mode tool list. `*` marks required.
pub fn signature(schema: &Value) -> String {
    let required: Vec<&str> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|r| r.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let Some(props) = schema.get("properties").and_then(Value::as_object) else {
        return String::new();
    };
    props
        .iter()
        .map(|(k, s)| {
            let t = types_of(s);
            let t = if t.is_empty() { "any".to_string() } else { t.join("|") };
            let star = if required.contains(&k.as_str()) { "*" } else { "" };
            format!("{k}: {t}{star}")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn args(v: Value) -> Map<String, Value> {
        serde_json::from_value(v).unwrap()
    }

    fn schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "depth": {"type": "integer"},
                "recursive": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "mode": {"type": "string", "enum": ["fast", "full"]},
                "opts": {"type": "object", "properties": {"limit": {"type": "number"}}}
            },
            "required": ["path"],
            "additionalProperties": false
        })
    }

    #[test]
    fn a_valid_call_passes_unchanged() {
        let mut a = args(json!({"path": "a", "depth": 2, "recursive": true}));
        let before = a.clone();
        validate(&schema(), &mut a).unwrap();
        assert_eq!(a, before);
    }

    #[test]
    fn a_missing_required_argument_is_named() {
        let err = validate(&schema(), &mut args(json!({"depth": 1}))).unwrap_err();
        assert!(err.contains("`path`"), "{err}");
        let err = validate(&schema(), &mut args(json!({"path": ""}))).unwrap_err();
        assert!(err.contains("`path`"), "{err}");
    }

    #[test]
    fn common_type_slips_are_repaired_in_place() {
        let mut a = args(json!({
            "path": 42, "depth": "3", "recursive": "TRUE", "tags": "one",
            "opts": "{\"limit\": \"2.5\"}"
        }));
        validate(&schema(), &mut a).unwrap();
        assert_eq!(a["path"], json!("42"));
        assert_eq!(a["depth"], json!(3));
        assert_eq!(a["recursive"], json!(true));
        assert_eq!(a["tags"], json!(["one"]));
        assert_eq!(a["opts"]["limit"], json!(2.5));
    }

    #[test]
    fn a_wrong_type_that_cannot_be_repaired_is_reported() {
        let err = validate(&schema(), &mut args(json!({"path": "a", "depth": "deep"}))).unwrap_err();
        assert!(err.contains("`depth` must be integer"), "{err}");
    }

    #[test]
    fn enum_and_unknown_arguments_are_reported_with_the_allowed_values() {
        let err = validate(&schema(), &mut args(json!({"path": "a", "mode": "slow"}))).unwrap_err();
        assert!(err.contains("\"fast\""), "{err}");
        let err = validate(&schema(), &mut args(json!({"path": "a", "colour": 1}))).unwrap_err();
        assert!(err.contains("unknown argument `colour`") && err.contains("depth"), "{err}");
    }

    #[test]
    fn null_optionals_are_dropped_and_nested_errors_carry_their_path() {
        let mut a = args(json!({"path": "a", "depth": null}));
        validate(&schema(), &mut a).unwrap();
        assert!(!a.contains_key("depth"));

        let err = validate(&schema(), &mut args(json!({"path": "a", "opts": {"limit": "x"}}))).unwrap_err();
        assert!(err.contains("`opts.limit`"), "{err}");
    }

    #[test]
    fn schemas_without_types_fail_open() {
        let mut a = args(json!({"anything": [1, 2]}));
        validate(&json!({}), &mut a).unwrap();
        validate(&json!({"type": "object"}), &mut a).unwrap();
    }

    #[test]
    fn signature_marks_required_arguments() {
        let s = signature(&schema());
        assert!(s.contains("path: string*"), "{s}");
        assert!(s.contains("depth: integer"), "{s}");
    }
}
