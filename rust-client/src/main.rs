use std::collections::HashMap;
use std::env;
use std::process::ExitCode;

use serde_json::{json, Map, Value};

const DEFAULT_URL: &str = "http://127.0.0.1:8000";

fn usage() -> String {
    format!(
        "scalarflow-client - talk to a ScalarFlow HTTP server

USAGE:
    scalarflow-client [--url URL] <COMMAND> [OPTIONS]

COMMANDS:
    health                                   show server status and model info
    model [--num-in N] [--layer-sizes 4,2,1] [--activation tanh]
                                             replace the in-memory network
    train --inputs JSON --outputs JSON [--epochs N]
                                             train the network
    predict --inputs JSON                    run a forward pass

GLOBAL OPTIONS:
    --url URL     base server URL (default: {DEFAULT_URL}, or $SCALARFLOW_URL)
    -h, --help    show this help message

EXAMPLES:
    scalarflow-client health
    scalarflow-client model --layer-sizes 4,2,1 --activation relu
    scalarflow-client train --inputs '[[0,0],[0,1],[1,0],[1,1]]' \\
        --outputs '[0,1,1,0]' --epochs 100
    scalarflow-client predict --inputs '[[0,0],[1,0]]'
"
    )
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let raw: Vec<String> = env::args().skip(1).collect();
    if raw.is_empty() || raw.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{}", usage());
        return Ok(());
    }

    let (positional, flags) = parse_args(&raw)?;
    let base_url = flags
        .get("url")
        .cloned()
        .or_else(|| env::var("SCALARFLOW_URL").ok())
        .unwrap_or_else(|| DEFAULT_URL.to_string());

    let command = positional.first().map(String::as_str).unwrap_or("help");
    let response = match command {
        "help" => {
            println!("{}", usage());
            return Ok(());
        }
        "health" => request(&base_url, "GET", "/health", None)?,
        "model" => {
            let body = model_body(&flags)?;
            request(&base_url, "POST", "/model", Some(&Value::Object(body)))?
        }
        "train" => {
            let body = train_body(&flags)?;
            request(&base_url, "POST", "/train", Some(&body))?
        }
        "predict" => {
            let inputs = required_json(&flags, "inputs")?;
            let body = json!({ "inputs": inputs });
            request(&base_url, "POST", "/predict", Some(&body))?
        }
        other => return Err(format!("unknown command '{other}'\n\n{}", usage())),
    };

    let pretty = serde_json::to_string_pretty(&response).map_err(|err| err.to_string())?;
    println!("{pretty}");
    Ok(())
}

fn parse_args(raw: &[String]) -> Result<(Vec<String>, HashMap<String, String>), String> {
    let mut positional = Vec::new();
    let mut flags = HashMap::new();
    let mut index = 0;

    while index < raw.len() {
        let arg = &raw[index];
        if let Some(body) = arg.strip_prefix("--") {
            if let Some((key, value)) = body.split_once('=') {
                flags.insert(key.to_string(), value.to_string());
            } else {
                let value = raw
                    .get(index + 1)
                    .ok_or_else(|| format!("--{body} requires a value"))?;
                flags.insert(body.to_string(), value.clone());
                index += 1;
            }
        } else {
            positional.push(arg.clone());
        }
        index += 1;
    }

    Ok((positional, flags))
}

fn required_json(flags: &HashMap<String, String>, name: &str) -> Result<Value, String> {
    let raw = flags
        .get(name)
        .ok_or_else(|| format!("missing --{name} (expected a JSON value)"))?;
    serde_json::from_str(raw).map_err(|err| format!("--{name} is not valid JSON: {err}"))
}

fn parse_integer(flags: &HashMap<String, String>, name: &str) -> Result<Option<i64>, String> {
    match flags.get(name) {
        Some(raw) => raw
            .parse::<i64>()
            .map(Some)
            .map_err(|_| format!("--{name} must be an integer")),
        None => Ok(None),
    }
}

fn parse_layer_sizes(raw: &str) -> Result<Vec<i64>, String> {
    raw.split(',')
        .map(|part| {
            part.trim()
                .parse::<i64>()
                .map_err(|_| format!("--layer-sizes must be comma-separated integers, got '{raw}'"))
        })
        .collect()
}

fn model_body(flags: &HashMap<String, String>) -> Result<Map<String, Value>, String> {
    let mut body = Map::new();

    if let Some(num_in) = parse_integer(flags, "num-in")? {
        body.insert("num_in".to_string(), json!(num_in));
    }
    if let Some(raw) = flags.get("layer-sizes") {
        body.insert("layer_sizes".to_string(), json!(parse_layer_sizes(raw)?));
    }
    if let Some(activation) = flags.get("activation") {
        body.insert("activation".to_string(), json!(activation));
    }

    Ok(body)
}

fn train_body(flags: &HashMap<String, String>) -> Result<Value, String> {
    let mut body = Map::new();
    body.insert("inputs".to_string(), required_json(flags, "inputs")?);
    body.insert("outputs".to_string(), required_json(flags, "outputs")?);

    if let Some(epochs) = parse_integer(flags, "epochs")? {
        body.insert("epochs".to_string(), json!(epochs));
    }

    Ok(Value::Object(body))
}

fn request(
    base_url: &str,
    method: &str,
    path: &str,
    body: Option<&Value>,
) -> Result<Value, String> {
    let url = format!("{}{}", base_url.trim_end_matches('/'), path);

    let response = match method {
        "GET" => ureq::get(&url).call(),
        "POST" => {
            let payload = body.ok_or_else(|| "missing request body".to_string())?;
            ureq::post(&url).send_json(payload.clone())
        }
        other => return Err(format!("unsupported HTTP method '{other}'")),
    };

    match response {
        Ok(resp) => resp
            .into_json::<Value>()
            .map_err(|err| format!("could not decode response from {url}: {err}")),
        Err(ureq::Error::Status(status, resp)) => {
            let detail = resp
                .into_json::<Value>()
                .ok()
                .and_then(|value| value.get("message").and_then(Value::as_str).map(str::to_string))
                .unwrap_or_else(|| "no response body".to_string());
            Err(format!("{url} returned HTTP {status}: {detail}"))
        }
        Err(ureq::Error::Transport(err)) => {
            Err(format!("could not reach {url}: {err}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| item.to_string()).collect()
    }

    fn flags(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect()
    }

    #[test]
    fn parse_args_collects_positionals_and_flags() {
        let (positional, parsed) =
            parse_args(&args(&["train", "--inputs", "[[0]]", "--epochs=5"])).unwrap();

        assert_eq!(positional, vec!["train".to_string()]);
        assert_eq!(parsed.get("inputs").map(String::as_str), Some("[[0]]"));
        assert_eq!(parsed.get("epochs").map(String::as_str), Some("5"));
    }

    #[test]
    fn parse_args_requires_flag_value() {
        assert!(parse_args(&args(&["--url"])).is_err());
    }

    #[test]
    fn parse_layer_sizes_splits_and_trims() {
        assert_eq!(parse_layer_sizes("4, 2,1").unwrap(), vec![4, 2, 1]);
    }

    #[test]
    fn parse_layer_sizes_rejects_non_integer() {
        assert!(parse_layer_sizes("4,x").is_err());
    }

    #[test]
    fn required_json_parses_value() {
        let parsed = required_json(&flags(&[("inputs", "[[0,1]]")]), "inputs").unwrap();

        assert_eq!(parsed, serde_json::json!([[0, 1]]));
    }

    #[test]
    fn required_json_rejects_invalid_json() {
        assert!(required_json(&flags(&[("inputs", "nope")]), "inputs").is_err());
    }

    #[test]
    fn model_body_includes_given_fields() {
        let body = model_body(&flags(&[
            ("num-in", "3"),
            ("layer-sizes", "4,2,1"),
            ("activation", "relu"),
        ]))
        .unwrap();

        assert_eq!(body.get("num_in"), Some(&serde_json::json!(3)));
        assert_eq!(body.get("layer_sizes"), Some(&serde_json::json!([4, 2, 1])));
        assert_eq!(body.get("activation"), Some(&serde_json::json!("relu")));
    }

    #[test]
    fn train_body_requires_inputs_and_outputs() {
        assert!(train_body(&flags(&[("inputs", "[[0]]")])).is_err());
    }

    #[test]
    fn train_body_includes_epochs() {
        let body = train_body(&flags(&[
            ("inputs", "[[0]]"),
            ("outputs", "[1]"),
            ("epochs", "10"),
        ]))
        .unwrap();

        assert_eq!(body["inputs"], serde_json::json!([[0]]));
        assert_eq!(body["outputs"], serde_json::json!([1]));
        assert_eq!(body["epochs"], serde_json::json!(10));
    }
}
