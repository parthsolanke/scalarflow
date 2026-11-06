use std::fmt;
use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const DEFAULT_BASE_URL: &str = "http://localhost:8000";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiError {
    pub status_code: u16,
    pub message: String,
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.message.is_empty() {
            write!(
                f,
                "scalarflow: request failed with status {}",
                self.status_code
            )
        } else {
            write!(
                f,
                "scalarflow: request failed with status {}: {}",
                self.status_code, self.message
            )
        }
    }
}

impl std::error::Error for ApiError {}

#[derive(Debug)]
pub enum Error {
    Api(ApiError),
    Transport(String),
    Encode(String),
    Decode(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Api(err) => write!(f, "{err}"),
            Error::Transport(msg) => write!(f, "scalarflow: send request: {msg}"),
            Error::Encode(msg) => write!(f, "scalarflow: encode request: {msg}"),
            Error::Decode(msg) => write!(f, "scalarflow: decode response: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Api(err) => Some(err),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub num_in: usize,
    pub layer_sizes: Vec<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit_activation: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Network {
    pub id: String,
    pub config: NetworkConfig,
    #[serde(default)]
    pub parameters: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FitResult {
    pub epochs: usize,
    pub losses: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct PredictResponse {
    outputs: Vec<Vec<f64>>,
}

pub struct Client {
    base_url: String,
    agent: ureq::Agent,
}

impl Client {
    pub fn new(base_url: impl Into<String>) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(30))
            .build();
        Self::with_agent(base_url, agent)
    }

    pub fn with_agent(base_url: impl Into<String>, agent: ureq::Agent) -> Self {
        let mut base_url = base_url.into();
        if base_url.trim().is_empty() {
            base_url = DEFAULT_BASE_URL.to_string();
        }
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            agent,
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn create_network(&self, config: &NetworkConfig) -> Result<Network, Error> {
        let body = serde_json::to_value(config).map_err(|err| Error::Encode(err.to_string()))?;
        let text = self.send("POST", "/networks", Some(body))?;
        self.decode(&text)
    }

    pub fn get_network(&self, id: &str) -> Result<Network, Error> {
        let path = format!("/networks/{}", encode_segment(id));
        let text = self.send("GET", &path, None)?;
        self.decode(&text)
    }

    pub fn delete_network(&self, id: &str) -> Result<(), Error> {
        let path = format!("/networks/{}", encode_segment(id));
        self.send("DELETE", &path, None)?;
        Ok(())
    }

    pub fn predict(&self, id: &str, inputs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Error> {
        let path = format!("/networks/{}/predict", encode_segment(id));
        let body = json!({ "inputs": inputs });
        let text = self.send("POST", &path, Some(body))?;
        let response: PredictResponse = self.decode(&text)?;
        Ok(response.outputs)
    }

    pub fn fit(
        &self,
        id: &str,
        inputs: &[Vec<f64>],
        outputs: &[Vec<f64>],
        epochs: usize,
    ) -> Result<FitResult, Error> {
        let path = format!("/networks/{}/fit", encode_segment(id));
        let body = json!({ "inputs": inputs, "outputs": outputs, "epochs": epochs });
        let text = self.send("POST", &path, Some(body))?;
        self.decode(&text)
    }

    fn decode<T: DeserializeOwned>(&self, text: &str) -> Result<T, Error> {
        serde_json::from_str(text).map_err(|err| Error::Decode(err.to_string()))
    }

    fn send(&self, method: &str, path: &str, body: Option<Value>) -> Result<String, Error> {
        let url = format!("{}{}", self.base_url, path);
        let request = match method {
            "GET" => self.agent.get(&url),
            "POST" => self.agent.post(&url),
            "DELETE" => self.agent.delete(&url),
            other => {
                return Err(Error::Encode(format!("unsupported method {other}")));
            }
        };
        let request = request.set("Accept", "application/json");

        let result = match body {
            Some(value) => {
                let payload =
                    serde_json::to_string(&value).map_err(|err| Error::Encode(err.to_string()))?;
                request
                    .set("Content-Type", "application/json")
                    .send_string(&payload)
            }
            None => request.call(),
        };

        match result {
            Ok(response) => response
                .into_string()
                .map_err(|err| Error::Transport(err.to_string())),
            Err(ureq::Error::Status(status_code, response)) => {
                let text = response.into_string().unwrap_or_default();
                Err(Error::Api(ApiError {
                    status_code,
                    message: error_message(&text),
                }))
            }
            Err(ureq::Error::Transport(err)) => Err(Error::Transport(err.to_string())),
        }
    }
}

fn error_message(text: &str) -> String {
    if let Ok(Value::Object(payload)) = serde_json::from_str::<Value>(text) {
        if let Some(message) = payload.get("error").and_then(Value::as_str) {
            return message.to_string();
        }
        if let Some(message) = payload.get("message").and_then(Value::as_str) {
            return message.to_string();
        }
    }
    text.trim().to_string()
}

fn encode_segment(segment: &str) -> String {
    let mut encoded = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            _ => {
                encoded.push('%');
                encoded.push_str(&format!("{byte:02X}"));
            }
        }
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_segment_escapes_reserved_characters() {
        assert_eq!(encode_segment("a/b c"), "a%2Fb%20c");
    }

    #[test]
    fn encode_segment_keeps_unreserved_characters() {
        assert_eq!(encode_segment("abc-_.~"), "abc-_.~");
    }

    #[test]
    fn error_message_prefers_the_error_field() {
        assert_eq!(
            error_message(r#"{"error":"bad","message":"worse"}"#),
            "bad"
        );
    }

    #[test]
    fn error_message_falls_back_to_the_message_field() {
        assert_eq!(error_message(r#"{"message":"worse"}"#), "worse");
    }

    #[test]
    fn error_message_falls_back_to_the_raw_body() {
        assert_eq!(error_message("  plain failure  "), "plain failure");
    }

    #[test]
    fn client_uses_the_default_base_url_for_blank_input() {
        let client = Client::new("   ");
        assert_eq!(client.base_url(), DEFAULT_BASE_URL);
    }

    #[test]
    fn client_trims_trailing_slashes() {
        let client = Client::new("http://example.com///");
        assert_eq!(client.base_url(), "http://example.com");
    }

    #[test]
    fn api_error_display_with_and_without_a_message() {
        let with_message = ApiError {
            status_code: 400,
            message: "bad request".to_string(),
        };
        assert_eq!(
            with_message.to_string(),
            "scalarflow: request failed with status 400: bad request"
        );

        let without_message = ApiError {
            status_code: 404,
            message: String::new(),
        };
        assert_eq!(
            without_message.to_string(),
            "scalarflow: request failed with status 404"
        );
    }

    #[test]
    fn error_display_variants() {
        assert_eq!(
            Error::Transport("boom".to_string()).to_string(),
            "scalarflow: send request: boom"
        );
        assert_eq!(
            Error::Encode("boom".to_string()).to_string(),
            "scalarflow: encode request: boom"
        );
        assert_eq!(
            Error::Decode("boom".to_string()).to_string(),
            "scalarflow: decode response: boom"
        );

        let err = Error::Api(ApiError {
            status_code: 500,
            message: "oops".to_string(),
        });
        assert_eq!(
            err.to_string(),
            "scalarflow: request failed with status 500: oops"
        );
        assert!(std::error::Error::source(&err).is_some());
    }
}
