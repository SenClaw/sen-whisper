//! Two error shapes, because two namespaces have two different contracts.
//!
//! `/api/whisper/*` is the daemon's old namespace, served **verbatim** — its
//! errors are the daemon's own flat `{"error": "<message>"}`
//! ([`sen_runtime_sdk::api::ErrorBody`], the same shape every other runtime
//! route answers with). `/v1/audio/transcriptions` is OpenAI-shaped, so its
//! errors nest under `error.message` the way a real OpenAI client expects.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use sen_runtime_sdk::api::ErrorBody;

/// `/api/whisper/*` error: `{"error": "<message>"}`.
pub struct ApiError(pub StatusCode, pub String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(ErrorBody::new(self.1))).into_response()
    }
}

/// `/v1/audio/transcriptions` error: `{"error": {"message": "...", "type": "..."}}`.
pub struct OpenAiError(pub StatusCode, pub String);

impl IntoResponse for OpenAiError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({ "error": { "message": self.1, "type": "invalid_request_error" } });
        (self.0, Json(body)).into_response()
    }
}
