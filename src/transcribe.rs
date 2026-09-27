//! The transcription path: one decode pipeline behind two routes.
//!
//! `/api/whisper/transcribe` is the daemon's old namespace, served **verbatim**
//! (path, multipart body, JSON response shape). `/v1/audio/transcriptions` is
//! OpenAI's shape. Both resolve to the same model (an explicit, installed id
//! when given; otherwise the user's selection, else the first installed
//! catalog entry) and the same backend dispatch: **MLX** on Apple Silicon —
//! several times faster on the same checkpoint — and **Candle** (pure Rust,
//! CPU) everywhere else, which is what makes a Windows or Linux build a real
//! transcriber instead of a 501 stub. `SENCLAW_ASR_BACKEND=candle` forces the
//! Candle path on macOS too, which is how the cross-platform backend gets
//! integration-tested on the machine this repo is developed on.
//!
//! The two backends read **different checkpoint families** from the same model
//! root: MLX loads `mlx-community/*`, Candle loads HF-layout `openai/*`. A
//! directory the chosen backend cannot load is a 400 naming the family to
//! download, not a load-time stack trace.
//!
//! Every handler runs its decode on `spawn_blocking`. Both backends are
//! blocking, CPU/Metal-bound workloads; on the async reactor either would
//! stall every other request this process is serving — including the daemon's
//! own health probe, which would then report a working transcription as a
//! process that is down.

use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use axum_extra::extract::Multipart;
use serde_json::json;

#[cfg(target_os = "macos")]
use crate::mlx::WhisperEngine;

use crate::api_error::{ApiError, OpenAiError};
use crate::AppState;

pub fn router() -> Router<Arc<AppState>> {
    Router::new().route("/api/whisper/transcribe", post(legacy_transcribe)).route("/v1/audio/transcriptions", post(openai_transcribe))
}

struct TranscribeOutcome {
    text: String,
    decode_ms: Option<f64>,
    audio_secs: Option<f32>,
}

/// Which decoder serves this request.
fn use_candle() -> bool {
    if cfg!(not(target_os = "macos")) {
        return true;
    }
    std::env::var("SENCLAW_ASR_BACKEND").map(|v| v.trim().eq_ignore_ascii_case("candle")).unwrap_or(false)
}

// ── `/api/whisper/transcribe` — old namespace, verbatim ──────────────────────

async fn legacy_transcribe(State(state): State<Arc<AppState>>, multipart: Multipart) -> Result<impl IntoResponse, ApiError> {
    let (filename, bytes, language_field) = read_legacy_multipart(multipart).await.map_err(|(s, m)| ApiError(s, m))?;

    let model_id = crate::models::selected_model(&state)?;
    let dir = crate::models::installed_model_dir(&state.env, &model_id);
    if !crate::models::is_installed(&dir) {
        return Err(ApiError(StatusCode::BAD_REQUEST, format!("model `{model_id}` is not installed")));
    }
    let language = language_field.or_else(|| crate::models::default_language(&state));

    let outcome = transcribe_bytes(dir, filename, bytes, language, true).await.map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
    Ok(Json(json!({ "ok": true, "text": outcome.text })))
}

async fn read_legacy_multipart(mut multipart: Multipart) -> Result<(String, Vec<u8>, Option<String>), (StatusCode, String)> {
    let mut audio: Option<(String, Vec<u8>)> = None;
    let mut language: Option<String> = None;

    while let Some(field) = multipart.next_field().await.map_err(|e| (StatusCode::BAD_REQUEST, format!("read multipart: {e}")))? {
        let name = field.name().unwrap_or("").to_string();
        if name == "language" {
            language = field.text().await.ok().filter(|s| !s.is_empty());
        } else {
            // Treat any other field as the audio payload.
            let filename = field.file_name().unwrap_or("audio.bin").to_string();
            let bytes = field.bytes().await.map_err(|e| (StatusCode::BAD_REQUEST, format!("read audio: {e}")))?;
            audio = Some((filename, bytes.to_vec()));
        }
    }
    let (filename, bytes) = audio.ok_or_else(|| (StatusCode::BAD_REQUEST, "no audio field".to_string()))?;
    Ok((filename, bytes, language))
}

// ── `/v1/audio/transcriptions` — OpenAI multipart ────────────────────────────

#[derive(Default)]
struct OpenAiFields {
    file: Option<(String, Vec<u8>)>,
    model: Option<String>,
    language: Option<String>,
    response_format: Option<String>,
}

async fn openai_transcribe(State(state): State<Arc<AppState>>, multipart: Multipart) -> Response {
    match openai_transcribe_inner(state, multipart).await {
        Ok(r) => r,
        Err(e) => e.into_response(),
    }
}

async fn openai_transcribe_inner(state: Arc<AppState>, multipart: Multipart) -> Result<Response, OpenAiError> {
    let fields = read_openai_multipart(multipart).await?;
    let (filename, bytes) = fields.file.ok_or_else(|| OpenAiError(StatusCode::BAD_REQUEST, "no `file` field".into()))?;

    // An explicit, *installed* id wins. OpenAI clients routinely send a
    // placeholder like "whisper-1" that names nothing we have installed —
    // falling back to the configured model rather than erroring on that is
    // what keeps them working unmodified.
    let model_id = match fields.model.filter(|m| crate::models::is_installed(&crate::models::installed_model_dir(&state.env, m))) {
        Some(m) => m,
        None => crate::models::selected_model(&state).map_err(|ApiError(status, msg)| OpenAiError(status, msg))?,
    };
    let dir = crate::models::installed_model_dir(&state.env, &model_id);
    if !crate::models::is_installed(&dir) {
        return Err(OpenAiError(StatusCode::BAD_REQUEST, format!("model `{model_id}` is not installed")));
    }
    let language = fields.language.or_else(|| crate::models::default_language(&state));
    let response_format = fields.response_format.unwrap_or_else(|| "json".to_string());
    if response_format != "json" && response_format != "text" {
        return Err(OpenAiError(StatusCode::BAD_REQUEST, format!("unsupported response_format `{response_format}` (only json, text)")));
    }

    let outcome = transcribe_bytes(dir, filename, bytes, language, false).await.map_err(|e| OpenAiError(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;

    Ok(if response_format == "text" {
        ([(axum::http::header::CONTENT_TYPE, "text/plain; charset=utf-8")], outcome.text).into_response()
    } else {
        Json(json!({ "text": outcome.text })).into_response()
    })
}

async fn read_openai_multipart(mut multipart: Multipart) -> Result<OpenAiFields, OpenAiError> {
    let mut out = OpenAiFields::default();
    while let Some(field) = multipart.next_field().await.map_err(|e| OpenAiError(StatusCode::BAD_REQUEST, format!("read multipart: {e}")))? {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                let filename = field.file_name().unwrap_or("audio.bin").to_string();
                let bytes = field.bytes().await.map_err(|e| OpenAiError(StatusCode::BAD_REQUEST, format!("read file: {e}")))?;
                out.file = Some((filename, bytes.to_vec()));
            }
            "model" => out.model = field.text().await.ok().filter(|s| !s.is_empty()),
            "language" => out.language = field.text().await.ok().filter(|s| !s.is_empty()),
            "response_format" => out.response_format = field.text().await.ok().filter(|s| !s.is_empty()),
            // `prompt`, `temperature`, `timestamp_granularities[]` etc. — accepted
            // and ignored rather than rejected, so a client sending the full
            // OpenAI field set still works.
            _ => {
                let _ = field.bytes().await;
            }
        }
    }
    Ok(out)
}

// ── Shared decode dispatch ────────────────────────────────────────────────────

async fn transcribe_bytes(dir: PathBuf, filename: String, bytes: Vec<u8>, language: Option<String>, timestamps: bool) -> anyhow::Result<TranscribeOutcome> {
    if bytes.is_empty() {
        anyhow::bail!("empty audio body");
    }
    if !dir.is_dir() {
        anyhow::bail!("model_dir does not exist: {}", dir.display());
    }

    let outcome = if use_candle() {
        transcribe_candle(dir, filename, bytes, language).await?
    } else {
        #[cfg(not(target_os = "macos"))]
        unreachable!("use_candle() is always true off macOS");
        #[cfg(target_os = "macos")]
        {
            transcribe_mlx(dir, filename, bytes, language, timestamps).await?
        }
    };
    tracing::debug!(
        chars = outcome.text.chars().count(),
        decode_ms = ?outcome.decode_ms,
        audio_secs = ?outcome.audio_secs,
        "transcription complete"
    );
    Ok(outcome)
}

/// Candle path: chunked greedy decode on the CPU. No per-segment stats; a
/// missing extra field beats a failed transcription.
async fn transcribe_candle(dir: PathBuf, filename: String, audio: Vec<u8>, language: Option<String>) -> anyhow::Result<TranscribeOutcome> {
    if !crate::candle_whisper::supports_dir(&dir) {
        anyhow::bail!(
            "`{}` is not a Candle-compatible Whisper checkpoint — download an HF-layout repo such as \
             `openai/whisper-large-v3-turbo` for this platform",
            dir.display()
        );
    }
    tokio::task::spawn_blocking(move || {
        let tmp = write_probe_file(&audio, Some(&filename))?;
        // Loaded per request and dropped after: an idle process must cost
        // megabytes, and a CPU reload is cheap relative to a CPU decode.
        let mut engine = crate::candle_whisper::CandleWhisper::load(&dir)?;
        let out = engine.transcribe_file(&tmp, language.as_deref());
        let _ = std::fs::remove_file(&tmp);
        out.map(|text| TranscribeOutcome { text, decode_ms: None, audio_secs: None })
    })
    .await?
}

#[cfg(target_os = "macos")]
async fn transcribe_mlx(dir: PathBuf, filename: String, audio: Vec<u8>, language: Option<String>, timestamps: bool) -> anyhow::Result<TranscribeOutcome> {
    tokio::task::spawn_blocking(move || {
        // Serialize the whole load+decode against any other concurrent MLX
        // work in this process. Each `WhisperEngine` only guards its own
        // instance's `loaded` state, and this process creates a fresh one per
        // request — without a process-wide lock, two transcriptions arriving
        // together would touch MLX's shared Metal device from two OS threads
        // at once, which corrupts Metal state and can SIGSEGV.
        let _mlx_serial = crate::mlx::mlx_serial::lock();
        let engine = WhisperEngine::new(dir);
        // Make sure the ~2 GB of weights goes back to the OS whichever way this
        // closure exits. Dropping the engine alone only returns buffers to
        // MLX's own cache; an explicit unload (which also clears that cache) is
        // what makes an idle process cost megabytes rather than the model size.
        struct UnloadOnExit<'a>(&'a WhisperEngine);
        impl Drop for UnloadOnExit<'_> {
            fn drop(&mut self) {
                self.0.unload();
            }
        }
        let _unload = UnloadOnExit(&engine);
        let tmp = write_probe_file(&audio, Some(&filename))?;
        let out = if timestamps {
            engine
                .transcribe_file_timed(&tmp, language.as_deref())
                .map(|(text, stats)| TranscribeOutcome { text, decode_ms: Some(stats.decode_ms), audio_secs: Some(stats.audio_secs) })
        } else {
            engine.transcribe_file(&tmp, language.as_deref()).map(|text| TranscribeOutcome { text, decode_ms: None, audio_secs: None })
        };
        let _ = std::fs::remove_file(&tmp);
        out
    })
    .await?
}

/// Write the upload where a decoder can probe it. The extension is
/// load-bearing: Symphonia (and the MLX path's own temp file) probe the
/// container by it, so a temp file written as `.audio` breaks decoding of a
/// perfectly good file.
fn write_probe_file(audio: &[u8], filename: Option<&str>) -> anyhow::Result<PathBuf> {
    let ext = filename.and_then(|f| std::path::Path::new(f).extension()).and_then(|e| e.to_str()).filter(|e| !e.is_empty()).unwrap_or("wav");
    let tmp = std::env::temp_dir().join(format!(
        "sen-whisper-{}-{}.{ext}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
    ));
    std::fs::write(&tmp, audio)?;
    Ok(tmp)
}
