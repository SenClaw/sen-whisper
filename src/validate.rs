//! Pre-download HuggingFace compatibility check for Whisper checkpoints.
//!
//! `GET /api/whisper/models/:id/validate` — the STT half of the daemon's old
//! `src/gateway/ui_server/hf_validate.rs` (the TTS and LLM halves stayed with
//! their own runtimes). Checks the *metadata only* — repo info, file tree, and
//! the small `config.json` — against what [`crate::mlx::mlx_asr::whisper`]
//! actually supports, so a client can say "this won't work" before downloading
//! hundreds of megabytes. Nothing here fetches weights.
//!
//! The support rule mirrors the real loader: `ModelDimensions` (MLX-converted
//! config with flat `n_mels`/`n_audio_state`/… keys). Keep it in sync when the
//! loader gains or loses a shape it accepts.

use axum::extract::Path as AxumPath;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Serialize;
use serde_json::Value;

use crate::api_error::ApiError;
use crate::models::normalize_hf_id;

const HF_BASE: &str = "https://huggingface.co";

#[derive(Debug, Serialize)]
pub struct ValidateReport {
    pub id: String,
    /// Whether the MLX Whisper loader can run this checkpoint.
    pub supported: bool,
    /// Human-readable explanation (what matched, or why it's rejected).
    pub reason: String,
    /// Architecture / model_type detected from config.json (if any).
    pub architecture: Option<String>,
    /// Sum of downloadable file sizes in bytes (what a download would fetch).
    pub total_size_bytes: u64,
    /// Repo requires accepting terms / auth — the downloader can't fetch it.
    pub gated: bool,
    /// Metadata could not be fully retrieved (network/HF hiccup); the verdict
    /// is advisory and a download may still be worth trying.
    pub inconclusive: bool,
}

pub async fn whisper_validate(AxumPath(id): AxumPath<String>) -> Result<impl IntoResponse, ApiError> {
    validate(&id).await.map(Json)
}

async fn validate(raw_id: &str) -> Result<ValidateReport, ApiError> {
    let id = normalize_hf_id(raw_id).map_err(|e| ApiError(StatusCode::BAD_REQUEST, e))?;
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // 1. Repo info — existence + gating.
    let info_url = format!("{HF_BASE}/api/models/{id}");
    let info = match client.get(&info_url).send().await {
        // HF answers 401 (not 404) for unknown repos so private repo names
        // don't leak — treat all three as "not there for us".
        Ok(r) if matches!(r.status(), StatusCode::NOT_FOUND | StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) => {
            return Ok(ValidateReport {
                id,
                supported: false,
                reason: "model not found on Hugging Face (or it is private) — check the org/repo id".into(),
                architecture: None,
                total_size_bytes: 0,
                gated: false,
                inconclusive: false,
            });
        }
        Ok(r) if r.status().is_success() => r.json::<Value>().await.ok(),
        _ => None,
    };
    let Some(info) = info else {
        return Ok(ValidateReport {
            id,
            supported: false,
            reason: "could not reach the Hugging Face API — verdict unavailable; you may still try downloading".into(),
            architecture: None,
            total_size_bytes: 0,
            gated: false,
            inconclusive: true,
        });
    };
    // `gated` is false | "auto" | "manual"; private repos also 401 on files.
    let gated = !matches!(info.get("gated"), None | Some(Value::Bool(false)));
    let private = info.get("private").and_then(Value::as_bool).unwrap_or(false);
    if gated || private {
        return Ok(ValidateReport {
            id,
            supported: false,
            reason: "repo is gated/private — the built-in downloader has no Hugging Face login".into(),
            architecture: None,
            total_size_bytes: 0,
            gated: true,
            inconclusive: false,
        });
    }

    // 2. File tree — names + total size.
    let tree_url = format!("{HF_BASE}/api/models/{id}/tree/main?recursive=true");
    let tree: Vec<Value> = match client.get(&tree_url).send().await {
        Ok(r) if r.status().is_success() => r.json().await.unwrap_or_default(),
        _ => Vec::new(),
    };
    let files: Vec<(String, u64)> =
        tree.iter().filter(|e| e["type"] == "file").map(|e| (e["path"].as_str().unwrap_or_default().to_string(), e["size"].as_u64().unwrap_or(0))).collect();
    let total_size_bytes: u64 = files.iter().map(|f| f.1).sum();
    let has_file = |name: &str| files.iter().any(|(p, _)| p == name);

    // 3. config.json (small) — the architecture source of truth.
    let cfg_url = format!("{HF_BASE}/{id}/resolve/main/config.json");
    let cfg: Option<Value> = match client.get(&cfg_url).send().await {
        Ok(r) if r.status().is_success() => r.json().await.ok(),
        _ => None,
    };

    let (supported, reason, architecture) = match &cfg {
        None => (false, "repo has no readable config.json — cannot determine the architecture".to_string(), None),
        Some(cfg) => check_whisper(cfg, &has_file),
    };

    Ok(ValidateReport { id, supported, reason, architecture, total_size_bytes, gated: false, inconclusive: cfg.is_none() && files.is_empty() })
}

/// Whisper rule — mirrors `ModelDimensions` (MLX-converted flat config).
fn check_whisper(cfg: &Value, has_file: &dyn Fn(&str) -> bool) -> (bool, String, Option<String>) {
    let mlx_dims = ["n_mels", "n_audio_state", "n_text_layer"].iter().all(|k| cfg.get(*k).is_some());
    let weights = has_file("weights.safetensors") || has_file("model.safetensors") || has_file("weights.npz");
    if mlx_dims && weights {
        let q = cfg["quantization"]["bits"].as_u64().map(|b| format!(", {b}-bit quantized")).unwrap_or_default();
        return (true, format!("MLX Whisper checkpoint (mlx-community layout{q}) — supported"), Some("whisper (MLX)".into()));
    }
    if cfg["model_type"].as_str() == Some("whisper") {
        return (
            false,
            "this is the HF transformers Whisper layout — use an mlx-community/whisper-* \
             conversion instead (its config.json carries flat n_mels/n_audio_state dims)"
                .into(),
            Some("whisper (transformers)".into()),
        );
    }
    let mt = cfg["model_type"].as_str().unwrap_or("unknown").to_string();
    (false, format!("not a Whisper checkpoint (model_type `{mt}`)"), Some(mt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn whisper_wants_mlx_layout() {
        let mlx = json!({"n_mels": 128, "n_audio_state": 1280, "n_text_layer": 4,
                         "quantization": {"bits": 4, "group_size": 64}});
        let (ok, reason, _) = check_whisper(&mlx, &|f| f == "weights.safetensors");
        assert!(ok, "{reason}");
        assert!(reason.contains("4-bit"));

        let hf = json!({"model_type": "whisper"});
        let (ok, reason, _) = check_whisper(&hf, &|_| true);
        assert!(!ok && reason.contains("mlx-community"), "{reason}");
    }

    #[test]
    fn a_foreign_architecture_is_rejected_by_name() {
        let (ok, reason, arch) = check_whisper(&json!({"model_type": "qwen3"}), &|_| false);
        assert!(!ok);
        assert!(reason.contains("qwen3"));
        assert_eq!(arch.as_deref(), Some("qwen3"));
    }
}
