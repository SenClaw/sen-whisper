//! Whisper model management: catalog, on-disk layout, and composite downloads.
//!
//! Ported from the daemon's old `src/gateway/ui_server/whisper.rs`. Whisper
//! checkpoints on mlx-community ship no tokenizer, so a download is
//! **composite**: weights + config from the mlx repo, `tokenizer.json` from the
//! paired `openai/whisper-*` repo, assembled into one model directory.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

use sen_runtime_sdk::env::LaunchEnv;

use crate::api_error::ApiError;
use crate::AppState;

const HF_BASE: &str = "https://huggingface.co";

/// A curated Whisper model: where to fetch weights, and where to borrow the
/// tokenizer.json from (mlx-community repos don't ship one).
struct CatalogEntry {
    /// Public id (also the weights HF repo) — what the caller passes back.
    id: &'static str,
    label: &'static str,
    approx_size_gb: f32,
    /// Repo to pull `tokenizer.json` from (transformers layout).
    tokenizer_repo: &'static str,
    /// Whisper language code default for this model.
    default_language: &'static str,
}

static CATALOG: &[CatalogEntry] = &[
    CatalogEntry {
        id: "mlx-community/whisper-large-v3-turbo",
        label: "Whisper large-v3-turbo (MLX, 128-mel, fast multilingual)",
        approx_size_gb: 1.6,
        tokenizer_repo: "openai/whisper-large-v3-turbo",
        default_language: "vi",
    },
    CatalogEntry {
        id: "mlx-community/whisper-large-v3-turbo-4bit",
        label: "Whisper large-v3-turbo 4-bit (MLX, smaller/faster download)",
        approx_size_gb: 0.46,
        tokenizer_repo: "openai/whisper-large-v3-turbo",
        default_language: "vi",
    },
    // ── HF-layout checkpoints, for the Candle backend ────────────────────────
    // The two decoders read different serializations of the same weights: MLX
    // loads the `mlx-community/*` entries above, Candle (every non-mac
    // platform) loads these. A Windows build offering only the MLX entries
    // would download a checkpoint its own backend cannot open.
    CatalogEntry {
        id: "openai/whisper-large-v3-turbo",
        label: "Whisper large-v3-turbo (HF — for Windows/Linux, CPU)",
        approx_size_gb: 1.6,
        tokenizer_repo: "openai/whisper-large-v3-turbo",
        default_language: "vi",
    },
    CatalogEntry {
        id: "openai/whisper-tiny",
        label: "Whisper tiny (HF — small & fast, lower accuracy)",
        approx_size_gb: 0.15,
        tokenizer_repo: "openai/whisper-tiny",
        default_language: "vi",
    },
];

fn catalog_get(id: &str) -> Option<&'static CatalogEntry> {
    CATALOG.iter().find(|e| e.id == id)
}

/// Whether this build's backend can open a catalog entry: the
/// `mlx-community/*` serialization needs the MLX decoder, which only a macOS
/// build has. Off macOS those entries would download a checkpoint Candle
/// cannot load, so they are not offered there.
fn offered_here(id: &str) -> bool {
    cfg!(target_os = "macos") || !id.starts_with("mlx-community/")
}

/// The catalog entries this build can transcribe with.
fn offered_catalog() -> impl Iterator<Item = &'static CatalogEntry> {
    CATALOG.iter().filter(|e| offered_here(e.id))
}

pub fn safe_dirname(id: &str) -> String {
    id.replace('/', "__")
}

fn unsafe_dirname(name: &str) -> Option<String> {
    let (org, repo) = name.split_once("__")?;
    if org.is_empty() || repo.is_empty() {
        return None;
    }
    Some(format!("{org}/{repo}"))
}

/// `SENCLAW_WHISPER_MODELS_DIR`, else `<SENCLAW_HOME>/whisper-models` — the
/// same default the in-daemon engine resolved (runtime protocol §6.1).
pub fn whisper_models_dir(env: &LaunchEnv) -> PathBuf {
    std::env::var("SENCLAW_WHISPER_MODELS_DIR")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| env.home.join("whisper-models"))
}

fn model_dir(env: &LaunchEnv, id: &str) -> PathBuf {
    whisper_models_dir(env).join(safe_dirname(id))
}

/// Before `whisper-models/` was its own directory, Whisper checkpoints landed
/// in the shared `local-models/` root beside LLMs. Existing installs there are
/// never re-downloaded.
fn legacy_model_dir(env: &LaunchEnv, id: &str) -> PathBuf {
    env.models_dir.join(safe_dirname(id))
}

pub fn installed_model_dir(env: &LaunchEnv, id: &str) -> PathBuf {
    let dir = model_dir(env, id);
    if is_installed(&dir) {
        return dir;
    }
    let legacy = legacy_model_dir(env, id);
    if is_installed(&legacy) {
        legacy
    } else {
        dir
    }
}

fn is_whisper_model_dir(dir: &Path) -> bool {
    let Ok(file) = std::fs::File::open(dir.join("config.json")) else {
        return false;
    };
    let Ok(cfg) = serde_json::from_reader::<_, serde_json::Value>(file) else {
        return false;
    };
    ["n_mels", "n_audio_ctx", "n_audio_state", "n_audio_layer", "n_text_ctx", "n_text_state", "n_text_layer", "n_vocab"]
        .iter()
        .all(|k| cfg.get(*k).and_then(|v| v.as_i64()).is_some())
}

/// A Whisper dir is "installed" once Whisper config + weights + tokenizer are present.
pub fn is_installed(dir: &Path) -> bool {
    is_whisper_model_dir(dir)
        && (dir.join("weights.safetensors").exists() || dir.join("model.safetensors").exists())
        && dir.join("tokenizer.json").exists()
}

/// Normalize a HuggingFace `org/repo` id from a bare id or full URL. Shared by
/// the download and validate routes.
pub fn normalize_hf_id(raw: &str) -> Result<String, String> {
    let s = raw.trim();
    if s.is_empty() {
        return Err("empty model id".into());
    }
    let stripped = s
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_start_matches("huggingface.co/")
        .trim_start_matches("hf.co/")
        .trim_end_matches('/');
    let parts: Vec<&str> = stripped.split('/').collect();
    if parts.len() < 2 {
        return Err(format!("expected `org/repo` form, got `{s}`"));
    }
    let (org, repo) = (parts[0], parts[1]);
    if org.is_empty() || repo.is_empty() {
        return Err(format!("invalid `org/repo` in `{s}`"));
    }
    for seg in [org, repo] {
        if seg.contains("..") || seg.contains('\\') {
            return Err(format!("unsafe path segment in `{s}`"));
        }
    }
    Ok(format!("{org}/{repo}"))
}

fn infer_tokenizer_repo(weights_repo: &str) -> String {
    let repo_name = weights_repo.split('/').next_back().unwrap_or(weights_repo).trim();
    let repo_name = repo_name
        .strip_suffix("-4bit")
        .or_else(|| repo_name.strip_suffix("-8bit"))
        .or_else(|| repo_name.strip_suffix("-fp16"))
        .or_else(|| repo_name.strip_suffix("-bf16"))
        .unwrap_or(repo_name);
    if repo_name.starts_with("whisper-") {
        format!("openai/{repo_name}")
    } else {
        "openai/whisper-large-v3-turbo".to_string()
    }
}

// ── Download progress (process-global) ───────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DownloadStatus {
    Queued,
    Listing,
    Downloading,
    Done,
    Error,
    Cancelled,
}

#[derive(Debug, Clone, Serialize)]
struct DownloadState {
    model_id: String,
    status: DownloadStatus,
    total_bytes: u64,
    downloaded_bytes: u64,
    current_file: Option<String>,
    files_total: u32,
    files_done: u32,
    error: Option<String>,
}

#[derive(Clone)]
struct DownloadHandle {
    state: Arc<Mutex<DownloadState>>,
    cancel: CancellationToken,
}

fn downloads() -> &'static Mutex<HashMap<String, DownloadHandle>> {
    static DOWNLOADS: OnceLock<Mutex<HashMap<String, DownloadHandle>>> = OnceLock::new();
    DOWNLOADS.get_or_init(|| Mutex::new(HashMap::new()))
}

// ── Routes: model listing ────────────────────────────────────────────────────

pub async fn list(State(state): State<Arc<AppState>>) -> Result<impl IntoResponse, ApiError> {
    let downloads = downloads().lock().unwrap();
    let mut models = Vec::new();
    for e in offered_catalog() {
        let dir = installed_model_dir(&state.env, e.id);
        let download = downloads.get(e.id).map(|h| h.state.lock().unwrap().clone());
        models.push(json!({
            "id": e.id,
            "label": e.label,
            "approx_size_gb": e.approx_size_gb,
            "default_language": e.default_language,
            "installed": is_installed(&dir),
            "on_disk_path": dir.to_string_lossy(),
            "download": download,
        }));
    }
    for root in [whisper_models_dir(&state.env), state.env.models_dir.clone()] {
        if let Ok(entries) = std::fs::read_dir(&root) {
            for entry in entries.flatten() {
                let Ok(file_type) = entry.file_type() else {
                    continue;
                };
                if !file_type.is_dir() {
                    continue;
                }
                let name = entry.file_name().to_string_lossy().to_string();
                let Some(id) = unsafe_dirname(&name) else {
                    continue;
                };
                if catalog_get(&id).is_some() || models.iter().any(|m| m["id"] == id) {
                    continue;
                }
                let dir = entry.path();
                let download = downloads.get(&id).map(|h| h.state.lock().unwrap().clone());
                let allow_legacy = root == whisper_models_dir(&state.env) || is_whisper_model_dir(&dir) || download.is_some();
                if allow_legacy && (is_installed(&dir) || download.is_some()) {
                    models.push(json!({
                        "id": id,
                        "label": format!("Whisper custom ({id})"),
                        "approx_size_gb": 0.0,
                        "default_language": "vi",
                        "installed": is_installed(&dir),
                        "on_disk_path": dir.to_string_lossy(),
                        "download": download,
                    }));
                }
            }
        }
    }
    for (id, handle) in downloads.iter() {
        if catalog_get(id).is_some() || models.iter().any(|m| m["id"] == *id) {
            continue;
        }
        let dir = installed_model_dir(&state.env, id);
        models.push(json!({
            "id": id,
            "label": format!("Whisper custom ({id})"),
            "approx_size_gb": 0.0,
            "default_language": "vi",
            "installed": is_installed(&dir),
            "on_disk_path": dir.to_string_lossy(),
            "download": handle.state.lock().unwrap().clone(),
        }));
    }
    Ok(Json(json!({ "models": models })))
}

// ── Routes: download (composite) ─────────────────────────────────────────────

pub async fn download(State(state): State<Arc<AppState>>, AxumPath(id): AxumPath<String>) -> Result<impl IntoResponse, ApiError> {
    let id = normalize_hf_id(&id).map_err(|e| ApiError(StatusCode::BAD_REQUEST, e))?;
    let tokenizer_repo = catalog_get(&id).map(|e| e.tokenizer_repo.to_string()).unwrap_or_else(|| infer_tokenizer_repo(&id));

    {
        let downloads = downloads().lock().unwrap();
        if let Some(h) = downloads.get(&id) {
            let s = h.state.lock().unwrap();
            if matches!(s.status, DownloadStatus::Queued | DownloadStatus::Listing | DownloadStatus::Downloading) {
                return Err(ApiError(StatusCode::CONFLICT, format!("download for {id} already in progress")));
            }
        }
    }

    let dir = model_dir(&state.env, &id);
    tokio::fs::create_dir_all(&dir).await.map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let progress = Arc::new(Mutex::new(DownloadState {
        model_id: id.clone(),
        status: DownloadStatus::Queued,
        total_bytes: 0,
        downloaded_bytes: 0,
        current_file: None,
        files_total: 0,
        files_done: 0,
        error: None,
    }));
    let cancel = CancellationToken::new();
    downloads().lock().unwrap().insert(id.clone(), DownloadHandle { state: progress.clone(), cancel: cancel.clone() });

    let weights_repo = id.clone();
    tokio::spawn(async move {
        let result = run_download(&weights_repo, &tokenizer_repo, &dir, progress.clone(), cancel).await;
        let mut s = progress.lock().unwrap();
        match result {
            Ok(()) if s.status != DownloadStatus::Cancelled => s.status = DownloadStatus::Done,
            Ok(()) => {}
            Err(e) => {
                s.status = DownloadStatus::Error;
                s.error = Some(e.to_string());
            }
        }
    });

    Ok(Json(json!({ "ok": true, "id": id })))
}

pub async fn status(AxumPath(id): AxumPath<String>) -> Result<impl IntoResponse, ApiError> {
    let downloads = downloads().lock().unwrap();
    let progress = downloads.get(&id).map(|h| h.state.lock().unwrap().clone());
    Ok(Json(json!({ "id": id, "download": progress })))
}

pub async fn cancel(AxumPath(id): AxumPath<String>) -> Result<impl IntoResponse, ApiError> {
    let downloads = downloads().lock().unwrap();
    if let Some(h) = downloads.get(&id) {
        h.cancel.cancel();
        h.state.lock().unwrap().status = DownloadStatus::Cancelled;
    }
    Ok(Json(json!({ "ok": true })))
}

pub async fn delete(State(state): State<Arc<AppState>>, AxumPath(id): AxumPath<String>) -> Result<impl IntoResponse, ApiError> {
    let dir = model_dir(&state.env, &id);
    if dir.exists() {
        tokio::fs::remove_dir_all(&dir).await.map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }
    let legacy = legacy_model_dir(&state.env, &id);
    if legacy.exists() {
        tokio::fs::remove_dir_all(&legacy).await.map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }
    downloads().lock().unwrap().remove(&id);
    Ok(Json(json!({ "ok": true })))
}

// ── Routes: settings ─────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct SettingsBody {
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    language: Option<String>,
}

pub async fn settings_get(State(state): State<Arc<AppState>>) -> Result<impl IntoResponse, ApiError> {
    let s = crate::settings::load(&state.env.data_dir, &state.env.config_path);
    Ok(Json(json!({ "model_id": s.model_id, "language": s.language.unwrap_or_else(|| "vi".to_string()) })))
}

pub async fn settings_put(State(state): State<Arc<AppState>>, Json(body): Json<SettingsBody>) -> Result<impl IntoResponse, ApiError> {
    let settings = crate::settings::Settings { model_id: body.model_id, language: body.language };
    crate::settings::save(&state.env.data_dir, &settings).map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(json!({ "ok": true })))
}

/// The model to transcribe with when a request does not name one: the user's
/// selection, else the first installed catalog model. Shared by both
/// transcription routes.
pub fn selected_model(state: &AppState) -> Result<String, ApiError> {
    let settings = crate::settings::load(&state.env.data_dir, &state.env.config_path);
    settings
        .model_id
        .clone()
        .or_else(|| offered_catalog().map(|e| e.id.to_string()).find(|id| is_installed(&installed_model_dir(&state.env, id))))
        .ok_or_else(|| ApiError(StatusCode::BAD_REQUEST, "no Whisper model selected or installed".into()))
}

pub fn default_language(state: &AppState) -> Option<String> {
    crate::settings::load(&state.env.data_dir, &state.env.config_path).language
}

// ── Composite download worker ────────────────────────────────────────────────

#[derive(Deserialize)]
struct HfTreeEntry {
    #[serde(rename = "type")]
    entry_type: String,
    path: String,
    #[serde(default)]
    size: u64,
}

fn should_skip(name: &str) -> bool {
    let lower = name.to_lowercase();
    matches!(lower.as_str(), ".gitattributes" | "readme.md" | "license" | "license.md" | "license.txt")
        || lower.ends_with(".png")
        || lower.ends_with(".jpg")
        || lower.ends_with(".jpeg")
        || lower.ends_with(".gif")
        || lower.ends_with(".svg")
}

async fn run_download(
    weights_repo: &str,
    tokenizer_repo: &str,
    dir: &PathBuf,
    progress: Arc<Mutex<DownloadState>>,
    cancel: CancellationToken,
) -> anyhow::Result<()> {
    let client = reqwest::Client::builder().connect_timeout(std::time::Duration::from_secs(30)).build()?;

    progress.lock().unwrap().status = DownloadStatus::Listing;

    let tree_url = format!("{HF_BASE}/api/models/{weights_repo}/tree/main?recursive=true");
    let tree: Vec<HfTreeEntry> = client.get(&tree_url).send().await?.error_for_status()?.json().await?;

    let mut files: Vec<(String, String, u64)> =
        tree.into_iter().filter(|e| e.entry_type == "file" && !should_skip(&e.path)).map(|e| (weights_repo.to_string(), e.path, e.size)).collect();
    // Append the tokenizer from the paired repo (size unknown -> 0).
    files.push((tokenizer_repo.to_string(), "tokenizer.json".to_string(), 0));

    {
        let mut s = progress.lock().unwrap();
        s.files_total = files.len() as u32;
        s.total_bytes = files.iter().map(|f| f.2).sum();
        s.status = DownloadStatus::Downloading;
    }

    for (repo, path, size) in files {
        if cancel.is_cancelled() {
            progress.lock().unwrap().status = DownloadStatus::Cancelled;
            return Ok(());
        }
        progress.lock().unwrap().current_file = Some(path.clone());

        let dst = dir.join(&path);
        if let Some(parent) = dst.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        // Resume: skip if a complete copy exists.
        if size > 0 {
            if let Ok(meta) = tokio::fs::metadata(&dst).await {
                if meta.len() == size {
                    let mut s = progress.lock().unwrap();
                    s.files_done += 1;
                    s.downloaded_bytes += size;
                    continue;
                }
            }
        }

        let url = format!("{HF_BASE}/{repo}/resolve/main/{path}");
        let resp = client.get(&url).send().await?.error_for_status()?;
        let mut stream = resp.bytes_stream();
        let mut file = tokio::fs::File::create(&dst).await?;
        while let Some(chunk) = stream.next().await {
            if cancel.is_cancelled() {
                drop(file);
                let _ = tokio::fs::remove_file(&dst).await;
                progress.lock().unwrap().status = DownloadStatus::Cancelled;
                return Ok(());
            }
            let bytes = chunk?;
            file.write_all(&bytes).await?;
            progress.lock().unwrap().downloaded_bytes += bytes.len() as u64;
        }
        file.flush().await?;
        progress.lock().unwrap().files_done += 1;
    }

    Ok(())
}

#[cfg(test)]
mod tests {

    #[test]
    fn mlx_checkpoints_are_offered_only_where_the_mlx_decoder_exists() {
        assert!(offered_here("openai/whisper-tiny"));
        assert_eq!(offered_here("mlx-community/whisper-large-v3-turbo"), cfg!(target_os = "macos"));
        assert!(offered_catalog().any(|e| e.id.starts_with("openai/")), "every build offers a checkpoint it can open");
    }

    use super::*;

    #[test]
    fn dirname_round_trips_and_rejects_foreign_names() {
        let id = "mlx-community/whisper-large-v3-turbo-4bit";
        assert_eq!(unsafe_dirname(&safe_dirname(id)).as_deref(), Some(id));
        assert_eq!(unsafe_dirname("hf-cache"), None);
    }

    #[test]
    fn normalizes_urls_and_rejects_garbage() {
        assert_eq!(normalize_hf_id("https://huggingface.co/openai/whisper-tiny/").unwrap(), "openai/whisper-tiny");
        assert!(normalize_hf_id("nonsense").is_err());
        assert!(normalize_hf_id("a/../b").is_err());
    }

    #[test]
    fn infers_the_paired_tokenizer_repo_from_the_weights_repo_name() {
        assert_eq!(infer_tokenizer_repo("mlx-community/whisper-large-v3-turbo-4bit"), "openai/whisper-large-v3-turbo");
        assert_eq!(infer_tokenizer_repo("mlx-community/some-other-model"), "openai/whisper-large-v3-turbo");
    }

    #[test]
    fn a_config_without_weights_or_tokenizer_is_not_installed() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"n_mels":128,"n_audio_ctx":1,"n_audio_state":1,"n_audio_layer":1,"n_text_ctx":1,"n_text_state":1,"n_text_layer":1,"n_vocab":1}"#,
        )
        .unwrap();
        assert!(!is_installed(dir.path()), "no weights, no tokenizer yet");
        std::fs::write(dir.path().join("model.safetensors"), b"x").unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
        assert!(is_installed(dir.path()));
    }

    #[test]
    fn a_non_whisper_config_is_never_installed() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"model_type":"qwen3"}"#).unwrap();
        std::fs::write(dir.path().join("model.safetensors"), b"x").unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
        assert!(!is_installed(dir.path()));
    }

    /// A legacy install under the shared `local-models` root must still be
    /// found — nothing already on disk should be treated as missing.
    #[test]
    fn installed_model_dir_falls_back_to_the_legacy_shared_root() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("home");
        let models_dir = tmp.path().join("local-models");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::create_dir_all(&models_dir).unwrap();
        let env = LaunchEnv::from_lookup("sen-whisper", "0.1.0", {
            let home = home.clone();
            let models_dir = models_dir.clone();
            move |k| match k {
                "SENCLAW_HOME" => Some(home.to_string_lossy().into_owned()),
                "SENCLAW_LOCAL_MODELS_DIR" => Some(models_dir.to_string_lossy().into_owned()),
                _ => None,
            }
        });
        let id = "mlx-community/whisper-large-v3-turbo-4bit";
        let legacy = legacy_model_dir(&env, id);
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(
            legacy.join("config.json"),
            r#"{"n_mels":128,"n_audio_ctx":1,"n_audio_state":1,"n_audio_layer":1,"n_text_ctx":1,"n_text_state":1,"n_text_layer":1,"n_vocab":1}"#,
        )
        .unwrap();
        std::fs::write(legacy.join("weights.safetensors"), b"x").unwrap();
        std::fs::write(legacy.join("tokenizer.json"), b"{}").unwrap();

        assert_eq!(installed_model_dir(&env, id), legacy);
    }
}
