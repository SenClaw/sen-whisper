//! Whisper ASR settings: selected model id + default language.
//!
//! Before the split these lived in the daemon's `config.json["whisperConfig"]`
//! (camelCase, `WhisperSettings` in `gateway/group_manager/types.rs`). This
//! runtime owns them now, in `<SENCLAW_RUNTIME_DATA_DIR>/settings.json`, seeded
//! from the old key on first start via [`sen_runtime_sdk::legacy::load_or_import`]
//! so an upgraded machine keeps the user's choice of model and language.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Same field names and casing as the daemon's old `WhisperSettings` — an
/// upgraded machine's `config.json` must parse into this unchanged.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Settings {
    #[serde(rename = "modelId", default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

pub fn settings_path(data_dir: &Path) -> PathBuf {
    data_dir.join("settings.json")
}

/// Read the stored settings, importing from the daemon's legacy `config.json`
/// the first time this runtime starts against a given `data_dir`. Any failure
/// (unreadable file, bad JSON, no legacy key) reads as defaults rather than an
/// error — a broken settings file must not stop transcription.
pub fn load(data_dir: &Path, config_path: &Path) -> Settings {
    sen_runtime_sdk::legacy::load_or_import(data_dir, config_path, "whisperConfig")
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_default()
}

pub fn save(data_dir: &Path, s: &Settings) -> std::io::Result<()> {
    std::fs::create_dir_all(data_dir)?;
    let body = serde_json::to_vec_pretty(s)?;
    // Write-then-rename: a reader on the transcribe path sees either the old
    // settings or the new ones, never a truncated file.
    let path = settings_path(data_dir);
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &body)?;
    std::fs::rename(&tmp, &path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imports_the_legacy_whisper_config_once_then_owns_its_own_file() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        std::fs::write(&config, r#"{"whisperConfig": {"modelId": "openai/whisper-tiny", "language": "vi"}}"#).unwrap();
        let data = dir.path().join("data");

        let first = load(&data, &config);
        assert_eq!(first.model_id.as_deref(), Some("openai/whisper-tiny"));
        assert_eq!(first.language.as_deref(), Some("vi"));
        assert!(data.join("settings.json").is_file());

        // The legacy file changing afterwards must not leak in.
        std::fs::write(&config, r#"{"whisperConfig": {"modelId": "openai/whisper-large-v3-turbo"}}"#).unwrap();
        let second = load(&data, &config);
        assert_eq!(second.model_id.as_deref(), Some("openai/whisper-tiny"));
    }

    #[test]
    fn missing_legacy_config_is_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let s = load(&dir.path().join("data"), &dir.path().join("nope.json"));
        assert!(s.model_id.is_none());
        assert!(s.language.is_none());
    }

    #[test]
    fn round_trips_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let s = Settings { model_id: Some("openai/whisper-tiny".into()), language: Some("en".into()) };
        save(dir.path(), &s).unwrap();
        let raw = std::fs::read_to_string(settings_path(dir.path())).unwrap();
        assert!(raw.contains("\"modelId\""), "must stay camelCase: {raw}");
        let back: Settings = serde_json::from_str(&raw).unwrap();
        assert_eq!(back.model_id.as_deref(), Some("openai/whisper-tiny"));
        assert_eq!(back.language.as_deref(), Some("en"));
    }
}
