//! The MLX speech runtime — Apple Silicon only.
//!
//! Everything in here is behind `#[cfg(target_os = "macos")]` at the point of
//! declaration in `main.rs`, so a Linux or Windows build of this sidecar
//! compiles none of it and pulls in none of `mlx-rs`, `mlx-sys`, `symphonia`,
//! `rubato` or `realfft`. That is the point of the split: the media sidecar
//! ships everywhere, and only picks up the heavy platform-specific stack where
//! it can actually run.
//!
//! Whisper's tokenizer and encoder/decoder came from the daemon's old
//! `src/local_model/`; the MLX TTS voices live one level up under
//! [`crate::tts`], because from the dispatcher's point of view they are just
//! two more backends.

pub mod mlx_asr;
pub mod mlx_serial;
pub mod whisper_transcribe;

pub use whisper_transcribe::WhisperEngine;
