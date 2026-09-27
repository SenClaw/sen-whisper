//! One lock for every MLX user in this process.
//!
//! Concurrent MLX work on separate threads corrupts Metal state and SIGSEGVs.
//! This used to guard the local LLM engine against Whisper and the TTS backends;
//! the LLM has since moved to [`apps/mlx-lm`](../../../apps/mlx-lm), which owns
//! its own copy of this lock in its own process. What is left here still needs
//! it, because there is still more than one of them:
//!
//! - Whisper ASR ([`super::whisper_transcribe`])
//! - MMS-VITS and ZipVoice TTS (this app's `zipvoice` / `mms_vits`)
//! - The cognitive MLX embedder
//!
//! The lock is per *process*, which is the boundary Metal state is corrupted
//! across. Whether two **processes** — this daemon and the engine app — can
//! safely drive Metal at the same time is a separate question, and one nothing
//! in this repo has measured.

use std::sync::Mutex;

static MLX_SERIAL: Mutex<()> = Mutex::new(());

/// Acquire the lock, recovering from poisoning.
///
/// A prior MLX panic must not wedge every future call — the guarded state is
/// Metal's, not ours, and refusing to proceed would take ASR and TTS down for
/// the rest of the process's life. Blocks until it is this caller's turn, so
/// use it only from blocking / `spawn_blocking` contexts.
pub fn lock() -> std::sync::MutexGuard<'static, ()> {
    MLX_SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Non-blocking acquire — `None` when MLX is busy.
///
/// For the async-invoked free paths (idle eviction, cache release), which must
/// neither block a tokio worker behind a long generation nor free Metal buffers
/// while one is running. The caller skips and retries on the next pass.
pub fn try_lock() -> Option<std::sync::MutexGuard<'static, ()>> {
    match MLX_SERIAL.try_lock() {
        Ok(g) => Some(g),
        Err(std::sync::TryLockError::WouldBlock) => None,
        Err(std::sync::TryLockError::Poisoned(g)) => Some(g.into_inner()),
    }
}
