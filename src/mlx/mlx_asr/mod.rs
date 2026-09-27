//! The MLX pieces the daemon still needs: Whisper, and nothing else.
//!
//! This was `mlx_lm` — a whole family of language-model architectures plus the
//! KV cache, sampler and prefill machinery around them. All of that moved to
//! [`apps/mlx-lm`](../../../apps/mlx-lm) when local LLM inference became a Space
//! App. What stayed is the part that was never a language model: the Whisper
//! encoder/decoder, its tokenizer, and the shared error type.
//!
//! Renamed rather than left as `mlx_lm` because the old name is now wrong in a
//! way that matters — someone looking for Gemma or Qwen here should find nothing
//! and go to the app, not find a directory whose name suggests they are missing.

pub mod error;
pub mod whisper;
pub mod whisper_tokenizer;
