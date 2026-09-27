//! Whisper on Candle — the cross-platform ASR backend.
//!
//! Pure Rust on the CPU, which is the whole reason it exists: MLX is Apple
//! Silicon only, and without this a Windows or Linux build of the sidecar could
//! only answer 501. On macOS the MLX backend stays the default (it is several
//! times faster on the same checkpoint); this one is selected everywhere else,
//! or anywhere via `SENCLAW_ASR_BACKEND=candle` — which is also how it gets
//! integration-tested on a Mac.
//!
//! ## Checkpoint format — the trap worth stating up front
//!
//! This backend loads **HuggingFace-layout** Whisper checkpoints
//! (`openai/whisper-*`: `config.json` with `d_model`/`encoder_layers`,
//! `model.safetensors` with `model.encoder.*` keys). The `mlx-community/*`
//! checkpoints the MLX backend uses are a *different serialization* of the same
//! weights — different key names, MLX quantization blocks — and do not load
//! here. The two backends sharing one model directory does not mean they share
//! checkpoints; [`supports_dir`] is what keeps a config mismatch from becoming
//! a load-time stack trace.
//!
//! The audio front-end *is* shared: [`crate::audio`] produces the same
//! whisper-standard log-mel both backends consume — this module only transposes
//! it (our layout is frame-major, Candle wants `[1, n_mels, n_frames]`).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as wsp, model::Whisper};
use tokenizers::Tokenizer;

use crate::audio;

/// Can this backend load the checkpoint in `dir`?
///
/// Answered from `config.json` alone — an HF Whisper config carries `d_model`
/// and `encoder_layers`, an MLX one carries `n_audio_state`-style keys and
/// often a `quantization` block. Cheap enough to call per request.
pub fn supports_dir(dir: &Path) -> bool {
    let Ok(raw) = std::fs::read_to_string(dir.join("config.json")) else {
        return false;
    };
    serde_json::from_str::<wsp::Config>(&raw).is_ok()
        && dir.join("tokenizer.json").exists()
        && (dir.join("model.safetensors").exists()
            || dir.join("model.safetensors.index.json").exists())
}

pub struct CandleWhisper {
    model: Whisper,
    tokenizer: Tokenizer,
    device: Device,
    suppress: Vec<u32>,
    sot: u32,
    eot: u32,
    transcribe: u32,
    no_timestamps: u32,
}

impl CandleWhisper {
    pub fn load(dir: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(dir.join("config.json"))
            .with_context(|| format!("reading {}/config.json", dir.display()))?;
        let config: wsp::Config = serde_json::from_str(&raw).map_err(|e| {
            anyhow!(
                "`{}` is not a Candle-compatible Whisper checkpoint ({e}). This backend \
                 loads HuggingFace-layout repos such as `openai/whisper-large-v3-turbo` \
                 or `openai/whisper-tiny` — the `mlx-community/*` checkpoints are \
                 MLX-only.",
                dir.display()
            )
        })?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("tokenizer.json: {e}"))?;

        let weights = weight_files(dir)?;
        let device = Device::Cpu;
        // SAFETY: mmap of files we just enumerated; nothing writes them while
        // the model is alive.
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&weights, wsp::DTYPE, &device)? };
        let suppress = config.suppress_tokens.clone();
        let model = Whisper::load(&vb, config).context("loading Whisper weights")?;

        let tok = |name: &str| -> Result<u32> {
            tokenizer
                .token_to_id(name)
                .ok_or_else(|| anyhow!("tokenizer has no `{name}` token"))
        };
        Ok(Self {
            sot: tok(wsp::SOT_TOKEN)?,
            eot: tok(wsp::EOT_TOKEN)?,
            transcribe: tok(wsp::TRANSCRIBE_TOKEN)?,
            no_timestamps: tok(wsp::NO_TIMESTAMPS_TOKEN)?,
            suppress,
            model,
            tokenizer,
            device,
        })
    }

    /// Transcribe a whole file. Mirrors the MLX engine's contract: decode any
    /// common container, chunk into 30 s windows, greedy-decode each, join.
    pub fn transcribe_file(&mut self, path: impl AsRef<Path>, language: Option<&str>) -> Result<String> {
        let pcm = audio::load_audio(path)?;
        if pcm.is_empty() {
            return Ok(String::new());
        }
        let n_mels = self.model.config.num_mel_bins;
        let mel = audio::log_mel_spectrogram(&pcm, n_mels, 0)?;

        // The language token is positional in Whisper's prompt, so it cannot be
        // skipped for a multilingual model. `<|vi|>`-style ids; an unknown code
        // is an input error, not a fallback to silence.
        let lang_token = match language.map(str::trim).filter(|l| !l.is_empty()) {
            Some(l) => Some(
                self.tokenizer
                    .token_to_id(&format!("<|{l}|>"))
                    .ok_or_else(|| anyhow!("model has no language token for `{l}`"))?,
            ),
            None => None,
        };

        let mut out = String::new();
        let mut frame = 0usize;
        while frame < mel.n_frames {
            let take = (mel.n_frames - frame).min(audio::N_FRAMES);
            let chunk = self.chunk_tensor(&mel, frame, take, n_mels)?;
            let text = self.decode_chunk(&chunk, lang_token)?;
            if !text.trim().is_empty() {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(text.trim());
            }
            frame += take;
        }
        Ok(out)
    }

    /// Our mel is frame-major `[n_frames][n_mels]`; Candle wants
    /// `[1, n_mels, n_frames]`, zero-padded to the full 30 s window (the
    /// encoder's positional embedding is sized for exactly `N_FRAMES`).
    fn chunk_tensor(
        &self,
        mel: &audio::MelSpectrogram,
        start: usize,
        take: usize,
        n_mels: usize,
    ) -> Result<Tensor> {
        let mut data = vec![0f32; n_mels * audio::N_FRAMES];
        for f in 0..take {
            let src = mel.frame(start + f);
            for (m, v) in src.iter().enumerate() {
                data[m * audio::N_FRAMES + f] = *v;
            }
        }
        Ok(Tensor::from_vec(data, (1, n_mels, audio::N_FRAMES), &self.device)?)
    }

    fn decode_chunk(&mut self, mel: &Tensor, lang_token: Option<u32>) -> Result<String> {
        let features = self.model.encoder.forward(mel, true)?;

        let mut tokens = vec![self.sot];
        if let Some(l) = lang_token {
            tokens.push(l);
        }
        tokens.push(self.transcribe);
        tokens.push(self.no_timestamps);

        let max_len = self.model.config.max_target_positions / 2;
        for i in 0.. {
            if tokens.len() >= max_len {
                break;
            }
            // The whole sequence every step, not just the newest token: this
            // model's kv_cache covers only **cross**-attention (the encoder
            // K/V projection, cleared by the flush on step 0); self-attention
            // recomputes from the input, so a one-token input would be decoded
            // as position zero with no history. Quadratic per chunk, but the
            // text budget is 224 tokens — negligible next to the encoder.
            let t = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let logits = self
                .model
                .decoder
                .forward(&t, &features, i == 0)?;
            let (_, seq, _) = logits.dims3()?;
            let logits = self
                .model
                .decoder
                .final_linear(&logits.i((.., seq - 1..seq))?)?
                .i(0)?
                .i(0)?;

            let mut logits: Vec<f32> = logits.to_vec1()?;
            for &s in &self.suppress {
                if let Some(v) = logits.get_mut(s as usize) {
                    *v = f32::NEG_INFINITY;
                }
            }
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .ok_or_else(|| anyhow!("empty logits"))?;
            if next == self.eot {
                break;
            }
            tokens.push(next);
        }

        // Strip the prompt; keep only generated text tokens.
        let prompt_len = 2 + lang_token.is_some() as usize + 1;
        let text = self
            .tokenizer
            .decode(&tokens[prompt_len.min(tokens.len())..], true)
            .map_err(|e| anyhow!("decoding tokens: {e}"))?;
        Ok(text)
    }
}

fn weight_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let raw = std::fs::read_to_string(&index)?;
        let v: serde_json::Value = serde_json::from_str(&raw)?;
        let mut shards: Vec<String> = v["weight_map"]
            .as_object()
            .map(|m| m.values().filter_map(|s| s.as_str().map(String::from)).collect())
            .unwrap_or_default();
        shards.sort();
        shards.dedup();
        if !shards.is_empty() {
            return Ok(shards.into_iter().map(|s| dir.join(s)).collect());
        }
    }
    bail!("no model.safetensors in {}", dir.display())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The config check is what turns "wrong checkpoint family" into a sentence
    /// instead of a key-mismatch stack trace at load time.
    #[test]
    fn an_mlx_checkpoint_is_recognised_as_not_ours() {
        let dir = tempfile::tempdir().unwrap();
        // Shape of mlx-community whisper configs: n_mels + quantization, none
        // of the HF field names.
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"n_mels":128,"n_audio_state":1280,"quantization":{"bits":4}}"#,
        )
        .unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        std::fs::write(dir.path().join("model.safetensors"), b"x").unwrap();
        assert!(!supports_dir(dir.path()));
    }

    #[test]
    fn an_hf_checkpoint_is_recognised() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"num_mel_bins":80,"max_source_positions":1500,"d_model":384,
                "encoder_attention_heads":6,"encoder_layers":4,"vocab_size":51865,
                "max_target_positions":448,"decoder_attention_heads":6,"decoder_layers":4}"#,
        )
        .unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        std::fs::write(dir.path().join("model.safetensors"), b"x").unwrap();
        assert!(supports_dir(dir.path()));
    }

    #[test]
    fn missing_weights_or_tokenizer_fail_the_support_check() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"num_mel_bins":80,"max_source_positions":1500,"d_model":384,
                "encoder_attention_heads":6,"encoder_layers":4,"vocab_size":51865,
                "max_target_positions":448,"decoder_attention_heads":6,"decoder_layers":4}"#,
        )
        .unwrap();
        assert!(!supports_dir(dir.path()), "no tokenizer, no weights");
    }
}
