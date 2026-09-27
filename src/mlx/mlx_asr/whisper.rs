//! OpenAI Whisper encoder/decoder ASR, ported to `mlx-rs`.
//!
//! Faithful port of `mlx_whisper/whisper.py` (Apple's mlx-examples). Targets
//! the `mlx-community/whisper-large-v3-turbo` checkpoint (n_mels=128, 32 encoder
//! / 4 decoder layers) but is dimension-driven so any Whisper variant loads.
//!
//! Layout notes that matter for correctness:
//! - `mlx_rs::nn::Conv1d` is channels-last (`[N, L, C]`) with weight
//!   `[out, kernel, in]` — identical to `mlx.nn.Conv1d`, so the mel produced by
//!   [`crate::audio`] (`[1, n_frames, n_mels]`) feeds straight in.
//! - The output projection is the **tied** token-embedding matrix
//!   (`Embedding::as_linear`), as in qwen3.rs.
//! - Attention scale `(n_state/n_head)^-0.25` is applied to **both** q and k.
//! - Self-attention KV cache grows by concat; cross-attention KV is computed
//!   once from the audio features and frozen for the whole decode.
//!
//! Weight keys are matched by struct-field path via `load_safetensors`, so the
//! field names below mirror the checkpoint exactly (`encoder.conv1`,
//! `decoder.blocks.{i}.cross_attn.query`, `decoder.positional_embedding`, …).

use std::path::Path;

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParameters, Param},
    nn,
    ops::{self, concatenate_axis, indexing::IndexOp, softmax_axis},
    quantization::MaybeQuantized,
    Array, Dtype,
};
use serde::Deserialize;

use super::error::Error;

/// Whisper `config.json` (mlx-examples layout).
#[derive(Debug, Clone, Deserialize)]
pub struct ModelDimensions {
    pub n_mels: i32,
    pub n_audio_ctx: i32,
    pub n_audio_state: i32,
    pub n_audio_head: i32,
    pub n_audio_layer: i32,
    pub n_vocab: i32,
    pub n_text_ctx: i32,
    pub n_text_state: i32,
    pub n_text_head: i32,
    pub n_text_layer: i32,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

/// MLX quantization parameters from `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

/// Per-layer KV cache slot. Self-attention grows it by concat; cross-attention
/// fills it once (compute-once) and reuses it across every decode step.
#[derive(Debug, Clone, Default)]
pub struct KvCache {
    pub k: Option<Array>,
    pub v: Option<Array>,
}

impl KvCache {
    pub fn empty() -> Self {
        Self { k: None, v: None }
    }
}

// ── Attention ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct MultiHeadAttention {
    pub n_head: i32,
    pub scale: f32,
    #[quantizable]
    #[param]
    pub query: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub key: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub value: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub out: MaybeQuantized<nn::Linear>,
}

impl MultiHeadAttention {
    fn new(n_state: i32, n_head: i32) -> Result<Self, Exception> {
        let query = MaybeQuantized::Original(nn::LinearBuilder::new(n_state, n_state).build()?);
        // Whisper's key projection has NO bias.
        let key = MaybeQuantized::Original(
            nn::LinearBuilder::new(n_state, n_state)
                .bias(false)
                .build()?,
        );
        let value = MaybeQuantized::Original(nn::LinearBuilder::new(n_state, n_state).build()?);
        let out = MaybeQuantized::Original(nn::LinearBuilder::new(n_state, n_state).build()?);
        let scale = ((n_state / n_head) as f32).powf(-0.25);
        Ok(Self {
            n_head,
            scale,
            query,
            key,
            value,
            out,
        })
    }

    /// `x`: query input `[B, Lq, n_state]`. `xa`: audio features for
    /// cross-attention (`None` for self-attention). `mask`: additive `[Lq, Lk]`.
    fn forward(
        &mut self,
        x: &Array,
        xa: Option<&Array>,
        mask: Option<&Array>,
        cache: Option<&mut KvCache>,
    ) -> Result<Array, Exception> {
        let q = self
            .query
            .forward(x)?
            .multiply(Array::from_f32(self.scale))?;

        let (k, v) = match xa {
            // Self-attention: project x, optionally append to the growing cache.
            None => {
                let mut k = self.key.forward(x)?.multiply(Array::from_f32(self.scale))?;
                let mut v = self.value.forward(x)?;
                if let Some(c) = cache {
                    if let (Some(pk), Some(pv)) = (c.k.as_ref(), c.v.as_ref()) {
                        k = concatenate_axis(&[pk, &k], 1)?;
                        v = concatenate_axis(&[pv, &v], 1)?;
                    }
                    c.k = Some(k.clone());
                    c.v = Some(v.clone());
                }
                (k, v)
            }
            // Cross-attention: derive K,V from audio features exactly once.
            Some(audio) => match cache {
                Some(c) => {
                    if c.k.is_none() {
                        c.k = Some(
                            self.key
                                .forward(audio)?
                                .multiply(Array::from_f32(self.scale))?,
                        );
                        c.v = Some(self.value.forward(audio)?);
                    }
                    (c.k.clone().unwrap(), c.v.clone().unwrap())
                }
                None => (
                    self.key
                        .forward(audio)?
                        .multiply(Array::from_f32(self.scale))?,
                    self.value.forward(audio)?,
                ),
            },
        };

        let out = self.qkv_attention(&q, &k, &v, mask)?;
        self.out.forward(&out)
    }

    fn qkv_attention(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let b = q.shape()[0];
        let lq = q.shape()[1];
        let lk = k.shape()[1];

        // [B, Lq, H, d] -> [B, H, Lq, d]
        let q = q
            .reshape(&[b, lq, self.n_head, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        // [B, Lk, H, d] -> [B, H, d, Lk]  (transposed for q @ k)
        let k = k
            .reshape(&[b, lk, self.n_head, -1])?
            .transpose_axes(&[0, 2, 3, 1])?;
        // [B, Lk, H, d] -> [B, H, Lk, d]
        let v = v
            .reshape(&[b, lk, self.n_head, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let mut qk = ops::matmul(&q, &k)?;
        if let Some(m) = mask {
            qk = qk.add(m)?;
        }
        let w = softmax_axis(&qk, -1, true)?;
        let out = ops::matmul(&w, &v)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, lq, -1])?;
        Ok(out)
    }
}

// ── Residual blocks (separate encoder/decoder types) ─────────────────────────

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct EncoderBlock {
    #[quantizable]
    #[param]
    pub attn: MultiHeadAttention,
    #[param]
    pub attn_ln: nn::LayerNorm,
    #[quantizable]
    #[param]
    pub mlp1: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub mlp2: MaybeQuantized<nn::Linear>,
    #[param]
    pub mlp_ln: nn::LayerNorm,
}

impl EncoderBlock {
    fn new(n_state: i32, n_head: i32) -> Result<Self, Exception> {
        Ok(Self {
            attn: MultiHeadAttention::new(n_state, n_head)?,
            attn_ln: nn::LayerNormBuilder::new(n_state).build()?,
            mlp1: MaybeQuantized::Original(nn::LinearBuilder::new(n_state, n_state * 4).build()?),
            mlp2: MaybeQuantized::Original(nn::LinearBuilder::new(n_state * 4, n_state).build()?),
            mlp_ln: nn::LayerNormBuilder::new(n_state).build()?,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let normed = self.attn_ln.forward(x)?;
        let attn = self.attn.forward(&normed, None, None, None)?;
        let x = x.add(&attn)?;
        let h = self.mlp1.forward(&self.mlp_ln.forward(&x)?)?;
        let h = self.mlp2.forward(&nn::gelu(&h)?)?;
        x.add(&h)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct DecoderBlock {
    #[quantizable]
    #[param]
    pub attn: MultiHeadAttention,
    #[param]
    pub attn_ln: nn::LayerNorm,
    #[quantizable]
    #[param]
    pub cross_attn: MultiHeadAttention,
    #[param]
    pub cross_attn_ln: nn::LayerNorm,
    #[quantizable]
    #[param]
    pub mlp1: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub mlp2: MaybeQuantized<nn::Linear>,
    #[param]
    pub mlp_ln: nn::LayerNorm,
}

impl DecoderBlock {
    fn new(n_state: i32, n_head: i32) -> Result<Self, Exception> {
        Ok(Self {
            attn: MultiHeadAttention::new(n_state, n_head)?,
            attn_ln: nn::LayerNormBuilder::new(n_state).build()?,
            cross_attn: MultiHeadAttention::new(n_state, n_head)?,
            cross_attn_ln: nn::LayerNormBuilder::new(n_state).build()?,
            mlp1: MaybeQuantized::Original(nn::LinearBuilder::new(n_state, n_state * 4).build()?),
            mlp2: MaybeQuantized::Original(nn::LinearBuilder::new(n_state * 4, n_state).build()?),
            mlp_ln: nn::LayerNormBuilder::new(n_state).build()?,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        audio: &Array,
        mask: Option<&Array>,
        self_cache: &mut KvCache,
        cross_cache: &mut KvCache,
    ) -> Result<Array, Exception> {
        let normed = self.attn_ln.forward(x)?;
        let attn = self.attn.forward(&normed, None, mask, Some(self_cache))?;
        let x = x.add(&attn)?;

        let normed = self.cross_attn_ln.forward(&x)?;
        let cross = self
            .cross_attn
            .forward(&normed, Some(audio), None, Some(cross_cache))?;
        let x = x.add(&cross)?;

        let h = self.mlp1.forward(&self.mlp_ln.forward(&x)?)?;
        let h = self.mlp2.forward(&nn::gelu(&h)?)?;
        x.add(&h)
    }
}

// ── Encoder / Decoder ────────────────────────────────────────────────────────

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct AudioEncoder {
    #[param]
    pub conv1: nn::Conv1d,
    #[param]
    pub conv2: nn::Conv1d,
    #[quantizable]
    #[param]
    pub blocks: Vec<EncoderBlock>,
    #[param]
    pub ln_post: nn::LayerNorm,
    // Sinusoidal positional embedding is computed, not stored.
    positional: Array,
}

impl AudioEncoder {
    fn new(dims: &ModelDimensions) -> Result<Self, Exception> {
        let n_state = dims.n_audio_state;
        let conv1 = nn::Conv1dBuilder::new(dims.n_mels, n_state, 3)
            .padding(1)
            .build()?;
        let conv2 = nn::Conv1dBuilder::new(n_state, n_state, 3)
            .stride(2)
            .padding(1)
            .build()?;
        let blocks = (0..dims.n_audio_layer)
            .map(|_| EncoderBlock::new(n_state, dims.n_audio_head))
            .collect::<Result<Vec<_>, _>>()?;
        let ln_post = nn::LayerNormBuilder::new(n_state).build()?;
        let positional = sinusoids(dims.n_audio_ctx, n_state)?;
        Ok(Self {
            conv1,
            conv2,
            blocks,
            ln_post,
            positional,
        })
    }

    /// `mel`: `[1, n_frames, n_mels]` (channels-last). Returns `[1, n_audio_ctx, n_state]`.
    pub fn forward(&mut self, mel: &Array) -> Result<Array, Exception> {
        let mut x = nn::gelu(&self.conv1.forward(mel)?)?;
        x = nn::gelu(&self.conv2.forward(&x)?)?;
        x = x.add(&self.positional.as_dtype(x.dtype())?)?;
        for block in self.blocks.iter_mut() {
            x = block.forward(&x)?;
        }
        self.ln_post.forward(&x)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct TextDecoder {
    #[quantizable]
    #[param]
    pub token_embedding: MaybeQuantized<nn::Embedding>,
    /// Learned positional embedding `[n_text_ctx, n_state]`.
    #[param]
    pub positional_embedding: Param<Array>,
    #[quantizable]
    #[param]
    pub blocks: Vec<DecoderBlock>,
    #[param]
    pub ln: nn::LayerNorm,
}

impl TextDecoder {
    fn new(dims: &ModelDimensions) -> Result<Self, Exception> {
        let n_state = dims.n_text_state;
        let token_embedding = MaybeQuantized::Original(nn::Embedding::new(dims.n_vocab, n_state)?);
        let positional_embedding =
            Param::new(mlx_rs::ops::zeros::<f32>(&[dims.n_text_ctx, n_state])?);
        let blocks = (0..dims.n_text_layer)
            .map(|_| DecoderBlock::new(n_state, dims.n_text_head))
            .collect::<Result<Vec<_>, _>>()?;
        let ln = nn::LayerNormBuilder::new(n_state).build()?;
        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
        })
    }

    /// `tokens`: `[1, L]`. `audio`: encoder output. `offset`: absolute position
    /// of the first token in `tokens` (KV-cache length so far).
    /// Returns logits `[1, L, n_vocab]`.
    pub fn forward(
        &mut self,
        tokens: &Array,
        audio: &Array,
        offset: i32,
        mask: Option<&Array>,
        self_caches: &mut [KvCache],
        cross_caches: &mut [KvCache],
    ) -> Result<Array, Exception> {
        let l = tokens.shape()[1];
        let emb = self.token_embedding.forward(tokens)?;
        let pos = self
            .positional_embedding
            .index((offset..offset + l, ..))
            .as_dtype(emb.dtype())?;
        let mut x = emb.add(&pos)?;

        for (i, block) in self.blocks.iter_mut().enumerate() {
            x = block.forward(&x, audio, mask, &mut self_caches[i], &mut cross_caches[i])?;
        }
        x = self.ln.forward(&x)?;
        // Tied output projection.
        match &mut self.token_embedding {
            MaybeQuantized::Original(embed) => embed.as_linear(&x),
            MaybeQuantized::Quantized(embed) => embed.as_linear(&x),
        }
    }
}

// ── Top-level model ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct WhisperModel {
    #[quantizable]
    #[param]
    pub encoder: AudioEncoder,
    #[quantizable]
    #[param]
    pub decoder: TextDecoder,
    pub dims: ModelDimensions,
}

impl WhisperModel {
    pub fn new(dims: ModelDimensions) -> Result<Self, Exception> {
        let encoder = AudioEncoder::new(&dims)?;
        let decoder = TextDecoder::new(&dims)?;
        Ok(Self {
            encoder,
            decoder,
            dims,
        })
    }

    /// Dtype the weights were loaded in (e.g. f16 for mlx-community turbo).
    pub fn dtype(&self) -> Dtype {
        self.encoder.conv1.weight.dtype()
    }

    /// Fresh self- and cross-attention cache vectors (one slot per decoder layer).
    pub fn new_caches(&self) -> (Vec<KvCache>, Vec<KvCache>) {
        let n = self.dims.n_text_layer as usize;
        (
            (0..n).map(|_| KvCache::empty()).collect(),
            (0..n).map(|_| KvCache::empty()).collect(),
        )
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Sinusoidal positional embedding, matching `mlx_whisper.sinusoids`.
fn sinusoids(length: i32, channels: i32) -> Result<Array, Exception> {
    debug_assert!(channels % 2 == 0);
    let half = channels / 2;
    let max_timescale = 10_000.0f32;
    let log_inc = max_timescale.ln() / (half as f32 - 1.0);
    // inv_timescales = exp(-log_inc * arange(half))
    let idx = mlx_rs::ops::arange::<_, f32>(None, half, None)?;
    let inv_timescales = ops::exp(&idx.multiply(Array::from_f32(-log_inc))?)?;
    // scaled_time[t, j] = t * inv_timescales[j]
    let times = mlx_rs::ops::arange::<_, f32>(None, length, None)?.reshape(&[length, 1])?;
    let inv = inv_timescales.reshape(&[1, half])?;
    let scaled = ops::matmul(&times, &inv)?; // [length, half]
    concatenate_axis(&[&ops::sin(&scaled)?, &ops::cos(&scaled)?], 1)
}

/// Additive causal mask `[n, n]`: 0 on/below the diagonal, -inf above.
fn causal_mask(n: i32, dtype: Dtype) -> Result<Array, Exception> {
    let neg = mlx_rs::ops::full::<f32>(&[n, n], Array::from_f32(f32::NEG_INFINITY))?;
    // triu(_, 1) keeps strictly-above-diagonal entries, zeros the rest.
    ops::triu(neg, 1)?.as_dtype(dtype)
}

/// Build the additive causal mask only when needed (prefill of >1 token).
pub fn maybe_causal_mask(l: i32, dtype: Dtype) -> Result<Option<Array>, Exception> {
    if l > 1 {
        Ok(Some(causal_mask(l, dtype)?))
    } else {
        Ok(None)
    }
}

// ── Loading ──────────────────────────────────────────────────────────────────

pub fn get_whisper_dims(model_dir: impl AsRef<Path>) -> Result<ModelDimensions, Error> {
    let file = std::fs::File::open(model_dir.as_ref().join("config.json"))?;
    Ok(serde_json::from_reader(file)?)
}

fn remap_quantized_key(key: &str) -> Option<String> {
    if let Some(stripped) = key.strip_suffix(".weight") {
        return Some(format!("{stripped}.inner.weight"));
    }
    if let Some(stripped) = key.strip_suffix(".bias") {
        return Some(format!("{stripped}.inner.bias"));
    }
    if key.ends_with(".scales") || key.ends_with(".biases") {
        return Some(key.to_string());
    }
    None
}

/// Load Whisper from a model dir containing `config.json` and a single
/// safetensors file (`weights.safetensors` for mlx-community, `model.safetensors`
/// as a fallback). Whisper checkpoints are not sharded.
pub fn load_whisper_model(model_dir: impl AsRef<Path>) -> Result<WhisperModel, Error> {
    let model_dir = model_dir.as_ref();
    let dims = get_whisper_dims(model_dir)?;
    let quantization = dims.quantization.clone();
    if std::env::var("SENCLAW_WHISPER_DEBUG").is_ok() {
        eprintln!("[whisper-debug] load dims quantization={quantization:?}");
    }
    if std::env::var("SENCLAW_WHISPER_DEBUG").is_ok() {
        eprintln!("[whisper-debug] build dense model start");
    }
    let mut model = WhisperModel::new(dims)?;
    if std::env::var("SENCLAW_WHISPER_DEBUG").is_ok() {
        eprintln!("[whisper-debug] build dense model done");
    }
    if let Some(q) = quantization.as_ref() {
        if std::env::var("SENCLAW_WHISPER_DEBUG").is_ok() {
            eprintln!(
                "[whisper-debug] quantize model start group_size={} bits={}",
                q.group_size,
                q.bits
            );
        }
        model = nn::quantize(model, Some(q.group_size), Some(q.bits))?;
        if std::env::var("SENCLAW_WHISPER_DEBUG").is_ok() {
            eprintln!("[whisper-debug] quantize model done");
        }
    }
    if std::env::var("SENCLAW_WHISPER_DEBUG").is_ok() {
        eprintln!("[whisper-debug] load base model built");
    }
    let is_quantized = quantization.is_some();

    let weights = {
        let primary = model_dir.join("weights.safetensors");
        if primary.exists() {
            primary
        } else {
            model_dir.join("model.safetensors")
        }
    };
    if std::env::var("SENCLAW_WHISPER_DEBUG").is_ok() {
        eprintln!(
            "[whisper-debug] load safetensors path={}",
            weights.display()
        );
    }
    let loaded = Array::load_safetensors(&weights)?;
    if std::env::var("SENCLAW_WHISPER_DEBUG").is_ok() {
        eprintln!("[whisper-debug] load safetensors decoded");
    }
    let mut params = model.parameters_mut().flatten();
    if std::env::var("SENCLAW_WHISPER_DEBUG").is_ok() {
        let mut sample = params
            .iter()
            .take(12)
            .map(|(k, v)| format!("{k}:{:?}", v.shape()))
            .collect::<Vec<_>>();
        sample.sort();
        eprintln!("[whisper-debug] param tree sample={}", sample.join(", "));
    }
    let mut token_embedding_weight: Option<Array> = None;
    let mut token_embedding_scales: Option<Array> = None;
    let mut token_embedding_biases: Option<Array> = None;
    let mut total_loaded = 0usize;
    let mut total_missed = 0usize;
    let mut unmatched_samples: Vec<String> = Vec::new();

    for (key, value) in loaded {
        match key.as_str() {
            "decoder.token_embedding.weight" => {
                token_embedding_weight = Some(value);
                total_loaded += 1;
                continue;
            }
            "decoder.token_embedding.scales" => {
                token_embedding_scales = Some(value);
                total_loaded += 1;
                continue;
            }
            "decoder.token_embedding.biases" => {
                token_embedding_biases = Some(value);
                total_loaded += 1;
                continue;
            }
            _ => {}
        }
        if let Some(slot) = params.get_mut(key.as_str()) {
            **slot = value;
            total_loaded += 1;
            continue;
        }
        if is_quantized {
            if let Some(remapped) = remap_quantized_key(&key) {
                if let Some(slot) = params.get_mut(remapped.as_str()) {
                    **slot = value;
                    total_loaded += 1;
                    continue;
                }
            }
        }
        total_missed += 1;
        if unmatched_samples.len() < 8 {
            unmatched_samples.push(key);
        }
    }

    if token_embedding_weight.is_some()
        || token_embedding_scales.is_some()
        || token_embedding_biases.is_some()
    {
        match &mut model.decoder.token_embedding {
            MaybeQuantized::Quantized(q) => {
                if let Some(w) = token_embedding_weight {
                    q.inner.weight.value = w;
                }
                if let Some(s) = token_embedding_scales {
                    q.scales.value = s;
                }
                if let Some(b) = token_embedding_biases {
                    q.biases.value = b;
                }
            }
            MaybeQuantized::Original(e) => {
                if let Some(w) = token_embedding_weight {
                    e.weight.value = w;
                }
            }
        }
    }

    if total_loaded == 0 {
        return Err(
            Exception::custom("no safetensor keys matched the Whisper parameter tree").into(),
        );
    }
    if !unmatched_samples.is_empty() {
        tracing::warn!(
            "[whisper] safetensor load missed {total_missed} key(s), sample: {}",
            unmatched_samples.join(", ")
        );
    }
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinusoids_shape() {
        let s = sinusoids(1500, 1280).unwrap();
        assert_eq!(s.shape(), &[1500, 1280]);
    }

    /// Loads the real downloaded checkpoint and exercises both forward passes.
    /// Run with: `SENCLAW_WHISPER_DIR=~/.senclaw/whisper-models/mlx-community__whisper-large-v3-turbo \
    ///   cargo test --features local-mlx -- --ignored --test-threads=1 load_real_model`
    #[test]
    #[ignore = "requires a downloaded checkpoint via SENCLAW_WHISPER_DIR"]
    fn load_real_model() {
        let dir = std::env::var("SENCLAW_WHISPER_DIR").expect("set SENCLAW_WHISPER_DIR");
        let mut m = load_whisper_model(&dir).expect("load");
        assert_eq!(m.dtype(), Dtype::Float16);

        // Encoder: zero mel [1, 3000, 128] -> [1, 1500, 1280].
        let mel = mlx_rs::ops::zeros::<f32>(&[1, 3000, m.dims.n_mels])
            .unwrap()
            .as_dtype(m.dtype())
            .unwrap();
        let feats = m.encoder.forward(&mel).unwrap();
        assert_eq!(
            feats.shape(),
            &[1, m.dims.n_audio_ctx, m.dims.n_audio_state]
        );

        // Decoder prefill of 4 tokens -> logits [1, 4, n_vocab].
        let toks = Array::from_slice(&[50258i32, 50278, 50360, 50364], &[1, 4]);
        let (mut sc, mut cc) = m.new_caches();
        let mask = maybe_causal_mask(4, m.dtype()).unwrap();
        let logits = m
            .decoder
            .forward(&toks, &feats, 0, mask.as_ref(), &mut sc, &mut cc)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 4, m.dims.n_vocab]);
    }

    #[test]
    fn causal_mask_shape_and_triangle() {
        let m = causal_mask(4, Dtype::Float32).unwrap();
        assert_eq!(m.shape(), &[4, 4]);
        // Diagonal is 0, strictly-upper is -inf.
        let v: Vec<f32> = m.as_slice().to_vec();
        assert_eq!(v[0], 0.0); // [0,0]
        assert!(v[1].is_infinite() && v[1] < 0.0); // [0,1]
        assert_eq!(v[5], 0.0); // [1,1]
        assert_eq!(v[4], 0.0); // [1,0] below diagonal
    }
}
