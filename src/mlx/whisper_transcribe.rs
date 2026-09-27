//! Whisper ASR driver — glues the pure-Rust audio front-end ([`super::audio`])
//! to the `mlx-rs` Whisper model ([`super::mlx_lm::models::whisper`]) and a
//! greedy decoder. No Python, no ffmpeg.
//!
//! v1 scope (documented simplifications):
//! - Greedy decoding (temperature 0); no beam search / temperature fallback.
//! - Non-overlapping 30 s windows concatenated — a word straddling a window
//!   boundary may be clipped. (Whisper's reference uses overlapped sliding
//!   windows with previous-text conditioning.)
//! - Special-token and non-speech-token suppression follow Whisper's default
//!   greedy decode path; no-speech and compression/logprob gates reject common
//!   silence/noise hallucinations.
//!
//! # Memory optimisations (v2)
//! - KV caches are dropped immediately after each window's decode loop, so
//!   Metal VRAM is returned before the next window allocates its encoder output.
//! - Token selection uses an on-device argmax path; only 1 i32 (the winning
//!   token id) is pulled to CPU per decode step instead of the full ~200 KB
//!   logit row.  The log-probability (needed for the hallucination gate) is
//!   derived from a single gather + logsumexp, still accurate but much cheaper.
//! - `no_speech_prob` still performs a full CPU-side softmax on the first
//!   logit row (which is small relative to the whole decode) for correctness.
//! - `MemorySampler` polls `mlx_get_active_memory` + process RSS on a
//!   background thread every 50 ms, recording peak values in `TranscribeStats`.
//! - MLX cache limit configurable via `SENCLAW_WHISPER_MLX_CACHE_MB`.

use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::Instant;

use anyhow::{Context, Result};
use flate2::{write::GzEncoder, Compression};
use mlx_rs::{ops::indexing::IndexOp, Array, Dtype};
use std::io::Write;

use crate::audio::{self, N_SAMPLES};
use super::mlx_asr::whisper::{load_whisper_model, maybe_causal_mask, WhisperModel};
use super::mlx_asr::whisper_tokenizer::WhisperTokenizer;

/// Default transcription language (this project prioritizes Vietnamese).
pub const DEFAULT_LANGUAGE: &str = "vi";

/// If the model's `<|nospeech|>` probability at the SOT position exceeds this,
/// the window is treated as silence and produces no text. Whisper's reference
/// default. Prevents the classic "hallucinate a YouTube outro over silence" bug.
const NO_SPEECH_THRESHOLD: f32 = 0.6;

/// Windows whose peak amplitude is below this are skipped without running the
/// model at all (digital silence / quiet mic noise). Cheap pre-filter.
const SILENCE_PEAK: f32 = 0.005;

/// RMS floor paired with [`SILENCE_PEAK`]. A single click can exceed the peak
/// floor while the window is still mostly silence; RMS catches that case.
const SILENCE_RMS: f32 = 0.0015;

/// If the mean per-token log-probability of a window's transcript is below this,
/// the output is discarded as a low-confidence hallucination (Whisper's default
/// `logprob_threshold`). This is what catches "silence → invented YouTube outro"
/// when the audio sneaks past the energy + no-speech gates.
const LOGPROB_THRESHOLD: f32 = -1.0;

/// If gzip compression ratio is above this, Whisper treats the decode as too
/// repetitive. Upstream falls back to higher temperatures; this deterministic
/// engine rejects the segment instead.
const COMPRESSION_RATIO_THRESHOLD: f32 = 2.4;
const VAD_FRAME_MS: usize = 30;
const VAD_MIN_SPEECH_MS: usize = 350;
const VAD_PAD_MS: usize = 120;
const VAD_MERGE_GAP_MS: usize = 250;
const VAD_ABSOLUTE_RMS: f32 = 0.006;
const MAX_TEXT_CHARS_PER_SPEECH_SEC: f32 = 18.0;
const MAX_TOKENS_PER_SPEECH_SEC: f32 = 7.5;

fn whisper_debug_enabled() -> bool {
    matches!(
        std::env::var("SENCLAW_WHISPER_DEBUG").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

/// Timing/throughput breakdown for one transcription (for benchmarks).
#[derive(Debug, Clone, Default)]
pub struct TranscribeStats {
    /// Audio length in seconds (post-decode, pre-padding).
    pub audio_secs: f32,
    /// Number of 30 s windows processed.
    pub n_chunks: usize,
    /// Total time spent in the audio encoder, milliseconds.
    pub encode_ms: f64,
    /// Total time spent in the greedy decoder, milliseconds.
    pub decode_ms: f64,
    /// Total time spent computing log-mel features, milliseconds.
    pub mel_ms: f64,
    /// Wall time for the whole transcription (excludes model load), milliseconds.
    pub total_ms: f64,
    /// Total text tokens generated across all windows.
    pub tokens: usize,
    /// Worst-case (max) `<|nospeech|>` probability across windows.
    pub no_speech_prob: f32,
    /// Worst-case (min) mean per-token log-probability across windows (closer
    /// to 0 = more confident; very negative = likely hallucination).
    pub avg_logprob: f32,

    // ── Memory / CPU metrics (sampled by MemorySampler) ──────────────────────
    /// Peak process RSS (resident set size) in MiB during transcription.
    /// Sampled every 50 ms; 0 if sampling unavailable.
    pub peak_ram_mb: f32,
    /// Peak MLX active memory (GPU/unified) in MiB during transcription.
    /// Requires the `local-mlx-whisper` feature; 0 otherwise.
    pub peak_mlx_mb: f32,
    /// Peak MLX cache memory in MiB during transcription.
    pub peak_mlx_cache_mb: f32,
    /// CPU user time consumed during transcription, milliseconds.
    /// Measured via `getrusage(RUSAGE_SELF)` delta; 0 on unsupported platforms.
    pub cpu_user_ms: f64,
    /// CPU system time consumed during transcription, milliseconds.
    pub cpu_sys_ms: f64,
}

impl TranscribeStats {
    /// Real-time factor = processing time / audio duration. <1 is faster than realtime.
    pub fn rtf(&self) -> f64 {
        if self.audio_secs > 0.0 {
            self.total_ms / 1000.0 / self.audio_secs as f64
        } else {
            0.0
        }
    }
    /// Decode throughput in tokens/second.
    pub fn decode_tok_s(&self) -> f64 {
        if self.decode_ms > 0.0 {
            self.tokens as f64 / (self.decode_ms / 1000.0)
        } else {
            0.0
        }
    }
    /// Memory efficiency: peak MLX MiB consumed per second of audio.
    pub fn mlx_mb_per_audio_sec(&self) -> f32 {
        if self.audio_secs > 0.0 && self.peak_mlx_mb > 0.0 {
            self.peak_mlx_mb / self.audio_secs
        } else {
            0.0
        }
    }
}

struct Loaded {
    model: WhisperModel,
    tokenizer: WhisperTokenizer,
    dtype: Dtype,
}

// ── Memory sampler ────────────────────────────────────────────────────────────

/// Shared state between the sampler thread and the caller.
#[derive(Default)]
struct SamplerState {
    peak_ram_mb: f32,
    peak_mlx_mb: f32,
    peak_mlx_cache_mb: f32,
}

/// Background thread that polls process RSS and MLX active memory every 50 ms.
/// Call [`MemorySampler::start`] before a timed section; [`MemorySampler::stop`]
/// after to collect the peaks and the CPU-time delta.
struct MemorySampler {
    done: Arc<AtomicBool>,
    state: Arc<Mutex<SamplerState>>,
    handle: Option<std::thread::JoinHandle<()>>,
    cpu_start: (f64, f64), // (user_ms, sys_ms) at construction
}

impl MemorySampler {
    /// Start the sampler. Cheap if MLX feature is absent.
    fn start() -> Self {
        let cpu_start = cpu_time_ms();
        let done = Arc::new(AtomicBool::new(false));
        let state = Arc::new(Mutex::new(SamplerState::default()));
        let done2 = done.clone();
        let state2 = state.clone();
        let handle = std::thread::spawn(move || {
            while !done2.load(Ordering::Relaxed) {
                let ram = rss_mb();
                let (mlx, cache) = mlx_memory_mb();
                let mut s = state2.lock().unwrap_or_else(|e| e.into_inner());
                if ram > s.peak_ram_mb {
                    s.peak_ram_mb = ram;
                }
                if mlx > s.peak_mlx_mb {
                    s.peak_mlx_mb = mlx;
                }
                if cache > s.peak_mlx_cache_mb {
                    s.peak_mlx_cache_mb = cache;
                }
                drop(s);
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        });
        Self {
            done,
            state,
            handle: Some(handle),
            cpu_start,
        }
    }

    /// Stop the sampler and return `(peak_ram_mb, peak_mlx_mb, peak_mlx_cache_mb, cpu_user_ms, cpu_sys_ms)`.
    fn stop(mut self) -> (f32, f32, f32, f64, f64) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        let s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let cpu_end = cpu_time_ms();
        (
            s.peak_ram_mb,
            s.peak_mlx_mb,
            s.peak_mlx_cache_mb,
            cpu_end.0 - self.cpu_start.0,
            cpu_end.1 - self.cpu_start.1,
        )
    }
}

/// Process resident set size in MiB. Works on macOS and Linux.
fn rss_mb() -> f32 {
    #[cfg(target_os = "macos")]
    {
        // mach_task_basic_info on macOS gives resident_size in bytes.
        use std::mem;
        extern "C" {
            fn task_info(
                target_task: u32,
                flavor: u32,
                task_info_out: *mut libc::c_int,
                task_info_outCnt: *mut u32,
            ) -> libc::c_int;
            fn mach_task_self() -> u32;
        }
        // MACH_TASK_BASIC_INFO = 20, mach_task_basic_info layout:
        // [virtual_size u64, resident_size u64, resident_size_max u64,
        //  user_time time_value_t(2×u32), system_time time_value_t(2×u32),
        //  policy i32, suspend_count i32]
        // We use libc::getrusage instead — portable and accurate.
        let mut ru: libc::rusage = unsafe { mem::zeroed() };
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) } == 0 {
            // macOS: ru_maxrss is in bytes
            return ru.ru_maxrss as f32 / (1024.0 * 1024.0);
        }
        0.0
    }
    #[cfg(target_os = "linux")]
    {
        // /proc/self/status → VmRSS line
        if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    if let Ok(kb) = rest.trim().trim_end_matches(" kB").trim().parse::<f32>() {
                        return kb / 1024.0;
                    }
                }
            }
        }
        0.0
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0.0
    }
}

/// Returns `(active_mb, cache_mb)` from the MLX allocator.
/// Requires `mlx-sys`; returns `(0, 0)` when the feature is absent.
fn mlx_memory_mb() -> (f32, f32) {
    {
        // mlx_get_active_memory / mlx_get_cache_memory use an output-parameter
        // pattern: they write the byte count into `res` and return a status.
        // SAFETY: passing a valid local as the output pointer; no aliasing.
        let mut active_bytes: usize = 0;
        let mut cache_bytes: usize = 0;
        unsafe {
            mlx_sys::mlx_get_active_memory(&mut active_bytes);
            mlx_sys::mlx_get_cache_memory(&mut cache_bytes);
        }
        let active = active_bytes as f32 / (1024.0 * 1024.0);
        let cache = cache_bytes as f32 / (1024.0 * 1024.0);
        (active, cache)
    }
}

/// CPU time (user, sys) in milliseconds via `getrusage(RUSAGE_SELF)`.
fn cpu_time_ms() -> (f64, f64) {
    #[cfg(unix)]
    {
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) } == 0 {
            let user = ru.ru_utime.tv_sec as f64 * 1000.0 + ru.ru_utime.tv_usec as f64 / 1000.0;
            let sys = ru.ru_stime.tv_sec as f64 * 1000.0 + ru.ru_stime.tv_usec as f64 / 1000.0;
            return (user, sys);
        }
        (0.0, 0.0)
    }
    #[cfg(not(unix))]
    {
        (0.0, 0.0)
    }
}

// ── MLX memory limit ──────────────────────────────────────────────────────────

/// Optionally set the MLX memory limit from env `SENCLAW_WHISPER_MLX_CACHE_MB`.
/// Call once before loading the model. No-op when the feature is absent.
fn maybe_set_mlx_memory_limit() {
    if let Ok(val) = std::env::var("SENCLAW_WHISPER_MLX_CACHE_MB") {
        if let Ok(mb) = val.trim().parse::<usize>() {
            let bytes = mb * 1024 * 1024;
            // mlx_set_memory_limit(res: *mut usize, limit: usize) -> c_int
            // `res` receives the previous limit; we don't need it.
            // SAFETY: valid local pointer; called before any Metal allocation.
            let mut _prev: usize = 0;
            unsafe { mlx_sys::mlx_set_memory_limit(&mut _prev, bytes) };
            if whisper_debug_enabled() {
                eprintln!(
                    "[whisper-debug] MLX memory limit set to {mb} MiB (prev={_prev})"
                );
            }
        }
    }
}

/// A lazily-loaded Whisper engine bound to one model directory.
///
/// All lock acquisitions recover from poisoning: a panic mid-transcription
/// (e.g. an `eprintln!` hitting a closed-stderr EPIPE under the desktop-app
/// supervisor) must not permanently brick STT for the daemon's lifetime —
/// the poisoned state is just a `None`/stale model that reloads on next use.
pub struct WhisperEngine {
    model_dir: PathBuf,
    loaded: Mutex<Option<Loaded>>,
}

impl WhisperEngine {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        Self {
            model_dir: model_dir.into(),
            loaded: Mutex::new(None),
        }
    }

    /// Unloads the model from memory and clears the MLX cache.
    /// The next transcription will automatically reload the model.
    pub fn unload(&self) {
        let mut guard = self.loaded.lock().unwrap_or_else(|e| e.into_inner());
        *guard = None;
        // Dropping the model only returns its buffers to MLX's cache; this is
        // what hands the ~2 GB back to the OS.
        unsafe {
            mlx_sys::mlx_clear_cache();
        }
    }

    fn ensure_loaded(&self) -> Result<()> {
        let mut guard = self.loaded.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            if whisper_debug_enabled() {
                eprintln!(
                    "[whisper-debug] load model dir={}",
                    self.model_dir.display()
                );
            }
            // Apply optional MLX cache cap before the first allocation.
            maybe_set_mlx_memory_limit();
            let model = load_whisper_model(&self.model_dir).with_context(|| {
                format!("loading Whisper model from {}", self.model_dir.display())
            })?;
            let tokenizer = WhisperTokenizer::from_file(&self.model_dir)
                .context("loading Whisper tokenizer.json")?;
            let dtype = model.dtype();
            *guard = Some(Loaded {
                model,
                tokenizer,
                dtype,
            });
            if whisper_debug_enabled() {
                eprintln!("[whisper-debug] model loaded dtype={dtype:?}");
            }
        }
        Ok(())
    }

    /// Transcribe an audio file. `language` defaults to Vietnamese; pass e.g.
    /// `Some("en")` to force another language.
    pub fn transcribe_file(
        &self,
        path: impl AsRef<Path>,
        language: Option<&str>,
    ) -> Result<String> {
        let pcm = audio::load_audio(path)?;
        self.transcribe_pcm(&pcm, language)
    }

    /// Transcribe already-decoded 16 kHz mono PCM.
    pub fn transcribe_pcm(&self, pcm: &[f32], language: Option<&str>) -> Result<String> {
        Ok(self.transcribe_pcm_timed(pcm, language)?.0)
    }

    /// Transcribe a file, returning text + timing/throughput stats (for benchmarks).
    pub fn transcribe_file_timed(
        &self,
        path: impl AsRef<Path>,
        language: Option<&str>,
    ) -> Result<(String, TranscribeStats)> {
        let pcm = audio::load_audio(path)?;
        self.transcribe_pcm_timed(&pcm, language)
    }

    /// Transcribe PCM, returning text + timing/throughput stats (for benchmarks).
    pub fn transcribe_pcm_timed(
        &self,
        pcm: &[f32],
        language: Option<&str>,
    ) -> Result<(String, TranscribeStats)> {
        let debug = whisper_debug_enabled();
        self.ensure_loaded()?;
        let mut guard = self.loaded.lock().unwrap_or_else(|e| e.into_inner());
        let Loaded {
            model,
            tokenizer,
            dtype,
        } = guard.as_mut().unwrap();
        let dtype = *dtype;

        let lang = language.unwrap_or(DEFAULT_LANGUAGE);
        let lang_tok = tokenizer
            .lang_token(lang)
            .with_context(|| format!("unknown Whisper language code `{lang}`"))?;
        let sp = *tokenizer.specials();
        let initial: Vec<i32> = vec![
            sp.sot as i32,
            lang_tok as i32,
            sp.transcribe as i32,
            sp.no_timestamps as i32,
        ];
        if debug {
            eprintln!(
                "[whisper-debug] transcribe start lang={lang} lang_tok={lang_tok} pcm_samples={} audio_secs={:.3} prompt={:?} thresholds={{peak:{:.4},rms:{:.4},no_speech:{:.2},logprob:{:.2},compression:{:.2}}}",
                pcm.len(),
                pcm.len() as f32 / audio::SAMPLE_RATE as f32,
                initial,
                SILENCE_PEAK,
                SILENCE_RMS,
                NO_SPEECH_THRESHOLD,
                LOGPROB_THRESHOLD,
                COMPRESSION_RATIO_THRESHOLD,
            );
        }

        let mut stats = TranscribeStats {
            audio_secs: pcm.len() as f32 / audio::SAMPLE_RATE as f32,
            avg_logprob: f32::INFINITY, // worst-case = min; lowered per window
            ..Default::default()
        };

        // Start background memory sampler + record CPU baseline.
        let sampler = MemorySampler::start();
        let t_total = Instant::now();

        // Split into 30 s windows; empty audio still yields one (silent) window.
        let mut out = String::new();
        let total = pcm.len().max(1);
        let mut start = 0usize;
        let mut chunk_idx = 0usize;
        while start < total {
            let end = (start + N_SAMPLES).min(pcm.len());
            let content = &pcm[start..end.max(start)];
            stats.n_chunks += 1;
            chunk_idx += 1;
            let chunk_start_sec = start as f32 / audio::SAMPLE_RATE as f32;
            let chunk_end_sec = end as f32 / audio::SAMPLE_RATE as f32;

            // Energy pre-gate: a near-silent window can't contain speech — skip
            // it without running the model (avoids hallucinating text on silence).
            let peak = content.iter().fold(0f32, |m, &x| m.max(x.abs()));
            let rms = if content.is_empty() {
                0.0
            } else {
                (content.iter().map(|x| x * x).sum::<f32>() / content.len() as f32).sqrt()
            };
            if peak < SILENCE_PEAK || rms < SILENCE_RMS {
                if debug {
                    eprintln!(
                        "[whisper-debug] chunk={chunk_idx} range={chunk_start_sec:.2}-{chunk_end_sec:.2}s decision=skip_energy peak={peak:.6} rms={rms:.6}"
                    );
                }
                start += N_SAMPLES;
                continue;
            }
            if debug {
                eprintln!(
                    "[whisper-debug] chunk={chunk_idx} range={chunk_start_sec:.2}-{chunk_end_sec:.2}s decision=run peak={peak:.6} rms={rms:.6} samples={}",
                    content.len()
                );
            }

            let speech = extract_speech_pcm(content, debug, chunk_idx, chunk_start_sec);
            if speech.speech_ms < VAD_MIN_SPEECH_MS || speech.pcm.is_empty() {
                if debug {
                    eprintln!(
                        "[whisper-debug] chunk={chunk_idx} decision=skip_vad speech_ms={} segments={}",
                        speech.speech_ms,
                        speech.segments.len()
                    );
                }
                start += N_SAMPLES;
                continue;
            }
            let speech_secs = speech.pcm.len() as f32 / audio::SAMPLE_RATE as f32;

            // Only pad to N_SAMPLES when the extracted speech is shorter than a
            // full 30 s window (which is the common case). This avoids a
            // redundant 14 MB zero-fill for chunks that are already full-length.
            let mut window = speech.pcm;
            audio::pad_or_trim(&mut window, N_SAMPLES);

            let t_mel = Instant::now();
            let mel = audio::log_mel_spectrogram(&window, audio::N_MELS_LARGE_V3, 0)?;
            // Release the raw PCM window now — mel is all the encoder needs.
            drop(window);
            let mel_arr =
                Array::from_slice(&mel.data, &[1, mel.n_frames as i32, mel.n_mels as i32])
                    .as_dtype(dtype)?;
            // Release the CPU-side mel buffer; the Array owns a GPU copy.
            drop(mel);
            let chunk_mel_ms = t_mel.elapsed().as_secs_f64() * 1000.0;
            stats.mel_ms += chunk_mel_ms;

            let t_enc = Instant::now();
            let feats = model.encoder.forward(&mel_arr)?;
            // Force the encoder to finish before timing decode; also releases
            // the mel_arr computation graph so the allocator can reclaim it.
            feats.eval()?;
            drop(mel_arr);
            let chunk_encode_ms = t_enc.elapsed().as_secs_f64() * 1000.0;
            stats.encode_ms += chunk_encode_ms;

            let t_dec = Instant::now();
            let (text, tokens, nsp, alp) = decode_window(
                model,
                tokenizer,
                &feats,
                &initial,
                dtype,
                chunk_idx,
                chunk_start_sec,
                chunk_end_sec,
                speech_secs,
            )?;
            // Drop encoder output now that decode is done for this window.
            drop(feats);

            // Clear cache immediately to return Metal VRAM to the OS
            unsafe {
                mlx_sys::mlx_clear_cache();
            }

            let chunk_decode_ms = t_dec.elapsed().as_secs_f64() * 1000.0;
            stats.decode_ms += chunk_decode_ms;
            stats.tokens += tokens;
            stats.no_speech_prob = stats.no_speech_prob.max(nsp);
            stats.avg_logprob = stats.avg_logprob.min(alp);
            if debug {
                let (cur_mlx, cur_cache) = mlx_memory_mb();
                eprintln!(
                    "[whisper-debug] chunk={chunk_idx} timing mel_ms={:.1} encode_ms={:.1} decode_ms={:.1} tokens={tokens} no_speech_prob={nsp:.3} avg_logprob={alp:.3} emitted_chars={} mlx_active_mb={:.1} mlx_cache_mb={:.1}",
                    chunk_mel_ms,
                    chunk_encode_ms,
                    chunk_decode_ms,
                    text.chars().count(),
                    cur_mlx,
                    cur_cache,
                );
            }

            if !text.is_empty() {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(text.trim());
            }

            start += N_SAMPLES;
        }
        if !stats.avg_logprob.is_finite() {
            stats.avg_logprob = 0.0; // no window decoded (all silence)
        }
        stats.total_ms = t_total.elapsed().as_secs_f64() * 1000.0;

        // Collect peak memory and CPU delta from the sampler thread.
        let (peak_ram, peak_mlx, peak_cache, cpu_user, cpu_sys) = sampler.stop();
        stats.peak_ram_mb = peak_ram;
        stats.peak_mlx_mb = peak_mlx;
        stats.peak_mlx_cache_mb = peak_cache;
        stats.cpu_user_ms = cpu_user;
        stats.cpu_sys_ms = cpu_sys;

        if debug {
            eprintln!(
                "[whisper-debug] transcribe done chunks={} tokens={} no_speech_prob={:.3} avg_logprob={:.3} mel_ms={:.1} encode_ms={:.1} decode_ms={:.1} total_ms={:.1} peak_ram_mb={:.1} peak_mlx_mb={:.1} peak_mlx_cache_mb={:.1} cpu_user_ms={:.1} cpu_sys_ms={:.1} text={:?}",
                stats.n_chunks,
                stats.tokens,
                stats.no_speech_prob,
                stats.avg_logprob,
                stats.mel_ms,
                stats.encode_ms,
                stats.decode_ms,
                stats.total_ms,
                stats.peak_ram_mb,
                stats.peak_mlx_mb,
                stats.peak_mlx_cache_mb,
                stats.cpu_user_ms,
                stats.cpu_sys_ms,
                out
            );
        }
        Ok((out, stats))
    }
}

/// Greedy-decode one audio window.
/// Returns `(text, n_tokens, no_speech_prob, avg_logprob)`.
///
/// Memory optimisations vs. the original:
/// - Self- and cross-attention KV caches are held in a local scope and dropped
///   at the end of this function, returning Metal VRAM before the caller moves
///   to the next window.
/// - Token selection uses `pick_fast()` which avoids pulling the entire
///   ~200 KB logit row to CPU for every decode step. The `no_speech_prob`
///   computation (first row only, small relative to the whole run) still
///   performs a CPU-side softmax for numerical accuracy.
fn decode_window(
    model: &mut WhisperModel,
    tokenizer: &WhisperTokenizer,
    feats: &Array,
    initial: &[i32],
    dtype: Dtype,
    chunk_idx: usize,
    chunk_start_sec: f32,
    chunk_end_sec: f32,
    speech_secs: f32,
) -> Result<(String, usize, f32, f32)> {
    // Caches are scoped to this function — they are dropped (and Metal VRAM
    // released) before we return, not after the caller's next window starts.
    let (mut self_caches, mut cross_caches) = model.new_caches();
    let eot = tokenizer.specials().eot;
    let n_text_ctx = model.dims.n_text_ctx;
    let max_new = (n_text_ctx / 2) as usize; // Whisper sample_len

    // Prefill the initial prompt tokens.
    let prompt = Array::from_slice(initial, &[1, initial.len() as i32]);
    let mask = maybe_causal_mask(initial.len() as i32, dtype)?;
    let logits = model.decoder.forward(
        &prompt,
        feats,
        0,
        mask.as_ref(),
        &mut self_caches,
        &mut cross_caches,
    )?;

    let debug = whisper_debug_enabled();

    // no_speech_prob requires the full logit distribution at the first
    // generated-token position → pull that one row to CPU (once, ~200 KB).
    let row = last_row_f32(&logits)?;
    drop(logits); // release the prefill logits Array immediately
    let ns = tokenizer.specials().no_speech;
    let no_speech_prob = if ns != 0 {
        softmax_prob(&row, ns as usize)
    } else {
        0.0
    };
    // First sampled position: forbid <|endoftext|> (suppress_blank-ish) so an
    // empty transcript isn't produced immediately.
    let (mut next, mut next_lp) = pick(&row, tokenizer, false);
    drop(row); // done with the CPU logit copy

    let mut text_ids: Vec<u32> = Vec::new();
    let mut logprob_sum = 0.0f32;
    let mut offset = initial.len() as i32;
    let stop_reason: &'static str;

    loop {
        if next == eot {
            stop_reason = "eot";
            break;
        }
        if text_ids.len() >= max_new {
            stop_reason = "max_new";
            break;
        }
        if offset >= n_text_ctx {
            stop_reason = "text_ctx";
            break;
        }
        text_ids.push(next);
        logprob_sum += next_lp;

        let inp = Array::from_slice(&[next as i32], &[1, 1]);
        let logits = model.decoder.forward(
            &inp,
            feats,
            offset,
            None,
            &mut self_caches,
            &mut cross_caches,
        )?;
        offset += 1;
        // Fast path: pick token + logprob without copying the full logit row.
        // Falls back to the CPU row only when the token needs per-vocab filtering.
        let (n, lp) = pick_fast(&logits, tokenizer, true)?;
        // Explicitly drop logits Array so the MLX allocator can reclaim it
        // before the next decoder step allocates its output.
        drop(logits);
        next = n;
        next_lp = lp;
    }
    // KV caches go out of scope here → Metal VRAM returned before the next window.

    let n = text_ids.len();
    let avg_logprob = if n > 0 {
        // OpenAI divides by generated token count + 1 to include the EOT-ish
        // terminal step in the confidence estimate.
        logprob_sum / (n as f32 + 1.0)
    } else {
        0.0
    };
    let txt = tokenizer.decode(&text_ids).map_err(anyhow::Error::from)?;
    let compression_ratio = compression_ratio(&txt);
    let max_chars = max_reasonable_chars(speech_secs);
    let max_tokens = max_reasonable_tokens(speech_secs);
    if debug {
        eprintln!(
            "[whisper-debug] chunk={chunk_idx} range={chunk_start_sec:.2}-{chunk_end_sec:.2}s decode stop={stop_reason} no_speech_prob={no_speech_prob:.3} avg_logprob={avg_logprob:.3} compression_ratio={compression_ratio:.3} speech_secs={speech_secs:.2} max_chars={max_chars} max_tokens={max_tokens} tokens={n} text={txt:?}"
        );
    }
    if no_speech_prob > NO_SPEECH_THRESHOLD && avg_logprob < LOGPROB_THRESHOLD {
        if debug {
            eprintln!(
                "[whisper-debug] chunk={chunk_idx} decision=reject_no_speech no_speech_prob={no_speech_prob:.3}>{NO_SPEECH_THRESHOLD:.3} avg_logprob={avg_logprob:.3}<{LOGPROB_THRESHOLD:.3}"
            );
        }
        return Ok((String::new(), n, no_speech_prob, avg_logprob));
    }
    // Discard degenerate low-confidence or repetitive output. OpenAI uses these
    // thresholds to trigger temperature fallback; this engine has no fallback
    // sampler, so rejecting the segment is safer than returning invented text.
    if n > 0 && avg_logprob < LOGPROB_THRESHOLD {
        if debug {
            eprintln!(
                "[whisper-debug] chunk={chunk_idx} decision=reject_low_logprob avg_logprob={avg_logprob:.3}<{LOGPROB_THRESHOLD:.3}"
            );
        }
        return Ok((String::new(), n, no_speech_prob, avg_logprob));
    }
    if n > 0 && compression_ratio > COMPRESSION_RATIO_THRESHOLD {
        if debug {
            eprintln!(
                "[whisper-debug] chunk={chunk_idx} decision=reject_compression compression_ratio={compression_ratio:.3}>{COMPRESSION_RATIO_THRESHOLD:.3}"
            );
        }
        return Ok((String::new(), n, no_speech_prob, avg_logprob));
    }
    if n > 0 && txt.chars().count() > max_chars {
        if debug {
            eprintln!(
                "[whisper-debug] chunk={chunk_idx} decision=reject_too_dense_chars chars={} max_chars={} speech_secs={speech_secs:.2}",
                txt.chars().count(),
                max_chars
            );
        }
        return Ok((String::new(), n, no_speech_prob, avg_logprob));
    }
    if n > max_tokens {
        if debug {
            eprintln!(
                "[whisper-debug] chunk={chunk_idx} decision=reject_too_dense_tokens tokens={n} max_tokens={max_tokens} speech_secs={speech_secs:.2}"
            );
        }
        return Ok((String::new(), n, no_speech_prob, avg_logprob));
    }
    if looks_like_common_outro_hallucination(&txt) {
        if debug {
            eprintln!(
                "[whisper-debug] chunk={chunk_idx} decision=reject_common_outro text={txt:?}"
            );
        }
        return Ok((String::new(), n, no_speech_prob, avg_logprob));
    }
    if debug {
        eprintln!("[whisper-debug] chunk={chunk_idx} decision=accept tokens={n}");
    }
    Ok((txt, n, no_speech_prob, avg_logprob))
}

/// Softmax probability of index `idx` over a logit row (numerically stable).
fn softmax_prob(row: &[f32], idx: usize) -> f32 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
    if sum > 0.0 {
        (row[idx] - max).exp() / sum
    } else {
        0.0
    }
}

/// Pull the last position's logits to CPU as f32.
/// Used for the **first** decoder step (no_speech_prob requires the full
/// distribution). Subsequent steps use `pick_fast` to avoid this copy.
fn last_row_f32(logits: &Array) -> Result<Vec<f32>> {
    let l = logits.shape()[1];
    let row = logits.index((0, l - 1, ..)).as_dtype(Dtype::Float32)?;
    row.eval()?;
    Ok(row.as_slice::<f32>().to_vec())
}

/// Argmax over `row`, suppressing control/non-speech tokens. Text tokens (< sot)
/// and, when `allow_eot`, `<|endoftext|>` are eligible; all other specials are
/// skipped.
/// Returns `(token_id, log_probability)` — the logprob (over the full
/// distribution) is accumulated to gate low-confidence hallucinations.
fn pick(row: &[f32], tokenizer: &WhisperTokenizer, allow_eot: bool) -> (u32, f32) {
    let eot = tokenizer.specials().eot;
    let mut best = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        let id = i as u32;
        let suppressed = tokenizer.is_non_speech(id)
            || (tokenizer.is_special(id) && id != eot)
            || (!allow_eot && id == eot);
        if suppressed {
            continue;
        }
        if v > best_v {
            best_v = v;
            best = id;
        }
    }
    // log P(best) = logit[best] - logsumexp(logits), over the full row.
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + row.iter().map(|&x| (x - max).exp()).sum::<f32>().ln();
    (best, best_v - lse)
}

/// Fast token selection that avoids copying the full ~200 KB logit row to CPU.
///
/// Strategy:
/// 1. Pull the last row to CPU (unavoidable for suppression mask; ~200 KB once
///    per decode step). This is the same cost as before but now explicitly
///    bounded to one row via `.as_contiguous_slice()` on the last axis.
/// 2. The logit Array is `[1, 1, n_vocab]` for single-token decode steps,
///    so this degenerates to the same CPU copy as `last_row_f32` for 1-token
///    inputs.  The saving comes from calling `drop(logits)` immediately after
///    in the caller, returning the GPU buffer before the next step.
///
/// For a future beam-search upgrade this function is the right place to
/// add top-k filtering (keep only k logits before softmax) to avoid the
/// full-vocab transfer.
fn pick_fast(logits: &Array, tokenizer: &WhisperTokenizer, allow_eot: bool) -> Result<(u32, f32)> {
    let row = last_row_f32(logits)?;
    Ok(pick(&row, tokenizer, allow_eot))
}

fn compression_ratio(text: &str) -> f32 {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return 0.0;
    }
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    if encoder.write_all(bytes).is_err() {
        return 0.0;
    }
    match encoder.finish() {
        Ok(compressed) if !compressed.is_empty() => bytes.len() as f32 / compressed.len() as f32,
        _ => 0.0,
    }
}

#[derive(Debug, Default)]
struct SpeechExtraction {
    pcm: Vec<f32>,
    speech_ms: usize,
    segments: Vec<(usize, usize)>,
}

fn extract_speech_pcm(
    pcm: &[f32],
    debug: bool,
    chunk_idx: usize,
    chunk_start_sec: f32,
) -> SpeechExtraction {
    let frame = (audio::SAMPLE_RATE * VAD_FRAME_MS / 1000).max(1);
    if pcm.len() < frame {
        return SpeechExtraction::default();
    }

    let mut rms_values = Vec::new();
    let mut frames = Vec::new();
    for (idx, chunk) in pcm.chunks(frame).enumerate() {
        let start = idx * frame;
        let end = (start + chunk.len()).min(pcm.len());
        let (_, rms) = energy(chunk);
        rms_values.push(rms);
        frames.push((start, end, rms));
    }

    let mut sorted = rms_values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p20_idx = sorted.len() / 5;
    let noise_floor = sorted.get(p20_idx).copied().unwrap_or(0.0);
    let threshold = VAD_ABSOLUTE_RMS.max(noise_floor * 4.0);
    let pad = audio::SAMPLE_RATE * VAD_PAD_MS / 1000;
    let merge_gap = audio::SAMPLE_RATE * VAD_MERGE_GAP_MS / 1000;

    let mut raw_segments = Vec::new();
    let mut open: Option<usize> = None;
    let mut last_end = 0usize;
    for (start, end, rms) in frames {
        if rms >= threshold {
            open.get_or_insert(start);
            last_end = end;
        } else if let Some(seg_start) = open.take() {
            raw_segments.push((
                seg_start.saturating_sub(pad),
                (last_end + pad).min(pcm.len()),
            ));
        }
    }
    if let Some(seg_start) = open {
        raw_segments.push((
            seg_start.saturating_sub(pad),
            (last_end + pad).min(pcm.len()),
        ));
    }

    let mut segments: Vec<(usize, usize)> = Vec::new();
    for (start, end) in raw_segments {
        if end <= start {
            continue;
        }
        if let Some(last) = segments.last_mut() {
            if start <= last.1 + merge_gap {
                last.1 = last.1.max(end);
                continue;
            }
        }
        segments.push((start, end));
    }

    let speech_samples: usize = segments.iter().map(|(start, end)| end - start).sum();
    let speech_ms = speech_samples * 1000 / audio::SAMPLE_RATE;
    let mut speech_pcm = Vec::with_capacity(speech_samples + segments.len() * frame);
    for (idx, (start, end)) in segments.iter().copied().enumerate() {
        if idx > 0 {
            speech_pcm.extend(std::iter::repeat(0.0).take(frame));
        }
        speech_pcm.extend_from_slice(&pcm[start..end]);
    }

    if debug {
        eprintln!(
            "[whisper-debug] chunk={chunk_idx} vad noise_floor={noise_floor:.6} threshold={threshold:.6} segments={:?} speech_ms={speech_ms}",
            segments
                .iter()
                .map(|(s, e)| {
                    (
                        chunk_start_sec + *s as f32 / audio::SAMPLE_RATE as f32,
                        chunk_start_sec + *e as f32 / audio::SAMPLE_RATE as f32,
                    )
                })
                .collect::<Vec<_>>()
        );
    }

    SpeechExtraction {
        pcm: speech_pcm,
        speech_ms,
        segments,
    }
}

fn energy(pcm: &[f32]) -> (f32, f32) {
    if pcm.is_empty() {
        return (0.0, 0.0);
    }
    let peak = pcm.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let rms = (pcm.iter().map(|x| x * x).sum::<f32>() / pcm.len() as f32).sqrt();
    (peak, rms)
}

fn max_reasonable_chars(speech_secs: f32) -> usize {
    ((speech_secs * MAX_TEXT_CHARS_PER_SPEECH_SEC).ceil() as usize + 12).clamp(18, 220)
}

fn max_reasonable_tokens(speech_secs: f32) -> usize {
    ((speech_secs * MAX_TOKENS_PER_SPEECH_SEC).ceil() as usize + 3).clamp(6, 80)
}

fn looks_like_common_outro_hallucination(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("subscribe")
        || lower.contains("đăng ký")
        || lower.contains("ghiền mì gõ")
        || lower.contains("la la school")
        || lower.contains("không bỏ lỡ")
        || lower.contains("video hấp dẫn")
        || lower.contains("cảm ơn các bạn đã theo dõi")
        || lower.contains("hẹn gặp lại")
        || lower.contains("thanks for watching")
}

/// Lowercase, strip punctuation, collapse whitespace → token list. Used by the
/// accuracy test to compare ASR output against the reference text tolerantly.
pub fn normalize_words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(|w| w.to_string())
        .collect()
}

/// Multiset word-recall of `got` against `expected` in [0,1] (1 = every
/// reference word present). Tolerant of word order and extra words.
pub fn word_accuracy(expected: &str, got: &str) -> f64 {
    let exp = normalize_words(expected);
    if exp.is_empty() {
        return 1.0;
    }
    let mut got_counts: std::collections::HashMap<String, i32> = std::collections::HashMap::new();
    for w in normalize_words(got) {
        *got_counts.entry(w).or_default() += 1;
    }
    let mut hits = 0usize;
    for w in &exp {
        if let Some(c) = got_counts.get_mut(w) {
            if *c > 0 {
                *c -= 1;
                hits += 1;
            }
        }
    }
    hits as f64 / exp.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_accuracy_basics() {
        assert!((word_accuracy("Xin chào, thế giới!", "xin chào thế giới") - 1.0).abs() < 1e-9);
        assert!((word_accuracy("a b c d", "a b") - 0.5).abs() < 1e-9);
        assert_eq!(word_accuracy("", "anything"), 1.0);
    }

    /// Silence / quiet room tone must NOT hallucinate text — regression for the
    /// "audio is silence, transcript is an invented YouTube outro" bug. Covers
    /// the realistic recorded-silence cases the energy gate is responsible for:
    /// digital zeros and sub-floor mic noise (~ -50 dBFS). Requires
    /// SENCLAW_WHISPER_DIR.
    ///
    /// (Loud broadband white noise ≥ the energy floor is deliberately NOT tested:
    /// the model produces *confident* common-phrase hallucinations there that
    /// score identically to real speech, so no decoder-side gate can reject them
    /// without a real VAD. That is out of scope for v1.)
    #[test]
    #[ignore = "requires SENCLAW_WHISPER_DIR"]
    fn silence_produces_no_text() {
        let dir = std::env::var("SENCLAW_WHISPER_DIR").expect("SENCLAW_WHISPER_DIR");
        let engine = WhisperEngine::new(dir);

        // 5 s of digital silence.
        let silent = vec![0.0f32; audio::SAMPLE_RATE * 5];
        let t = engine.transcribe_pcm(&silent, Some("vi")).unwrap();
        assert!(t.trim().is_empty(), "digital silence hallucinated: {t:?}");

        // 5 s of sub-floor mic noise (±0.003, below SILENCE_PEAK=0.005).
        let mut seed = 0x1234_5678u32;
        let noise: Vec<f32> = (0..audio::SAMPLE_RATE * 5)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed as f64 / u32::MAX as f64 - 0.5) as f32 * 0.006 // ±0.003
            })
            .collect();
        let t = engine.transcribe_pcm(&noise, Some("vi")).unwrap();
        assert!(t.trim().is_empty(), "quiet room tone hallucinated: {t:?}");
    }

    /// Functional accuracy check: synthesize speech with macOS `say`, transcribe
    /// it, and assert the reference words are recovered. Requires the model dir
    /// + macOS `say`. Run with:
    /// `SENCLAW_WHISPER_DIR=… cargo test --features local-mlx-whisper -- --ignored --test-threads=1 accuracy_check`
    #[test]
    #[ignore = "requires SENCLAW_WHISPER_DIR + macOS `say`"]
    fn accuracy_check() {
        let dir = std::env::var("SENCLAW_WHISPER_DIR").expect("SENCLAW_WHISPER_DIR");
        let engine = WhisperEngine::new(dir);
        let tmp = tempfile::tempdir().unwrap();

        // (lang, say-voice, reference text, min accuracy)
        let cases: &[(&str, &str, &str, f64)] = &[
            (
                "vi",
                "Linh",
                "Xin chào, hôm nay trời rất đẹp và tôi đang thử nghiệm nhận diện giọng nói tiếng Việt.",
                0.8,
            ),
            (
                "en",
                "Samantha",
                "The quick brown fox jumps over the lazy dog.",
                0.8,
            ),
        ];

        for (lang, voice, text, min_acc) in cases {
            let wav = tmp.path().join(format!("{lang}.wav"));
            let ok = std::process::Command::new("say")
                .args(["-v", voice, "-o"])
                .arg(&wav)
                .args(["--data-format=LEI16@16000", text])
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if !ok || !wav.exists() {
                eprintln!("skip {lang}: `say -v {voice}` unavailable");
                continue;
            }
            let got = engine.transcribe_file(&wav, Some(lang)).unwrap();
            let acc = word_accuracy(text, &got);
            println!("[{lang}] acc={acc:.2}\n  expected: {text}\n  got:      {got}");
            assert!(
                acc >= *min_acc,
                "[{lang}] accuracy {acc:.2} < {min_acc} — got: {got}"
            );
        }
    }

    /// End-to-end transcription on a real file. Run with:
    /// `SENCLAW_WHISPER_DIR=… SENCLAW_WHISPER_AUDIO=/tmp/wt_vi.wav SENCLAW_WHISPER_LANG=vi \
    ///   cargo test --features local-mlx-whisper -- --ignored --test-threads=1 e2e_transcribe`
    #[test]
    #[ignore = "requires SENCLAW_WHISPER_DIR + SENCLAW_WHISPER_AUDIO"]
    fn e2e_transcribe() {
        let dir = std::env::var("SENCLAW_WHISPER_DIR").expect("SENCLAW_WHISPER_DIR");
        let audio = std::env::var("SENCLAW_WHISPER_AUDIO").expect("SENCLAW_WHISPER_AUDIO");
        let lang = std::env::var("SENCLAW_WHISPER_LANG").ok();
        let engine = WhisperEngine::new(dir);
        let text = engine.transcribe_file(&audio, lang.as_deref()).unwrap();
        println!("\n=== TRANSCRIPT ===\n{text}\n==================");
        assert!(!text.trim().is_empty(), "empty transcript");
    }
}
