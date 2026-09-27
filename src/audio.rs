//! Whisper-family ASR audio front-end — **pure-Rust, CPU only**.
//!
//! Turns an arbitrary audio file (or raw PCM) into the exact log-mel
//! spectrogram the Whisper encoder expects, with **no Python and no `ffmpeg`
//! subprocess** (the reference `mlx_whisper/audio.py` shells out to `ffmpeg`).
//!
//! Pipeline:
//! ```text
//!   file ──symphonia──▶ interleaved f32 ──avg──▶ mono f32 @ src_hz
//!        ──rubato──▶ mono f32 @ 16 kHz ──STFT(realfft)──▶ |X|²
//!        ──mel filterbank──▶ log10 + clamp + normalize ──▶ [n_frames, n_mels]
//! ```
//!
//! Numerically matched to `mlx_whisper/audio.py` (`log_mel_spectrogram`):
//! Hann window `hanning(N_FFT+1)[:-1]`, reflect padding of `N_FFT/2`, the last
//! STFT frame dropped, power spectrum `|X|²`, a librosa-Slaney mel filterbank
//! computed in-tree (so we ship no `mel_filters.npz` asset), then
//! `log10` floored at `1e-10`, dynamic-range clamp to `max-8`, and `(x+4)/4`.
//!
//! Output layout is **frame-major** `[n_frames][n_mels]` — exactly the shape
//! `magnitudes @ filters.T` produces in the reference, and the channels-last
//! layout `mlx_rs::nn::Conv1d` (`[N, L, C_in]`) wants for the encoder stem.

use std::path::Path;

use anyhow::{anyhow, Context, Result};

// ── Whisper hard-coded audio hyperparameters (mlx_whisper/audio.py) ──────────
/// Target sample rate fed to the encoder.
pub const SAMPLE_RATE: usize = 16_000;
/// FFT window length (samples).
pub const N_FFT: usize = 400;
/// STFT hop length (samples) — 10 ms frames.
pub const HOP_LENGTH: usize = 160;
/// Max chunk length the encoder accepts (seconds).
pub const CHUNK_LENGTH: usize = 30;
/// Samples in a 30-second chunk (`480_000`).
pub const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE;
/// Mel frames produced from a full chunk (`3_000`).
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH;
/// Frequency bins kept from the real FFT (`N_FFT/2 + 1` = `201`).
pub const N_FREQS: usize = N_FFT / 2 + 1;
/// Mel filters for `whisper-large-v3` / `-turbo`. Older models use 80.
pub const N_MELS_LARGE_V3: usize = 128;

/// A log-mel spectrogram in row-major **`[n_frames][n_mels]`** order.
#[derive(Debug, Clone)]
pub struct MelSpectrogram {
    /// `n_frames * n_mels` values, frame-major (`data[frame * n_mels + mel]`).
    pub data: Vec<f32>,
    pub n_frames: usize,
    pub n_mels: usize,
}

impl MelSpectrogram {
    /// View of a single frame's `n_mels` values.
    #[inline]
    pub fn frame(&self, f: usize) -> &[f32] {
        &self.data[f * self.n_mels..(f + 1) * self.n_mels]
    }
}

// ── Public entry points ──────────────────────────────────────────────────────

/// Decode an audio file to mono f32 PCM at 16 kHz.
///
/// Any container/codec enabled in the `symphonia` feature set (wav, mp3, m4a,
/// flac, ogg/vorbis, …) is accepted. Multi-channel audio is down-mixed by
/// averaging; non-16 kHz audio is resampled with a high-quality sinc filter.
pub fn load_audio(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    let path = path.as_ref();
    let (samples, src_hz) =
        decode_to_mono_f32(path).with_context(|| format!("decoding audio {}", path.display()))?;
    resample_to_16k(samples, src_hz)
}

/// Compute the Whisper log-mel spectrogram for already-decoded 16 kHz mono PCM.
///
/// `n_mels` must be 80 or 128. `padding` zero-samples are appended on the right
/// before the STFT (the reference pads to align chunk boundaries).
pub fn log_mel_spectrogram(
    samples: &[f32],
    n_mels: usize,
    padding: usize,
) -> Result<MelSpectrogram> {
    if n_mels != 80 && n_mels != N_MELS_LARGE_V3 {
        return Err(anyhow!("unsupported n_mels {n_mels} (expected 80 or 128)"));
    }

    // Right-pad with zeros, then reflect-pad N_FFT/2 each side (STFT centering).
    let mut padded = Vec::with_capacity(samples.len() + padding + N_FFT);
    padded.extend_from_slice(samples);
    padded.resize(samples.len() + padding, 0.0);
    let padded = reflect_pad(&padded, N_FFT / 2);

    // Number of STFT frames, then drop the final one (`freqs[:-1, :]`).
    if padded.len() < N_FFT {
        return Err(anyhow!("audio too short for one STFT frame"));
    }
    let t = (padded.len() - N_FFT + HOP_LENGTH) / HOP_LENGTH;
    let n_frames = t.saturating_sub(1);

    let window = hann_window();
    let filters = mel_filters(n_mels); // [n_mels][N_FREQS], row-major

    // realfft plan for a 400-point real transform → N_FREQS complex bins.
    let mut planner = realfft::RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut frame_buf = fft.make_input_vec(); // len N_FFT
    let mut spec_buf = fft.make_output_vec(); // len N_FREQS

    let mut data = vec![0.0f32; n_frames * n_mels];
    let mut power = [0.0f32; N_FREQS];
    let mut log_max = f32::NEG_INFINITY;

    for f in 0..n_frames {
        let start = f * HOP_LENGTH;
        for i in 0..N_FFT {
            frame_buf[i] = padded[start + i] * window[i];
        }
        // realfft overwrites the input buffer; refill it every frame above.
        fft.process(&mut frame_buf, &mut spec_buf)
            .map_err(|e| anyhow!("rfft failed: {e}"))?;
        for k in 0..N_FREQS {
            power[k] = spec_buf[k].re * spec_buf[k].re + spec_buf[k].im * spec_buf[k].im;
        }

        let row = &mut data[f * n_mels..(f + 1) * n_mels];
        for m in 0..n_mels {
            let filt = &filters[m * N_FREQS..(m + 1) * N_FREQS];
            let mut acc = 0.0f32;
            for k in 0..N_FREQS {
                acc += power[k] * filt[k];
            }
            // log10 floored at 1e-10 (matches mx.maximum(mel, 1e-10).log10()).
            let v = acc.max(1e-10).log10();
            if v > log_max {
                log_max = v;
            }
            row[m] = v;
        }
    }

    // Dynamic-range clamp to (global max − 8) and normalize to ≈[-1, 1].
    let floor = log_max - 8.0;
    for v in data.iter_mut() {
        *v = (v.max(floor) + 4.0) / 4.0;
    }

    Ok(MelSpectrogram {
        data,
        n_frames,
        n_mels,
    })
}

/// Decode a file and compute its log-mel spectrogram in one call.
pub fn file_to_log_mel(path: impl AsRef<Path>, n_mels: usize) -> Result<MelSpectrogram> {
    let samples = load_audio(path)?;
    log_mel_spectrogram(&samples, n_mels, 0)
}

/// Pad with zeros or trim (in place) to exactly `length` samples — the encoder
/// processes fixed 30-second (`N_SAMPLES`) chunks.
pub fn pad_or_trim(samples: &mut Vec<f32>, length: usize) {
    if samples.len() > length {
        samples.truncate(length);
    } else if samples.len() < length {
        samples.resize(length, 0.0);
    }
}

// ── Internals ────────────────────────────────────────────────────────────────

/// `np.hanning(N_FFT + 1)[:-1]` → periodic Hann window of length `N_FFT`.
fn hann_window() -> [f32; N_FFT] {
    let mut w = [0.0f32; N_FFT];
    for (n, wn) in w.iter_mut().enumerate() {
        // 0.5 - 0.5*cos(2πn / N_FFT)  (denominator is N_FFT because of the +1).
        *wn = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos() as f32;
    }
    w
}

/// NumPy `mode="reflect"` padding by `pad` samples on each side (edge sample
/// not duplicated): `[x[pad], …, x[1], x[0..], x[len-2], …, x[len-1-pad]]`.
fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    // prefix: x[1..=pad] reversed
    for i in (1..=pad).rev() {
        out.push(x[i]);
    }
    out.extend_from_slice(x);
    // suffix: x[n-1-pad .. n-1] reversed  (i.e. x[n-2], x[n-3], …, x[n-1-pad])
    for i in (n - 1 - pad..n - 1).rev() {
        out.push(x[i]);
    }
    out
}

/// librosa-style Slaney mel filterbank, computed in-tree:
/// `librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels)` with
/// `htk=False, norm="slaney"`. Returns row-major `[n_mels][N_FREQS]`.
fn mel_filters(n_mels: usize) -> Vec<f32> {
    let sr = SAMPLE_RATE as f64;

    // FFT bin center frequencies: i * sr / n_fft  (== linspace(0, sr/2, N_FREQS)).
    let fftfreqs: Vec<f64> = (0..N_FREQS).map(|i| i as f64 * sr / N_FFT as f64).collect();

    // Mel band edges (n_mels + 2 points spanning [0, sr/2]).
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(sr / 2.0);
    let n_pts = n_mels + 2;
    let freq_pts: Vec<f64> = (0..n_pts)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f64 / (n_pts as f64 - 1.0);
            mel_to_hz(mel)
        })
        .collect();
    let fdiff: Vec<f64> = freq_pts.windows(2).map(|w| w[1] - w[0]).collect(); // len n_mels+1

    let mut weights = vec![0.0f32; n_mels * N_FREQS];
    for m in 0..n_mels {
        let enorm = 2.0 / (freq_pts[m + 2] - freq_pts[m]); // Slaney normalization
        for k in 0..N_FREQS {
            let lower = (fftfreqs[k] - freq_pts[m]) / fdiff[m];
            let upper = (freq_pts[m + 2] - fftfreqs[k]) / fdiff[m + 1];
            let tri = lower.min(upper).max(0.0);
            weights[m * N_FREQS + k] = (tri * enorm) as f32;
        }
    }
    weights
}

/// Hz → mel on the Slaney (auditory) scale used by librosa with `htk=False`.
fn hz_to_mel(f: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP; // 15.0
    let logstep = (6.4f64).ln() / 27.0;
    if f >= MIN_LOG_HZ {
        min_log_mel + (f / MIN_LOG_HZ).ln() / logstep
    } else {
        f / F_SP
    }
}

/// Inverse of [`hz_to_mel`].
fn mel_to_hz(mel: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP; // 15.0
    let logstep = (6.4f64).ln() / 27.0;
    if mel >= min_log_mel {
        MIN_LOG_HZ * ((mel - min_log_mel) * logstep).exp()
    } else {
        F_SP * mel
    }
}

/// Decode any supported file to interleaved-then-averaged mono f32 + its rate.
fn decode_to_mono_f32(path: &Path) -> Result<(Vec<f32>, u32)> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::errors::Error as SymError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("unsupported or corrupt audio container")?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("no decodable audio track"))?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("no decoder for audio codec")?;

    let mut sample_rate = track.codec_params.sample_rate.unwrap_or(SAMPLE_RATE as u32);
    let mut samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            // Clean end-of-stream surfaces as an unexpected-EOF IoError.
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(anyhow!("demux error: {e}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                sample_rate = spec.rate;
                let channels = spec.channels.count().max(1);
                let mut sbuf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                sbuf.copy_interleaved_ref(decoded);
                // Down-mix to mono by averaging channels.
                for frame in sbuf.samples().chunks(channels) {
                    let sum: f32 = frame.iter().copied().sum();
                    samples.push(sum / channels as f32);
                }
            }
            // Recoverable decode hiccups: skip the packet, keep going.
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(anyhow!("decode error: {e}")),
        }
    }

    if samples.is_empty() {
        return Err(anyhow!("decoded zero audio samples"));
    }
    Ok((samples, sample_rate))
}

/// Resample mono f32 to 16 kHz with a high-quality sinc filter (no-op if already 16 kHz).
fn resample_to_16k(samples: Vec<f32>, src_hz: u32) -> Result<Vec<f32>> {
    if src_hz as usize == SAMPLE_RATE || samples.is_empty() {
        return Ok(samples);
    }

    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    let ratio = SAMPLE_RATE as f64 / src_hz as f64;
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    const CHUNK: usize = 1024;
    let mut resampler =
        SincFixedIn::<f32>::new(ratio, 2.0, params, CHUNK, 1).context("constructing resampler")?;

    let mut out: Vec<f32> = Vec::with_capacity((samples.len() as f64 * ratio) as usize + CHUNK);
    let mut pos = 0;
    while pos + CHUNK <= samples.len() {
        let chunk = [&samples[pos..pos + CHUNK]];
        let res = resampler.process(&chunk, None).context("resample chunk")?;
        out.extend_from_slice(&res[0]);
        pos += CHUNK;
    }
    if pos < samples.len() {
        // Final short chunk: process_partial zero-pads internally.
        let tail = [&samples[pos..]];
        let res = resampler
            .process_partial(Some(&tail), None)
            .context("resample tail")?;
        out.extend_from_slice(&res[0]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_window_endpoints_and_peak() {
        let w = hann_window();
        assert_eq!(w.len(), N_FFT);
        assert!(w[0].abs() < 1e-6, "hann[0] should be 0, got {}", w[0]);
        // Peak at the center (cos(π) = −1 → 1.0).
        assert!((w[N_FFT / 2] - 1.0).abs() < 1e-6, "hann[200] should be 1");
        // Symmetric about the implicit (N_FFT+1)/2 point: w[n] == w[N_FFT-n].
        for n in 1..N_FFT {
            assert!((w[n] - w[N_FFT - n]).abs() < 1e-6);
        }
    }

    #[test]
    fn mel_scale_roundtrip_and_anchor() {
        // 1000 Hz is the Slaney linear/log knee → exactly 15.0 mel.
        assert!((hz_to_mel(1000.0) - 15.0).abs() < 1e-9);
        for &f in &[0.0, 100.0, 440.0, 1000.0, 4000.0, 8000.0] {
            let back = mel_to_hz(hz_to_mel(f));
            assert!((back - f).abs() < 1e-6, "roundtrip {f} → {back}");
        }
    }

    #[test]
    fn mel_filterbank_shape_and_properties() {
        for &n_mels in &[80usize, 128] {
            let filt = mel_filters(n_mels);
            assert_eq!(filt.len(), n_mels * N_FREQS);
            // Every weight non-negative.
            assert!(filt.iter().all(|&w| w >= 0.0));
            // Every mel band has positive total weight (no empty filters).
            for m in 0..n_mels {
                let s: f32 = filt[m * N_FREQS..(m + 1) * N_FREQS].iter().sum();
                assert!(s > 0.0, "mel band {m} is empty for n_mels={n_mels}");
            }
        }
    }

    #[test]
    fn frame_count_matches_whisper_chunk() {
        // A full 30 s chunk must yield exactly N_FRAMES (3000) mel frames.
        let samples = vec![0.0f32; N_SAMPLES];
        let mel = log_mel_spectrogram(&samples, N_MELS_LARGE_V3, 0).unwrap();
        assert_eq!(mel.n_frames, N_FRAMES);
        assert_eq!(mel.n_mels, N_MELS_LARGE_V3);
        assert_eq!(mel.data.len(), N_FRAMES * N_MELS_LARGE_V3);
    }

    #[test]
    fn normalization_dynamic_range_within_bounds() {
        // 440 Hz tone, 1 s, padded to a full chunk.
        let mut samples: Vec<f32> = (0..SAMPLE_RATE)
            .map(|n| {
                (2.0 * std::f64::consts::PI * 440.0 * n as f64 / SAMPLE_RATE as f64).sin() as f32
            })
            .collect();
        pad_or_trim(&mut samples, N_SAMPLES);
        let mel = log_mel_spectrogram(&samples, N_MELS_LARGE_V3, 0).unwrap();

        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in &mel.data {
            assert!(v.is_finite());
            lo = lo.min(v);
            hi = hi.max(v);
        }
        // Reference clamps the log10 dynamic range to 8 decades, then divides
        // by 4, so the post-norm span can never exceed 2.0. (The absolute level
        // is *not* bounded to 1.0 — a loud tone with mel energy > 1 pushes
        // log10 > 0, hence (x+4)/4 > 1.)
        assert!(hi - lo <= 2.0 + 1e-4, "span {} exceeds 2.0", hi - lo);
    }

    /// Write a minimal 16-bit PCM mono WAV so the symphonia decode path is
    /// exercised end-to-end without pulling in an encoder dependency.
    fn write_wav(path: &std::path::Path, sr: u32, samples: &[i16]) {
        use std::io::Write;
        let data_len = (samples.len() * 2) as u32;
        let byte_rate = sr * 2;
        let mut buf: Vec<u8> = Vec::with_capacity(44 + data_len as usize);
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(36 + data_len).to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
        buf.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
        buf.extend_from_slice(&1u16.to_le_bytes()); // channels = mono
        buf.extend_from_slice(&sr.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes()); // block align
        buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_len.to_le_bytes());
        for &s in samples {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        std::fs::File::create(path)
            .unwrap()
            .write_all(&buf)
            .unwrap();
    }

    fn tone_i16(freq: f64, sr: u32, secs: f64) -> Vec<i16> {
        let n = (sr as f64 * secs) as usize;
        (0..n)
            .map(|i| {
                let v = (2.0 * std::f64::consts::PI * freq * i as f64 / sr as f64).sin();
                (v * 16000.0) as i16
            })
            .collect()
    }

    #[test]
    fn decode_wav_16k_no_resample() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tone16k.wav");
        write_wav(&path, 16_000, &tone_i16(440.0, 16_000, 0.5));

        let pcm = load_audio(&path).unwrap();
        // 0.5 s @ 16 kHz, no resampling → ~8000 samples.
        assert!(
            (pcm.len() as i64 - 8000).abs() < 16,
            "got {} samples",
            pcm.len()
        );
        assert!(pcm.iter().all(|v| v.is_finite() && v.abs() <= 1.0));

        let mel = file_to_log_mel(&path, N_MELS_LARGE_V3).unwrap();
        assert!(mel.n_frames > 0);
        assert_eq!(mel.n_mels, N_MELS_LARGE_V3);
    }

    #[test]
    fn decode_wav_8k_resamples_to_16k() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tone8k.wav");
        write_wav(&path, 8_000, &tone_i16(440.0, 8_000, 1.0));

        let pcm = load_audio(&path).unwrap();
        // 1 s @ 8 kHz resampled to 16 kHz → ~16000 samples (allow filter slack).
        assert!(
            (pcm.len() as i64 - 16_000).abs() < 600,
            "expected ~16000 samples after resample, got {}",
            pcm.len()
        );
    }

    #[test]
    fn tone_energy_lands_in_expected_mel_band() {
        // A pure 1 kHz tone should peak in the mel band whose center is nearest
        // 1 kHz, not at DC or Nyquist.
        let mut samples: Vec<f32> = (0..SAMPLE_RATE)
            .map(|n| {
                (2.0 * std::f64::consts::PI * 1000.0 * n as f64 / SAMPLE_RATE as f64).sin() as f32
            })
            .collect();
        pad_or_trim(&mut samples, N_SAMPLES);
        let mel = log_mel_spectrogram(&samples, N_MELS_LARGE_V3, 0).unwrap();

        // Average each mel band over a mid-signal frame run, find the argmax.
        let n_mels = mel.n_mels;
        let mut band_energy = vec![0.0f32; n_mels];
        let frames = 100..200;
        for f in frames.clone() {
            for (m, e) in band_energy.iter_mut().enumerate() {
                *e += mel.frame(f)[m];
            }
        }
        let peak = band_energy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        // 1 kHz is well below Nyquist (8 kHz); the peak band must be in the
        // lower portion of the 128-band range, and not band 0 (DC).
        assert!(
            peak > 0 && peak < n_mels / 2,
            "peak band {peak} unexpected for 1 kHz"
        );
    }
}
