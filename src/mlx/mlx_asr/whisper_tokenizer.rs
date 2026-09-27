//! Thin wrapper around the HuggingFace `tokenizers` fast tokenizer for Whisper.
//!
//! mlx-community Whisper checkpoints ship no tokenizer; we pair them with the
//! `tokenizer.json` from the matching `openai/whisper-*` repo. Special tokens
//! are resolved **by string** (`token_to_id`) so the ids always match the
//! checkpoint's vocab — we never hardcode the language-id arithmetic.

use std::path::Path;

use tokenizers::Tokenizer as HfTokenizer;

use super::error::Error;

/// The fixed special tokens of a multilingual Whisper transcription prompt.
#[derive(Debug, Clone, Copy)]
pub struct SpecialTokens {
    pub sot: u32,           // <|startoftranscript|>
    pub transcribe: u32,    // <|transcribe|>
    pub translate: u32,     // <|translate|>
    pub no_timestamps: u32, // <|notimestamps|>
    pub eot: u32,           // <|endoftext|>
    pub no_speech: u32,     // <|nospeech|> — high prob ⇒ segment is silence
}

pub struct WhisperTokenizer {
    inner: HfTokenizer,
    specials: SpecialTokens,
    non_speech_tokens: Vec<u32>,
}

impl WhisperTokenizer {
    pub fn from_file(model_dir: impl AsRef<Path>) -> Result<Self, Error> {
        let path = model_dir.as_ref().join("tokenizer.json");
        let inner = HfTokenizer::from_file(&path).map_err(Error::from)?;
        let specials = SpecialTokens {
            sot: lookup(&inner, "<|startoftranscript|>")?,
            transcribe: lookup(&inner, "<|transcribe|>")?,
            translate: lookup(&inner, "<|translate|>")?,
            no_timestamps: lookup(&inner, "<|notimestamps|>")?,
            eot: lookup(&inner, "<|endoftext|>")?,
            // Some checkpoints name it <|nocaptions|>; fall back, else 0 (disabled).
            no_speech: inner
                .token_to_id("<|nospeech|>")
                .or_else(|| inner.token_to_id("<|nocaptions|>"))
                .unwrap_or(0),
        };
        let mut non_speech_tokens = build_non_speech_tokens(&inner);
        non_speech_tokens.sort_unstable();
        non_speech_tokens.dedup();
        Ok(Self {
            inner,
            specials,
            non_speech_tokens,
        })
    }

    pub fn specials(&self) -> &SpecialTokens {
        &self.specials
    }

    /// Resolve a language token id, e.g. `lang_token("vi")` → `<|vi|>`.
    pub fn lang_token(&self, lang: &str) -> Option<u32> {
        self.inner.token_to_id(&format!("<|{lang}|>"))
    }

    /// Resolve an arbitrary special token by its literal string.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Decode generated token ids to text, dropping special tokens.
    pub fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        self.inner.decode(ids, true).map_err(Error::from)
    }

    /// True if `id` is any `<|...|>` special / control token (id ≥ sot, plus eot).
    /// Used to suppress non-text tokens during greedy decoding.
    pub fn is_special(&self, id: u32) -> bool {
        id >= self.specials.sot || id == self.specials.eot
    }

    /// True if `id` is one of OpenAI Whisper's non-speech annotation tokens
    /// (music symbols, bracket/parenthesis starts, speaker-tag punctuation, …).
    pub fn is_non_speech(&self, id: u32) -> bool {
        self.non_speech_tokens.binary_search(&id).is_ok()
    }
}

fn lookup(tk: &HfTokenizer, token: &str) -> Result<u32, Error> {
    tk.token_to_id(token)
        .ok_or_else(|| Error::Custom(format!("Whisper tokenizer missing special token {token}")))
}

fn first_token(tk: &HfTokenizer, text: &str) -> Option<u32> {
    tk.encode(text, false)
        .ok()
        .and_then(|enc| enc.get_ids().first().copied())
}

/// Port of OpenAI Whisper's `Tokenizer.non_speech_tokens` list. Suppressing
/// these prevents the decoder from emitting common hallucinated annotations
/// like music markers, speaker labels, and bracketed stage directions.
fn build_non_speech_tokens(tk: &HfTokenizer) -> Vec<u32> {
    let mut symbols: Vec<String> = "\"#()*+/:;<=>@[\\]^_`{|}~「」『』"
        .chars()
        .map(|c| c.to_string())
        .collect();
    symbols.extend(
        "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪"
            .split_whitespace()
            .map(str::to_string),
    );

    let miscellaneous = "♩♪♫♬♭♮♯";
    let mut result = Vec::new();

    // Allow hyphens and single quotes inside words, but suppress their
    // whitespace-prefixed forms at the start of a generated phrase.
    for text in [" -", " '"] {
        if let Some(id) = first_token(tk, text) {
            result.push(id);
        }
    }

    for symbol in symbols {
        for text in [symbol.clone(), format!(" {symbol}")] {
            if let Ok(enc) = tk.encode(text, false) {
                let ids = enc.get_ids();
                if ids.len() == 1 {
                    result.push(ids[0]);
                }
            }
        }
    }

    // For miscellaneous music symbols, suppress the first token even if the
    // tokenizer splits them into multiple pieces; upstream Whisper does the
    // same because these symbols share the same UTF-8 prefix bytes.
    for symbol in miscellaneous.chars() {
        for text in [symbol.to_string(), format!(" {symbol}")] {
            if let Some(id) = first_token(tk, &text) {
                result.push(id);
            }
        }
    }

    result
}
