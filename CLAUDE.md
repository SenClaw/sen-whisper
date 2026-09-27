# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project overview

`sen-whisper` is SenClaw's speech-to-text runtime: a standalone binary the
SenClaw daemon (`../senclaw`) launches as a child process and drives over
loopback HTTP. Unlike `sen-mlx` (one process per loaded LLM), this runtime is
`mode: service` — **one process manages its own models** over its own API,
the same shape the old daemon's `/api/whisper/*` namespace had before the
split.

History: Whisper ASR started in the daemon's `src/local_model/`, then moved to
`crates/senclaw-media` — a sidecar binary spawned on demand by
`src/media_sidecar.rs`, reached only through the daemon's own
`/api/whisper/*` proxy (`src/gateway/ui_server/whisper.rs`, which owned model
management: catalog, composite HF download, settings). This repo **merges**
those two halves into one process: the old sidecar's decode engines
(`src/audio.rs`, `src/candle_whisper.rs`, `src/mlx/`) plus the old daemon's
model management (`src/models.rs`, `src/validate.rs`, `src/settings.rs`), so
there is no longer a daemon-to-sidecar HTTP hop for a transcription.

Contract: [`../senclaw/docs/runtime-protocol.md`](../senclaw/docs/runtime-protocol.md)
§4.5. SDK: [`../senclaw/crates/sen-runtime-sdk`](../senclaw/crates/sen-runtime-sdk)
(path dependency, frozen — read-only from here).

## Build & run

```bash
cargo build --release                        # or `make build`
cargo test                                   # or `make test` — see below
make package                                 # dist/sen-whisper-<version>-darwin-arm64.tar.gz
make run-dev
```

Share `CARGO_TARGET_DIR` with a `sen-mlx` checkout while developing both: they
pin `mlx-rs`/`mlx-sys` to the same tag, and a shared target dir reuses one
compile of every *pure-Rust* dependency between them. It does **not** reuse
the MLX C++ build itself — `sen-mlx` and `sen-whisper` are separate workspaces
with separate `Cargo.lock` files, so `mlx-sys`'s build-script fingerprint (and
its `mlx-sys-<hash>` output directory) differs between the two even on the
same tag. Budget one full `mlx-sys` build (~1.2 GB, several minutes) per repo.

`[profile.dev.build-override] debug = false` (in `Cargo.toml`) keeps that
build's fingerprint stable across `build`/`check`/`test` — without it, Cargo's
default debuginfo handling for build-script units varies by which of those
three you run, which silently triggers a fresh from-scratch `mlx-sys` rebuild
on the next invocation even when nothing changed. Do not remove it.

In a disk-constrained environment, prefix a routine `cargo build`/`test`/
`check` with `CMAKE=/usr/bin/false`: a still-valid cached build is reused
without ever consulting it, while a Cargo-decided rebuild fails fast against
the fake `cmake` instead of silently spending ~1.2 GB. Unset it for the one
invocation where a real `mlx-sys` build is actually intended.

## Architecture

- `src/main.rs` — the `serve` subcommand, mounts every route, `Readiness::ready()`
  immediately (this process never reads weights before a request names a model).
- `src/models.rs` — the curated catalog, on-disk layout (`model_dir` under
  `SENCLAW_WHISPER_MODELS_DIR`, `legacy_model_dir` under the shared
  `SENCLAW_LOCAL_MODELS_DIR` for installs that predate the dedicated
  directory), composite download (weights from the mlx-community repo,
  `tokenizer.json` borrowed from the paired `openai/whisper-*` repo — MLX
  checkpoints ship no tokenizer of their own), and the download-progress
  registry.
- `src/validate.rs` — `GET .../validate`: checks a HF repo's metadata against
  what the MLX loader accepts *before* a download, so a client can say "this
  won't work" without spending hundreds of megabytes.
- `src/settings.rs` — `{modelId, language}`, camelCase, seeded once from the
  daemon's old `config.json["whisperConfig"]` via `sen_runtime_sdk::legacy`.
  **Different shape from `sen-mlx`'s `settings.json`** (snake_case, sampling
  knobs) — the two runtimes each own an unrelated file that happens to share a
  name; do not try to unify them.
- `src/transcribe.rs` — the shared decode dispatch behind both
  `/api/whisper/transcribe` (old namespace, verbatim) and
  `/v1/audio/transcriptions` (OpenAI multipart). Picks MLX or Candle
  (`SENCLAW_ASR_BACKEND=candle` forces Candle even on macOS, which is how the
  cross-platform path gets exercised on the dev machine), runs the decode on
  `spawn_blocking`.
- `src/audio.rs` — container decode (Symphonia) → mono 16 kHz → Whisper
  log-mel, pure Rust, numerically matched to `mlx_whisper/audio.py`.
- `src/candle_whisper.rs` — the CPU decoder for HF-layout checkpoints
  (`openai/whisper-*`); what makes a non-macOS build a real transcriber.
- `src/mlx/` — the MLX-accelerated decoder (`mlx-community/*` checkpoints),
  compiled only under `#[cfg(target_os = "macos")]`.

## Testing

`cargo test` — unit tests co-located in `#[cfg(test)]` modules, plus
`tests/manifest.rs`. A couple of ignored tests need a real model directory
(`SENCLAW_WHISPER_DIR`) and are meant to be run by hand, not in CI.

### MLX and test concurrency — and a bug this port fixed

Rules for Claude:

- **The same MLX-concurrency hazard `sen-mlx` documents applies here.**
  [`.cargo/config.toml`](.cargo/config.toml) sets `RUST_TEST_THREADS=1` so a
  plain `cargo test` does not intermittently SIGSEGV. Do not remove it.
- **`mlx::mlx_serial::lock()` must be held for the whole load+decode, not just
  part of it.** The old `senclaw-media` sidecar defined this lock
  (`src/mlx/mlx_serial.rs`) but **never called it anywhere** — a real bug
  carried forward from before the split, most likely never triggered in
  practice because voice messages rarely overlapped. `transcribe.rs`'s
  `transcribe_mlx` now acquires it for the whole `spawn_blocking` closure
  (load, decode, and the `UnloadOnExit` drop guard, which must run *before*
  the lock is released — declare the lock guard first so Rust's reverse drop
  order keeps it held that long). Two transcriptions arriving concurrently
  without this serialize instead of racing MLX's shared Metal device from two
  OS threads, which corrupts Metal state and can SIGSEGV — exactly the
  failure mode the test-concurrency issue above reproduces.
- **This is a same-process constraint, not a same-machine one** (see
  `sen-mlx`'s CLAUDE.md for the fuller version of this rule): a `sen-whisper`
  process and a `sen-mlx` process generating at the same time are fine, since
  Metal isolates command queues per process.

## Carried-over rules from the old monorepo

Rules for Claude:

- **Symphonia probes audio by file extension.** `write_probe_file` keeps the
  original upload's extension on the temp file it writes — a `filename` query
  param or multipart field name is what makes this possible, and writing
  `.audio` or some other made-up extension breaks decoding of a perfectly good
  file. Never normalize or strip the extension before probing.
- **Weights load on the first request, never in `main`.** `Readiness::ready()`
  is set immediately; every model-loading path is behind a route handler.
- **This runtime is not reaped when idle** (`idleTimeoutSecs: 0` in the
  manifest) — it drops decoder weights after each transcription (`unload()`
  inside `UnloadOnExit`, or Candle's per-request `CandleWhisper::load`), so an
  idle process costs a few megabytes. Respawning would trade that for a cold
  start on the next voice message; keep the `0`.
- **The two decode backends read different checkpoint families from the same
  model root.** MLX loads `mlx-community/*` (flat `n_mels`/`n_audio_state`
  config keys); Candle loads HF-layout `openai/*` (`d_model`/`encoder_layers`).
  `candle_whisper::supports_dir` / the MLX path's own config check are what
  turn a mismatched checkpoint into a clear 400 instead of a load-time crash —
  keep both checks whenever a new catalog entry is added.
- **A Whisper install is not "complete" without a tokenizer.**
  `models::is_installed` requires `config.json` (with the Whisper-shaped keys),
  a weights file, *and* `tokenizer.json` — an mlx-community repo ships no
  tokenizer of its own, so `models::run_download`'s composite fetch (weights
  from the target repo, `tokenizer.json` borrowed from the paired
  `openai/whisper-*` repo) is what makes a "downloaded" model actually usable.

## Porting conventions

- `anyhow::Result` for fallible functions; the two-shape error convention in
  `src/api_error.rs` (`ApiError` flat, for `/api/whisper/*`; `OpenAiError`
  nested, for `/v1/audio/transcriptions`) exists because those two namespaces
  have two different wire contracts — do not merge them.
- Comments explain *why*, not *what* — match that density in new code.
- No plan ids, phase numbers, or finding codes in code, comments, test names,
  or commit messages: explain the invariant or behavior directly.
