# sen-whisper

Speech-to-text runtime for [SenClaw](https://github.com/SenClaw/senclaw):
Whisper ASR, MLX-accelerated on Apple Silicon (Candle/CPU everywhere the code
builds, though this repo currently packages the MLX build only). The daemon
launches this as a child process and manages it like every other runtime —
it owns its own models, settings, and downloads over its own HTTP API.

Contract: [`senclaw/docs/runtime-protocol.md`](../senclaw/docs/runtime-protocol.md)
(esp. §4.5), built on [`sen-runtime-sdk`](../senclaw/crates/sen-runtime-sdk).

## Requirements

- Packages for macOS Apple Silicon (`darwin-arm64`), Linux x64 and Windows x64. Rust (stable) everywhere.
- macOS: Xcode command line tools (Metal, cmake) for the `mlx-sys` native build; the package ships the MLX
  backend and its `mlx.metallib`.
- Linux / Windows: nothing native — the package is the pure-Rust Candle backend (`src/candle_whisper.rs`) on
  the CPU, and the model list offers the `openai/whisper-*` checkpoints it reads (the `mlx-community/*` ones
  are macOS-only).

## Build

```bash
make build                    # cargo build --release
make test                     # cargo test (single-threaded — see CLAUDE.md)
make package                  # dist/sen-whisper-<version>-<platform>.tar.gz + .sha256
```

Share `CARGO_TARGET_DIR` with a `sen-mlx` checkout while developing both: they
pin `mlx-rs`/`mlx-sys` to the same tag, and a shared target dir reuses one
compile of every *pure-Rust* dependency between them. It does not reuse the
`mlx-sys` C++ build itself — separate `Cargo.lock` per repo means separate
build-script fingerprints — so budget one full `mlx-sys` build per repo.

```bash
make build CARGO_TARGET_DIR=/path/to/.cargo-target-mlx CARGO_BUILD_JOBS=4
```

## Run

The daemon launches this with `sen-whisper serve --host {host} --port {port}`
plus the environment in
[`sen_runtime_sdk::env`](../senclaw/crates/sen-runtime-sdk/src/env.rs). Every
variable has a standalone default, so it also runs by hand:

```bash
make run-dev
# -> serves on 127.0.0.1:4963, no auth token, no parent watchdog
```

`GET /health` answers immediately — this runtime manages its own models on
demand and never reads weights at startup. Model management is the old daemon
namespace, served verbatim:

```
GET    /api/whisper/models
GET    /api/whisper/models/:id/validate
POST   /api/whisper/models/:id/download
GET    /api/whisper/models/:id/status
POST   /api/whisper/models/:id/cancel
DELETE /api/whisper/models/:id
GET|PUT /api/whisper/settings
POST   /api/whisper/transcribe          (multipart, {ok, text} response)
```

Plus the OpenAI shape:

```
POST /v1/audio/transcriptions           (multipart file, model?, language?, response_format json|text)
```

Both transcription routes share one decode path: MLX on Apple Silicon, Candle
everywhere `SENCLAW_ASR_BACKEND=candle` or off macOS.

## Install into a daemon's runtime directory

```bash
make install-local            # `senclaw runtime install-local dist/…` if senclaw
                               # is on PATH, else extracts to
                               # ~/.senclaw/runtimes/sen-whisper/<version>/ by hand
```

## Layout

| Path | What |
|---|---|
| `src/main.rs` | CLI, env resolution, mounts every route behind the SDK's server scaffold |
| `src/models.rs` | Catalog, on-disk layout, composite HF download (weights + paired tokenizer) |
| `src/validate.rs` | Pre-download compatibility check (`GET .../validate`) |
| `src/settings.rs` | `{modelId, language}`, seeded from the daemon's old `config.json["whisperConfig"]` |
| `src/transcribe.rs` | The shared decode dispatch behind both transcription routes |
| `src/audio.rs` | Container decode → mono 16 kHz → Whisper log-mel, pure Rust |
| `src/candle_whisper.rs` | Cross-platform CPU decoder (HF-layout checkpoints) |
| `src/mlx/` | MLX-accelerated decoder (macOS only; `mlx-community/*` checkpoints) |
| `senclaw-runtime.json` | The package manifest the daemon reads |

See [`CLAUDE.md`](CLAUDE.md) for the rules this port depends on.
