//! `sen-whisper` — Speech-to-text runtime for SenClaw.
//!
//! `sen-whisper serve --host {host} --port {port}` — mode `service`: one
//! process manages its own models over its own API, so `/health` answers
//! before a single byte of weights is read. Weights load on first use and drop
//! after each transcription (the policy the old `senclaw-media` sidecar and
//! `media_sidecar.rs` used); the process itself may stay up between requests.
//! Full contract: `senclaw/docs/runtime-protocol.md` §4.5.

mod api_error;
mod audio;
mod candle_whisper;
#[cfg(target_os = "macos")]
mod mlx;
mod models;
mod settings;
mod transcribe;
mod validate;

use std::sync::Arc;

use axum::extract::DefaultBodyLimit;
use axum::routing::{delete, get, post};
use axum::Router;

use sen_runtime_sdk::env::LaunchEnv;
use sen_runtime_sdk::manifest::{Capability, RunMode};
use sen_runtime_sdk::server::{serve, Readiness, ServeArgs, ServeOptions};

/// Shared across every handler: just the launch environment, since this
/// runtime keeps no other process-wide state besides the download registry
/// (process-global, in [`models`]) and per-request engine instances.
pub struct AppState {
    pub env: LaunchEnv,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    sen_runtime_sdk::server::init_tracing();

    // `serve` is the only subcommand this runtime has. Accept a bare flag list
    // too (`sen-whisper --port 4963`), the way `cargo run --` is used during
    // development, but refuse anything else by name rather than silently
    // mis-parsing a typo as a flag.
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    match argv.first().map(String::as_str) {
        Some("serve") => {
            argv.remove(0);
        }
        Some(other) if !other.starts_with('-') => {
            anyhow::bail!("unknown subcommand `{other}` (sen-whisper has only `serve`)");
        }
        _ => {}
    }
    let parsed = ServeArgs::parse(argv).map_err(|e| anyhow::anyhow!(e))?;

    let env = LaunchEnv::from_env("sen-whisper", env!("CARGO_PKG_VERSION"));
    let state = Arc::new(AppState { env: env.clone() });

    let routes = Router::new()
        .route("/api/whisper/models", get(models::list))
        .route("/api/whisper/models/:id/download", post(models::download))
        .route("/api/whisper/models/:id/status", get(models::status))
        .route("/api/whisper/models/:id/cancel", post(models::cancel))
        .route("/api/whisper/models/:id/validate", get(validate::whisper_validate))
        .route("/api/whisper/models/:id", delete(models::delete))
        .route("/api/whisper/settings", get(models::settings_get).put(models::settings_put))
        .merge(transcribe::router())
        // Audio uploads: a few minutes of WAV is tens of megabytes, and axum's
        // 2 MB default would reject a transcription that worked before the split.
        .layer(DefaultBodyLimit::max(256 * 1024 * 1024))
        .with_state(state);

    // Service mode: this process manages its own models over its own API, and
    // never loads anything until a request names a model — so it is ready the
    // instant it can accept connections.
    let readiness = Readiness::ready();

    serve(
        routes,
        ServeOptions {
            env,
            mode: RunMode::Service,
            capabilities: vec![Capability::Asr],
            readiness,
            info_detail: None,
            args: parsed,
        },
    )
    .await
}
