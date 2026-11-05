//! Credential helpers for on-disk login state.
//!
//! ## Codex (chatgpt.com) login support
//! Holds the Codex (`~/.codex/auth.json`) reader + route builder: a user who has
//! logged in with the Codex CLI (or imported those credentials) can run the
//! engine against the ChatGPT-backed backend (`chatgpt.com/backend-api`). This is
//! a supported production code path, part of the default build.
//!
//! The other production path resolves credentials from standard provider env keys
//! (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `LLM_BROWSER_*` / etc.) and builds an
//! [`crate::route::Route`] via the first-party [`crate::providers`] facades
//! (`OpenAi`, `Anthropic`, `OpenAiCompatible`). See
//! `browser-use-agent::turn::model_path`.

/// Codex `auth.json` reader + route builder (chatgpt.com login support).
pub mod codex;
/// Google Application Default Credentials + the Gemini Enterprise Agent Platform route builder.
pub mod google_adc;

pub use codex::{
    codex_auth_path, codex_route, load_codex_auth, load_codex_auth_file, CodexAuth, CodexAuthError,
    CODEX_BASE_URL,
};
pub use google_adc::{
    enterprise_requested as google_enterprise_requested, google_enterprise_route,
    resolve_enterprise_credentials as resolve_google_enterprise_credentials, GoogleAdcError,
    GoogleEnterpriseCredentials,
};
