//! Google Application Default Credentials (ADC) and the Gemini Enterprise Agent Platform route.
//!
//! The Gemini Developer API (`generativelanguage.googleapis.com`) authenticates with an API key. The Gemini
//! Enterprise Agent Platform (formerly Vertex AI) serves the same `generateContent` wire format under
//! `https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/publishers/google/models/{model}`
//! and authenticates with an OAuth2 bearer token. This module resolves that token from ADC and builds the
//! matching [`Route`] on top of the unchanged Gemini protocol codec.
//!
//! ADC resolution follows google-auth's order:
//! 1. the file named by `GOOGLE_APPLICATION_CREDENTIALS` (a service-account key or an authorized-user file),
//! 2. the gcloud user file written by `gcloud auth application-default login`
//!    (`$CLOUDSDK_CONFIG/application_default_credentials.json`, default `~/.config/gcloud/`),
//! 3. the GCE metadata server.
//!
//! The token is exchanged when the provider is resolved (run start) and is valid for the lifetime Google
//! grants it (one hour). Like the codex OAuth path in this crate, no refresh happens during a run.

use std::fmt;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine;
use serde::Deserialize;

use crate::protocols::GeminiGenerateContentProtocol;
use crate::route::{Auth, Endpoint, Route};

/// Either of the google-genai SDK's platform-selection flags (`true` / `1`) selects the Enterprise platform.
pub const ENTERPRISE_FLAG_ENV_KEYS: [&str; 2] =
    ["GOOGLE_GENAI_USE_ENTERPRISE", "GOOGLE_GENAI_USE_VERTEXAI"];
/// Project id override (otherwise the credentials file's project, then the metadata server).
pub const PROJECT_ENV_KEY: &str = "GOOGLE_CLOUD_PROJECT";
/// Location override; defaults to [`DEFAULT_LOCATION`].
pub const LOCATION_ENV_KEY: &str = "GOOGLE_CLOUD_LOCATION";
/// Explicit credentials file (service account key or authorized user file).
pub const CREDENTIALS_FILE_ENV_KEY: &str = "GOOGLE_APPLICATION_CREDENTIALS";
/// The location the google-genai SDK uses when none is configured.
pub const DEFAULT_LOCATION: &str = "global";

const CLOUD_PLATFORM_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";
const OAUTH_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const METADATA_TOKEN_URL: &str =
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token";
const METADATA_PROJECT_URL: &str =
    "http://metadata.google.internal/computeMetadata/v1/project/project-id";
const TOKEN_REQUEST_TIMEOUT: Duration = Duration::from_secs(20);
const METADATA_REQUEST_TIMEOUT: Duration = Duration::from_secs(3);
const JWT_LIFETIME_SECS: u64 = 3600;

/// Whether the environment selects the Enterprise platform (`GOOGLE_GENAI_USE_ENTERPRISE` or the legacy
/// `GOOGLE_GENAI_USE_VERTEXAI` set to `true` / `1`, case-insensitive).
pub fn enterprise_requested(env: &dyn Fn(&str) -> Option<String>) -> bool {
    ENTERPRISE_FLAG_ENV_KEYS.iter().any(|key| {
        env(key)
            .is_some_and(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "true" | "1"))
    })
}

/// The platform API base for a location: the global endpoint, or the regional one.
pub fn enterprise_base_url(location: &str) -> String {
    if location == DEFAULT_LOCATION {
        "https://aiplatform.googleapis.com/v1beta1".to_string()
    } else {
        format!("https://{location}-aiplatform.googleapis.com/v1beta1")
    }
}

/// A ready [`Route`] for `model` on the Enterprise platform: the publisher-model path under the project and
/// location, `streamGenerateContent?alt=sse` like the Developer API route, and a bearer token instead of an
/// API key. `base_url` overrides the platform host (gateway / proxy); the path shape stays the same.
pub fn google_enterprise_route(
    project: &str,
    location: &str,
    access_token: &str,
    base_url: Option<&str>,
    model: &str,
) -> Route {
    let model = model.trim();
    let model = model.strip_prefix("models/").unwrap_or(model);
    let base_url = base_url
        .map(str::to_string)
        .unwrap_or_else(|| enterprise_base_url(location));
    Route::new(
        Box::new(GeminiGenerateContentProtocol::new()),
        Endpoint::new(
            base_url,
            format!(
                "/projects/{project}/locations/{location}/publishers/google/models/{model}:streamGenerateContent"
            ),
        )
        .with_query("alt", "sse"),
        Auth::bearer(access_token),
    )
}

/// Why ADC could not produce a usable platform credential. Messages never carry key material.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GoogleAdcError {
    /// No credentials file was configured or found, and no metadata server answered.
    NotFound(String),
    /// The credentials file could not be read.
    Io(String),
    /// The credentials file is not a supported ADC document.
    Parse(String),
    /// The token endpoint refused the exchange or answered with something that is not a token.
    Token(String),
    /// The credentials carry no project id and `GOOGLE_CLOUD_PROJECT` is unset.
    MissingProject,
}

impl fmt::Display for GoogleAdcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GoogleAdcError::NotFound(why) => write!(f, "{why}"),
            GoogleAdcError::Io(why) => write!(f, "cannot read Application Default Credentials: {why}"),
            GoogleAdcError::Parse(why) => {
                write!(f, "Application Default Credentials are not usable: {why}")
            }
            GoogleAdcError::Token(why) => write!(f, "access token exchange failed: {why}"),
            GoogleAdcError::MissingProject => write!(
                f,
                "no Google Cloud project: set GOOGLE_CLOUD_PROJECT (the credentials carry no project id)"
            ),
        }
    }
}

impl std::error::Error for GoogleAdcError {}

/// The credential kinds ADC can hand us. `Debug` redacts every secret field.
#[derive(Clone, PartialEq, Eq)]
pub enum AdcCredentials {
    /// A service-account key file (`"type": "service_account"`).
    ServiceAccount {
        client_email: String,
        private_key_pem: String,
        token_uri: String,
        project_id: Option<String>,
    },
    /// A user credential written by `gcloud auth application-default login` (`"type": "authorized_user"`).
    AuthorizedUser {
        client_id: String,
        client_secret: String,
        refresh_token: String,
        quota_project_id: Option<String>,
    },
}

impl fmt::Debug for AdcCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdcCredentials::ServiceAccount {
                client_email,
                token_uri,
                project_id,
                ..
            } => f
                .debug_struct("ServiceAccount")
                .field("client_email", client_email)
                .field("token_uri", token_uri)
                .field("project_id", project_id)
                .field("private_key_pem", &"<redacted>")
                .finish(),
            AdcCredentials::AuthorizedUser {
                client_id,
                quota_project_id,
                ..
            } => f
                .debug_struct("AuthorizedUser")
                .field("client_id", client_id)
                .field("quota_project_id", quota_project_id)
                .field("client_secret", &"<redacted>")
                .field("refresh_token", &"<redacted>")
                .finish(),
        }
    }
}

#[derive(Deserialize)]
struct AdcFile {
    #[serde(rename = "type")]
    kind: String,
    client_email: Option<String>,
    private_key: Option<String>,
    token_uri: Option<String>,
    project_id: Option<String>,
    client_id: Option<String>,
    client_secret: Option<String>,
    refresh_token: Option<String>,
    quota_project_id: Option<String>,
}

impl AdcCredentials {
    /// Parse an ADC JSON document (the two credential types google-auth writes for local development).
    pub fn parse(json: &str) -> Result<Self, GoogleAdcError> {
        let file: AdcFile = serde_json::from_str(json)
            .map_err(|e| GoogleAdcError::Parse(format!("not an ADC JSON document: {e}")))?;
        let required = |value: Option<String>, name: &str| {
            value.filter(|v| !v.trim().is_empty()).ok_or_else(|| {
                GoogleAdcError::Parse(format!("missing `{name}` for type `{}`", file.kind))
            })
        };
        match file.kind.as_str() {
            "service_account" => Ok(AdcCredentials::ServiceAccount {
                client_email: required(file.client_email.clone(), "client_email")?,
                private_key_pem: required(file.private_key.clone(), "private_key")?,
                token_uri: file
                    .token_uri
                    .clone()
                    .filter(|v| !v.trim().is_empty())
                    .unwrap_or_else(|| OAUTH_TOKEN_URL.to_string()),
                project_id: file.project_id.clone().filter(|v| !v.trim().is_empty()),
            }),
            "authorized_user" => Ok(AdcCredentials::AuthorizedUser {
                client_id: required(file.client_id.clone(), "client_id")?,
                client_secret: required(file.client_secret.clone(), "client_secret")?,
                refresh_token: required(file.refresh_token.clone(), "refresh_token")?,
                quota_project_id: file.quota_project_id.clone().filter(|v| !v.trim().is_empty()),
            }),
            other => Err(GoogleAdcError::Parse(format!(
                "unsupported credential type `{other}` (supported: service_account, authorized_user)"
            ))),
        }
    }

    /// The project the credentials belong to, when the document names one.
    pub fn project_id(&self) -> Option<&str> {
        match self {
            AdcCredentials::ServiceAccount { project_id, .. } => project_id.as_deref(),
            AdcCredentials::AuthorizedUser {
                quota_project_id, ..
            } => quota_project_id.as_deref(),
        }
    }
}

/// Where ADC looks for a credentials file: the explicit `GOOGLE_APPLICATION_CREDENTIALS` path (returned
/// even if it does not exist, so a typo is reported rather than silently falling through), otherwise the
/// gcloud user file when it exists.
pub fn adc_credentials_path(env: &dyn Fn(&str) -> Option<String>) -> Option<PathBuf> {
    let non_empty = |value: Option<String>| {
        value
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    };
    if let Some(explicit) = non_empty(env(CREDENTIALS_FILE_ENV_KEY)) {
        return Some(PathBuf::from(explicit));
    }
    let config_dir = non_empty(env("CLOUDSDK_CONFIG"))
        .map(PathBuf::from)
        .or_else(|| {
            if cfg!(windows) {
                non_empty(env("APPDATA")).map(|dir| PathBuf::from(dir).join("gcloud"))
            } else {
                non_empty(env("HOME"))
                    .map(|home| PathBuf::from(home).join(".config").join("gcloud"))
            }
        })?;
    let path = config_dir.join("application_default_credentials.json");
    path.is_file().then_some(path)
}

/// Load the ADC credentials file, if one is configured or present.
pub fn load_adc_credentials(
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<Option<AdcCredentials>, GoogleAdcError> {
    let Some(path) = adc_credentials_path(env) else {
        return Ok(None);
    };
    let json = std::fs::read_to_string(&path)
        .map_err(|e| GoogleAdcError::Io(format!("{}: {e}", path.display())))?;
    AdcCredentials::parse(&json).map(Some).map_err(|e| match e {
        GoogleAdcError::Parse(why) => GoogleAdcError::Parse(format!("{}: {why}", path.display())),
        other => other,
    })
}

/// An OAuth2 access token. `Debug` redacts the token.
#[derive(Clone, PartialEq, Eq)]
pub struct AccessToken {
    pub token: String,
    pub expires_in_secs: Option<u64>,
}

impl fmt::Debug for AccessToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AccessToken")
            .field("token", &"<redacted>")
            .field("expires_in_secs", &self.expires_in_secs)
            .finish()
    }
}

/// Exchange the credentials for an access token at the OAuth2 token endpoint (service-account JWT
/// assertion, or authorized-user refresh token). `token_endpoint_override` exists for tests.
pub fn exchange_access_token(
    credentials: &AdcCredentials,
    token_endpoint_override: Option<&str>,
) -> Result<AccessToken, GoogleAdcError> {
    match credentials {
        AdcCredentials::ServiceAccount {
            client_email,
            private_key_pem,
            token_uri,
            ..
        } => {
            let endpoint = token_endpoint_override.unwrap_or(token_uri);
            let assertion =
                sign_service_account_jwt(client_email, private_key_pem, endpoint, unix_now())?;
            let form = vec![
                (
                    "grant_type".to_string(),
                    "urn:ietf:params:oauth:grant-type:jwt-bearer".to_string(),
                ),
                ("assertion".to_string(), assertion),
            ];
            parse_token_response(post_form(endpoint, form, TOKEN_REQUEST_TIMEOUT)?)
        }
        AdcCredentials::AuthorizedUser {
            client_id,
            client_secret,
            refresh_token,
            ..
        } => {
            let endpoint = token_endpoint_override.unwrap_or(OAUTH_TOKEN_URL);
            let form = vec![
                ("grant_type".to_string(), "refresh_token".to_string()),
                ("client_id".to_string(), client_id.clone()),
                ("client_secret".to_string(), client_secret.clone()),
                ("refresh_token".to_string(), refresh_token.clone()),
            ];
            parse_token_response(post_form(endpoint, form, TOKEN_REQUEST_TIMEOUT)?)
        }
    }
}

/// The GCE metadata server's token and project, when the process runs on Google Cloud. `Ok(None)` when
/// the metadata host is unreachable (the normal case on a workstation).
pub fn metadata_server_credentials() -> Result<Option<(AccessToken, Option<String>)>, GoogleAdcError>
{
    let headers = [("Metadata-Flavor", "Google")];
    let (status, body) = match get_text(METADATA_TOKEN_URL, &headers, METADATA_REQUEST_TIMEOUT) {
        Ok(response) => response,
        Err(_) => return Ok(None),
    };
    if status != 200 {
        return Ok(None);
    }
    let token = parse_token_response((status, body))?;
    let project = get_text(METADATA_PROJECT_URL, &headers, METADATA_REQUEST_TIMEOUT)
        .ok()
        .filter(|(status, _)| *status == 200)
        .map(|(_, text)| text.trim().to_string())
        .filter(|text| !text.is_empty());
    Ok(Some((token, project)))
}

/// Everything the platform route needs. `Debug` redacts the token.
#[derive(Clone, PartialEq, Eq)]
pub struct GoogleEnterpriseCredentials {
    pub project: String,
    pub location: String,
    pub access_token: String,
}

impl fmt::Debug for GoogleEnterpriseCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GoogleEnterpriseCredentials")
            .field("project", &self.project)
            .field("location", &self.location)
            .field("access_token", &"<redacted>")
            .finish()
    }
}

/// Resolve project, location and a fresh access token for the Enterprise platform from the environment and
/// ADC (file first, then the metadata server). `env` abstracts process env for testability.
pub fn resolve_enterprise_credentials(
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<GoogleEnterpriseCredentials, GoogleAdcError> {
    let non_empty = |value: Option<String>| {
        value
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    };
    let location = non_empty(env(LOCATION_ENV_KEY)).unwrap_or_else(|| DEFAULT_LOCATION.to_string());
    let env_project = non_empty(env(PROJECT_ENV_KEY));

    let (token, credential_project) = match load_adc_credentials(env)? {
        Some(credentials) => (
            exchange_access_token(&credentials, None)?,
            credentials.project_id().map(str::to_string),
        ),
        None => match metadata_server_credentials()? {
            Some((token, project)) => (token, project),
            None => {
                return Err(GoogleAdcError::NotFound(
                    "no Application Default Credentials: set GOOGLE_APPLICATION_CREDENTIALS to a \
                     service-account key file, run `gcloud auth application-default login`, or run on \
                     Google Cloud"
                        .to_string(),
                ))
            }
        },
    };
    let project = env_project
        .or(credential_project)
        .ok_or(GoogleAdcError::MissingProject)?;
    Ok(GoogleEnterpriseCredentials {
        project,
        location,
        access_token: token.token,
    })
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Build and sign the RS256 JWT assertion a service account trades for an access token.
fn sign_service_account_jwt(
    client_email: &str,
    private_key_pem: &str,
    audience: &str,
    now_secs: u64,
) -> Result<String, GoogleAdcError> {
    let der = pem_to_der(private_key_pem)?;
    let key_pair = ring::signature::RsaKeyPair::from_pkcs8(&der).map_err(|e| {
        GoogleAdcError::Parse(format!(
            "service account private_key is not a PKCS#8 RSA key: {e}"
        ))
    })?;
    let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
    let header = engine.encode(br#"{"alg":"RS256","typ":"JWT"}"#);
    let claims = serde_json::json!({
        "iss": client_email,
        "scope": CLOUD_PLATFORM_SCOPE,
        "aud": audience,
        "iat": now_secs,
        "exp": now_secs + JWT_LIFETIME_SECS,
    });
    let claims = engine
        .encode(serde_json::to_vec(&claims).map_err(|e| GoogleAdcError::Token(e.to_string()))?);
    let signing_input = format!("{header}.{claims}");
    let mut signature = vec![0u8; key_pair.public().modulus_len()];
    key_pair
        .sign(
            &ring::signature::RSA_PKCS1_SHA256,
            &ring::rand::SystemRandom::new(),
            signing_input.as_bytes(),
            &mut signature,
        )
        .map_err(|_| {
            GoogleAdcError::Token("signing the service account assertion failed".to_string())
        })?;
    Ok(format!("{signing_input}.{}", engine.encode(&signature)))
}

fn pem_to_der(pem: &str) -> Result<Vec<u8>, GoogleAdcError> {
    let body: String = pem
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with("-----"))
        .collect();
    base64::engine::general_purpose::STANDARD
        .decode(body)
        .map_err(|e| GoogleAdcError::Parse(format!("private_key PEM body is not base64: {e}")))
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    expires_in: Option<u64>,
}

fn parse_token_response((status, body): (u16, String)) -> Result<AccessToken, GoogleAdcError> {
    if !(200..300).contains(&status) {
        let detail: String = body.chars().take(300).collect();
        return Err(GoogleAdcError::Token(format!(
            "token endpoint answered HTTP {status}: {detail}"
        )));
    }
    let token: TokenResponse = serde_json::from_str(&body).map_err(|e| {
        GoogleAdcError::Token(format!(
            "token endpoint answered without an access_token: {e}"
        ))
    })?;
    if token.access_token.trim().is_empty() {
        return Err(GoogleAdcError::Token(
            "token endpoint answered with an empty access_token".to_string(),
        ));
    }
    Ok(AccessToken {
        token: token.access_token,
        expires_in_secs: token.expires_in,
    })
}

/// `reqwest::blocking` owns its own runtime; running it on a dedicated thread keeps it clear of whatever
/// async runtime the caller may be on.
fn run_blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, GoogleAdcError> + Send + 'static,
) -> Result<T, GoogleAdcError> {
    std::thread::spawn(work)
        .join()
        .map_err(|_| GoogleAdcError::Token("credential request thread panicked".to_string()))?
}

fn post_form(
    url: &str,
    form: Vec<(String, String)>,
    timeout: Duration,
) -> Result<(u16, String), GoogleAdcError> {
    let url = url.to_string();
    run_blocking(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| GoogleAdcError::Token(e.to_string()))?;
        let response = client
            .post(&url)
            .form(&form)
            .send()
            .map_err(|e| GoogleAdcError::Token(format!("POST {url}: {e}")))?;
        let status = response.status().as_u16();
        let text = response
            .text()
            .map_err(|e| GoogleAdcError::Token(e.to_string()))?;
        Ok((status, text))
    })
}

fn get_text(
    url: &str,
    headers: &[(&str, &str)],
    timeout: Duration,
) -> Result<(u16, String), GoogleAdcError> {
    let url = url.to_string();
    let headers: Vec<(String, String)> = headers
        .iter()
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect();
    run_blocking(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| GoogleAdcError::Token(e.to_string()))?;
        let mut request = client.get(&url);
        for (name, value) in &headers {
            request = request.header(name.as_str(), value.as_str());
        }
        let response = request
            .send()
            .map_err(|e| GoogleAdcError::Token(format!("GET {url}: {e}")))?;
        let status = response.status().as_u16();
        let text = response
            .text()
            .map_err(|e| GoogleAdcError::Token(e.to_string()))?;
        Ok((status, text))
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;

    use super::*;

    /// A throwaway 2048-bit RSA key generated for these tests; it has never guarded anything.
    const TEST_KEY_PEM: &str = "-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDOeMvZEOmJKyf9
5oq156HYz6ntf++kl/24x9w8hAIbTapPem4JT0f0LKhez+n0D5i5bToWMEzc36Za
hsQuzzW1wZvcaYPBB1sS3MUDulg7O31AtlIMaUaUOP1m9WRjg43PajEQwxBopI/D
H1PRW+5ob3nKh+AL3oVCMJ3rfvKYM4LteJvUr/cGFimulK0W44xlPADwXkIuj/+q
3BOsfTV0mDmlrveMmLyFQ6ebUdnDE2/z22rbyxR5/Dc7sV9jPXC1CGx2Pb+NhJtD
3x7rEpaGZrh8/QS9yHf7yNBjvbWhRCWtlaB4GOSGh7g8QpU0Df/LlN9kawpeMCzl
u2sSZfIXAgMBAAECggEAJHipBXxbzEuPwmVNXkl+yNYR3a6KLchdXJ0NNbFZz+nd
qjFlq4ZSHvseeLUATgjNQP/e0VmR0jg9YeNJVz1p7uO2vO30hI3glAnkUT7C5B+j
heFOx/t2qKsSc/Pn2j48OAW5igzy4pEjBeQMQcCb8d2zJUfwFxik8Ga3FoizoJez
kM8r2AFHW1Hg2ZD8kwdrIx/Ct+t0gF5Ep1EWgspJOY3bN6pGwQXsHnVFADS8Rk+V
+ZsgygMnGOfarE0OTVVXuZoKM+7vftLpLhsGINQKzbHfYiuLCu0dbQJYi1GJkH+F
Cr8/80sf/hcp7S986VIZ8V7LOCBwXcXifHnjBax2mQKBgQDnqSxY229pSuFofVJn
4ThRjFLZNgdSjtwj0Rk66NbgioRbdwNPvZAr31pvs3AmNOw5MOT3KbxALhh5pQZe
/lE8LVtClfxbCgh/Pv7pSqSEhGv2LuP6EzrA8gnt4RkDCKr1aZOt4TAmQFt+TmT3
YDdBOY66BlfhdBLko8+juQHb6QKBgQDkKiHPlIKcSn+BqSnPNdLl9nIqHLhLKuzE
oRmuJIYIwhJvgJ/QvBP70GTeT/oatwiXUSyOOpyZJA2ALNk6Y9jSOXcZiNYvyeWo
imTriavPHpGYYzIrVDT29IV/l4GBlIQ8H7C6/plTjzMz6uSuAgjCils6CNhLCtv8
UmjdgX6d/wKBgQDNnOG3nEwucD/QtirF+Kg7+Qgvsjydkh7hy11RZAgUE92WMPPH
esDTr9JdAi6wOpjPyg3vMiPb8w3WKhqVhbQkbT1qxCTPyP/xsMAf4BhVZhNM1iGS
fC7RwgH+GKRShQb3+1UQJ6bMzKJoYLxMtCFWyvdbNPYbLSL3vE03S6XSwQKBgQDX
rTP0ws+L6agJ6txIsPpAwus7/5X+B/90B0z9Hi1NDhbbK9zmNUM2n9EZjMRAiJAo
aWUStTjeUgQpCzv0Q/nrDsMUchL1bYYYAWpQc/zidmTuhaYg6yJ95rpi19ZlpPwy
0UoAGTbak9y00O+FxQp+ysZdX4sFtKgqJcRWzlfgBQKBgETeSWJmSxJf1s+bdMWx
4jK73Yvr7MGTcwi5or8mT7LaaGdUJkNmmdM7jKW8+ACadv9T5H/wwVnZSlqq8Cjh
aoOmcGzN4ZpnDGd/saNl/I8j77dZO8ko3QuB96/WNgGUfrvZnBQwH88WOkr3dFsH
MyKiAO0d/TYbtchTmDLTmU+M
-----END PRIVATE KEY-----";

    fn env_from(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |key: &str| map.get(key).cloned()
    }

    fn header(route: &Route, name: &str) -> Option<String> {
        route
            .auth
            .headers()
            .into_iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v)
    }

    /// One-shot local HTTP server: records the raw request, answers `body` as JSON.
    fn serve_once(body: &'static str) -> (String, mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut raw = Vec::new();
            let mut chunk = [0u8; 4096];
            loop {
                let n = stream.read(&mut chunk).expect("read");
                if n == 0 {
                    break;
                }
                raw.extend_from_slice(&chunk[..n]);
                if let Some(end) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                    let head = String::from_utf8_lossy(&raw[..end]).to_string();
                    let length = head
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.trim()
                                .eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    if raw.len() >= end + 4 + length {
                        break;
                    }
                }
            }
            tx.send(String::from_utf8_lossy(&raw).to_string())
                .expect("send");
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).expect("write");
        });
        (format!("http://127.0.0.1:{port}/token"), rx)
    }

    #[test]
    fn enterprise_flag_accepts_both_sdk_keys_and_truthy_spellings() {
        assert!(enterprise_requested(&env_from(&[(
            "GOOGLE_GENAI_USE_ENTERPRISE",
            "true"
        )])));
        assert!(enterprise_requested(&env_from(&[(
            "GOOGLE_GENAI_USE_VERTEXAI",
            "1"
        )])));
        assert!(enterprise_requested(&env_from(&[(
            "GOOGLE_GENAI_USE_ENTERPRISE",
            " TRUE "
        )])));
        assert!(!enterprise_requested(&env_from(&[(
            "GOOGLE_GENAI_USE_ENTERPRISE",
            "false"
        )])));
        assert!(!enterprise_requested(&env_from(&[(
            "GOOGLE_GENAI_USE_ENTERPRISE",
            ""
        )])));
        assert!(!enterprise_requested(&env_from(&[])));
    }

    #[test]
    fn base_url_is_global_or_regional() {
        assert_eq!(
            enterprise_base_url("global"),
            "https://aiplatform.googleapis.com/v1beta1"
        );
        assert_eq!(
            enterprise_base_url("us-central1"),
            "https://us-central1-aiplatform.googleapis.com/v1beta1"
        );
    }

    #[test]
    fn route_targets_the_platform_publisher_model_with_a_bearer_token() {
        let route = google_enterprise_route(
            "proj",
            "global",
            "tok-secret",
            None,
            "models/gemini-3-flash-preview",
        );
        assert_eq!(
            route.endpoint.url(),
            "https://aiplatform.googleapis.com/v1beta1/projects/proj/locations/global/publishers/google/models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
        );
        assert_eq!(
            header(&route, "authorization").as_deref(),
            Some("Bearer tok-secret")
        );
        assert!(header(&route, "x-goog-api-key").is_none());

        let regional = google_enterprise_route(
            "proj",
            "europe-west4",
            "t",
            Some("https://gw.example.com/v1beta1"),
            "gemini-3.1-pro-preview",
        );
        assert_eq!(
            regional.endpoint.url(),
            "https://gw.example.com/v1beta1/projects/proj/locations/europe-west4/publishers/google/models/gemini-3.1-pro-preview:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn parses_service_account_and_authorized_user_documents() {
        let service_account = AdcCredentials::parse(
            r#"{"type":"service_account","project_id":"proj","client_email":"sa@proj.iam.gserviceaccount.com","private_key":"-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n","token_uri":"https://oauth2.googleapis.com/token"}"#,
        )
        .expect("service account parses");
        assert_eq!(service_account.project_id(), Some("proj"));
        assert!(
            matches!(service_account, AdcCredentials::ServiceAccount { ref client_email, .. } if client_email == "sa@proj.iam.gserviceaccount.com")
        );
        assert!(
            !format!("{service_account:?}").contains("abc"),
            "debug must redact the key"
        );

        let user = AdcCredentials::parse(
            r#"{"type":"authorized_user","client_id":"cid","client_secret":"csecret","refresh_token":"rtoken","quota_project_id":"quota-proj"}"#,
        )
        .expect("authorized user parses");
        assert_eq!(user.project_id(), Some("quota-proj"));
        let debug = format!("{user:?}");
        assert!(
            !debug.contains("csecret") && !debug.contains("rtoken"),
            "debug must redact secrets"
        );

        let err = AdcCredentials::parse(r#"{"type":"external_account"}"#).unwrap_err();
        assert!(matches!(err, GoogleAdcError::Parse(ref why) if why.contains("external_account")));
        let err =
            AdcCredentials::parse(r#"{"type":"service_account","client_email":"x"}"#).unwrap_err();
        assert!(matches!(err, GoogleAdcError::Parse(ref why) if why.contains("private_key")));
    }

    #[test]
    fn credentials_path_prefers_explicit_env_then_the_gcloud_config_dir() {
        let explicit = adc_credentials_path(&env_from(&[(
            "GOOGLE_APPLICATION_CREDENTIALS",
            "/tmp/sa.json",
        )]));
        assert_eq!(
            explicit,
            Some(PathBuf::from("/tmp/sa.json")),
            "explicit path wins even if absent"
        );

        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("application_default_credentials.json");
        std::fs::write(&file, "{}").expect("write");
        let found = adc_credentials_path(&env_from(&[(
            "CLOUDSDK_CONFIG",
            dir.path().to_str().unwrap(),
        )]));
        assert_eq!(found, Some(file));

        let empty = tempfile::tempdir().expect("tempdir");
        assert_eq!(
            adc_credentials_path(&env_from(&[(
                "CLOUDSDK_CONFIG",
                empty.path().to_str().unwrap()
            )])),
            None
        );
    }

    #[test]
    fn service_account_assertion_is_rs256_signed_and_verifiable() {
        let jwt = sign_service_account_jwt(
            "sa@proj.iam.gserviceaccount.com",
            TEST_KEY_PEM,
            "https://oauth2.googleapis.com/token",
            1_700_000_000,
        )
        .expect("signs");
        let parts: Vec<&str> = jwt.split('.').collect();
        assert_eq!(parts.len(), 3);
        let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let header: serde_json::Value =
            serde_json::from_slice(&engine.decode(parts[0]).unwrap()).unwrap();
        assert_eq!(header["alg"], "RS256");
        let claims: serde_json::Value =
            serde_json::from_slice(&engine.decode(parts[1]).unwrap()).unwrap();
        assert_eq!(claims["iss"], "sa@proj.iam.gserviceaccount.com");
        assert_eq!(claims["aud"], "https://oauth2.googleapis.com/token");
        assert_eq!(claims["scope"], CLOUD_PLATFORM_SCOPE);
        assert_eq!(claims["iat"], 1_700_000_000u64);
        assert_eq!(claims["exp"], 1_700_003_600u64);

        let key_pair =
            ring::signature::RsaKeyPair::from_pkcs8(&pem_to_der(TEST_KEY_PEM).unwrap()).unwrap();
        let public_key = ring::signature::UnparsedPublicKey::new(
            &ring::signature::RSA_PKCS1_2048_8192_SHA256,
            key_pair.public().as_ref(),
        );
        let signing_input = format!("{}.{}", parts[0], parts[1]);
        public_key
            .verify(signing_input.as_bytes(), &engine.decode(parts[2]).unwrap())
            .expect("signature verifies with the key's public half");
    }

    #[test]
    fn service_account_exchange_posts_a_jwt_bearer_assertion() {
        let (endpoint, requests) =
            serve_once(r#"{"access_token":"ya29.token","expires_in":3599,"token_type":"Bearer"}"#);
        let credentials = AdcCredentials::ServiceAccount {
            client_email: "sa@proj.iam.gserviceaccount.com".into(),
            private_key_pem: TEST_KEY_PEM.into(),
            token_uri: endpoint.clone(),
            project_id: Some("proj".into()),
        };
        let token = exchange_access_token(&credentials, None).expect("exchange succeeds");
        assert_eq!(token.token, "ya29.token");
        assert_eq!(token.expires_in_secs, Some(3599));
        assert!(
            !format!("{token:?}").contains("ya29"),
            "debug must redact the token"
        );

        let request = requests.recv().expect("request recorded");
        assert!(request.starts_with("POST /token HTTP/1.1"), "{request}");
        assert!(
            request.contains("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer"),
            "{request}"
        );
        assert!(request.contains("&assertion=eyJ"), "{request}");
    }

    #[test]
    fn authorized_user_exchange_posts_the_refresh_token() {
        let (endpoint, requests) = serve_once(r#"{"access_token":"ya29.user","expires_in":3599}"#);
        let credentials = AdcCredentials::AuthorizedUser {
            client_id: "cid".into(),
            client_secret: "csecret".into(),
            refresh_token: "rtoken".into(),
            quota_project_id: None,
        };
        let token =
            exchange_access_token(&credentials, Some(&endpoint)).expect("exchange succeeds");
        assert_eq!(token.token, "ya29.user");
        let request = requests.recv().expect("request recorded");
        assert!(request.contains("grant_type=refresh_token"), "{request}");
        assert!(request.contains("client_id=cid"), "{request}");
        assert!(request.contains("client_secret=csecret"), "{request}");
        assert!(request.contains("refresh_token=rtoken"), "{request}");
    }

    #[test]
    fn token_endpoint_errors_are_typed_and_truncated() {
        let (endpoint, _requests) = serve_once(r#"{"error":"invalid_grant"}"#);
        // The fake answers 200 with no access_token: that is a malformed token response.
        let credentials = AdcCredentials::AuthorizedUser {
            client_id: "cid".into(),
            client_secret: "csecret".into(),
            refresh_token: "rtoken".into(),
            quota_project_id: None,
        };
        let err = exchange_access_token(&credentials, Some(&endpoint)).unwrap_err();
        assert!(
            matches!(err, GoogleAdcError::Token(ref why) if why.contains("without an access_token")),
            "{err}"
        );
    }

    #[test]
    fn resolve_uses_env_project_and_location_over_the_credentials_file() {
        let (endpoint, _requests) = serve_once(r#"{"access_token":"ya29.sa","expires_in":3599}"#);
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("sa.json");
        let document = serde_json::json!({
            "type": "service_account",
            "project_id": "file-project",
            "client_email": "sa@proj.iam.gserviceaccount.com",
            "private_key": TEST_KEY_PEM,
            "token_uri": endpoint,
        });
        std::fs::write(&file, serde_json::to_vec(&document).unwrap()).expect("write");

        let resolved = resolve_enterprise_credentials(&env_from(&[
            ("GOOGLE_APPLICATION_CREDENTIALS", file.to_str().unwrap()),
            ("GOOGLE_CLOUD_PROJECT", "env-project"),
            ("GOOGLE_CLOUD_LOCATION", "us-central1"),
        ]))
        .expect("resolves");
        assert_eq!(resolved.project, "env-project");
        assert_eq!(resolved.location, "us-central1");
        assert_eq!(resolved.access_token, "ya29.sa");
        assert!(
            !format!("{resolved:?}").contains("ya29"),
            "debug must redact the token"
        );
    }

    #[test]
    fn resolve_falls_back_to_the_file_project_and_global_location() {
        let (endpoint, _requests) = serve_once(r#"{"access_token":"ya29.sa","expires_in":3599}"#);
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("sa.json");
        let document = serde_json::json!({
            "type": "service_account",
            "project_id": "file-project",
            "client_email": "sa@proj.iam.gserviceaccount.com",
            "private_key": TEST_KEY_PEM,
            "token_uri": endpoint,
        });
        std::fs::write(&file, serde_json::to_vec(&document).unwrap()).expect("write");

        let resolved = resolve_enterprise_credentials(&env_from(&[(
            "GOOGLE_APPLICATION_CREDENTIALS",
            file.to_str().unwrap(),
        )]))
        .expect("resolves");
        assert_eq!(resolved.project, "file-project");
        assert_eq!(resolved.location, "global");
    }

    #[test]
    fn missing_explicit_file_is_an_io_error_not_a_silent_fallthrough() {
        let err = resolve_enterprise_credentials(&env_from(&[(
            "GOOGLE_APPLICATION_CREDENTIALS",
            "/nonexistent/sa.json",
        )]))
        .unwrap_err();
        assert!(matches!(err, GoogleAdcError::Io(_)), "{err}");
    }
}
