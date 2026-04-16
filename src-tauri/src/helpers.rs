pub(crate) fn lm_base_url() -> String {
    let url = std::env::var("LM_STUDIO_URL")
        .unwrap_or_else(|_| "http://localhost:1234/api/v1/chat".to_string());
    url.trim_end_matches('/')
        .trim_end_matches("/chat")
        .trim_end_matches("/v1")
        .trim_end_matches("/api")
        .to_string()
}

pub(crate) fn default_true() -> bool { true }
pub(crate) fn default_priority() -> u8 { 128 }
