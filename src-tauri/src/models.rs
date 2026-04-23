use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use crate::agent_config::{AgentLoadConfig, BackendInstance};
use crate::helpers::{backend_api_key, backend_extra_instances, backend_type, lm_base_url};
use crate::state::AppState;

#[derive(Serialize)]
pub struct ModelInfo {
    pub key: String,
    pub display_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_context_length: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quantization: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    params_string: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vision: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    trained_for_tool_use: Option<bool>,
}

#[derive(Serialize)]
pub struct LoadedInstance {
    instance_id: String,
    context_length: Option<u64>,
    flash_attention: Option<bool>,
}

#[derive(Serialize)]
pub struct LoadedModelInfo {
    key: String,
    display_name: String,
    instances: Vec<LoadedInstance>,
}

fn auth_header(client: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    if let Some(key) = backend_api_key() {
        client.header("Authorization", format!("Bearer {}", key))
    } else {
        client
    }
}

fn parse_lmstudio_models(data: &Value) -> Result<Vec<ModelInfo>, String> {
    let models = data
        .get("models")
        .and_then(|v| v.as_array())
        .ok_or("Invalid models response")?;
    Ok(models
        .iter()
        .map(|m| {
            let capabilities = m.get("capabilities");
            ModelInfo {
                key: m["key"].as_str().unwrap_or("").to_string(),
                display_name: m
                    .get("display_name")
                    .or_else(|| m.get("key"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                model_type: m["type"].as_str().map(|s| s.to_string()),
                max_context_length: m["max_context_length"].as_u64(),
                format: m["format"].as_str().map(|s| s.to_string()),
                quantization: m
                    .get("quantization")
                    .and_then(|q| q.get("name"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                params_string: m["params_string"].as_str().map(|s| s.to_string()),
                vision: capabilities.and_then(|c| c["vision"].as_bool()),
                trained_for_tool_use: capabilities
                    .and_then(|c| c["trained_for_tool_use"].as_bool()),
            }
        })
        .collect())
}

fn parse_ollama_models(data: &Value) -> Result<Vec<ModelInfo>, String> {
    let models = data
        .get("models")
        .and_then(|v| v.as_array())
        .ok_or("Invalid Ollama models response")?;
    Ok(models
        .iter()
        .map(|m| {
            let details = m.get("details");
            let name = m["name"].as_str().unwrap_or("").to_string();
            ModelInfo {
                key: name.clone(),
                display_name: name,
                model_type: details
                    .and_then(|d| d["family"].as_str())
                    .map(|s| s.to_string()),
                max_context_length: None,
                format: details
                    .and_then(|d| d["format"].as_str())
                    .map(|s| s.to_string()),
                quantization: details
                    .and_then(|d| d["quantization_level"].as_str())
                    .map(|s| s.to_string()),
                params_string: details
                    .and_then(|d| d["parameter_size"].as_str())
                    .map(|s| s.to_string()),
                vision: None,
                trained_for_tool_use: None,
            }
        })
        .collect())
}

fn parse_openai_models(data: &Value) -> Result<Vec<ModelInfo>, String> {
    let models = data
        .get("data")
        .and_then(|v| v.as_array())
        .ok_or("Invalid OpenAI-format models response")?;
    Ok(models
        .iter()
        .map(|m| {
            let id = m["id"].as_str().unwrap_or("").to_string();
            ModelInfo {
                key: id.clone(),
                display_name: id,
                model_type: m["owned_by"].as_str().map(|s| s.to_string()),
                max_context_length: None,
                format: None,
                quantization: None,
                params_string: None,
                vision: None,
                trained_for_tool_use: None,
            }
        })
        .collect())
}

#[tauri::command]
pub async fn fetch_models() -> Result<Vec<ModelInfo>, String> {
    let base = lm_base_url();
    let btype = backend_type();
    let client = reqwest::Client::new();
    let url = match btype.as_str() {
        "ollama" => format!("{}/api/tags", base),
        "llamacpp" => format!("{}/v1/models", base),
        _ => format!("{}/api/v1/models", base),
    };
    let resp = auth_header(client.get(&url))
        .send()
        .await
        .map_err(|e| format!("Failed to fetch models: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!("Failed to fetch models: {}", resp.status()));
    }
    let data: Value = resp
        .json()
        .await
        .map_err(|e| format!("Invalid response: {}", e))?;
    let mut models = match btype.as_str() {
        "ollama" => parse_ollama_models(&data),
        "llamacpp" => parse_openai_models(&data),
        _ => parse_lmstudio_models(&data),
    }?;

    // Fan out to extra backend instances and merge results (deduplicate by key).
    let mut seen_keys: std::collections::HashSet<String> =
        models.iter().map(|m| m.key.clone()).collect();
    for (extra_url, extra_type) in backend_extra_instances() {
        let extra_url_str = match extra_type.as_str() {
            "ollama" => format!("{}/api/tags", extra_url),
            _ => format!("{}/v1/models", extra_url),
        };
        if let Ok(resp) = client.get(&extra_url_str).send().await {
            if resp.status().is_success() {
                if let Ok(extra_data) = resp.json::<Value>().await {
                    let extra_models = match extra_type.as_str() {
                        "ollama" => parse_ollama_models(&extra_data),
                        _ => parse_openai_models(&extra_data),
                    }
                    .unwrap_or_default();
                    for m in extra_models {
                        if seen_keys.insert(m.key.clone()) {
                            models.push(m);
                        }
                    }
                }
            }
        }
    }

    Ok(models)
}

#[tauri::command]
pub async fn fetch_loaded_models() -> Result<Vec<LoadedModelInfo>, String> {
    let base = lm_base_url();
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/api/v1/models", base))
        .send()
        .await
        .map_err(|e| format!("Failed to fetch models: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!("Failed to fetch models: {}", resp.status()));
    }
    let data: Value = resp
        .json()
        .await
        .map_err(|e| format!("Invalid response: {}", e))?;
    let models = data
        .get("models")
        .and_then(|v| v.as_array())
        .ok_or("Invalid models response")?;
    Ok(models
        .iter()
        .filter_map(|m| {
            let instances = m.get("loaded_instances").and_then(|v| v.as_array())?;
            if instances.is_empty() {
                return None;
            }
            Some(LoadedModelInfo {
                key: m["key"].as_str().unwrap_or("").to_string(),
                display_name: m
                    .get("display_name")
                    .or_else(|| m.get("key"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                instances: instances
                    .iter()
                    .map(|inst| LoadedInstance {
                        instance_id: inst["id"].as_str().unwrap_or("").to_string(),
                        context_length: inst["context_length"].as_u64(),
                        flash_attention: inst["flash_attention"].as_bool(),
                    })
                    .collect(),
            })
        })
        .collect())
}

#[tauri::command]
pub async fn set_model(
    state: tauri::State<'_, Arc<AppState>>,
    model: String,
) -> Result<bool, String> {
    if !model.is_empty() {
        *state.current_model.lock().unwrap() = model;
    }
    Ok(true)
}

#[derive(Deserialize)]
pub struct LoadConfig {
    pub model: String,
    #[serde(default)]
    pub context_length: Option<u64>,
    #[serde(default)]
    pub eval_batch_size: Option<u64>,
    #[serde(default)]
    pub flash_attention: Option<bool>,
    #[serde(default)]
    pub num_experts: Option<u64>,
    #[serde(default)]
    pub offload_kv_cache_to_gpu: Option<bool>,
}

pub async fn load_model_internal(
    config: &AgentLoadConfig,
    model_key: &str,
) -> Result<(), String> {
    let url = format!("{}/api/v1/models/load", lm_base_url());
    let mut body = json!({ "model": model_key });
    if let Some(v) = config.context_length {
        body["context_length"] = json!(v);
    }
    if let Some(v) = config.eval_batch_size {
        body["eval_batch_size"] = json!(v);
    }
    if let Some(v) = config.flash_attention {
        body["flash_attention"] = json!(v);
    }
    if let Some(v) = config.num_experts {
        body["num_experts"] = json!(v);
    }
    if let Some(v) = config.offload_kv_cache_to_gpu {
        body["offload_kv_cache_to_gpu"] = json!(v);
    }
    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Load failed: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!(
            "Load failed: {}",
            resp.text().await.unwrap_or_default()
        ));
    }
    Ok(())
}

pub async fn unload_model_internal(instance_id: &str) -> Result<(), String> {
    let url = format!("{}/api/v1/models/unload", lm_base_url());
    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .json(&json!({ "instance_id": instance_id }))
        .send()
        .await
        .map_err(|e| format!("Unload failed: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!(
            "Unload failed: {}",
            resp.text().await.unwrap_or_default()
        ));
    }
    Ok(())
}

// Maximum chars per embedding chunk (~1800 tokens with 4 chars/token, leaving headroom in a
// 2048-token context window for model overhead and the prompt template).
const EMBEDDING_CHUNK_CHARS: usize = 7_200;

fn split_into_chunks(text: &str, max_chars: usize) -> Vec<String> {
    if text.chars().count() <= max_chars {
        return vec![text.to_string()];
    }
    let mut chunks = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if !current.is_empty() && current.chars().count() + 1 + word.chars().count() > max_chars {
            chunks.push(current.trim().to_string());
            current = String::new();
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }
    chunks
}

fn mean_pool(embeddings: Vec<Vec<f32>>) -> Vec<f32> {
    if embeddings.is_empty() {
        return vec![];
    }
    if embeddings.len() == 1 {
        return embeddings.into_iter().next().unwrap();
    }
    let dim = embeddings[0].len();
    let n = embeddings.len() as f32;
    let mut result = vec![0f32; dim];
    for emb in &embeddings {
        for (r, v) in result.iter_mut().zip(emb.iter()) {
            *r += v;
        }
    }
    for r in &mut result {
        *r /= n;
    }
    // Re-normalise so cosine similarity stays valid after pooling.
    let norm: f32 = result.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for r in &mut result {
            *r /= norm;
        }
    }
    result
}

async fn embed_single_batch(
    client: &reqwest::Client,
    url: &str,
    model_key: &str,
    batch: &[String],
) -> Result<Vec<Vec<f32>>, String> {
    let resp = client
        .post(url)
        .json(&json!({ "model": model_key, "input": batch }))
        .send()
        .await
        .map_err(|e| format!("Embedding request failed: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!(
            "Embedding request failed: {}",
            resp.text().await.unwrap_or_default()
        ));
    }
    let data: Value = resp
        .json()
        .await
        .map_err(|e| format!("Invalid embeddings response: {}", e))?;
    data["data"]
        .as_array()
        .ok_or_else(|| "Invalid embeddings response: missing data array".to_string())?
        .iter()
        .map(|item| {
            item["embedding"]
                .as_array()
                .ok_or_else(|| "Invalid embeddings response: missing embedding".to_string())
                .and_then(|values| {
                    values
                        .iter()
                        .map(|v| {
                            v.as_f64()
                                .map(|f| f as f32)
                                .ok_or_else(|| "Invalid embeddings response: non-numeric value".to_string())
                        })
                        .collect::<Result<Vec<f32>, _>>()
                })
        })
        .collect()
}

pub async fn create_embeddings(
    model_key: &str,
    inputs: &[String],
) -> Result<Vec<Vec<f32>>, String> {
    if model_key.trim().is_empty() {
        return Err("Embedding model is not configured.".to_string());
    }
    if inputs.is_empty() {
        return Ok(vec![]);
    }

    let url = format!("{}/v1/embeddings", lm_base_url());
    let client = reqwest::Client::new();

    let mut results = Vec::with_capacity(inputs.len());
    for input in inputs {
        let chunks = split_into_chunks(input, EMBEDDING_CHUNK_CHARS);
        if chunks.len() == 1 {
            let mut batch = embed_single_batch(&client, &url, model_key, &chunks).await?;
            results.push(batch.remove(0));
        } else {
            // Embed each chunk individually then mean-pool into one vector.
            let mut chunk_embeddings = Vec::with_capacity(chunks.len());
            for chunk in &chunks {
                let mut batch =
                    embed_single_batch(&client, &url, model_key, &[chunk.clone()]).await?;
                chunk_embeddings.push(batch.remove(0));
            }
            results.push(mean_pool(chunk_embeddings));
        }
    }
    Ok(results)
}

#[tauri::command]
pub async fn load_model(config: LoadConfig) -> Result<(), String> {
    load_model_internal(
        &AgentLoadConfig {
            context_length: config.context_length,
            eval_batch_size: config.eval_batch_size,
            flash_attention: config.flash_attention,
            num_experts: config.num_experts,
            offload_kv_cache_to_gpu: config.offload_kv_cache_to_gpu,
        },
        &config.model,
    )
    .await
}

#[tauri::command]
pub async fn unload_model(instance_id: String) -> Result<(), String> {
    unload_model_internal(&instance_id).await
}

#[tauri::command]
pub async fn download_model(model: String) -> Result<(), String> {
    let url = format!("{}/api/v1/models/download", lm_base_url());
    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .json(&json!({ "model": model }))
        .send()
        .await
        .map_err(|e| format!("Download failed: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!(
            "Download failed: {}",
            resp.text().await.unwrap_or_default()
        ));
    }
    Ok(())
}

// ── Backend detection ─────────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct BackendDetected {
    pub ok: bool,
    pub version: Option<String>,
    pub model: Option<String>,
    pub parallel_slots: Option<u32>,
    pub features: Vec<String>,
    pub error: Option<String>,
}

fn strip_base_url(url: &str) -> String {
    url.trim_end_matches('/')
        .trim_end_matches("/chat")
        .trim_end_matches("/v1")
        .trim_end_matches("/api")
        .to_string()
}

async fn detect_lmstudio(client: &reqwest::Client, base: &str) -> BackendDetected {
    match client
        .get(format!("{}/api/v1/models", base))
        .send()
        .await
    {
        Err(e) => BackendDetected {
            ok: false,
            version: None,
            model: None,
            parallel_slots: None,
            features: vec![],
            error: Some(format!("Connection failed: {}", e)),
        },
        Ok(resp) => {
            let status = resp.status();
            if !status.is_success() {
                return BackendDetected {
                    ok: false,
                    version: None,
                    model: None,
                    parallel_slots: None,
                    features: vec![],
                    error: Some(format!("Server returned {}", status)),
                };
            }
            let data: Value = resp.json().await.unwrap_or(Value::Null);
            let model_count = data
                .get("models")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            BackendDetected {
                ok: true,
                version: None,
                model: None,
                parallel_slots: None,
                features: vec![format!("{} model(s) available", model_count)],
                error: None,
            }
        }
    }
}

async fn detect_ollama(client: &reqwest::Client, base: &str) -> BackendDetected {
    match client.get(format!("{}/api/version", base)).send().await {
        Err(e) => BackendDetected {
            ok: false,
            version: None,
            model: None,
            parallel_slots: None,
            features: vec![],
            error: Some(format!("Connection failed: {}", e)),
        },
        Ok(resp) => {
            let status = resp.status();
            if !status.is_success() {
                return BackendDetected {
                    ok: false,
                    version: None,
                    model: None,
                    parallel_slots: None,
                    features: vec![],
                    error: Some(format!("Server returned {}", status)),
                };
            }
            let data: Value = resp.json().await.unwrap_or(Value::Null);
            let version = data["version"].as_str().map(|s| s.to_string());
            BackendDetected {
                ok: true,
                version,
                model: None,
                parallel_slots: None,
                features: vec!["OpenAI-compatible endpoints".to_string()],
                error: None,
            }
        }
    }
}

async fn detect_llamacpp(client: &reqwest::Client, base: &str) -> BackendDetected {
    // Try /health first.
    let health_ok = client
        .get(format!("{}/health", base))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);
    if !health_ok {
        return BackendDetected {
            ok: false,
            version: None,
            model: None,
            parallel_slots: None,
            features: vec![],
            error: Some("Connection failed (/health unreachable)".to_string()),
        };
    }

    // Probe /props for capability flags.
    let props: Value = if let Ok(resp) = client.get(format!("{}/props", base)).send().await {
        resp.json().await.unwrap_or(Value::Null)
    } else {
        Value::Null
    };

    let parallel_slots = props["total_slots"].as_u64().map(|n| n as u32);
    let model_alias = props["model_alias"]
        .as_str()
        .unwrap_or("")
        .to_string();

    let mut features = vec!["OpenAI-compatible endpoints".to_string()];

    if let Some(slots) = parallel_slots {
        if slots > 1 {
            features.push(format!("Parallel inference ({} slots)", slots));
        }
    }

    // KV cache quantization (llama.cpp exposes kv_cache_type_k / _v in /props).
    let kv_k = props["kv_cache_type_k"].as_str().unwrap_or("");
    let kv_v = props["kv_cache_type_v"].as_str().unwrap_or("");
    if !kv_k.is_empty() || !kv_v.is_empty() {
        let label = match (kv_k, kv_v) {
            (k, v) if k == v && !k.is_empty() => format!("KV cache quantization ({})", k),
            (k, v) if !k.is_empty() && !v.is_empty() => {
                format!("KV cache quantization (K:{} V:{})", k, v)
            }
            (k, _) if !k.is_empty() => format!("KV cache quantization (K:{})", k),
            (_, v) => format!("KV cache quantization (V:{})", v),
        };
        features.push(label);
    }

    // imatrix / IQ quantization detected from the loaded model name.
    if model_alias.contains("IQ") || model_alias.to_lowercase().contains("imatrix") {
        features.push("imatrix quantization".to_string());
    }

    // Derive a clean display name from the model alias (strip path + extension).
    let model = if model_alias.is_empty() {
        None
    } else {
        Some(
            Path::new(&model_alias)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(&model_alias)
                .to_string(),
        )
    };

    BackendDetected {
        ok: true,
        version: None,
        model,
        parallel_slots,
        features,
        error: None,
    }
}

pub async fn detect_backend_capabilities(
    base_url: &str,
    backend_type: &str,
) -> BackendDetected {
    let base = strip_base_url(base_url);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_default();
    match backend_type {
        "ollama" => detect_ollama(&client, &base).await,
        "llamacpp" => detect_llamacpp(&client, &base).await,
        _ => detect_lmstudio(&client, &base).await,
    }
}

// ── Process-based backend discovery ──────────────────────────────────────────

fn find_arg_value<'a>(tokens: &[&'a str], flags: &[&str]) -> Option<&'a str> {
    for (i, token) in tokens.iter().enumerate() {
        // --flag value  or  -f value
        if flags.contains(token) {
            return tokens.get(i + 1).copied();
        }
        // --flag=value
        for flag in flags {
            if let Some(val) = token.strip_prefix(&format!("{}=", flag)) {
                return Some(val);
            }
        }
    }
    None
}

fn parse_llama_processes() -> Vec<BackendInstance> {
    // Collect output of `ps -A -o pid,args` to get all process command lines.
    let output = Command::new("ps")
        .args(["-A", "-o", "pid,args"])
        .output();
    let stdout = match output {
        Ok(o) => String::from_utf8_lossy(&o.stdout).into_owned(),
        Err(_) => return vec![],
    };

    let mut instances = Vec::new();
    for line in stdout.lines() {
        let trimmed = line.trim();
        // Match any variant of the llama.cpp server binary name.
        if !trimmed.contains("llama-server") && !trimmed.contains("llama-ser ") {
            continue;
        }
        // Skip our own grep / ps invocation in case it leaks.
        if trimmed.contains("grep") || trimmed.contains("ps -A") {
            continue;
        }

        // Split into tokens; skip the leading PID.
        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        let args = if tokens.len() > 1 { &tokens[1..] } else { continue };

        let port = find_arg_value(args, &["--port", "-p"])
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(8080);

        let model_path = find_arg_value(args, &["--model", "-m", "-mo"]).unwrap_or("");
        let model_hint = if model_path.is_empty() {
            String::new()
        } else {
            Path::new(model_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(model_path)
                .to_string()
        };

        instances.push(BackendInstance {
            url: format!("http://localhost:{}", port),
            model_hint,
        });
    }

    instances
}

/// Discovers running llama.cpp server instances by inspecting the process list.
/// For each found instance, it also queries `/props` to confirm the server is
/// reachable and refine the detected model name.
pub async fn discover_backend_instances() -> Vec<BackendInstance> {
    // Parse processes on a blocking thread so we don't hold the async executor.
    let mut instances =
        tokio::task::spawn_blocking(parse_llama_processes)
            .await
            .unwrap_or_default();

    if instances.is_empty() {
        return instances;
    }

    // Validate each instance by probing /props, and prefer the server's own
    // model_alias over the path-derived hint when available.
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .unwrap_or_default();

    for inst in &mut instances {
        let base = inst.url.trim_end_matches('/').to_string();
        if let Ok(resp) = client.get(format!("{}/props", base)).send().await {
            if let Ok(props) = resp.json::<Value>().await {
                if let Some(alias) = props["model_alias"].as_str() {
                    if !alias.is_empty() {
                        // Use the server's alias; strip path components if present.
                        inst.model_hint = Path::new(alias)
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or(alias)
                            .to_string();
                    }
                }
            }
        }
    }

    instances
}
