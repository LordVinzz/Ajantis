use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;

use crate::agent_config::AgentLoadConfig;
use crate::helpers::lm_base_url;
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

#[tauri::command]
pub async fn fetch_models() -> Result<Vec<ModelInfo>, String> {
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
    let resp = client
        .post(&url)
        .json(&json!({
            "model": model_key,
            "input": inputs,
        }))
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
    let items = data["data"]
        .as_array()
        .ok_or("Invalid embeddings response: missing data array")?;

    items
        .iter()
        .map(|item| {
            item["embedding"]
                .as_array()
                .ok_or("Invalid embeddings response: missing embedding")
                .and_then(|values| {
                    values
                        .iter()
                        .map(|value| {
                            value
                                .as_f64()
                                .map(|v| v as f32)
                                .ok_or("Invalid embeddings response: non-numeric value")
                        })
                        .collect::<Result<Vec<f32>, _>>()
                })
                .map_err(|e| e.to_string())
        })
        .collect()
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
