use axum::{extract::State, http::StatusCode, routing::post, Json, Router};
use open_mind_format::{omf_reader::read_omf, OmfModel, Tensor};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap, env, error::Error, fs, fs::File, io::{self, BufRead, Write},
    net::SocketAddr, path::{Path, PathBuf}, sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone)]
struct RuntimeModel {
    id: String,
    config: Value,
    tensors: BTreeMap<String, Tensor>,
}

impl RuntimeModel {
    fn from_omf(model: OmfModel, id: String) -> Result<Self, String> {
        let config = model.config;
        for key in ["hidden_size", "layers", "vocab", "heads", "intermediate_size", "context"] {
            if config.get(key).and_then(Value::as_u64).is_none() {
                return Err(format!("CONFIG is missing integer field {key}"));
            }
        }
        if model.tokenizer.get("type").and_then(Value::as_str) != Some("byte") {
            return Err("runtime v0.1 supports the byte tokenizer only".into());
        }
        Ok(Self { id, config, tensors: model.tensors })
    }

    fn int(&self, key: &str) -> Result<usize, String> {
        self.config[key].as_u64().and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| format!("invalid CONFIG field {key}"))
    }

    fn weight(&self, name: &str, shape: &[usize]) -> Result<&[f32], String> {
        let tensor = self.tensors.get(name).ok_or_else(|| format!("missing tensor {name}"))?;
        let actual: Vec<usize> = tensor.shape.iter().map(|&d| usize::try_from(d).unwrap_or(usize::MAX)).collect();
        if actual != shape { return Err(format!("tensor {name} has shape {actual:?}, expected {shape:?}")); }
        Ok(&tensor.values)
    }

    fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        seed: u64,
    ) -> Result<String, String> {
        let vocab = self.int("vocab")?;
        let context = self.int("context")?;
        if vocab != 256 { return Err("byte tokenizer requires vocab=256".into()); }
        let mut tokens: Vec<usize> = prompt.as_bytes().iter().map(|&b| b as usize).collect();
        if tokens.is_empty() { tokens.push(b' ' as usize); }
        let mut generated = Vec::with_capacity(max_tokens);
        let mut rng = seed.max(1);
        for _ in 0..max_tokens {
            if tokens.len() > context { tokens.drain(..tokens.len() - context); }
            let logits = self.next_logits(&tokens)?;
            let next = sample_token(&logits, temperature, top_k, &mut rng)?;
            if next >= vocab { return Err("model produced token outside vocabulary".into()); }
            generated.push(next as u8);
            tokens.push(next);
        }
        Ok(String::from_utf8_lossy(&generated).into_owned())
    }

    fn next_logits(&self, tokens: &[usize]) -> Result<Vec<f32>, String> {
        let h = self.int("hidden_size")?;
        let layers = self.int("layers")?;
        let heads = self.int("heads")?;
        let f = self.int("intermediate_size")?;
        let vocab = self.int("vocab")?;
        let context = self.int("context")?;
        if h == 0 || heads == 0 || h % heads != 0 || tokens.len() > context {
            return Err("incompatible model dimensions".into());
        }
        let seq = tokens.len();
        let embed = self.weight("token_embedding.weight", &[vocab, h])?;
        let pos = self.weight("position_embedding.weight", &[context, h])?;
        let mut x = vec![0.0; seq * h];
        for (t, &id) in tokens.iter().enumerate() {
            for j in 0..h { x[t * h + j] = embed[id * h + j] + pos[t * h + j]; }
        }

        for layer in 0..layers {
            let prefix = format!("layers.{layer}");
            let norm1 = layer_norm(
                &x, seq, h,
                self.weight(&format!("{prefix}.ln1.weight"), &[h])?,
                self.weight(&format!("{prefix}.ln1.bias"), &[h])?,
            );
            let qkv = linear(
                &norm1, seq, h, 3 * h,
                self.weight(&format!("{prefix}.attn.qkv.weight"), &[3 * h, h])?,
                self.weight(&format!("{prefix}.attn.qkv.bias"), &[3 * h])?,
            );
            let head_dim = h / heads;
            let mut attended = vec![0.0; seq * h];
            for t in 0..seq {
                for head in 0..heads {
                    let base = head * head_dim;
                    let mut scores = Vec::with_capacity(t + 1);
                    for s in 0..=t {
                        let mut dot = 0.0;
                        for d in 0..head_dim {
                            dot += qkv[t * 3 * h + base + d] * qkv[s * 3 * h + h + base + d];
                        }
                        scores.push(dot / (head_dim as f32).sqrt());
                    }
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0;
                    for score in &mut scores { *score = (*score - max).exp(); sum += *score; }
                    for (s, score) in scores.iter().enumerate() {
                        let probability = *score / sum;
                        for d in 0..head_dim {
                            attended[t * h + base + d] += probability * qkv[s * 3 * h + 2 * h + base + d];
                        }
                    }
                }
            }
            let projected = linear(
                &attended, seq, h, h,
                self.weight(&format!("{prefix}.attn.proj.weight"), &[h, h])?,
                self.weight(&format!("{prefix}.attn.proj.bias"), &[h])?,
            );
            for (dst, add) in x.iter_mut().zip(projected) { *dst += add; }
            let norm2 = layer_norm(
                &x, seq, h,
                self.weight(&format!("{prefix}.ln2.weight"), &[h])?,
                self.weight(&format!("{prefix}.ln2.bias"), &[h])?,
            );
            let mut hidden = linear(
                &norm2, seq, h, f,
                self.weight(&format!("{prefix}.mlp.fc1.weight"), &[f, h])?,
                self.weight(&format!("{prefix}.mlp.fc1.bias"), &[f])?,
            );
            for value in &mut hidden { *value = gelu(*value); }
            let output = linear(
                &hidden, seq, f, h,
                self.weight(&format!("{prefix}.mlp.fc2.weight"), &[h, f])?,
                self.weight(&format!("{prefix}.mlp.fc2.bias"), &[h])?,
            );
            for (dst, add) in x.iter_mut().zip(output) { *dst += add; }
        }

        let x = layer_norm(
            &x, seq, h,
            self.weight("final_norm.weight", &[h])?,
            self.weight("final_norm.bias", &[h])?,
        );
        let last = &x[(seq - 1) * h..seq * h];
        let mut logits = Vec::with_capacity(vocab);
        for token in 0..vocab {
            let row = &embed[token * h..(token + 1) * h];
            logits.push(last.iter().zip(row).map(|(a, b)| a * b).sum());
        }
        Ok(logits)
    }
}

fn sample_token(logits: &[f32], temperature: f32, top_k: usize, rng: &mut u64) -> Result<usize, String> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err("model produced invalid logits".into());
    }
    if temperature == 0.0 {
        return logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index).ok_or_else(|| "model produced no logits".into());
    }
    let limit = top_k.clamp(1, logits.len());
    let mut candidates: Vec<usize> = (0..logits.len()).collect();
    candidates.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    candidates.truncate(limit);
    let max = logits[candidates[0]];
    let weights: Vec<f64> = candidates.iter()
        .map(|&index| ((logits[index] - max) as f64 / temperature as f64).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err("could not normalize model probabilities".into());
    }

    // xorshift64* gives seeded, dependency-free sampling suitable for local decoding.
    *rng ^= *rng >> 12;
    *rng ^= *rng << 25;
    *rng ^= *rng >> 27;
    let random = rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
    let mut choice = ((random >> 11) as f64) / ((1u64 << 53) as f64) * total;
    for (&index, weight) in candidates.iter().zip(weights) {
        if choice < weight { return Ok(index); }
        choice -= weight;
    }
    Ok(candidates[0])
}

fn linear(input: &[f32], rows: usize, cols: usize, out_cols: usize, weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; rows * out_cols];
    for row in 0..rows {
        for o in 0..out_cols {
            let mut value = bias[o];
            for i in 0..cols { value += input[row * cols + i] * weight[o * cols + i]; }
            out[row * out_cols + o] = value;
        }
    }
    out
}

fn layer_norm(x: &[f32], rows: usize, width: usize, weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for row in 0..rows {
        let v = &x[row * width..(row + 1) * width];
        let mean = v.iter().sum::<f32>() / width as f32;
        let variance = v.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / width as f32;
        let inv = (variance + 1e-5).sqrt().recip();
        for j in 0..width { out[row * width + j] = (v[j] - mean) * inv * weight[j] + bias[j]; }
    }
    out
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.79788456 * (x + 0.044715 * x * x * x)).tanh())
}

#[derive(Deserialize)]
struct GenerateRequest {
    model: Option<String>,
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
}

async fn generate(
    State(model): State<Arc<RuntimeModel>>,
    Json(request): Json<GenerateRequest>,
) -> Result<Json<Value>, (StatusCode, String)> {
    if let Some(request_model) = request.model.as_deref() {
        if request_model != model.id && Path::new(request_model).file_name().and_then(|s| s.to_str()) != Some(&model.id) {
            return Err((StatusCode::BAD_REQUEST, format!("loaded model is {}; requested {request_model}", model.id)));
        }
    }
    let max_tokens = request.max_tokens.unwrap_or(64).clamp(1, 1024);
    let temperature = request.temperature.unwrap_or(0.8);
    if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
        return Err((StatusCode::BAD_REQUEST, "temperature must be between 0 and 2".into()));
    }
    let top_k = request.top_k.unwrap_or(40).max(1);
    let seed = request.seed.unwrap_or_else(|| {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64
    });
    let prompt = request.prompt;
    let prompt_tokens = prompt.len();
    let model_for_infer = model.clone();
    let generated = tokio::task::spawn_blocking(move || {
        model_for_infer.generate(&prompt, max_tokens, temperature, top_k, seed)
    })
        .await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, e))?;
    let completion_tokens = generated.len();
    Ok(Json(json!({
        "id": "omf-local-generation",
        "object": "text_completion",
        "created": 0,
        "model": model.id,
        "choices": [{"index": 0, "text": generated, "finish_reason": "length"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}
    })))
}

#[cfg(test)]
mod tests {
    use super::sample_token;

    #[test]
    fn greedy_decoding_picks_the_highest_logit() {
        assert_eq!(sample_token(&[0.1, 0.8, 0.3], 0.0, 3, &mut 1).unwrap(), 1);
    }

    #[test]
    fn top_k_one_only_samples_the_highest_logit() {
        assert_eq!(sample_token(&[0.1, 0.8, 0.3], 0.8, 1, &mut 7).unwrap(), 1);
    }
}

fn usage() -> &'static str {
    "Open Mind local model manager\n\
     Usage:\n\
       om add FILE.omf\n\
       om list\n\
       om show MODEL\n\
       om run MODEL [--prompt TEXT] [--max-tokens N] [--temperature T] [--top-k N] [--seed N]\n\
       om serve [--model MODEL] [--host 127.0.0.1] [--port 8080]\n\
       om --help\n\
     Models are stored locally in ~/.open-mind/models (override with OPEN_MIND_HOME)."
}

enum CliCommand {
    Help,
    Serve { model: String, address: SocketAddr },
    Run {
        model: String,
        prompt: Option<String>,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        seed: Option<u64>,
    },
    Add(PathBuf),
    List,
    Show(String),
}

fn parse_command() -> Result<CliCommand, String> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || matches!(args[0].as_str(), "--help" | "-h" | "help") {
        return Ok(CliCommand::Help);
    }
    if args[0] == "--model" {
        args.insert(0, "serve".into());
    }
    let command = args.remove(0);
    match command.as_str() {
        "serve" => {
            let (model, address) = parse_serve_args(args)?;
            Ok(CliCommand::Serve { model, address })
        }
        "add" => {
            if args.len() != 1 {
                return Err("usage: om add FILE.omf".into());
            }
            Ok(CliCommand::Add(PathBuf::from(&args[0])))
        }
        "list" | "ls" => {
            if !args.is_empty() { return Err("usage: om list".into()); }
            Ok(CliCommand::List)
        }
        "show" => {
            if args.len() != 1 { return Err("usage: om show MODEL".into()); }
            Ok(CliCommand::Show(args.remove(0)))
        }
        "run" => parse_run_args(args),
        _ => Err(format!("unknown command {command}\n{}", usage())),
    }
}

fn parse_serve_args(args: Vec<String>) -> Result<(String, SocketAddr), String> {
    let mut model = None;
    let mut host = "127.0.0.1".to_string();
    let mut port = 8080u16;
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--model" => model = Some(iter.next().ok_or("--model requires a name or path")?),
            "--host" => host = iter.next().ok_or("--host requires an address")?,
            "--port" => port = iter.next().ok_or("--port requires a number")?.parse().map_err(|e| format!("invalid port: {e}"))?,
            "--help" | "-h" => return Err(usage().into()),
            _ => return Err(format!("unknown serve option {arg}\n{}", usage())),
        }
    }
    let model = model.or_else(default_model)
        .ok_or("specify --model or install exactly one model with `om add FILE.omf`")?;
    let address = format!("{host}:{port}").parse().map_err(|e| format!("invalid bind address: {e}"))?;
    Ok((model, address))
}

fn parse_run_args(args: Vec<String>) -> Result<CliCommand, String> {
    let Some(model) = args.first().cloned() else {
        return Err("usage: om run MODEL [--prompt TEXT] [--max-tokens N] [--temperature T] [--top-k N] [--seed N]".into());
    };
    let mut prompt = None;
    let mut max_tokens = 128usize;
    let mut temperature = 0.8f32;
    let mut top_k = 40usize;
    let mut seed = None;
    let mut iter = args.into_iter().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--prompt" => prompt = Some(iter.next().ok_or("--prompt requires text")?),
            "--max-tokens" => max_tokens = iter.next().ok_or("--max-tokens requires a number")?.parse().map_err(|e| format!("invalid token count: {e}"))?,
            "--temperature" => temperature = iter.next().ok_or("--temperature requires a value")?.parse().map_err(|e| format!("invalid temperature: {e}"))?,
            "--top-k" => top_k = iter.next().ok_or("--top-k requires a number")?.parse().map_err(|e| format!("invalid top-k: {e}"))?,
            "--seed" => seed = Some(iter.next().ok_or("--seed requires a number")?.parse().map_err(|e| format!("invalid seed: {e}"))?),
            "--help" | "-h" => return Err(usage().into()),
            _ => return Err(format!("unknown run option {arg}\n{}", usage())),
        }
    }
    if max_tokens == 0 || max_tokens > 1024 { return Err("--max-tokens must be between 1 and 1024".into()); }
    if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) { return Err("--temperature must be between 0 and 2".into()); }
    if top_k == 0 { return Err("--top-k must be at least 1".into()); }
    Ok(CliCommand::Run { model, prompt, max_tokens, temperature, top_k, seed })
}

fn home_dir() -> Result<PathBuf, String> {
    if let Some(path) = env::var_os("OPEN_MIND_HOME") {
        return Ok(PathBuf::from(path));
    }
    env::var_os("HOME").map(|home| PathBuf::from(home).join(".open-mind"))
        .ok_or("HOME is not set; set OPEN_MIND_HOME to a local directory".into())
}

fn model_dir() -> Result<PathBuf, String> {
    Ok(home_dir()?.join("models"))
}

fn default_model() -> Option<String> {
    let entries = fs::read_dir(model_dir().ok()?).ok()?;
    let models: Vec<String> = entries.filter_map(Result::ok)
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "omf"))
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect();
    if models.len() == 1 { Some(models[0].clone()) } else { None }
}

fn resolve_model(name: &str) -> Result<PathBuf, String> {
    let supplied = PathBuf::from(name);
    if supplied.is_file() { return Ok(supplied); }
    let installed = model_dir()?.join(name);
    if installed.is_file() { return Ok(installed); }
    Err(format!("model {name:?} not found; add a local .omf file with `om add FILE.omf`"))
}

fn load_model(name: &str) -> Result<RuntimeModel, Box<dyn Error>> {
    let path = resolve_model(name).map_err(io::Error::other)?;
    let id = path.file_name().and_then(|s| s.to_str()).unwrap_or(name).to_string();
    let model = RuntimeModel::from_omf(read_omf(File::open(&path)?)?, id)
        .map_err(io::Error::other)?;
    Ok(model)
}

fn add_model(source: PathBuf) -> Result<(), Box<dyn Error>> {
    let id = source.file_name().and_then(|name| name.to_str())
        .ok_or_else(|| io::Error::other("model path has no valid filename"))?;
    if source.extension().is_none_or(|ext| ext != "omf") {
        return Err(io::Error::other("only .omf files can be installed").into());
    }
    let _validated = RuntimeModel::from_omf(read_omf(File::open(&source)?)?, id.to_string())
        .map_err(io::Error::other)?;
    let destination_dir = model_dir().map_err(io::Error::other)?;
    fs::create_dir_all(&destination_dir)?;
    let destination = destination_dir.join(id);
    if destination.exists() {
        return Err(io::Error::other(format!("{} already exists; remove it before replacing", destination.display())).into());
    }
    fs::copy(&source, &destination)?;
    println!("Installed {} at {}", id, destination.display());
    Ok(())
}

fn list_models() -> Result<(), Box<dyn Error>> {
    let dir = model_dir().map_err(io::Error::other)?;
    if !dir.exists() {
        println!("No models installed. Add one with: om add FILE.omf");
        return Ok(());
    }
    let mut models = Vec::new();
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "omf") {
            models.push((entry.file_name().to_string_lossy().into_owned(), entry.metadata()?.len()));
        }
    }
    models.sort_by(|a, b| a.0.cmp(&b.0));
    if models.is_empty() {
        println!("No models installed. Add one with: om add FILE.omf");
    } else {
        for (name, bytes) in models {
            println!("{name:<32} {:>7.1} MiB", bytes as f64 / (1024.0 * 1024.0));
        }
    }
    Ok(())
}

fn run_interactive(
    model: RuntimeModel,
    prompt: Option<String>,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    seed: Option<u64>,
) -> Result<(), Box<dyn Error>> {
    if let Some(prompt) = prompt {
        let seed = seed.unwrap_or_else(random_seed);
        println!("{}", model.generate(&prompt, max_tokens, temperature, top_k, seed).map_err(io::Error::other)?);
        return Ok(());
    }
    println!("Open Mind local generation. Enter a prompt; Ctrl+D to exit.");
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    loop {
        print!("> ");
        io::stdout().flush()?;
        let mut prompt = String::new();
        if reader.read_line(&mut prompt)? == 0 { break; }
        let prompt = prompt.trim_end();
        if prompt.is_empty() { continue; }
        let seed = seed.unwrap_or_else(random_seed);
        let output = model.generate(prompt, max_tokens, temperature, top_k, seed)
            .map_err(io::Error::other)?;
        println!("{prompt}{output}\n");
    }
    Ok(())
}

fn random_seed() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    match parse_command().map_err(io::Error::other)? {
        CliCommand::Help => println!("{}", usage()),
        CliCommand::Add(source) => add_model(source)?,
        CliCommand::List => list_models()?,
        CliCommand::Show(name) => {
            let path = resolve_model(&name).map_err(io::Error::other)?;
            let model = RuntimeModel::from_omf(read_omf(File::open(&path)?)?, path.file_name()
                .and_then(|s| s.to_str()).unwrap_or(&name).to_string()).map_err(io::Error::other)?;
            println!("Name: {}", model.id);
            println!("Path: {}", path.display());
            println!("Config: {}", serde_json::to_string_pretty(&model.config)?);
            println!("Tensors: {}", model.tensors.len());
            println!("Parameters: {}", model.tensors.values().map(|t| t.values.len() as u64).sum::<u64>());
        }
        CliCommand::Run { model, prompt, max_tokens, temperature, top_k, seed } => {
            run_interactive(load_model(&model)?, prompt, max_tokens, temperature, top_k, seed)?;
        }
        CliCommand::Serve { model: model_path, address } => {
            let model = load_model(&model_path)?;
            let model_id = model.id.clone();
            let model = Arc::new(model);
            let app = Router::new()
                .route("/v1/generate", post(generate))
                .route("/v1/completions", post(generate))
                .with_state(model);
            let listener = tokio::net::TcpListener::bind(address).await?;
            eprintln!("om serving {model_id} at http://{address}");
            axum::serve(listener, app).await?;
        }
    }
    Ok(())
}
