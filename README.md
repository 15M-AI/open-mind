# Open Mind

**Open Mind** is a local ecosystem for training, storing, and running models in
the native **OMF** binary format. It is organized as a three-component
monorepo, does not download model weights, and does not use GGUF.

> **Status:** Experimental v0.1. The runtime does not yet provide streaming,
> authentication, or TLS. Keep the server bound to `127.0.0.1` unless you know
> how to secure the network where you expose it.

## Components

| Project | Description |
|---|---|
| `open-mind-format` | OMF v0.1 specification, Rust reader/writer, and PyTorch writer |
| `open-mind-runtime` | Native CPU inference, the `om` CLI, and the local OMA API |
| `open-mind-train` | PyTorch causal training, mmap dataset access, and OMF checkpoints |

## Requirements

- Stable Rust and Cargo.
- Python 3.10+ and local PyTorch (`pip install -e ./open-mind-train` installs missing dependencies).
- Training data in a UTF-8 text file.

No model weights are downloaded. To avoid writing to the system Python
installation, use a virtual environment:

```bash
cd ~/OpenMind
python -m venv .venv
source .venv/bin/activate
pip install -e ./open-mind-train
```

## Train a model

The repository includes `data.txt` as a tiny end-to-end smoke-test corpus; it
cannot produce a useful conversational model. Use a larger local corpus that
you have the right to use:

```bash
python train.py --device cpu --dataset corpus.txt --output pico.omf \
  --steps 5000 --checkpoint-every 500
```

The `15M` preset contains 14,983,680 parameters. CPU training uses gradient
accumulation and 8-bit quantized SGD momentum; model weights remain in memory
during training. The writer supports `fp16` (default), `q4_0`, and `q8_0`
checkpoints. Periodic checkpoints are written as `pico-v1.omf`, and the final
checkpoint is written as `pico.omf`.

## Install and run a model

Build the local `om` executable:

```bash
cargo build --release
./target/release/om --help
```

Install an OMF model from a local path. The file is copied to
`~/.open-mind/models`; model weights are not included in this repository:

```bash
./target/release/om add pico.omf
./target/release/om list
./target/release/om show pico.omf
./target/release/om run pico.omf --prompt "Hello" --max-tokens 64
```

Running `om run MODEL` without `--prompt` opens an interactive mode; enter one
prompt per line and press `Ctrl+D` to exit. Generation options:

```bash
./target/release/om run pico.omf --prompt "Hello" \
  --temperature 0.8 --top-k 40 --seed 123
```

Temperature `0` selects greedy decoding. A fixed seed reproduces a sampled
sequence; changing it does not add knowledge to the model. Set
`OPEN_MIND_HOME=/local/path` to change the directory where `om add` and
`om list` store and find models.

## Serve the OMA API

```bash
./target/release/om serve --model pico.omf
```

The server listens on `127.0.0.1:8080` by default. If exactly one model is
installed, `om serve` can resolve it without `--model`. Try the OpenAI-compatible
endpoint:

```bash
curl http://127.0.0.1:8080/v1/generate \
  -H 'content-type: application/json' \
  -d '{"model":"pico.omf","prompt":"Hello","max_tokens":32,"temperature":0.8,"top_k":40}'
```

`/v1/completions` is also available. `max_tokens` is capped at 1024;
`temperature` must be between 0 and 2. OMA v0.1 returns a complete JSON
response and does not support streaming.

## Optional terminal interface

The separate `omt` repository provides a keyboard-driven terminal
manager inspired by local model tools such as Ollama. It manages OMF files
through the `om` runtime and does not depend on Ollama:

```bash
cd ~/OpenMind-TUI
make install
omt
```

See that repository's README for controls, configuration, and troubleshooting.

## Monorepo layout

```text
OpenMind/
├── Cargo.toml                   # Rust workspace
├── open-mind-format/
│   ├── OMF_SPEC.md
│   ├── src/                     # Rust reader and writer
│   └── python/src/open_mind_format/
├── open-mind-runtime/
│   └── src/main.rs              # `om` CLI, inference, and OMA API
├── open-mind-train/
│   ├── pyproject.toml
│   └── src/open_mind/
├── Pico/                        # 15M cloud training recipe and model card
├── train.py
└── data.txt                     # Demo corpus, not model weights
```

## Quality and limitations

Test the Rust workspace and OMF reader/writer:

```bash
cargo test --workspace
```

This is a byte-level transformer trained from scratch. It is not a replacement
for pretrained models and cannot answer with knowledge absent from its
training data. The runtime performs inference on CPU; the 15M preset may be
slow on modest laptops.

The cloud training recipe for Pico is in `Pico/README.md`. Its model weights
remain OMF and are not checked into this repository.

To prepare a CUDA runner, follow that README and start with a short pilot. The
recipe does not provision a cloud machine or download training data.

## Publishing to GitHub

This project is designed as a **monorepo**, so create one repository:
`open-mind` (repository: `github.com/15M-AI/open-mind`). The component
names remain as directories: `open-mind-format`, `open-mind-runtime`, and
`open-mind-train`. The publisher script lives outside this repository in
`~/open-mind-publisher/`.

Choose and add a license before publishing if you want to grant reuse
permissions. No license is included by default.
