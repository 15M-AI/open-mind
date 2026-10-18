# Pico 15M

Pico is Open Mind's experimental 15-million-parameter, byte-level causal
language model. The training recipe is designed for a CUDA cloud runner, and
all model checkpoints are written directly as **OMF** files. The training
script does not convert through another model format.

> **Important:** A 15M model trained from scratch is a small research model, not
> a state-of-the-art assistant. Its quality depends heavily on a large,
> high-quality, legally usable corpus and sufficient compute. The bundled demo
> text is not a training corpus. This repository does not download datasets or
> run training automatically.

## Model specification

| Setting | Value |
|---|---:|
| Parameters | 14,983,680 |
| Transformer blocks | 8 |
| Hidden size | 384 |
| Attention heads | 6 |
| Feed-forward width | 1,632 |
| Context window | 256 byte tokens |
| Vocabulary | 256 UTF-8 byte values |
| Cloud optimizer | AdamW, cosine decay, warmup, gradient clipping |
| Output format | OMF v0.1, fp16 |

The byte tokenizer is reversible, simple, and already supported by the OMF
runtime. It is less token-efficient than a subword tokenizer, especially for
non-English text. Do not describe this release as a capable general-purpose
chat assistant. The chat template in the TUI is a convenience, not instruction
tuning.

## Cloud runner preparation

Use a GPU VM or notebook with Python 3.10+, CUDA-enabled PyTorch, enough disk
space for the corpus and checkpoints, and a persistent output volume. Clone the
core repository, then stage a UTF-8 plain-text corpus that you have permission
to use. Keep source archives and private data outside the Git checkout.

1. Review `DATASET_PROVENANCE_TEMPLATE.csv` and record every dataset, exact
   source/version, license, preprocessing operation, and SHA-256.
2. Prepare one UTF-8 text file on the runner's persistent disk. Shuffle
   complete documents before concatenating them so the trainer's deterministic
   contiguous 95%/5% train/validation split is representative; the trainer uses
   memory-mapped reads and does not download or concatenate datasets.
3. Install CUDA PyTorch for the runner first. `run_cloud.sh` intentionally uses
   `pip --no-deps` so it will not replace the runner's CUDA build.
4. Start a pilot with a small step count and inspect training/validation loss,
   output behavior, VRAM, and checkpoint recovery before a longer run.
5. Run the full configured recipe only after reviewing the pilot and estimating
   runtime and storage.

The default run consumes up to approximately 819 million byte-tokens
(`100,000 × 8 × 256 × 4`) if every step completes. This is an initial training
recipe, not a quality guarantee or a compute-optimal token budget. Reaching
modern language-model quality is not realistic to promise for a 15M model
trained from scratch; measure cost and output quality on your chosen GPU before
committing to a long run.

The model repository includes no cloud vendor credentials, dataset, weights,
or provider-specific assumptions. The runner must retain
`Pico/output/pico-0.1.0-v1.omf` between checkpoints and upload artifacts to
your chosen private object storage if the VM disk is ephemeral.

## Train

From the root of the cloned `open-mind` repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-deps -e ./open-mind-train
python -m pip install --no-deps -e ./open-mind-format/python

bash Pico/run_cloud.sh /persistent/data/pico-corpus.txt \
  --steps 100000 \
  --checkpoint-every 1000 \
  --output /persistent/checkpoints/pico-0.1.0.omf
```

The script requires a visible CUDA device and stops before training if CUDA is
not available. The final default configuration is recorded in `config.json`.
`--steps` is deliberately overridable so you can run a short pilot first. The
trainer logs loss and validation loss, applies gradient accumulation, clips
gradients, warms up and cosine-decays AdamW, and writes:

- `pico-0.1.0-v1.omf` at each checkpoint interval.
- `pico-0.1.0.omf` at the final step.

Do not commit checkpoints to Git. Upload model artifacts separately after
quality, provenance, safety, and license review.

## Validate the result

The minimum release checks are:

- Verify the OMF file can be loaded by the current Rust OMF reader/runtime.
- Record the exact training config, final step, train loss, validation loss,
  dataset manifest and artifact SHA-256.
- Generate samples for prompts absent from training and review them for
  repetition, memorization, and encoding problems.
- Evaluate against a held-out dataset and publish the task, split, and metric.
- Confirm that dataset terms allow model training and redistribution.

Do not infer assistant quality from training loss alone. Do not publish the
weights until the license field in `config.json` has been resolved based on the
data actually used.

## Files

| File | Purpose |
|---|---|
| `config.json` | Reproducible model and training defaults |
| `train.py` | Pico-specific CLI; writes checkpoints via the OMF writer |
| `run_cloud.sh` | CUDA preflight and cloud-runner entrypoint |
| `MODEL_CARD.md` | Scope, evaluation, limitations, and release status |
| `DATASET_CARD_TEMPLATE.md` | Dataset documentation and provenance checklist |
| `DATASET_PROVENANCE_TEMPLATE.csv` | Machine-readable source/license manifest |

## License status

Pico is intended to be open source, but its model-weight license is
**undecided** until the training dataset and its terms are recorded. The code
and weights must not be labeled with a permissive license by assumption.
Complete `MODEL_CARD.md` and the provenance manifest before publishing a model
release. Training data and checkpoints are excluded from Git by the local
`.gitignore`.
