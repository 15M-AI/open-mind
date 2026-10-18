# Pico 0.1.0 model card

## Summary

Pico is an experimental, byte-level, causal language model with 14,983,680
parameters. It is trained from scratch with the Open Mind PyTorch trainer and
is serialized directly to Open Mind Format (OMF) v0.1.

## Intended use

- Small-scale experimentation with language-model training and the OMF runtime.
- Local inference on CPU for testing the end-to-end Open Mind pipeline.
- Research on data preparation, small-model behavior, and OMF tooling.

Pico is not represented as a reliable assistant, a source of factual advice,
or a production system.

## Architecture and tokenizer

The model has 8 causal-transformer blocks, hidden size 384, 6 attention heads,
feed-forward width 1,632, context length 256, and tied token/output embeddings.
The tokenizer maps each UTF-8 byte to one of 256 token ids. This preserves
arbitrary text bytes but is inefficient for natural language and can produce
poor word and sentence structure at this parameter scale.

## Training

Defaults are defined in `config.json`. The cloud recipe uses AdamW, gradient
accumulation, gradient clipping, warmup, cosine learning-rate decay, and a
contiguous held-out validation split. Training uses only the corpus supplied
by the operator; there is no automatic dataset download.

Training dataset, data license, hardware, runtime, final step, and metrics are
**not yet recorded**. Populate those fields from the actual run before claiming
reproducibility or releasing weights.

## Evaluation

No trained release has been evaluated. Do not report quality claims based only
on training loss. A release should document held-out loss and a manual sample
review for repetition, memorization, and UTF-8 behavior.

## Limitations and risks

- A 15M-parameter model trained from scratch is far below modern assistant
  capability expectations.
- A byte vocabulary consumes context on individual bytes rather than words.
- Text is not instruction-tuned or preference-aligned.
- Output can be repetitive, inaccurate, biased, or unsuitable for a use case.
- Training-data rights and memorization risk depend on the corpus selected by
  the operator.
- The OMA API/runtime does not provide authentication or streaming in v0.1.

## Dataset and model licenses

The Pico training code and the Open Mind core repository are licensed under
Apache-2.0. The model-weight license is **UNDECIDED**. Record dataset provenance
and license terms in `DATASET_PROVENANCE_TEMPLATE.csv`, then determine whether
those terms permit training and redistribution of derived weights. No dataset
or trained weights are included in this source repository.

## Release checklist

- [ ] Dataset card and provenance manifest completed.
- [ ] Dataset terms reviewed for training and weight redistribution.
- [ ] Training configuration, source revision, hardware, and artifact hashes recorded.
- [ ] Held-out evaluation and sample review completed.
- [ ] Weight license selected and included with the released artifact.
- [ ] OMF reader/runtime load test completed for the final artifact.
