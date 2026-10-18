# Open Mind Format (OMF) v0.1

OMF is a little-endian binary container for local models. Readers must reject
unknown versions, duplicate sections, truncated lengths, and tensors whose
shapes do not match their data. Records do not require alignment.

## Header

| Field | Size | Value |
|---|---:|---|
| Magic | 3 bytes | ASCII `OMF` |
| Version | u16 | `1` |
| Parameter count | u64 | Total number of logical scalar values in WEIGHTS |

## Sections

Each section starts with `type:u8`, followed by `length:u64` and exactly
`length` payload bytes. In v0.1, the required order is CONFIG, TOKENIZER,
WEIGHTS, METADATA; ids are `1`, `2`, `3`, `4`.

* **CONFIG (1):** UTF-8 JSON object. The reference model contains
  `hidden_size`, `layers`, `vocab`, `heads`, `intermediate_size`, and `context`.
* **TOKENIZER (2):** UTF-8 JSON object. The reference tokenizer declares
  `{"type":"byte","vocab":256}`; each UTF-8 byte is one token and vice versa.
* **WEIGHTS (3):** `tensor_count:u32`, followed by the tensor records below.
* **METADATA (4):** UTF-8 JSON object (for example, model name, step, and date).

Each tensor record has `name_len:u16`, a UTF-8 name (`name_len` bytes),
`dtype:u8`, `rank:u8`, `rank` dimensions as `u64`, `data_len:u64`, then the
data. Weights are row-major. `data_len` must agree with the dtype and element
count.

| dtype | Id | Encoding |
|---|---:|---|
| fp16 | 1 | IEEE-754 binary16 little-endian, 2 bytes per value |
| q4_0 | 2 | Blocks of 32: binary16 scale + 16 bytes with two nibbles; q = clamp(round(x/scale), -8, 7), nibble = q+8; the low nibble stores the first value. |
| q8_0 | 3 | Blocks of 32: binary16 scale + 32 int8 bytes; q = clamp(round(x/scale), -127, 127). |

For quantization, `scale=max(abs(block))/7` for q4_0 and `/127` for q8_0.
Incomplete final blocks are zero-padded; padding values are discarded during
decoding. A zero block uses scale 0 and quantized values 0 (nibble 8 for q4_0).

The reader materializes weights as float32 for inference. `param_count` is the
sum of logical tensor elements and does not include quantization padding.
