"""Write OMF v0.1 models from PyTorch tensors."""

from __future__ import annotations

import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

_DTYPES = {"fp16": 1, "q4_0": 2, "q8_0": 3}


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _quantize(values: np.ndarray, dtype: str) -> bytes:
    padded = np.zeros(((values.size + 31) // 32) * 32, dtype=np.float32)
    padded[: values.size] = values
    out = bytearray()
    for block in padded.reshape(-1, 32):
        max_abs = float(np.max(np.abs(block)))
        divisor = 7.0 if dtype == "q4_0" else 127.0
        scale = max_abs / divisor if max_abs else 0.0
        out += struct.pack("<e", scale)
        if scale:
            scaled = block / scale
            rounded = np.where(scaled >= 0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
            upper = 7 if dtype == "q4_0" else 127
            quantized = np.clip(rounded, -8 if dtype == "q4_0" else -127, upper).astype(np.int8)
        else:
            quantized = np.zeros(32, dtype=np.int8)
        if dtype == "q4_0":
            nibbles = (quantized.astype(np.int16) + 8).astype(np.uint8)
            out += (nibbles[0::2] | (nibbles[1::2] << 4)).tobytes()
        else:
            out += quantized.view(np.uint8).tobytes()
    return bytes(out)


def write_omf(
    path: str | os.PathLike[str],
    tensors: Mapping[str, torch.Tensor],
    config: Mapping[str, object],
    tokenizer: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
    dtype: str = "fp16",
) -> Path:
    """Atomically serialize a mapping of named PyTorch tensors to OMF v0.1.

    ``dtype`` can be ``fp16``, ``q4_0`` or ``q8_0``. Quantization applies to
    every tensor in this call; tensor names and shapes are preserved.
    """
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported OMF dtype {dtype!r}; choose one of {tuple(_DTYPES)}")
    if not tensors:
        raise ValueError("at least one tensor is required")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    param_count = 0
    encoded_tensors: list[bytes] = []
    for name, tensor in sorted(tensors.items()):
        if not isinstance(name, str) or not name or len(name.encode("utf-8")) > 65535:
            raise ValueError("tensor names must be non-empty UTF-8 strings of at most 65535 bytes")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} is not a PyTorch tensor")
        if tensor.ndim > 255:
            raise ValueError(f"{name} rank exceeds OMF limit")
        array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().reshape(-1)
        param_count += int(array.size)
        if dtype == "fp16":
            data = array.astype("<f2", copy=False).tobytes()
        else:
            data = _quantize(array, dtype)
        name_bytes = name.encode("utf-8")
        record = bytearray(struct.pack("<H", len(name_bytes)))
        record += name_bytes
        record += struct.pack("<BB", _DTYPES[dtype], tensor.ndim)
        for dim in tensor.shape:
            record += struct.pack("<Q", int(dim))
        record += struct.pack("<Q", len(data))
        record += data
        encoded_tensors.append(bytes(record))

    weights = struct.pack("<I", len(encoded_tensors)) + b"".join(encoded_tensors)
    sections = [
        (1, _json_bytes(dict(config))),
        (2, _json_bytes(dict(tokenizer or {"type": "byte", "vocab": 256}))),
        (3, weights),
        (4, _json_bytes(dict(metadata or {}))),
    ]
    payload = bytearray(b"OMF" + struct.pack("<HQ", 1, param_count))
    for section_id, section_data in sections:
        payload += struct.pack("<BQ", section_id, len(section_data))
        payload += section_data

    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path
