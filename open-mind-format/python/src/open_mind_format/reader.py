"""Read OMF v0.1 model files into NumPy arrays."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

_MAX_SECTION_BYTES = 8 * 1024 * 1024 * 1024
_BLOCK_SIZE = 32


@dataclass
class OMFFile:
    config: dict
    tokenizer: dict
    metadata: dict
    tensors: dict[str, np.ndarray]


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated OMF file")
    return data


def _unpack(stream: BinaryIO, fmt: str):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, _read_exact(stream, size))


def _read_json(payload: bytes, section_name: str) -> dict:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {section_name} JSON section") from error
    if not isinstance(value, dict):
        raise ValueError(f"{section_name} must be a JSON object")
    return value


def _decode_tensor(dtype: int, count: int, data: bytes) -> np.ndarray:
    if dtype == 1:
        if len(data) != count * 2:
            raise ValueError("invalid fp16 tensor length")
        return np.frombuffer(data, dtype="<f2").astype(np.float32)

    block_bytes = 18 if dtype == 2 else 34 if dtype == 3 else 0
    if not block_bytes:
        raise ValueError(f"unsupported OMF tensor dtype {dtype}")
    blocks = (count + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    if len(data) != blocks * block_bytes:
        raise ValueError("invalid quantized tensor length")
    output = np.empty(blocks * _BLOCK_SIZE, dtype=np.float32)
    for block_index in range(blocks):
        offset = block_index * block_bytes
        scale = np.frombuffer(data, dtype="<f2", count=1, offset=offset).astype(np.float32)[0]
        if dtype == 2:
            packed = np.frombuffer(data, dtype=np.uint8, count=16, offset=offset + 2)
            quantized = np.empty(_BLOCK_SIZE, dtype=np.int8)
            quantized[0::2] = (packed & 0x0F).astype(np.int8) - 8
            quantized[1::2] = (packed >> 4).astype(np.int8) - 8
        else:
            quantized = np.frombuffer(data, dtype=np.int8, count=_BLOCK_SIZE, offset=offset + 2)
        output[block_index * _BLOCK_SIZE : (block_index + 1) * _BLOCK_SIZE] = quantized * scale
    return output[:count]


def read_omf(path: str | Path) -> OMFFile:
    """Read and validate OMF v0.1, dequantizing weights to float32 arrays."""
    tensors: dict[str, np.ndarray] = {}
    with Path(path).open("rb") as source:
        if _read_exact(source, 3) != b"OMF":
            raise ValueError("invalid OMF magic")
        (version,) = _unpack(source, "<H")
        if version != 1:
            raise ValueError(f"unsupported OMF version {version}")
        (param_count,) = _unpack(source, "<Q")
        sections: dict[int, bytes] = {}
        for expected_id in range(1, 5):
            (section_id,) = _unpack(source, "<B")
            (section_length,) = _unpack(source, "<Q")
            if section_id != expected_id:
                raise ValueError("missing, duplicate, or out-of-order OMF section")
            if section_length > _MAX_SECTION_BYTES:
                raise ValueError("OMF section exceeds the 8 GiB safety limit")
            sections[section_id] = _read_exact(source, section_length)
        if source.read(1):
            raise ValueError("trailing bytes after OMF sections")

    config = _read_json(sections[1], "CONFIG")
    tokenizer = _read_json(sections[2], "TOKENIZER")
    metadata = _read_json(sections[4], "METADATA")
    weights = memoryview(sections[3])
    if len(weights) < 4:
        raise ValueError("truncated WEIGHTS section")
    (tensor_count,) = struct.unpack_from("<I", weights, 0)
    if tensor_count > 1_000_000:
        raise ValueError("too many tensors in OMF file")
    offset = 4
    actual_params = 0
    for _ in range(tensor_count):
        if offset + 2 > len(weights):
            raise ValueError("truncated tensor name length")
        (name_len,) = struct.unpack_from("<H", weights, offset)
        offset += 2
        if offset + name_len + 2 > len(weights):
            raise ValueError("truncated tensor header")
        try:
            name = bytes(weights[offset : offset + name_len]).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("tensor name is not valid UTF-8") from error
        offset += name_len
        dtype, rank = struct.unpack_from("<BB", weights, offset)
        offset += 2
        shape_bytes = rank * 8
        if offset + shape_bytes + 8 > len(weights):
            raise ValueError(f"truncated tensor shape for {name}")
        shape = struct.unpack_from(f"<{rank}Q", weights, offset) if rank else ()
        offset += shape_bytes
        (data_len,) = struct.unpack_from("<Q", weights, offset)
        offset += 8
        if data_len > len(weights) - offset:
            raise ValueError(f"truncated tensor data for {name}")
        data = bytes(weights[offset : offset + data_len])
        offset += data_len
        elements = math.prod(shape) if shape else 1
        if name in tensors:
            raise ValueError(f"duplicate tensor name {name}")
        values = _decode_tensor(dtype, elements, data)
        tensors[name] = values.reshape(shape)
        actual_params += elements
    if offset != len(weights):
        raise ValueError("trailing bytes in WEIGHTS section")
    if actual_params != param_count:
        raise ValueError("OMF parameter count does not match tensor data")
    return OMFFile(config=config, tokenizer=tokenizer, metadata=metadata, tensors=tensors)
