"""CPU-first byte-level language-model trainer for the OMF v0.1 runtime."""

from __future__ import annotations

import mmap
import os
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from open_mind_format import write_omf

PRESETS = {
    "15M": {
        "hidden_size": 384,
        "layers": 8,
        "vocab": 256,
        "heads": 6,
        "intermediate_size": 1632,
        "context": 256,
    },
    "tiny": {
        "hidden_size": 64,
        "layers": 2,
        "vocab": 256,
        "heads": 4,
        "intermediate_size": 192,
        "context": 128,
    },
}


class Block(nn.Module):
    def __init__(self, config: Mapping[str, int]) -> None:
        super().__init__()
        hidden = config["hidden_size"]
        self.heads = config["heads"]
        self.head_dim = hidden // self.heads
        self.ln1 = nn.LayerNorm(hidden, eps=1e-5)
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(hidden, 3 * hidden)
        self.attn.proj = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden, eps=1e-5)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(hidden, config["intermediate_size"])
        self.mlp.fc2 = nn.Linear(config["intermediate_size"], hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = x.shape
        qkv = self.attn.qkv(self.ln1(x)).view(
            batch, length, 3, self.heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (item.transpose(1, 2) for item in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        attended = (scores.softmax(dim=-1) @ v).transpose(1, 2).contiguous()
        attended = attended.view(batch, length, hidden)
        x = x + self.attn.proj(attended)
        hidden_state = self.mlp.fc1(self.ln2(x))
        hidden_state = F.gelu(hidden_state, approximate="tanh")
        return x + self.mlp.fc2(hidden_state)


class ByteTransformer(nn.Module):
    def __init__(self, config: Mapping[str, int]) -> None:
        super().__init__()
        hidden = config["hidden_size"]
        self.config = dict(config)
        self.token_embedding = nn.Embedding(config["vocab"], hidden)
        self.position_embedding = nn.Embedding(config["context"], hidden)
        self.layers = nn.ModuleList(Block(config) for _ in range(config["layers"]))
        self.final_norm = nn.LayerNorm(hidden, eps=1e-5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return F.linear(x, self.token_embedding.weight)


class SGD8Bit:
    """SGD with momentum stored as per-tensor int8 plus a float32 scale."""

    def __init__(self, parameters, lr: float, momentum: float = 0.9) -> None:
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.buffers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def step(self) -> None:
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            key = id(parameter)
            if key in self.buffers:
                quantized, scale = self.buffers[key]
                buffer = quantized.to(torch.float32) * scale
                buffer.mul_(self.momentum).add_(parameter.grad)
            else:
                buffer = parameter.grad.detach().clone()
            max_abs = buffer.abs().max()
            scale = max_abs / 127.0 if max_abs.item() else buffer.new_tensor(0.0)
            if scale.item():
                quantized = torch.clamp(torch.round(buffer / scale), -127, 127).to(torch.int8)
                update = buffer
            else:
                quantized = torch.zeros_like(buffer, dtype=torch.int8)
                update = buffer
            self.buffers[key] = (quantized, scale.detach())
            parameter.add_(update, alpha=-self.lr)


class _MMapBytes:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.file = open(path, "rb")
        if os.fstat(self.file.fileno()).st_size < 2:
            self.file.close()
            raise ValueError("dataset must contain at least two bytes")
        self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def batch(self, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        last = len(self.mapping) - seq_len - 1
        if last < 0:
            raise ValueError(f"dataset needs at least {seq_len + 1} bytes for this sequence length")
        starts = torch.randint(last + 1, (batch_size,)).tolist()
        inputs, targets = [], []
        for start in starts:
            inputs.append(torch.frombuffer(bytearray(self.mapping[start : start + seq_len]), dtype=torch.uint8))
            targets.append(torch.frombuffer(bytearray(self.mapping[start + 1 : start + seq_len + 1]), dtype=torch.uint8))
        return torch.stack(inputs).to(device=device, dtype=torch.long), torch.stack(targets).to(device=device, dtype=torch.long)

    def close(self) -> None:
        self.mapping.close()
        self.file.close()


class Trainer:
    """Train a byte-level causal transformer and write checkpoints directly as OMF."""

    def __init__(
        self,
        model: str = "15M",
        device: str = "cpu",
        dataset: str | os.PathLike[str] = "data.txt",
        output: str | os.PathLike[str] = "pico.omf",
        steps: int = 100,
        checkpoint_every: int = 100,
        batch_size: int = 1,
        seq_len: int = 128,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 1e-3,
        dtype: str = "fp16",
        seed: int = 42,
    ) -> None:
        if model not in PRESETS:
            raise ValueError(f"unknown model preset {model!r}; choose from {tuple(PRESETS)}")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available")
        if steps < 1 or checkpoint_every < 1 or batch_size < 1 or gradient_accumulation_steps < 1:
            raise ValueError("steps, checkpoint interval, batch size, and accumulation must be positive")
        if seq_len < 1 or seq_len > PRESETS[model]["context"]:
            raise ValueError(f"seq_len must be between 1 and {PRESETS[model]['context']}")
        self.model_name = model
        self.device = torch.device(device)
        self.dataset = Path(dataset)
        self.output = Path(output)
        self.steps = steps
        self.checkpoint_every = checkpoint_every
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.dtype = dtype
        self.seed = seed

    def _save(self, model: ByteTransformer, path: Path, step: int, loss: float) -> None:
        weights = {name: tensor.detach() for name, tensor in model.state_dict().items()}
        write_omf(
            path,
            weights,
            model.config,
            tokenizer={"type": "byte", "vocab": 256, "encoding": "utf-8"},
            metadata={
                "model": self.model_name,
                "step": step,
                "loss": loss,
                "optimizer": "SGD8Bit",
            },
            dtype=self.dtype,
        )
        print(f"checkpoint: {path} (step {step}, loss {loss:.4f})", flush=True)

    def train(self) -> Path:
        torch.manual_seed(self.seed)
        if self.device.type == "cpu":
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        config = dict(PRESETS[self.model_name])
        config["context"] = max(config["context"], self.seq_len)
        model = ByteTransformer(config).to(self.device)
        optimizer = SGD8Bit(model.parameters(), lr=self.learning_rate)
        periodic_checkpoint = self.output.with_name(f"{self.output.stem}-v1.omf")
        data = _MMapBytes(self.dataset)
        started = time.monotonic()
        last_loss = float("nan")
        try:
            model.train()
            for step in range(1, self.steps + 1):
                for parameter in model.parameters():
                    parameter.grad = None
                total_loss = 0.0
                for _ in range(self.gradient_accumulation_steps):
                    x, y = data.batch(self.batch_size, self.seq_len, self.device)
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, config["vocab"]), y.reshape(-1))
                    (loss / self.gradient_accumulation_steps).backward()
                    total_loss += loss.detach().item() / self.gradient_accumulation_steps
                optimizer.step()
                last_loss = total_loss
                if step == 1 or step % 10 == 0:
                    print(f"step {step}/{self.steps} loss={last_loss:.4f}", flush=True)
                if step % self.checkpoint_every == 0:
                    self._save(model, periodic_checkpoint, step, last_loss)
                if step == self.steps:
                    self._save(model, self.output, step, last_loss)
            print(f"training finished in {time.monotonic() - started:.1f}s", flush=True)
        finally:
            data.close()
        return self.output
