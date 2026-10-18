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

from open_mind_format import read_omf, write_omf

_PICO_CONFIG = {
    "hidden_size": 384,
    "layers": 8,
    "vocab": 256,
    "heads": 6,
    "intermediate_size": 1632,
    "context": 256,
    "dropout": 0.1,
}

PRESETS = {
    "15M": dict(_PICO_CONFIG),
    "Pico": dict(_PICO_CONFIG),
    "tiny": {
        "hidden_size": 64,
        "layers": 2,
        "vocab": 256,
        "heads": 4,
        "intermediate_size": 192,
        "context": 128,
        "dropout": 0.1,
    },
}


class Block(nn.Module):
    def __init__(self, config: Mapping[str, int | float]) -> None:
        super().__init__()
        hidden = config["hidden_size"]
        self.heads = config["heads"]
        self.head_dim = hidden // self.heads
        self.ln1 = nn.LayerNorm(hidden, eps=1e-5)
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(hidden, 3 * hidden)
        self.attn.proj = nn.Linear(hidden, hidden)
        self.attn_dropout = nn.Dropout(config["dropout"])
        self.resid_dropout = nn.Dropout(config["dropout"])
        self.ln2 = nn.LayerNorm(hidden, eps=1e-5)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(hidden, config["intermediate_size"])
        self.mlp.fc2 = nn.Linear(config["intermediate_size"], hidden)
        self.mlp_dropout = nn.Dropout(config["dropout"])

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
        probabilities = self.attn_dropout(scores.softmax(dim=-1))
        attended = (probabilities @ v).transpose(1, 2).contiguous()
        attended = attended.view(batch, length, hidden)
        x = x + self.resid_dropout(self.attn.proj(attended))
        hidden_state = self.mlp.fc1(self.ln2(x))
        hidden_state = F.gelu(hidden_state, approximate="tanh")
        return x + self.resid_dropout(self.mlp.fc2(self.mlp_dropout(hidden_state)))


class ByteTransformer(nn.Module):
    def __init__(self, config: Mapping[str, int | float]) -> None:
        super().__init__()
        hidden = config["hidden_size"]
        self.config = dict(config)
        self.token_embedding = nn.Embedding(config["vocab"], hidden)
        self.position_embedding = nn.Embedding(config["context"], hidden)
        self.embedding_dropout = nn.Dropout(config["dropout"])
        self.layers = nn.ModuleList(Block(config) for _ in range(config["layers"]))
        self.final_norm = nn.LayerNorm(hidden, eps=1e-5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        x = self.embedding_dropout(
            self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        )
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
    def __init__(self, path: str | os.PathLike[str], val_fraction: float = 0.05) -> None:
        self.file = open(path, "rb")
        if os.fstat(self.file.fileno()).st_size < 2:
            self.file.close()
            raise ValueError("dataset must contain at least two bytes")
        self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.split_at = int(len(self.mapping) * (1.0 - val_fraction))
        if self.split_at < 2 or len(self.mapping) - self.split_at < 2:
            self.close()
            raise ValueError("dataset is too small for the requested train/validation split")

    def batch(
        self, batch_size: int, seq_len: int, device: torch.device, split: str = "train"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start_offset, end_offset = (0, self.split_at) if split == "train" else (self.split_at, len(self.mapping))
        start_count = end_offset - start_offset - seq_len
        if start_count < 1:
            raise ValueError(f"{split} split needs at least {seq_len + 1} bytes")
        starts = (torch.randint(start_count, (batch_size,)) + start_offset).tolist()
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
        optimizer: str = "sgd8bit",
        weight_decay: float = 0.1,
        validation_fraction: float = 0.05,
        eval_interval: int = 500,
        eval_batches: int = 20,
        warmup_steps: int = 500,
        min_learning_rate_ratio: float = 0.1,
        gradient_clip: float = 1.0,
        resume_from: str | os.PathLike[str] | None = None,
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
        if optimizer not in {"sgd8bit", "adamw"}:
            raise ValueError("optimizer must be 'sgd8bit' or 'adamw'")
        if not 0.0 < validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")
        if eval_interval < 1 or eval_batches < 1 or warmup_steps < 0:
            raise ValueError("evaluation interval/batches must be positive and warmup non-negative")
        if not 0.0 <= min_learning_rate_ratio <= 1.0 or gradient_clip <= 0:
            raise ValueError("min_learning_rate_ratio must be in [0, 1] and gradient_clip must be positive")
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
        self.optimizer_name = optimizer
        self.weight_decay = weight_decay
        self.validation_fraction = validation_fraction
        self.eval_interval = eval_interval
        self.eval_batches = eval_batches
        self.warmup_steps = warmup_steps
        self.min_learning_rate_ratio = min_learning_rate_ratio
        self.gradient_clip = gradient_clip
        self.resume_from = Path(resume_from) if resume_from is not None else None
        if not 0.0 <= PRESETS[model]["dropout"] < 1.0:
            raise ValueError("model dropout must be in [0, 1)")

    def _save(
        self, model: ByteTransformer, path: Path, step: int, loss: float, validation_loss: float | None
    ) -> None:
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
                "validation_loss": validation_loss,
                "optimizer": self.optimizer_name,
            },
            dtype=self.dtype,
        )
        validation = f", val_loss={validation_loss:.4f}" if validation_loss is not None else ""
        print(f"checkpoint: {path} (step {step}, loss={loss:.4f}{validation})", flush=True)

    def train(self) -> Path:
        torch.manual_seed(self.seed)
        if self.device.type == "cpu":
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        config = dict(PRESETS[self.model_name])
        config["context"] = max(config["context"], self.seq_len)
        model = ByteTransformer(config).to(self.device)
        start_step = 0
        if self.resume_from is not None:
            checkpoint = read_omf(self.resume_from)
            if checkpoint.config != config:
                raise ValueError("resume checkpoint CONFIG does not match the selected model configuration")
            model.load_state_dict(
                {name: torch.from_numpy(values.copy()) for name, values in checkpoint.tensors.items()},
                strict=True,
            )
            start_step = int(checkpoint.metadata.get("step", 0))
            if start_step < 0 or start_step >= self.steps:
                raise ValueError(f"resume checkpoint step {start_step} must be below target steps {self.steps}")
            print(
                f"resuming weights at step {start_step}; optimizer moments restart from zero",
                flush=True,
            )
        if self.optimizer_name == "adamw":
            decay = []
            no_decay = []
            for parameter in model.parameters():
                (decay if parameter.ndim >= 2 else no_decay).append(parameter)
            optimizer = torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": self.weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=self.learning_rate,
                betas=(0.9, 0.95),
            )
        else:
            optimizer = SGD8Bit(model.parameters(), lr=self.learning_rate)
        use_bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        amp_enabled = self.device.type == "cuda" and (
            use_bf16 or self.optimizer_name == "adamw"
        )
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled and amp_dtype == torch.float16
        )

        def forward_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            with torch.autocast(
                device_type=self.device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                logits = model(x)
                return F.cross_entropy(logits.reshape(-1, config["vocab"]), y.reshape(-1))

        periodic_checkpoint = self.output.with_name(f"{self.output.stem}-v1.omf")
        data = _MMapBytes(self.dataset, self.validation_fraction)
        started = time.monotonic()
        last_loss = float("nan")
        last_validation_loss: float | None = None
        warmup_steps = min(self.warmup_steps, max(0, self.steps - 1))
        try:
            model.train()
            for step in range(start_step + 1, self.steps + 1):
                if self.optimizer_name == "adamw":
                    if step <= warmup_steps:
                        lr_ratio = step / max(1, warmup_steps)
                    else:
                        progress = min(1.0, (step - warmup_steps) / max(1, self.steps - warmup_steps))
                        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
                        lr_ratio = self.min_learning_rate_ratio + (1.0 - self.min_learning_rate_ratio) * cosine
                    for group in optimizer.param_groups:
                        group["lr"] = self.learning_rate * float(lr_ratio)
                for parameter in model.parameters():
                    parameter.grad = None
                total_loss = 0.0
                for _ in range(self.gradient_accumulation_steps):
                    x, y = data.batch(self.batch_size, self.seq_len, self.device, "train")
                    loss = forward_loss(x, y)
                    scaler.scale(loss / self.gradient_accumulation_steps).backward()
                    total_loss += loss.detach().item() / self.gradient_accumulation_steps
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                last_loss = total_loss
                should_evaluate = step == 1 or step % self.eval_interval == 0 or step == self.steps
                if should_evaluate:
                    model.eval()
                    validation_losses = []
                    with torch.no_grad():
                        for _ in range(self.eval_batches):
                            x, y = data.batch(self.batch_size, self.seq_len, self.device, "validation")
                            validation_losses.append(forward_loss(x, y).item())
                    last_validation_loss = float(np.mean(validation_losses))
                    model.train()
                if step == 1 or step % 10 == 0:
                    validation = f" val_loss={last_validation_loss:.4f}" if last_validation_loss is not None else ""
                    print(f"step {step}/{self.steps} loss={last_loss:.4f}{validation}", flush=True)
                if step % self.checkpoint_every == 0:
                    self._save(model, periodic_checkpoint, step, last_loss, last_validation_loss)
                if step == self.steps:
                    self._save(model, self.output, step, last_loss, last_validation_loss)
            print(f"training finished in {time.monotonic() - started:.1f}s", flush=True)
        finally:
            data.close()
        return self.output
