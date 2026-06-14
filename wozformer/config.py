"""Dataclass-based configuration for each architecture.

Replaces the scattered constant blocks at the top of every notebook with one
typed config per architecture. Construct directly, override fields by keyword,
serialize to disk for reproducibility.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict
import json


# ---- Project-wide constants ---------------------------------------------------
SEED = 1337
DEFAULT_CORPUS = "data/tinyshakespeare.txt"


# ---- Training config (shared by all architectures) ---------------------------
@dataclass
class TrainConfig:
    batch_size: int = 64
    block_size: int = 16
    lr: float = 3e-3
    weight_decay: float = 0.01
    n_steps: int = 12000
    eval_every: int = 500
    eval_batches: int = 20
    seed: int = SEED


# ---- Per-architecture configs -------------------------------------------------
@dataclass
class TransformerConfig:
    """nb05/nb07b dense transformer (the baseline teacher candidate)."""
    vocab_size: int = 128
    d_model: int = 32
    num_heads: int = 1
    n_layers: int = 2
    mlp_mult: int = 2
    dropout: float = 0.0


@dataclass
class MoEConfig:
    """nb09 sparse mixture-of-experts."""
    vocab_size: int = 128
    d_model: int = 32
    num_heads: int = 1
    n_layers: int = 2
    n_experts: int = 4
    expert_inner: int = 16
    dropout: float = 0.0
    lb_loss_weight: float = 0.01


@dataclass
class RWKVConfig:
    """nb10 linear-recurrence RWKV-style."""
    vocab_size: int = 128
    d_model: int = 32
    d_inner_cm: int = 64
    n_layers: int = 2


@dataclass
class HDCConfig:
    """nb11 Hebbian-trained HDC (random vocab, prototype sums)."""
    vocab_size: int = 128
    D: int = 1024
    T: int = 8
    vocab_seed: int = 0x57415A48   # 'WAZH'


@dataclass
class HDCRWKVConfig:
    """nb12c shipping architecture: bipolar HDC × RWKV recurrence."""
    vocab_size: int = 256
    d: int = 256
    n_layers: int = 1
    block_size: int = 16
    # Channel mixing not in shipping; off by default
    use_channel_mix: bool = False
    cm_inner: int = 64


@dataclass
class DistillationConfig:
    """Teacher → student knowledge distillation."""
    temperature: float = 4.0
    alpha_nll: float = 0.5         # weight on hard-target NLL
    alpha_distill: float = 0.5     # weight on soft-target KL


# ---- Serialization -----------------------------------------------------------
def save_config(cfg: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(cfg), indent=2))


def load_config(cls: type, path: str | Path) -> Any:
    return cls(**json.loads(Path(path).read_text()))
