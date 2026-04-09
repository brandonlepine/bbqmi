from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from transformers import AutoModel, AutoTokenizer


PoolingStrategy = Literal["mean_last_token", "mean_all_tokens", "cls_token"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    layer: int = -1
    max_length: int = 256
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(spec: ModelSpec):
    tok = AutoTokenizer.from_pretrained(spec.name, use_fast=True)
    model = AutoModel.from_pretrained(spec.name, output_hidden_states=True)
    model.eval()
    model.to(resolve_device(spec.device))
    return model, tok


@torch.no_grad()
def extract_pooled_hidden_states(
    *,
    model,
    tokenizer,
    texts: list[str],
    layer: int,
    max_length: int,
    device: torch.device,
    pooling: PoolingStrategy,
) -> torch.Tensor:
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    hidden = out.hidden_states[layer]  # (batch, seq, hidden)
    attn = enc.get("attention_mask")  # (batch, seq)

    if pooling == "mean_all_tokens":
        if attn is None:
            return hidden.mean(dim=1)
        weights = attn.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return summed / denom

    if pooling == "cls_token":
        return hidden[:, 0, :]

    # mean_last_token: pool the last non-pad token when attention_mask exists,
    # otherwise take the last position.
    if attn is None:
        return hidden[:, -1, :]
    lengths = attn.sum(dim=1).clamp_min(1)  # (batch,)
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
    return hidden.gather(dim=1, index=idx).squeeze(1)

