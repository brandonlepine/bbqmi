from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def get_num_layers(model: Any) -> int:
    return int(getattr(model.config, "num_hidden_layers"))


def get_num_heads(model: Any) -> int:
    return int(getattr(model.config, "num_attention_heads"))


def get_hidden_size(model: Any) -> int:
    return int(getattr(model.config, "hidden_size"))


def get_decoder_layers(model: Any) -> Sequence[Any]:
    """Return a sequence of per-layer decoder blocks.

    Supports common HF layouts for Llama/Gemma-like models.\n
    Raises a clear error if the expected module tree is not present.
    """
    # Common for Llama/Gemma in transformers: AutoModelForCausalLM -> model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        if isinstance(layers, Sequence):
            return layers
        return list(layers)

    # Some models expose layers directly
    if hasattr(model, "layers"):
        layers = model.layers
        if isinstance(layers, Sequence):
            return layers
        return list(layers)

    # Encoder-decoder / decoder naming variants
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        layers = model.model.decoder.layers
        if isinstance(layers, Sequence):
            return layers
        return list(layers)

    raise AttributeError(
        "Unsupported model architecture: could not locate decoder layers. "
        "Expected one of: model.model.layers, model.layers, model.model.decoder.layers."
    )

