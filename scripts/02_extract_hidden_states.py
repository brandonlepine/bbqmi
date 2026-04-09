from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from rich.progress import track

from bbqmi.io import read_csv, write_npz_dated
from bbqmi.schema import normalize_dataset
from bbqmi.text_model import ModelSpec, extract_pooled_hidden_states, load_model_and_tokenizer, resolve_device


def main() -> int:
    p = argparse.ArgumentParser(description="Extract pooled hidden states for each row in a dataset.")
    p.add_argument("--input_csv", type=str, required=True, help="Processed dataset CSV (id,text,group, ...).")
    p.add_argument("--model_name", type=str, required=True, help="Hugging Face model id (e.g. gpt2).")
    p.add_argument("--layer", type=int, default=-1, help="Which layer of hidden states to extract.")
    p.add_argument("--max_length", type=int, default=256, help="Tokenizer truncation length.")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    p.add_argument(
        "--pooling",
        type=str,
        default="mean_last_token",
        choices=["mean_last_token", "mean_all_tokens", "cls_token"],
        help="Pooling strategy to get one vector per example.",
    )
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    args = p.parse_args()

    df = normalize_dataset(read_csv(Path(args.input_csv)))
    texts = df["text"].tolist()
    ids = df["id"].tolist()
    groups = df["group"].tolist()

    spec = ModelSpec(name=args.model_name, layer=args.layer, max_length=args.max_length, device=args.device)
    device = resolve_device(spec.device)
    model, tok = load_model_and_tokenizer(spec)

    pooled: list[np.ndarray] = []
    for start in track(range(0, len(texts), args.batch_size), description="Batches"):
        batch_texts = texts[start : start + args.batch_size]
        vecs = extract_pooled_hidden_states(
            model=model,
            tokenizer=tok,
            texts=batch_texts,
            layer=spec.layer,
            max_length=spec.max_length,
            device=device,
            pooling=args.pooling,  # type: ignore[arg-type]
        )
        pooled.append(vecs.detach().to(torch.float32).cpu().numpy())

    hidden_states = np.concatenate(pooled, axis=0).astype(np.float32)
    out_path = write_npz_dated(
        dir_path=Path("artifacts"),
        stem="hidden_states",
        hidden_states=hidden_states,
        id=np.asarray(ids, dtype=object),
        group=np.asarray(groups, dtype=object),
        model_name=np.asarray([args.model_name], dtype=object),
        layer=np.asarray([args.layer], dtype=np.int32),
    )
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

