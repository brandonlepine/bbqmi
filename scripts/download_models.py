"""
download_models.py — Download gated HF LLMs into ./models/
=========================================================

Convenience helper for RunPod (or any machine) to download one or more
Hugging Face models using your credentials.

Default destination: <repo_root>/models/<model_key>/

Token handling (in order):
  1) --hf_token
  2) HF_TOKEN environment variable
  3) HUGGINGFACE_HUB_TOKEN environment variable

Examples
--------
List available model keys:
  python scripts/download_models.py --list

Download two models:
  python scripts/download_models.py --models llama2-13b-chat llama3.1-8b

Download everything:
  python scripts/download_models.py --all

Specify destination root (e.g., /workspace/bbqmi/models on RunPod):
  python scripts/download_models.py --all --models_dir /workspace/bbqmi/models
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


MODEL_CATALOG: dict[str, str] = {
    # Llama 2
    "llama2-13b-base": "meta-llama/Llama-2-13b-hf",
    "llama2-7b-base": "meta-llama/Llama-2-7b-hf",
    "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    # Llama 3.x
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B",
    # Gemma
    "gemma-4-31b-it": "google/gemma-4-31B-it",
}


def _get_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def main() -> int:
    p = argparse.ArgumentParser(description="Download Hugging Face models into ./models/")
    p.add_argument(
        "--models_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models",
        help="Root directory to store models (default: <repo_root>/models).",
    )
    p.add_argument("--hf_token", type=str, default=None, help="Hugging Face token (or set HF_TOKEN env var).")
    p.add_argument("--revision", type=str, default=None, help="Optional model revision (branch/tag/commit).")
    p.add_argument("--list", action="store_true", help="List available model keys and exit.")
    p.add_argument("--all", action="store_true", help="Download all models in the catalog.")
    p.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="One or more model keys from the catalog (see --list).",
    )
    args = p.parse_args()

    if args.list:
        width = max(len(k) for k in MODEL_CATALOG)
        for k, repo in sorted(MODEL_CATALOG.items()):
            print(f"{k:<{width}}  {repo}")
        return 0

    selected: list[str]
    if args.all:
        selected = sorted(MODEL_CATALOG.keys())
    else:
        selected = list(dict.fromkeys(args.models))  # de-dupe, preserve order

    if not selected:
        raise SystemExit("No models selected. Use --models ... or --all (or --list).")

    unknown = [k for k in selected if k not in MODEL_CATALOG]
    if unknown:
        raise SystemExit(f"Unknown model keys: {unknown}. Use --list to see valid keys.")

    token = _get_token(args.hf_token)
    if token is None:
        raise SystemExit(
            "Missing Hugging Face token. Provide --hf_token or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN."
        )

    args.models_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    for key in selected:
        repo_id = MODEL_CATALOG[key]
        local_dir = args.models_dir / key
        local_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n==> Downloading {key} ({repo_id})")
        print(f"    -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=token,
            revision=args.revision,
            local_dir_use_symlinks=False,
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

