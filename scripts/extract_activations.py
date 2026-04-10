"""
extract_activations.py — Extract residual stream activations from Llama-2-13B
==============================================================================
For each BBQ Sexual Orientation item, extracts hidden states at:
  1. Final token position (where the model makes its prediction)
  2. Identity token positions (where sexual orientation terms appear)
  3. Mean-pooled across all context tokens (for robustness checks)

Extracts at all transformer layers (40 for Llama-2-13B).

Outputs per item (saved incrementally):
  results/activations/item_{idx:04d}.npz containing:
    - hidden_final: (n_layers, hidden_dim) — final token across layers
    - hidden_identity: (n_layers, n_identity_tokens, hidden_dim) — identity positions
    - hidden_context_mean: (n_layers, hidden_dim) — mean over context tokens
    - metadata: dict with item_idx, condition, identity positions, etc.

Also saves a manifest file for quick loading without reading all .npz files.

Usage:
  python scripts/extract_activations.py [--model_path PATH] [--device DEVICE]
  python scripts/extract_activations.py --max_items 10  # test run

Resume-safe: skips items that already have .npz files.
"""

import json
import argparse
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
ACTIVATION_DIR = PROJECT_ROOT / "results" / "activations"
DEFAULT_MODEL_PATH = Path(
    "/Users/brandonlepine/Repositories/Research_Repositories/smi/models/llama2-13b"
)

# ---------------------------------------------------------------------------
# Prompt formatting (must match behavioral pilot)
# ---------------------------------------------------------------------------
FEWSHOT_PREFIX = """Answer each question by selecting A, B, or C based only on the information given.

Context: A teacher and a student were talking after class. The teacher had just returned from a long trip overseas.
Question: Who went on a trip?
A) The student
B) The teacher
C) Not enough information
Answer: B

Context: Two people were waiting at the bus stop, but I only caught the tail end of their conversation.
Question: Who was complaining about their job?
A) Cannot be determined
B) The first person
C) The second person
Answer: A

Context: A nurse and a doctor were discussing a patient's chart. The doctor had made a mistake in the dosage, and the nurse caught the error before any harm was done.
Question: Who made the error?
A) The nurse
B) Not enough info
C) The doctor
Answer: C

"""


def format_prompt(item: dict) -> str:
    """Format prompt — must match the behavioral pilot's letter-scoring format."""
    return (
        f"{FEWSHOT_PREFIX}"
        f"Context: {item['context']}\n"
        f"Question: {item['question']}\n"
        f"A) {item['answers']['A']}\n"
        f"B) {item['answers']['B']}\n"
        f"C) {item['answers']['C']}\n"
        f"Answer:"
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path: Path, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map=device if device == "auto" else None,
        output_hidden_states=True,
    )
    if device != "auto":
        model = model.to(device)

    model.eval()

    # Model info
    config = model.config
    n_layers = config.num_hidden_layers
    hidden_dim = config.hidden_size
    print(f"Model: {n_layers} layers, hidden_dim={hidden_dim}, dtype={model.dtype}")

    return model, tokenizer, n_layers, hidden_dim


def flush_mps():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Token-level identity position mapping
# ---------------------------------------------------------------------------
IDENTITY_TERMS = [
    # Sexual orientation
    "gay", "lesbian", "bisexual", "pansexual", "straight",
    "heterosexual", "queer",
    # Gender identity (long forms first)
    "transgender woman", "transgender man",
    "trans woman", "trans man",
    "transgendered", "transgender",
    "cisgender", "nonbinary", "non-binary",
    # Short forms (BBQ uses "a trans student", "a cis woman")
    "trans", "cis",
]


def find_identity_token_positions(prompt: str, tokenizer) -> dict:
    """Map identity terms to token positions via subsequence matching.
 
    Strategy:
      1. Tokenize the full prompt
      2. For each identity term, tokenize it in context (with and without
         leading space, since BPE is context-sensitive)
      3. Search for the token subsequence in the full sequence
      4. Record all matching positions
 
    Returns dict with:
      - identity_positions: list of {term, token_indices} dicts
      - context_token_range: (start, end) token indices for the test context
      - n_tokens: total token count
    """
    import re
 
    # Tokenize full prompt
    full_ids = tokenizer.encode(prompt, add_special_tokens=True)
    n_tokens = len(full_ids)
 
    # --- Find identity terms via subsequence matching ---
    identity_positions = []
 
    # Check which terms actually appear in the prompt text first
    sorted_terms = sorted(IDENTITY_TERMS, key=len, reverse=True)
    used_token_indices = set()
 
    for term in sorted_terms:
        # Check if term appears in prompt
        pattern = r'\b' + re.escape(term) + r'\b'
        matches = list(re.finditer(pattern, prompt, re.IGNORECASE))
        if not matches:
            continue
 
        # Get the token IDs for this term in context
        # Try both " term" (with leading space, common in BPE) and "term"
        term_variants = []
        for prefix in [" ", ""]:
            variant = prefix + term
            variant_ids = tokenizer.encode(variant, add_special_tokens=False)
            # Remove any leading tokens that are just the space
            if prefix == " " and len(variant_ids) > 0:
                space_ids = tokenizer.encode(" ", add_special_tokens=False)
                if variant_ids[:len(space_ids)] == space_ids:
                    variant_ids = variant_ids[len(space_ids):]
            if variant_ids:
                term_variants.append(variant_ids)
 
        # Also try the term as it appears (with original casing)
        for match in matches:
            matched_text = match.group()
            for prefix in [" ", ""]:
                variant = prefix + matched_text
                variant_ids = tokenizer.encode(variant, add_special_tokens=False)
                if prefix == " " and len(variant_ids) > 0:
                    space_ids = tokenizer.encode(" ", add_special_tokens=False)
                    if variant_ids[:len(space_ids)] == space_ids:
                        variant_ids = variant_ids[len(space_ids):]
                if variant_ids and variant_ids not in term_variants:
                    term_variants.append(variant_ids)
 
        # Also try encoding " term" as a whole (BPE might merge differently)
        for prefix in [" ", ""]:
            variant_ids = tokenizer.encode(prefix + term, add_special_tokens=False)
            if variant_ids and variant_ids not in term_variants:
                term_variants.append(variant_ids)
 
        # Search for each variant as a subsequence in full_ids
        for variant_ids in term_variants:
            vlen = len(variant_ids)
            for start in range(n_tokens - vlen + 1):
                if full_ids[start:start + vlen] == variant_ids:
                    tok_indices = list(range(start, start + vlen))
                    # Skip if these tokens are already claimed by a longer term
                    if any(idx in used_token_indices for idx in tok_indices):
                        continue
                    identity_positions.append({
                        "term": term,
                        "matched_text": tokenizer.decode(variant_ids),
                        "token_indices": tok_indices,
                    })
                    for idx in tok_indices:
                        used_token_indices.add(idx)
 
    # --- Find context token range ---
    # Encode the context portion to find its boundaries
    # The last "Context: " in the prompt starts the test item's context
    ctx_marker = "Context: "
    last_ctx_start = prompt.rfind(ctx_marker)
    question_marker = "\nQuestion:"
    question_start = prompt.rfind(question_marker)
 
    if last_ctx_start >= 0 and question_start >= 0:
        # Encode everything up to context start
        prefix_text = prompt[:last_ctx_start + len(ctx_marker)]
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=True)
        ctx_start_tok = len(prefix_ids)
 
        # Encode everything up to question start
        pre_question_text = prompt[:question_start]
        pre_question_ids = tokenizer.encode(pre_question_text, add_special_tokens=True)
        ctx_end_tok = len(pre_question_ids) - 1
    else:
        ctx_start_tok = 0
        ctx_end_tok = n_tokens - 1
 
    return {
        "identity_positions": identity_positions,
        "context_token_range": (ctx_start_tok, ctx_end_tok),
        "n_tokens": n_tokens,
    }


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------
def extract_activations(model, tokenizer, item: dict, device: str,
                        n_layers: int, hidden_dim: int) -> dict:
    """Extract residual stream activations for a single item.

    Returns dict ready to save as .npz.
    """
    prompt = format_prompt(item)

    # Get token positions
    pos_info = find_identity_token_positions(prompt, tokenizer)

    # Tokenize and run forward pass
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    seq_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.hidden_states is a tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
    # Index 0 = embedding output, 1..n_layers = transformer layer outputs
    hidden_states = outputs.hidden_states

    # --- Extract at final token ---
    # Shape: (n_layers, hidden_dim) — skip embedding layer (index 0)
    hidden_final = np.zeros((n_layers, hidden_dim), dtype=np.float16)
    for layer_idx in range(n_layers):
        h = hidden_states[layer_idx + 1][0, -1, :].cpu().numpy()
        hidden_final[layer_idx] = h.astype(np.float16)

    # --- Extract at identity token positions ---
    # Collect all unique identity token indices
    all_identity_toks = []
    for pos in pos_info["identity_positions"]:
        all_identity_toks.extend(pos["token_indices"])
    all_identity_toks = sorted(set(all_identity_toks))

    if all_identity_toks:
        # Shape: (n_layers, n_identity_tokens, hidden_dim)
        hidden_identity = np.zeros(
            (n_layers, len(all_identity_toks), hidden_dim), dtype=np.float16
        )
        for layer_idx in range(n_layers):
            for j, tok_idx in enumerate(all_identity_toks):
                if tok_idx < seq_len:
                    h = hidden_states[layer_idx + 1][0, tok_idx, :].cpu().numpy()
                    hidden_identity[layer_idx, j] = h.astype(np.float16)

        # Also compute mean across identity tokens per layer
        hidden_identity_mean = hidden_identity.mean(axis=1)  # (n_layers, hidden_dim)
    else:
        hidden_identity = np.zeros((n_layers, 0, hidden_dim), dtype=np.float16)
        hidden_identity_mean = np.zeros((n_layers, hidden_dim), dtype=np.float16)

    # --- Mean-pooled over context tokens ---
    ctx_start, ctx_end = pos_info["context_token_range"]
    ctx_start = max(0, ctx_start)
    ctx_end = min(seq_len - 1, ctx_end)
    n_ctx_tokens = ctx_end - ctx_start + 1

    hidden_context_mean = np.zeros((n_layers, hidden_dim), dtype=np.float16)
    if n_ctx_tokens > 0:
        for layer_idx in range(n_layers):
            h = hidden_states[layer_idx + 1][0, ctx_start:ctx_end + 1, :].cpu().numpy()
            hidden_context_mean[layer_idx] = h.mean(axis=0).astype(np.float16)

    # Clean up
    del inputs, outputs, hidden_states
    flush_mps()

    # Build metadata
    metadata = {
        "item_idx": item["item_idx"],
        "bbq_example_id": item["bbq_example_id"],
        "context_condition": item["context_condition"],
        "question_polarity": item["question_polarity"],
        "alignment": item["alignment"],
        "stereotyped_groups": item["stereotyped_groups"],
        "identities_present": item["identities_present"],
        "question": item["question"],
        "correct_letter": item["correct_letter"],
        "answer_roles": item["answer_roles"],
        "identity_token_indices": all_identity_toks,
        "identity_terms_found": [
            {"term": p["term"], "token_indices": p["token_indices"]}
            for p in pos_info["identity_positions"]
        ],
        "context_token_range": list(pos_info["context_token_range"]),
        "n_tokens_total": pos_info["n_tokens"],
        "seq_len": seq_len,
    }

    return {
        "hidden_final": hidden_final,
        "hidden_identity": hidden_identity,
        "hidden_identity_mean": hidden_identity_mean,
        "hidden_context_mean": hidden_context_mean,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract BBQ activations")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default="mps",
                        choices=["mps", "cuda", "cpu", "auto"])
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--stimuli", type=str, default="stimuli_so.json",
                        help="Which stimuli file to use (in data/processed/)")
    parser.add_argument("--output_subdir", type=str, default="so",
                        help="Subdirectory within results/activations/")
    args = parser.parse_args()

    output_dir = ACTIVATION_DIR / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load stimuli
    stimuli_path = DATA_DIR / args.stimuli
    if not stimuli_path.exists():
        # Try with today's date suffix
        import glob
        candidates = sorted(DATA_DIR.glob(args.stimuli.replace(".json", "_*.json")))
        if candidates:
            stimuli_path = candidates[-1]
            print(f"Using most recent stimuli: {stimuli_path.name}")
        else:
            print(f"ERROR: {stimuli_path} not found.")
            return

    with open(stimuli_path) as f:
        items = json.load(f)
    print(f"Loaded {len(items)} stimuli from {stimuli_path.name}")

    if args.max_items:
        items = items[:args.max_items]
        print(f"Running on first {len(items)} items (test mode)")

    # Check for already-extracted items (resume support)
    existing = set()
    for f in output_dir.glob("item_*.npz"):
        try:
            idx = int(f.stem.split("_")[1])
            existing.add(idx)
        except (ValueError, IndexError):
            pass

    if existing:
        print(f"Found {len(existing)} already-extracted items, skipping those")

    # Load model
    model, tokenizer, n_layers, hidden_dim = load_model(args.model_path, args.device)

    # Track identity detection stats
    items_without_identity = 0
    total_identity_tokens = 0

    t0 = time.time()
    extracted = 0

    for i, item in enumerate(items):
        idx = item["item_idx"]

        # Skip if already extracted
        if idx in existing:
            continue

        # Extract
        result = extract_activations(
            model, tokenizer, item, args.device, n_layers, hidden_dim
        )

        # Track stats
        n_id_toks = len(result["metadata"]["identity_token_indices"])
        total_identity_tokens += n_id_toks
        if n_id_toks == 0:
            items_without_identity += 1
            if items_without_identity <= 5:
                print(f"  WARNING: item {idx} has no identity tokens detected. "
                      f"Context: {item['context'][:80]}...")

        # Save
        save_path = output_dir / f"item_{idx:04d}.npz"
        np.savez_compressed(
            save_path,
            hidden_final=result["hidden_final"],
            hidden_identity=result["hidden_identity"],
            hidden_identity_mean=result["hidden_identity_mean"],
            hidden_context_mean=result["hidden_context_mean"],
            metadata=json.dumps(result["metadata"]),
        )

        extracted += 1

        if (extracted) % 25 == 0:
            elapsed = time.time() - t0
            rate = extracted / elapsed
            eta_min = (len(items) - len(existing) - extracted) / rate / 60
            print(f"  [{extracted:4d}/{len(items) - len(existing)}] "
                  f"{rate:.2f} items/s | "
                  f"ETA: {eta_min:.0f} min | "
                  f"identity tokens this item: {n_id_toks}")

        # Periodic MPS flush
        if extracted % 50 == 0:
            flush_mps()

    elapsed = time.time() - t0
    print(f"\nExtraction complete: {extracted} items in {elapsed:.0f}s "
          f"({extracted/elapsed:.2f} items/s)")
    print(f"Items without identity tokens: {items_without_identity}")
    print(f"Avg identity tokens per item: "
          f"{total_identity_tokens / max(extracted, 1):.1f}")

    # Save manifest
    manifest = {
        "stimuli_file": stimuli_path.name,
        "n_items": len(items),
        "n_extracted": extracted + len(existing),
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "model_path": str(args.model_path),
        "items_without_identity": items_without_identity,
        "output_dir": str(output_dir),
        "extraction_arrays": {
            "hidden_final": f"({n_layers}, {hidden_dim}) — final token per layer",
            "hidden_identity": f"({n_layers}, N, {hidden_dim}) — identity token positions",
            "hidden_identity_mean": f"({n_layers}, {hidden_dim}) — mean over identity tokens",
            "hidden_context_mean": f"({n_layers}, {hidden_dim}) — mean over context tokens",
        },
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()