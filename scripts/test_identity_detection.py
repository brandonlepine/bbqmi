"""
test_identity_detection.py — Verify identity token detection on sample items
=============================================================================
Run this BEFORE the full extraction to confirm the fix works.
Only needs the tokenizer, not the full model.

Usage:
  python scripts/test_identity_detection.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_PATH = Path(
    "/Users/brandonlepine/Repositories/Research_Repositories/smi/models/llama2-13b"
)

# Must match extract_activations.py
IDENTITY_TERMS = [
    # Sexual orientation
    "gay", "lesbian", "bisexual", "pansexual", "straight",
    "heterosexual", "queer",
    # Gender identity
    "transgender woman", "transgender man",
    "trans woman", "trans man",
    "transgendered", "transgender",
    "cisgender", "nonbinary", "non-binary",
]

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


def format_prompt(item):
    return (
        f"{FEWSHOT_PREFIX}"
        f"Context: {item['context']}\n"
        f"Question: {item['question']}\n"
        f"A) {item['answers']['A']}\n"
        f"B) {item['answers']['B']}\n"
        f"C) {item['answers']['C']}\n"
        f"Answer:"
    )


# Paste the fixed function here (or import from extract_activations)
import re

def find_identity_token_positions(prompt, tokenizer):
    full_ids = tokenizer.encode(prompt, add_special_tokens=True)
    n_tokens = len(full_ids)

    identity_positions = []
    sorted_terms = sorted(IDENTITY_TERMS, key=len, reverse=True)
    used_token_indices = set()

    for term in sorted_terms:
        pattern = r'\b' + re.escape(term) + r'\b'
        matches = list(re.finditer(pattern, prompt, re.IGNORECASE))
        if not matches:
            continue

        term_variants = []
        for prefix in [" ", ""]:
            variant = prefix + term
            variant_ids = tokenizer.encode(variant, add_special_tokens=False)
            if prefix == " " and len(variant_ids) > 0:
                space_ids = tokenizer.encode(" ", add_special_tokens=False)
                if variant_ids[:len(space_ids)] == space_ids:
                    variant_ids = variant_ids[len(space_ids):]
            if variant_ids:
                term_variants.append(variant_ids)

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

        for prefix in [" ", ""]:
            variant_ids = tokenizer.encode(prefix + term, add_special_tokens=False)
            if variant_ids and variant_ids not in term_variants:
                term_variants.append(variant_ids)

        for variant_ids in term_variants:
            vlen = len(variant_ids)
            for start in range(n_tokens - vlen + 1):
                if full_ids[start:start + vlen] == variant_ids:
                    tok_indices = list(range(start, start + vlen))
                    if any(idx in used_token_indices for idx in tok_indices):
                        continue
                    identity_positions.append({
                        "term": term,
                        "matched_text": tokenizer.decode(variant_ids),
                        "token_indices": tok_indices,
                    })
                    for idx in tok_indices:
                        used_token_indices.add(idx)

    ctx_marker = "Context: "
    last_ctx_start = prompt.rfind(ctx_marker)
    question_marker = "\nQuestion:"
    question_start = prompt.rfind(question_marker)

    if last_ctx_start >= 0 and question_start >= 0:
        prefix_text = prompt[:last_ctx_start + len(ctx_marker)]
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=True)
        ctx_start_tok = len(prefix_ids)

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


def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_MODEL_PATH))

    # Load stimuli
    import glob
    candidates = sorted(DATA_DIR.glob("stimuli_gi*.json"))
    if not candidates:
        print("ERROR: No stimuli files found")
        return
    stimuli_path = candidates[-1]

    with open(stimuli_path) as f:
        items = json.load(f)
    print(f"Loaded {len(items)} items from {stimuli_path.name}")

    # Test on first 50 items
    n_test = min(50, len(items))
    n_found = 0
    n_missing = 0

    for i in range(n_test):
        item = items[i]
        prompt = format_prompt(item)
        result = find_identity_token_positions(prompt, tokenizer)

        n_id = len(result["identity_positions"])
        if n_id > 0:
            n_found += 1
        else:
            n_missing += 1

        if i < 10 or n_id == 0:
            terms_str = ", ".join(
                f"{p['term']}@tok{p['token_indices']}"
                for p in result["identity_positions"]
            )
            status = "OK" if n_id > 0 else "MISSING"
            print(f"  [{status}] Item {i}: {n_id} identity terms found: {terms_str}")
            if n_id > 0:
                # Verify by decoding the token indices
                full_ids = tokenizer.encode(prompt, add_special_tokens=True)
                for p in result["identity_positions"]:
                    decoded = tokenizer.decode([full_ids[idx] for idx in p["token_indices"]])
                    print(f"         {p['term']} -> decoded: '{decoded}'")

    print(f"\nResults: {n_found}/{n_test} items have identity tokens, "
          f"{n_missing}/{n_test} missing")

    if n_missing > 0:
        print("\nWARNING: Some items still missing identity tokens!")
        print("Checking which terms are problematic...")

        # Check each term individually
        for term in IDENTITY_TERMS:
            # How does the tokenizer encode this term?
            for prefix in [" ", ""]:
                ids = tokenizer.encode(prefix + term, add_special_tokens=False)
                decoded = tokenizer.decode(ids)
                print(f"  '{prefix}{term}' -> tokens: {ids} -> decoded: '{decoded}'")
    else:
        print("\nAll items detected successfully!")


if __name__ == "__main__":
    main()