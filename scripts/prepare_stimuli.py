"""
prepare_stimuli.py — Prepare BBQ stimuli for mechanistic interpretability study
================================================================================
Loads BBQ Sexual_orientation and Gender_identity (trans-targeted) subsets,
formats them into standardized prompts for Llama-2-13B base, and identifies
identity token positions for localized activation extraction.

Outputs (written to `data/processed/` with a date suffix):
  - stimuli_so_YYYY-MM-DD.json: 864 sexual orientation items with metadata + prompts
  - stimuli_gi_YYYY-MM-DD.json: 864 gender identity (trans) items with metadata + prompts
  - stimuli_summary_YYYY-MM-DD.txt: counts and sanity checks

Usage:
  python scripts/prepare_stimuli.py

Expects BBQ repo at data/BBQ/ (clone from https://github.com/nyu-mll/BBQ.git)
"""

from __future__ import annotations

import argparse
import json
import re
import hashlib
from collections import Counter
from pathlib import Path

from bbqmi.dates import today_yyyy_mm_dd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BBQ_DIR = DATA_DIR / "BBQ" / "data"
OUTPUT_DIR = DATA_DIR / "processed"

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
# Llama-2-13B base is a completion model. We use a minimal few-shot template
# that establishes the 3-choice QA pattern without biasing toward any answer.

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


def format_prompt(context: str, question: str,
                  ans_a: str, ans_b: str, ans_c: str) -> str:
    """Format a BBQ item as a few-shot 3-choice QA prompt."""
    return (
        f"{FEWSHOT_PREFIX}"
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"A) {ans_a}\n"
        f"B) {ans_b}\n"
        f"C) {ans_c}\n"
        f"Answer:"
    )
 
 
# --- Add this new function ---
def balanced_answer_order(item: dict, item_idx: int) -> tuple[list[int], dict]:
    """Produce a deterministic but balanced permutation of answer positions.
 
    Strategy: hash the example_id to get a deterministic rotation, then
    apply one of 6 permutations (all orderings of 3 elements). The hash
    ensures each item always gets the same permutation across runs, and
    the distribution across items is roughly uniform.
 
    Returns:
        perm: list of 3 ints — new positions for original ans0, ans1, ans2.
              e.g. [2, 0, 1] means ans0 -> C, ans1 -> A, ans2 -> B
        new_answer_map: dict mapping new letters to original answer keys.
              e.g. {"A": "ans1", "B": "ans2", "C": "ans0"}
    """
    PERMUTATIONS = [
        [0, 1, 2],  # A B C  (original order)
        [0, 2, 1],  # A C B
        [1, 0, 2],  # B A C
        [1, 2, 0],  # B C A
        [2, 0, 1],  # C A B
        [2, 1, 0],  # C B A
    ]
 
    # Deterministic hash based on example_id
    h = int(hashlib.md5(str(item["example_id"]).encode()).hexdigest(), 16)
    perm_idx = h % 6
    perm = PERMUTATIONS[perm_idx]
 
    # perm[i] = new position for original ans_i
    # So if perm = [2, 0, 1]: ans0 goes to position 2 (C),
    #   ans1 goes to position 0 (A), ans2 goes to position 1 (B)
 
    # Build reverse map: for each new position, which original answer is there?
    letters = ["A", "B", "C"]
    orig_keys = ["ans0", "ans1", "ans2"]
 
    new_answer_map = {}
    for orig_idx, new_pos in enumerate(perm):
        new_answer_map[letters[new_pos]] = orig_keys[orig_idx]
 
    return perm, new_answer_map


# ---------------------------------------------------------------------------
# Identity term detection
# ---------------------------------------------------------------------------
IDENTITY_TERMS_SO = [
    "gay",
    "lesbian",
    "bisexual",
    "pansexual",
    "straight",
    "heterosexual",
    "queer",
]

IDENTITY_TERMS_GI = [
    "transgender",
    "transgendered",
    "transgender woman",
    "transgender man",
    "trans woman",
    "trans man",
    "trans",
    "cisgender",
    "nonbinary",
    "non-binary",
]


def find_identity_spans(text: str, term_list: list[str]) -> list[dict]:
    """Find character-level spans of identity terms in text.

    Returns list of {term, start, end} dicts, longest-match-first to handle
    'transgender woman' before 'transgender'.
    """
    spans: list[dict] = []
    sorted_terms = sorted(term_list, key=len, reverse=True)
    used_ranges: list[tuple[int, int]] = []

    for term in sorted_terms:
        pattern = r"\b" + re.escape(term) + r"\b"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start, end = match.start(), match.end()
            if any(s <= start < e or s < end <= e for s, e in used_ranges):
                continue
            spans.append(
                {
                    "term": term,
                    "matched_text": match.group(),
                    "char_start": start,
                    "char_end": end,
                }
            )
            used_ranges.append((start, end))

    spans.sort(key=lambda x: x["char_start"])
    return spans


# ---------------------------------------------------------------------------
# Item processing
# ---------------------------------------------------------------------------
def classify_alignment(item: dict) -> str:
    """Classify disambiguated items as stereotype-aligned or stereotype-conflicting.

    For ambiguous items, returns 'ambiguous'.
    """
    if item["context_condition"] == "ambig":
        return "ambiguous"

    label = item["label"]
    correct_identity = item["answer_info"][f"ans{label}"][1]
    stereotyped_groups = item["additional_metadata"]["stereotyped_groups"]
    polarity = item["question_polarity"]

    if correct_identity == "unknown":
        return "unknown_correct"

    if polarity == "neg":
        is_aligned = correct_identity.lower() in [g.lower() for g in stereotyped_groups]
    else:
        is_aligned = correct_identity.lower() not in [g.lower() for g in stereotyped_groups]

    return "aligned" if is_aligned else "conflicting"


def process_item(item: dict, identity_terms: list[str], item_idx: int) -> dict:
    """Process a single BBQ item with balanced answer positions."""
 
    # Determine shuffled answer order
    perm, new_answer_map = balanced_answer_order(item, item_idx)
    letters = ["A", "B", "C"]
    orig_keys = ["ans0", "ans1", "ans2"]
 
    # Build shuffled answers
    shuffled_answers = {}
    for new_letter in letters:
        orig_key = new_answer_map[new_letter]
        shuffled_answers[new_letter] = item[orig_key]
 
    # Build prompt with shuffled answers
    prompt = format_prompt(
        item["context"],
        item["question"],
        shuffled_answers["A"],
        shuffled_answers["B"],
        shuffled_answers["C"],
    )
 
    # Find correct letter after shuffle
    orig_correct_key = f"ans{item['label']}"  # e.g. "ans1"
    correct_letter = None
    for new_letter, orig_key in new_answer_map.items():
        if orig_key == orig_correct_key:
            correct_letter = new_letter
            break
    correct_text = shuffled_answers[correct_letter]
 
    # Build answer roles after shuffle
    answer_roles = {}
    for new_letter in letters:
        orig_key = new_answer_map[new_letter]
        identity_info = item["answer_info"][orig_key][1]
        if identity_info == "unknown":
            answer_roles[new_letter] = "unknown"
        elif identity_info.lower() in [g.lower() for g in item["additional_metadata"]["stereotyped_groups"]]:
            answer_roles[new_letter] = "stereotyped_target"
        else:
            answer_roles[new_letter] = "non_target"
 
    # Classify alignment (same logic, but on shuffled data — alignment is
    # a property of the item content, not answer position)
    alignment = classify_alignment(item)
 
    # Find identity terms
    identity_spans = find_identity_spans(item["context"], identity_terms)
    prompt_identity_spans = find_identity_spans(prompt, identity_terms)
 
    # Extract identity pair
    identities_present = []
    for ans_key in ["ans0", "ans1", "ans2"]:
        identity = item["answer_info"][ans_key][1]
        if identity != "unknown":
            identities_present.append(identity)
 
    return {
        # Identifiers
        "item_idx": item_idx,
        "bbq_example_id": item["example_id"],
        "category": item["category"],
 
        # Experimental conditions
        "context_condition": item["context_condition"],
        "question_polarity": item["question_polarity"],
        "alignment": alignment,
 
        # Stereotype metadata
        "stereotyped_groups": item["additional_metadata"]["stereotyped_groups"],
        "identities_present": identities_present,
        "stereotype_source": item["additional_metadata"].get("source", ""),
 
        # Prompt and answers (SHUFFLED)
        "context": item["context"],
        "question": item["question"],
        "prompt": prompt,
        "answers": shuffled_answers,
        "correct_letter": correct_letter,
        "correct_text": correct_text,
        "answer_roles": answer_roles,
        "answer_permutation": perm,
        "answer_map": new_answer_map,
 
        # Identity token positions
        "identity_spans_in_context": identity_spans,
        "identity_spans_in_prompt": prompt_identity_spans,
    }


def process_subset(
    jsonl_path: Path, identity_terms: list[str], filter_fn=None, label: str = ""
) -> list[dict]:
    """Load and process a BBQ JSONL file."""
    raw_items: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if filter_fn is None or filter_fn(item):
                raw_items.append(item)

    print(f"  Loaded {len(raw_items)} items from {jsonl_path.name} {label}")

    processed: list[dict] = []
    for i, item in enumerate(raw_items):
        processed.append(process_item(item, identity_terms, i))

    return processed


# ---------------------------------------------------------------------------
# Validation and summary
# ---------------------------------------------------------------------------
def validate_and_summarize(items: list[dict], name: str) -> str:
    """Run sanity checks and produce a summary string."""
    lines = [f"\n{'='*60}", f"  {name}: {len(items)} items", f"{'='*60}"]

    cond = Counter((i["context_condition"], i["question_polarity"]) for i in items)
    lines.append("\nCondition breakdown:")
    for (ctx, pol), n in sorted(cond.items()):
        lines.append(f"  {ctx:10s} × {pol:8s}: {n}")

    align = Counter(i["alignment"] for i in items)
    lines.append("\nAlignment breakdown:")
    for a, n in sorted(align.items()):
        lines.append(f"  {a}: {n}")

    terms_found = Counter()
    items_without_spans = 0
    for item in items:
        spans = item["identity_spans_in_context"]
        if not spans:
            items_without_spans += 1
        for s in spans:
            terms_found[s["term"]] += 1
    lines.append("\nIdentity terms found in contexts:")
    for term, n in terms_found.most_common():
        lines.append(f"  {term}: {n}")
    if items_without_spans:
        lines.append(f"  WARNING: {items_without_spans} items with no identity terms found!")

    sg = Counter()
    for item in items:
        for g in item["stereotyped_groups"]:
            sg[g] += 1
    lines.append("\nStereotyped groups targeted:")
    for g, n in sg.most_common():
        lines.append(f"  {g}: {n}")

    questions = Counter(i["question"] for i in items)
    lines.append(f"\nUnique questions: {len(questions)}")

    prompt_lens = [len(i["prompt"].split()) for i in items]
    lines.append(
        f"\nPrompt length (words): min={min(prompt_lens)}, max={max(prompt_lens)}, "
        f"mean={sum(prompt_lens)/len(prompt_lens):.0f}"
    )

    n_missing_stereo = sum(1 for i in items if "stereotyped_target" not in i["answer_roles"].values())
    n_missing_unknown = sum(1 for i in items if "unknown" not in i["answer_roles"].values())
    lines.append("\nAnswer role validation:")
    lines.append(f"  Items missing stereotyped_target answer: {n_missing_stereo}")
    lines.append(f"  Items missing unknown answer: {n_missing_unknown}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BBQ stimuli")
    parser.add_argument("--bbq_dir", type=Path, default=BBQ_DIR, help="Path to BBQ data/ directory")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Processing BBQ stimuli...")

    so_items = process_subset(args.bbq_dir / "Sexual_orientation.jsonl", IDENTITY_TERMS_SO, label="(all)")

    gi_items = process_subset(
        args.bbq_dir / "Gender_identity.jsonl",
        IDENTITY_TERMS_GI,
        filter_fn=lambda i: any("trans" in g.lower() for g in i["additional_metadata"]["stereotyped_groups"]),
        label="(trans-targeted only)",
    )

    summary_so = validate_and_summarize(so_items, "Sexual Orientation")
    summary_gi = validate_and_summarize(gi_items, "Gender Identity (Trans)")

    print(summary_so)
    print(summary_gi)

    date_suffix = today_yyyy_mm_dd()
    so_path = args.output_dir / f"stimuli_so_{date_suffix}.json"
    gi_path = args.output_dir / f"stimuli_gi_{date_suffix}.json"
    summary_path = args.output_dir / f"stimuli_summary_{date_suffix}.txt"

    with open(so_path, "w", encoding="utf-8") as f:
        json.dump(so_items, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(so_items)} SO items to {so_path}")

    with open(gi_path, "w", encoding="utf-8") as f:
        json.dump(gi_items, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(gi_items)} GI items to {gi_path}")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_so + "\n\n" + summary_gi + "\n")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

