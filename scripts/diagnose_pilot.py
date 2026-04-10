"""
diagnose_pilot.py — Analyze behavioral pilot failures
======================================================
Checks for prompt artifacts that might be depressing accuracy:
  1. Position bias: does the model over-select A, B, or C?
  2. Unknown-position effect: is accuracy worse when 'unknown' is in certain slots?
  3. Correct-answer position: is accuracy correlated with where the correct answer sits?
  4. Failure mode: on errors, is the model selecting 'unknown' when it shouldn't,
     or selecting the wrong person?
  5. Sample wrong answers: show concrete examples of failures

Usage:
  python scripts/diagnose_pilot.py
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def newest_file(dir_path: Path, pattern: str) -> Path | None:
    files = sorted(dir_path.glob(pattern))
    return files[-1] if files else None


def main():
    parser = argparse.ArgumentParser(description="Diagnose behavioral pilot results.")
    parser.add_argument("--model_path", type=Path, default=None, help="Optional model path to resolve model_id/run_dir.")
    parser.add_argument("--model_id", type=str, default=None, help="Override model id used for results/runs/<model_id>/")
    parser.add_argument("--run_date", type=str, default=None, help="Run date (YYYY-MM-DD). Defaults to newest for model_id.")
    parser.add_argument("--run_dir", type=Path, default=None, help="Explicit run directory override.")
    parser.add_argument(
        "--results_json",
        type=Path,
        default=None,
        help="Path to behavioral_results.json; defaults to newest in run_dir/behavioral_pilot/.",
    )
    parser.add_argument(
        "--stimuli_json",
        type=Path,
        default=None,
        help="Path to stimuli_so_YYYY-MM-DD.json; defaults to run metadata or newest in data/processed/.",
    )
    args = parser.parse_args()

    from bbqmi.run_paths import resolve_run_dir

    run_dir, model_id, run_date = resolve_run_dir(
        project_root=PROJECT_ROOT,
        run_dir_arg=args.run_dir,
        model_path=args.model_path,
        model_id_arg=args.model_id,
        run_date_arg=args.run_date,
        must_exist=False,
    )
    behavioral_dir = run_dir / "behavioral_pilot"

    # Load results
    results_path = args.results_json or (behavioral_dir / "behavioral_results.json")
    if not results_path.exists():
        raise FileNotFoundError(
            f"Could not find results JSON at {results_path}. "
            f"Pass --results_json or set --run_dir/--model_id to a completed run."
        )
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    # Load stimuli for additional context
    stimuli_path = args.stimuli_json
    if stimuli_path is None:
        # Try to read from run metadata first
        meta_path = run_dir / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                stim_name = (
                    meta.get("steps", {})
                    .get("behavioral_pilot", {})
                    .get("stimuli_file")
                )
                if stim_name:
                    stimuli_path = DATA_DIR / stim_name
            except Exception:
                stimuli_path = None
    if stimuli_path is None:
        stimuli_path = newest_file(DATA_DIR, "stimuli_so_*.json") or (DATA_DIR / "stimuli_so.json")
    if not stimuli_path.exists():
        raise FileNotFoundError(
            f"Could not find stimuli JSON. Looked for newest stimuli_so_*.json in {DATA_DIR} "
            f"and also {DATA_DIR/'stimuli_so.json'}."
        )
    with open(stimuli_path, encoding="utf-8") as f:
        stimuli = json.load(f)

    # Index stimuli by item_idx
    stim_by_idx = {s["item_idx"]: s for s in stimuli}

    print("=" * 70)
    print("  Behavioral pilot diagnostics")
    print(f"  model_id: {model_id}  run_date: {run_date}")
    print(f"  run_dir: {run_dir}")
    print(f"  Results: {results_path.name}")
    print(f"  Stimuli: {stimuli_path.name}")
    print("=" * 70)

    # --- 1. Overall prediction distribution ---
    pred_dist = Counter(r["predicted_letter"] for r in results)
    print("\n1. PREDICTION DISTRIBUTION (all items):")
    for letter in ["A", "B", "C"]:
        print(f"   {letter}: {pred_dist.get(letter, 0):4d} ({pred_dist.get(letter, 0)/len(results):.1%})")

    # By condition
    for cond in ["ambig", "disambig"]:
        subset = [r for r in results if r["context_condition"] == cond]
        dist = Counter(r["predicted_letter"] for r in subset)
        print(f"   {cond:8s}: A={dist.get('A',0):3d} B={dist.get('B',0):3d} C={dist.get('C',0):3d}")

    # --- 2. Where is the 'unknown' answer? ---
    print("\n2. UNKNOWN ANSWER POSITION:")
    unknown_pos = Counter()
    for r in results:
        for letter, role in r["answer_roles"].items():
            if role == "unknown":
                unknown_pos[letter] += 1
    for letter in ["A", "B", "C"]:
        print(f"   'Unknown' is option {letter}: {unknown_pos.get(letter, 0)} items")

    # --- 3. Correct answer position ---
    print("\n3. CORRECT ANSWER POSITION (disambig only):")
    disambig = [r for r in results if r["context_condition"] == "disambig"]
    correct_pos = Counter(r["correct_letter"] for r in disambig)
    for letter in ["A", "B", "C"]:
        n = correct_pos.get(letter, 0)
        subset = [r for r in disambig if r["correct_letter"] == letter]
        acc = sum(r["correct"] for r in subset) / len(subset) if subset else 0
        print(f"   Correct={letter}: n={n:3d}, accuracy={acc:.3f}")

    # --- 4. Failure mode analysis (disambig) ---
    print("\n4. FAILURE MODE (disambig errors):")
    errors = [r for r in disambig if not r["correct"]]
    print(f"   Total disambig errors: {len(errors)} / {len(disambig)}")

    error_modes = Counter()
    for r in errors:
        pred_role = r["answer_roles"].get(r["predicted_letter"], "???")
        error_modes[pred_role] += 1
    for mode, n in error_modes.most_common():
        print(f"   Selected {mode:25s}: {n:3d} ({n/len(errors):.1%})")

    # --- 5. Ambig failure mode ---
    print("\n5. AMBIGUOUS FAILURE MODE:")
    ambig = [r for r in results if r["context_condition"] == "ambig"]
    ambig_errors = [r for r in ambig if not r["correct"]]
    print(f"   Total ambig errors: {len(ambig_errors)} / {len(ambig)} "
          f"(correct = selecting 'unknown')")

    ambig_error_roles = Counter()
    for r in ambig_errors:
        pred_role = r["answer_roles"].get(r["predicted_letter"], "???")
        ambig_error_roles[pred_role] += 1
    for mode, n in ambig_error_roles.most_common():
        print(f"   Selected {mode:25s}: {n:3d} ({n/len(ambig_errors):.1%})")

    # --- 6. Confidence analysis (disambig) ---
    print("\n6. CONFIDENCE ANALYSIS (disambig):")
    correct_items = [r for r in disambig if r["correct"]]
    error_items = [r for r in disambig if not r["correct"]]

    # Newer behavioral_pilot stores per-letter mean logprob scores under answer_scores.
    has_answer_scores = bool(disambig and isinstance(disambig[0].get("answer_scores"), dict))
    if has_answer_scores:
        def mean_max_score(items):
            vals = []
            for r in items:
                scores = [v.get("score", float("-inf")) for v in r["answer_scores"].values()]
                vals.append(max(scores))
            return sum(vals) / len(vals) if vals else float("nan")

        def mean_margin_score(items):
            vals = []
            for r in items:
                scores = sorted([v.get("score", float("-inf")) for v in r["answer_scores"].values()], reverse=True)
                if len(scores) >= 2:
                    vals.append(scores[0] - scores[1])
            return sum(vals) / len(vals) if vals else float("nan")

        print(f"   Correct items: mean max score = {mean_max_score(correct_items):.3f}, "
              f"mean margin = {mean_margin_score(correct_items):.3f}")
        print(f"   Error items:   mean max score = {mean_max_score(error_items):.3f}, "
              f"mean margin = {mean_margin_score(error_items):.3f}")
    else:
        print("   (No per-option scores found in results; skipping confidence analysis.)")

    # --- 7. Sample errors ---
    print("\n7. SAMPLE DISAMBIG ERRORS (first 10):")
    for r in errors[:10]:
        stim = stim_by_idx.get(r["item_idx"], {})
        ctx = stim.get("context", "")
        print(f"\n   Item {r['item_idx']}:")
        print(f"   Context: {ctx[:200]}...")
        print(f"   Question: {r['question']}")
        print(f"   Correct: {r['correct_letter']} | Predicted: {r['predicted_letter']}")
        print(f"   Probs: A={r['probs']['A']:.3f} B={r['probs']['B']:.3f} C={r['probs']['C']:.3f}")
        print(f"   Top5: {r['top5_tokens']}")
        pred_role = r["answer_roles"].get(r["predicted_letter"], "???")
        print(f"   Error type: selected {pred_role}")

    # --- 8. Check if "Can't be determined" wording variants cause issues ---
    print("\n8. UNKNOWN ANSWER WORDING (sample):")
    unknown_wordings = Counter()
    for s in stimuli:
        for letter in ["A", "B", "C"]:
            ans_text = s["answers"][letter]
            info = s.get("answer_roles", {}).get(letter)
            # Check raw answer_info
            if "unknown" in str(s.get("identity_spans_in_context", "")).lower():
                continue
            # Heuristic: if it contains "can't", "cannot", "not enough", "unknown", "undetermined"
            lower = ans_text.lower()
            if any(phrase in lower for phrase in ["can't", "cannot", "not enough",
                                                   "unknown", "undetermined",
                                                   "not answerable", "no answer"]):
                unknown_wordings[ans_text] += 1

    print(f"   Unique 'unknown' wordings: {len(unknown_wordings)}")
    for wording, n in unknown_wordings.most_common(15):
        print(f"   [{n:3d}] '{wording}'")


if __name__ == "__main__":
    main()