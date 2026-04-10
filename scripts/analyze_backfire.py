"""
analyze_backfire.py — Investigate why ablating the bisexual direction amplifies bias
=====================================================================================

Experiment 4: Project final-token hidden states onto each group's identity
  direction. Split by whether the model selected the stereotype answer.
  If stereotyped items have SMALLER projections, the direction is protective.

Experiment 5: Per-item flip analysis between baseline and bisexual ablation.
  Which items flipped from correct/neutral → stereotyped? Are they clustered
  by stereotype template?

No model needed — uses saved activations + behavioral/intervention results.

Usage:
  python scripts/analyze_backfire.py
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTIVATION_DIR = PROJECT_ROOT / "results" / "activations" / "so"
BEHAVIORAL_DIR = PROJECT_ROOT / "results" / "behavioral_pilot"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

SO_TERMS = {"gay", "lesbian", "bisexual", "pansexual", "straight"}

COLORS = {
    "gay": "#0072B2",
    "lesbian": "#CC79A7",
    "bisexual": "#E69F00",
    "pansexual": "#009E73",
    "baseline": "#999999",
    "stereotyped": "#D55E00",
    "non_stereotyped": "#56B4E9",
}


def log(msg):
    print(msg, flush=True)


def cosine_sim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 1e-10 else 0.0


def project_out(vector, direction):
    direction = direction.astype(np.float64)
    d_norm = np.linalg.norm(direction)
    if d_norm < 1e-10:
        return vector.astype(np.float64)
    d_unit = direction / d_norm
    return float(np.dot(vector.astype(np.float64), d_unit))


# ===========================================================================
# Data loading
# ===========================================================================
def load_all_data():
    """Load activations, behavioral results, and intervention results."""

    # Behavioral results
    behav_candidates = sorted(BEHAVIORAL_DIR.glob("behavioral_results*.json"))
    if not behav_candidates:
        raise FileNotFoundError(f"No behavioral_results*.json found in {BEHAVIORAL_DIR}. Run behavioral_pilot.py first.")
    behav_path = behav_candidates[-1]
    with open(behav_path) as f:
        behavioral = json.load(f)
    behav_by_idx = {r["item_idx"]: r for r in behavioral}
    log(f"Loaded behavioral: {behav_path.name} ({len(behavioral)} items)")

    # Intervention results (if available)
    intervention_path = RESULTS_DIR / "pergroup_ablation_results.json"
    intervention = None
    if intervention_path.exists():
        with open(intervention_path) as f:
            intervention = json.load(f)
        log(f"Loaded intervention results: {len(intervention.get('scores', {}))} conditions")

    # Load activations
    items = []
    for npz_path in sorted(ACTIVATION_DIR.glob("item_*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        meta = json.loads(str(data["metadata"]))
        idx = meta["item_idx"]
        behav = behav_by_idx.get(idx, {})

        # Per-term hidden states
        term_to_tokens = defaultdict(list)
        for entry in meta["identity_terms_found"]:
            t = entry["term"].lower()
            if t in SO_TERMS:
                term_to_tokens[t].append(entry["token_indices"])

        all_identity_indices = meta["identity_token_indices"]
        hidden_identity_raw = data["hidden_identity"]
        idx_to_pos = {ti: p for p, ti in enumerate(all_identity_indices)}

        term_hidden = {}
        for term, token_lists in term_to_tokens.items():
            tok_indices = sorted(set(ti for tl in token_lists for ti in tl))
            positions = [idx_to_pos[ti] for ti in tok_indices if ti in idx_to_pos]
            if positions and hidden_identity_raw.shape[1] > 0:
                term_hidden[term] = (
                    hidden_identity_raw[:, positions, :]
                    .mean(axis=1)
                    .astype(np.float32)
                )

        items.append({
            "idx": idx,
            "hidden_final": data["hidden_final"].astype(np.float32),
            "term_hidden": term_hidden,
            "stereotyped_groups": [g.lower() for g in meta["stereotyped_groups"]],
            "context_condition": meta["context_condition"],
            "question_polarity": meta["question_polarity"],
            "alignment": meta["alignment"],
            "question": meta["question"],
            "correct_letter": meta["correct_letter"],
            "predicted_letter": behav.get("predicted_letter", ""),
            "correct": behav.get("correct", False),
            "answer_roles": meta["answer_roles"],
        })

    log(f"Loaded {len(items)} activation files")
    return items, intervention


def compute_directions(items, n_layers):
    """Compute per-group DIM directions from within-item deltas."""
    groups = ["gay", "lesbian", "bisexual", "pansexual"]

    # Compute deltas
    deltas_by_group = defaultdict(list)
    for item in items:
        if len(item["term_hidden"]) < 2:
            continue
        stereo_groups = item["stereotyped_groups"]
        terms = list(item["term_hidden"].keys())
        stereo_term = next((t for t in terms if t in stereo_groups), None)
        non_stereo_term = next((t for t in terms if t != stereo_term), None)
        if not stereo_term or not non_stereo_term:
            continue

        h_s = item["term_hidden"][stereo_term]
        h_ns = item["term_hidden"][non_stereo_term]
        norms_mean = np.maximum(
            (np.linalg.norm(h_s, axis=1, keepdims=True)
             + np.linalg.norm(h_ns, axis=1, keepdims=True)) / 2,
            1e-10,
        )
        delta_normed = (h_s - h_ns) / norms_mean
        group = stereo_groups[0]
        deltas_by_group[group].append(delta_normed)

    # Mean directions, unit normalized
    directions = {}
    for g in groups:
        if deltas_by_group[g]:
            stacked = np.stack(deltas_by_group[g])
            mean_dir = stacked.mean(axis=0).astype(np.float64)
            for layer in range(n_layers):
                norm = np.linalg.norm(mean_dir[layer])
                if norm > 1e-10:
                    mean_dir[layer] /= norm
            directions[g] = mean_dir

    # Gender-projected directions
    gender_dir = (directions["gay"] - directions["lesbian"]) / 2.0
    proj_directions = {}
    for g in groups:
        proj = np.zeros_like(directions[g])
        for layer in range(n_layers):
            residual = directions[g][layer] - (
                np.dot(directions[g][layer], gender_dir[layer])
                / max(np.dot(gender_dir[layer], gender_dir[layer]), 1e-10)
                * gender_dir[layer]
            )
            norm = np.linalg.norm(residual)
            if norm > 1e-10:
                residual /= norm
            proj[layer] = residual
        proj_directions[g] = proj.astype(np.float64)

    return directions, proj_directions, gender_dir


# ===========================================================================
# Experiment 4: Final-token projection analysis
# ===========================================================================
def experiment_4_projection(items, directions, proj_directions, n_layers):
    log("\n" + "=" * 70)
    log("  EXPERIMENT 4: Final-token projection onto identity directions")
    log("=" * 70)

    groups = ["gay", "lesbian", "bisexual", "pansexual"]
    target_layers = [l for l in [10, 15, 20, 25, 30] if l < n_layers]

    results = {}

    for target_group in groups:
        group_items = [
            it for it in items
            if target_group in it["stereotyped_groups"]
            and it["context_condition"] == "ambig"
        ]

        if len(group_items) < 10:
            log(f"  {target_group}: too few ambig items ({len(group_items)}), skipping")
            continue

        # Split by whether model selected stereotype
        stereo_items = [
            it for it in group_items
            if it["answer_roles"].get(it["predicted_letter"], "") == "stereotyped_target"
        ]
        non_stereo_items = [
            it for it in group_items
            if it["answer_roles"].get(it["predicted_letter"], "") != "stereotyped_target"
            and it["answer_roles"].get(it["predicted_letter"], "") != "unknown"
        ]
        unknown_items = [
            it for it in group_items
            if it["answer_roles"].get(it["predicted_letter"], "") == "unknown"
        ]

        log(f"\n  {target_group.upper()}: {len(group_items)} ambig items "
            f"(stereo={len(stereo_items)}, anti-stereo={len(non_stereo_items)}, "
            f"unknown={len(unknown_items)})")

        group_results = {}

        for layer in target_layers:
            # Project final-token hidden state onto own-group direction
            # and onto each other group's direction
            layer_results = {}

            for dir_name, dir_set in [("raw", directions), ("proj", proj_directions)]:
                direction = dir_set[target_group][layer]

                # Projections for stereotype-selecting items
                stereo_projs = [
                    float(np.dot(it["hidden_final"][layer].astype(np.float64), direction))
                    for it in stereo_items
                ]
                # Projections for non-stereotype items
                non_stereo_projs = [
                    float(np.dot(it["hidden_final"][layer].astype(np.float64), direction))
                    for it in non_stereo_items
                ]
                # Unknown items
                unknown_projs = [
                    float(np.dot(it["hidden_final"][layer].astype(np.float64), direction))
                    for it in unknown_items
                ]

                stereo_mean = float(np.mean(stereo_projs)) if stereo_projs else 0
                non_stereo_mean = float(np.mean(non_stereo_projs)) if non_stereo_projs else 0
                unknown_mean = float(np.mean(unknown_projs)) if unknown_projs else 0

                # Cohen's d
                if stereo_projs and non_stereo_projs:
                    pooled_std = np.sqrt(
                        (np.var(stereo_projs) * (len(stereo_projs) - 1)
                         + np.var(non_stereo_projs) * (len(non_stereo_projs) - 1))
                        / (len(stereo_projs) + len(non_stereo_projs) - 2)
                    )
                    cohens_d = (stereo_mean - non_stereo_mean) / pooled_std if pooled_std > 1e-10 else 0
                else:
                    cohens_d = 0

                layer_results[dir_name] = {
                    "stereo_mean": stereo_mean,
                    "non_stereo_mean": non_stereo_mean,
                    "unknown_mean": unknown_mean,
                    "stereo_std": float(np.std(stereo_projs)) if stereo_projs else 0,
                    "non_stereo_std": float(np.std(non_stereo_projs)) if non_stereo_projs else 0,
                    "cohens_d": cohens_d,
                    "n_stereo": len(stereo_projs),
                    "n_non_stereo": len(non_stereo_projs),
                }

            group_results[layer] = layer_results

            # Print
            raw = layer_results["raw"]
            proj = layer_results["proj"]
            log(f"    Layer {layer}:")
            log(f"      Raw dir:  stereo={raw['stereo_mean']:+.2f} "
                f"non-stereo={raw['non_stereo_mean']:+.2f} "
                f"unknown={raw['unknown_mean']:+.2f} "
                f"d={raw['cohens_d']:+.3f}")
            log(f"      Proj dir: stereo={proj['stereo_mean']:+.2f} "
                f"non-stereo={proj['non_stereo_mean']:+.2f} "
                f"unknown={proj['unknown_mean']:+.2f} "
                f"d={proj['cohens_d']:+.3f}")

        results[target_group] = group_results

    # Cross-direction analysis: project bisexual items onto gay direction and vice versa
    log("\n  --- Cross-direction projections (layer 20) ---")
    layer = 20
    for target_group in groups:
        group_ambig = [
            it for it in items
            if target_group in it["stereotyped_groups"]
            and it["context_condition"] == "ambig"
        ]
        if not group_ambig:
            continue

        stereo_items = [
            it for it in group_ambig
            if it["answer_roles"].get(it["predicted_letter"], "") == "stereotyped_target"
        ]
        non_stereo_items = [
            it for it in group_ambig
            if it["answer_roles"].get(it["predicted_letter"], "") not in ["stereotyped_target", "unknown"]
        ]

        log(f"\n  {target_group.upper()} items projected onto each group's direction:")
        for dir_group in groups:
            direction = proj_directions[dir_group][layer]

            s_projs = [float(np.dot(it["hidden_final"][layer].astype(np.float64), direction))
                       for it in stereo_items] if stereo_items else [0]
            ns_projs = [float(np.dot(it["hidden_final"][layer].astype(np.float64), direction))
                        for it in non_stereo_items] if non_stereo_items else [0]

            log(f"    → {dir_group:12s} dir: stereo={np.mean(s_projs):+.2f}  "
                f"non-stereo={np.mean(ns_projs):+.2f}  "
                f"Δ={np.mean(s_projs) - np.mean(ns_projs):+.2f}")

    return results


# ===========================================================================
# Experiment 5: Per-item flip analysis
# ===========================================================================
def experiment_5_flips(items, intervention):
    log("\n" + "=" * 70)
    log("  EXPERIMENT 5: Per-item flip analysis (baseline vs bisexual ablation)")
    log("=" * 70)

    if intervention is None:
        log("  No intervention results found, skipping.")
        return {}

    scores = intervention.get("scores", {})

    # We need per-item results, but we only have aggregate scores
    # Check if we have the raw per-item intervention results saved
    # If not, we can still analyze which TYPES of items show the biggest
    # aggregate bias shift

    # Analyze by question template
    log("\n  Analyzing baseline behavioral patterns by question template...")

    # Group items by stereotyped group and question
    bisexual_items = [
        it for it in items
        if "bisexual" in it["stereotyped_groups"]
        and it["context_condition"] == "ambig"
    ]

    log(f"  Bisexual ambig items: {len(bisexual_items)}")

    # Analyze by question template
    by_question = defaultdict(list)
    for it in bisexual_items:
        q = it["question"][:60]  # truncate for grouping
        by_question[q].append(it)

    log(f"\n  Bisexual stereotype selection rate by question template:")
    question_stats = []
    for q, q_items in sorted(by_question.items(), key=lambda x: -len(x[1])):
        n = len(q_items)
        n_stereo = sum(
            1 for it in q_items
            if it["answer_roles"].get(it["predicted_letter"], "") == "stereotyped_target"
        )
        n_non_stereo = sum(
            1 for it in q_items
            if it["answer_roles"].get(it["predicted_letter"], "") not in ["stereotyped_target", "unknown"]
        )
        n_unknown = sum(
            1 for it in q_items
            if it["answer_roles"].get(it["predicted_letter"], "") == "unknown"
        )
        rate = n_stereo / max(n_stereo + n_non_stereo, 1)

        question_stats.append({
            "question": q,
            "n": n,
            "n_stereo": n_stereo,
            "n_non_stereo": n_non_stereo,
            "n_unknown": n_unknown,
            "stereo_rate": rate,
        })
        log(f"    [{n:3d} items] stereo={n_stereo:2d} anti={n_non_stereo:2d} "
            f"unk={n_unknown:2d} rate={rate:.2f} | {q}")

    # Same analysis for ALL groups
    log("\n\n  Stereotype selection rate by group and question:")
    for group in ["gay", "lesbian", "bisexual", "pansexual"]:
        group_ambig = [
            it for it in items
            if group in it["stereotyped_groups"]
            and it["context_condition"] == "ambig"
        ]
        n_stereo = sum(
            1 for it in group_ambig
            if it["answer_roles"].get(it["predicted_letter"], "") == "stereotyped_target"
        )
        n_non = sum(
            1 for it in group_ambig
            if it["answer_roles"].get(it["predicted_letter"], "") not in ["stereotyped_target", "unknown"]
        )
        n_unk = sum(
            1 for it in group_ambig
            if it["answer_roles"].get(it["predicted_letter"], "") == "unknown"
        )
        rate = n_stereo / max(n_stereo + n_non, 1)
        log(f"  {group:12s}: n={len(group_ambig):3d}  stereo={n_stereo:3d}  "
            f"anti={n_non:3d}  unk={n_unk:3d}  rate={rate:.3f}")

    # Analyze: which questions have the HIGHEST stereotype rate for bisexual
    # vs the LOWEST for gay?
    log("\n\n  Questions where bisexual is most stereotyped:")
    high_stereo = [qs for qs in question_stats if qs["stereo_rate"] > 0.5 and qs["n"] >= 4]
    for qs in sorted(high_stereo, key=lambda x: -x["stereo_rate"]):
        log(f"    rate={qs['stereo_rate']:.2f} (n={qs['n']}) | {qs['question']}")

    return {"question_stats": question_stats}


# ===========================================================================
# Plotting
# ===========================================================================
def plot_all(exp4_results, exp5_results, items, directions, proj_directions,
             n_layers, figures_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    groups = ["gay", "lesbian", "bisexual", "pansexual"]

    # --- Figure 1: Projection distributions for each group ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    layer = 20

    for ax, group in zip(axes.flat, groups):
        group_ambig = [
            it for it in items
            if group in it["stereotyped_groups"]
            and it["context_condition"] == "ambig"
        ]

        direction = proj_directions[group][layer]

        stereo_projs = []
        non_stereo_projs = []
        unknown_projs = []

        for it in group_ambig:
            proj = float(np.dot(it["hidden_final"][layer].astype(np.float64), direction))
            role = it["answer_roles"].get(it["predicted_letter"], "")
            if role == "stereotyped_target":
                stereo_projs.append(proj)
            elif role == "unknown":
                unknown_projs.append(proj)
            else:
                non_stereo_projs.append(proj)

        bins = np.linspace(
            min(stereo_projs + non_stereo_projs + unknown_projs) - 1,
            max(stereo_projs + non_stereo_projs + unknown_projs) + 1,
            30,
        )

        if stereo_projs:
            ax.hist(stereo_projs, bins=bins, alpha=0.6, color=COLORS["stereotyped"],
                    label=f"Stereotype (n={len(stereo_projs)})", edgecolor="white")
        if non_stereo_projs:
            ax.hist(non_stereo_projs, bins=bins, alpha=0.6, color=COLORS["non_stereotyped"],
                    label=f"Anti-stereo (n={len(non_stereo_projs)})", edgecolor="white")
        if unknown_projs:
            ax.hist(unknown_projs, bins=bins, alpha=0.4, color=COLORS["baseline"],
                    label=f"Unknown (n={len(unknown_projs)})", edgecolor="white")

        # Means
        if stereo_projs:
            ax.axvline(np.mean(stereo_projs), color=COLORS["stereotyped"],
                       linewidth=2, linestyle="--")
        if non_stereo_projs:
            ax.axvline(np.mean(non_stereo_projs), color=COLORS["non_stereotyped"],
                       linewidth=2, linestyle="--")

        ax.set_xlabel("Projection onto group direction", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"{group.capitalize()} (ambig items, layer {layer})", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Final-token projection onto gender-projected identity direction\n"
                 "Split by model's answer choice", fontsize=13)
    fig.tight_layout()
    fig.savefig(figures_dir / "backfire_1_projection_distributions.png", dpi=150)
    plt.close(fig)
    log("  Saved backfire_1_projection_distributions.png")

    # --- Figure 2: Cohen's d across layers for each group ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    target_layers = sorted(exp4_results.get("bisexual", {}).keys())

    for group in groups:
        if group not in exp4_results:
            continue
        raw_ds = [exp4_results[group][l]["raw"]["cohens_d"] for l in target_layers]
        proj_ds = [exp4_results[group][l]["proj"]["cohens_d"] for l in target_layers]
        ax1.plot(target_layers, raw_ds, color=COLORS[group], linewidth=2,
                 label=group, marker="o", markersize=5)
        ax2.plot(target_layers, proj_ds, color=COLORS[group], linewidth=2,
                 label=group, marker="o", markersize=5)

    for ax, title in [(ax1, "Raw direction"), (ax2, "Gender-projected direction")]:
        ax.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Cohen's d (stereo − non-stereo)", fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Effect size: projection difference between stereotype-selecting\n"
                 "and non-stereotype-selecting items (+ = stereo projects MORE)", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures_dir / "backfire_2_cohens_d_by_layer.png", dpi=150)
    plt.close(fig)
    log("  Saved backfire_2_cohens_d_by_layer.png")

    # --- Figure 3: Cross-direction projection heatmap ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    layer = 20

    # For each target group's items, project onto each group's direction
    # Compute: mean(stereo) - mean(non-stereo) for each combination
    matrix = np.zeros((4, 4))

    for i, target_group in enumerate(groups):
        group_ambig = [
            it for it in items
            if target_group in it["stereotyped_groups"]
            and it["context_condition"] == "ambig"
        ]
        stereo = [it for it in group_ambig
                  if it["answer_roles"].get(it["predicted_letter"], "") == "stereotyped_target"]
        non_stereo = [it for it in group_ambig
                      if it["answer_roles"].get(it["predicted_letter"], "")
                      not in ["stereotyped_target", "unknown"]]

        for j, dir_group in enumerate(groups):
            direction = proj_directions[dir_group][layer]
            s_mean = np.mean([float(np.dot(it["hidden_final"][layer].astype(np.float64), direction))
                              for it in stereo]) if stereo else 0
            ns_mean = np.mean([float(np.dot(it["hidden_final"][layer].astype(np.float64), direction))
                               for it in non_stereo]) if non_stereo else 0
            matrix[i, j] = s_mean - ns_mean

    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-5, vmax=5)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([f"{g}\ndirection" for g in groups], fontsize=10)
    ax.set_yticklabels([f"{g}\nitems" for g in groups], fontsize=10)
    ax.set_xlabel("Projection direction", fontsize=11)
    ax.set_ylabel("Item group", fontsize=11)

    for i in range(4):
        for j in range(4):
            val = matrix[i, j]
            color = "white" if abs(val) > 3 else "black"
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Mean projection: stereo − non-stereo")
    ax.set_title("Cross-direction projection difference (layer 20)\n"
                 "(+ = stereotyped items project MORE onto this direction)", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures_dir / "backfire_3_cross_projection.png", dpi=150)
    plt.close(fig)
    log("  Saved backfire_3_cross_projection.png")

    # --- Figure 4: Stereotype rate by group and question template ---
    if exp5_results and "question_stats" in exp5_results:
        stats = exp5_results["question_stats"]
        stats_sorted = sorted(stats, key=lambda x: x["stereo_rate"])

        fig, ax = plt.subplots(1, 1, figsize=(10, max(6, len(stats) * 0.3)))
        y_pos = range(len(stats_sorted))
        rates = [s["stereo_rate"] for s in stats_sorted]
        labels = [f"{s['question'][:50]}... (n={s['n']})" for s in stats_sorted]

        colors_bar = [COLORS["stereotyped"] if r > 0.5 else COLORS["non_stereotyped"]
                      for r in rates]
        ax.barh(y_pos, rates, color=colors_bar, edgecolor="white", height=0.7)
        ax.axvline(0.5, color="#999999", linewidth=1, linestyle="--",
                   label="Chance (no bias)")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Stereotype selection rate (among non-unknown)", fontsize=11)
        ax.set_title("Bisexual items: stereotype rate by question template", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis="x")

        fig.tight_layout()
        fig.savefig(figures_dir / "backfire_4_bisexual_by_question.png", dpi=150)
        plt.close(fig)
        log("  Saved backfire_4_bisexual_by_question.png")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, default=None, help="Optional model path to resolve model_id/run_dir.")
    parser.add_argument("--model_id", type=str, default=None, help="Override model id used for results/runs/<model_id>/")
    parser.add_argument("--run_date", type=str, default=None, help="Run date (YYYY-MM-DD). Defaults to newest for model_id.")
    parser.add_argument("--run_dir", type=Path, default=None, help="Explicit run directory override.")
    args = parser.parse_args()

    from bbqmi.run_paths import ensure_run_subdirs, resolve_run_dir

    run_dir, model_id, run_date = resolve_run_dir(
        project_root=PROJECT_ROOT,
        run_dir_arg=args.run_dir,
        model_path=args.model_path,
        model_id_arg=args.model_id,
        run_date_arg=args.run_date,
        must_exist=False,
    )
    subdirs = ensure_run_subdirs(run_dir)

    global ACTIVATION_DIR, BEHAVIORAL_DIR, FIGURES_DIR, RESULTS_DIR
    ACTIVATION_DIR = subdirs.activations_so_dir
    BEHAVIORAL_DIR = subdirs.behavioral_dir
    FIGURES_DIR = subdirs.figures_dir
    RESULTS_DIR = subdirs.analysis_dir

    log(f"Run: model_id={model_id}  run_date={run_date}")
    log(f"Run dir: {run_dir}")
    log(f"Activations (SO): {ACTIVATION_DIR}")
    log(f"Behavioral dir: {BEHAVIORAL_DIR}")
    log(f"Analysis outputs: {RESULTS_DIR}")
    log(f"Figures: {FIGURES_DIR}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log("Loading data...")
    items, intervention = load_all_data()
    if not items:
        log(f"ERROR: No activation files found in {ACTIVATION_DIR} (expected item_*.npz).")
        return
    n_layers = items[0]["hidden_final"].shape[0]

    log("Computing directions...")
    directions, proj_directions, gender_dir = compute_directions(items, n_layers)

    # Experiment 4
    exp4 = experiment_4_projection(items, directions, proj_directions, n_layers)

    # Experiment 5
    exp5 = experiment_5_flips(items, intervention)

    # Save
    save_data = {
        "experiment_4": {
            group: {
                str(layer): {
                    dir_type: {k: v for k, v in vals.items()}
                    for dir_type, vals in layer_data.items()
                }
                for layer, layer_data in group_data.items()
            }
            for group, group_data in exp4.items()
        },
    }
    with open(RESULTS_DIR / "backfire_analysis.json", "w") as f:
        json.dump(save_data, f, indent=2)
    log("\nSaved backfire_analysis.json")

    # Plot
    log("\nGenerating figures...")
    plot_all(exp4, exp5, items, directions, proj_directions, n_layers, FIGURES_DIR)

    log("\nDone.")


if __name__ == "__main__":
    main()