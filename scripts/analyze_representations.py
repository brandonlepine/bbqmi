"""
analyze_representations_v2.py — Within-item representation geometry
====================================================================
Uses within-item contrasts at identity token positions rather than
across-item classification (which trivially detects token presence).
 
Core approach:
  Each BBQ item mentions two identity groups (e.g., "a gay man and a
  straight man"). We extract hidden states at both identity token
  positions and compute within-item delta vectors. These deltas isolate
  how the model's representation of the identity differs with all other
  context held constant.
 
Analyses:
  1. Identity direction: DIM across within-item identity deltas per layer.
     Does the model encode a consistent orientation axis beyond token-level?
 
  2. Entanglement: Does the identity delta direction correlate with the
     stereotype direction differently in ambig vs disambig contexts?
 
  3. Error analysis: On disambig-conflicting errors, is the identity delta
     larger/more stereotype-aligned than on correct trials?
 
  4. Group-level: Does the identity delta differ for items targeting
     gay vs bisexual vs lesbian vs pansexual stereotypes?
 
Usage:
  python scripts/analyze_representations_v2.py
"""
 
import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
 
import numpy as np
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTIVATION_DIR = PROJECT_ROOT / "results" / "activations" / "so"
BEHAVIORAL_DIR = PROJECT_ROOT / "results" / "behavioral_pilot"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
 
COLORS = {
    "orange": "#E69F00",
    "blue": "#0072B2",
    "green": "#009E73",
    "purple": "#CC79A7",
    "red": "#D55E00",
    "yellow": "#F0E442",
    "gray": "#999999",
    "black": "#000000",
}
 
IDENTITY_TERMS = {"gay", "lesbian", "bisexual", "pansexual", "straight"}
 
 
# ---------------------------------------------------------------------------
# Data loading — with per-identity-term activations
# ---------------------------------------------------------------------------
def load_data(activation_dir: Path, behavioral_path: Path):
    """Load activations with per-identity-term hidden states."""
    with open(behavioral_path) as f:
        behavioral = json.load(f)
    behav_by_idx = {r["item_idx"]: r for r in behavioral}
 
    items = []
    npz_files = sorted(activation_dir.glob("item_*.npz"))
    print(f"Loading {len(npz_files)} files...")
 
    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        meta = json.loads(str(data["metadata"]))
        idx = meta["item_idx"]
        behav = behav_by_idx.get(idx, {})
 
        # Parse identity term positions
        # Each entry in identity_terms_found has {term, token_indices}
        # Group token indices by term
        term_to_tokens = defaultdict(list)
        for entry in meta["identity_terms_found"]:
            term = entry["term"].lower()
            if term in IDENTITY_TERMS:
                term_to_tokens[term].append(entry["token_indices"])
 
        # hidden_identity has shape (n_layers, n_all_identity_tokens, hidden_dim)
        # We need to figure out which columns correspond to which terms
        all_identity_indices = meta["identity_token_indices"]
 
        # Build per-term hidden states by mapping token indices back
        hidden_identity_raw = data["hidden_identity"]  # (n_layers, N, hidden_dim)
        n_layers = hidden_identity_raw.shape[0]
        hidden_dim = hidden_identity_raw.shape[2] if hidden_identity_raw.shape[1] > 0 else 5120
 
        # Map: for each identity token index, which position is it in the array?
        idx_to_pos = {tok_idx: pos for pos, tok_idx in enumerate(all_identity_indices)}
 
        term_hidden = {}  # term -> (n_layers, hidden_dim) mean over that term's tokens
        for term, token_lists in term_to_tokens.items():
            # Flatten all token indices for this term
            term_tok_indices = []
            for tl in token_lists:
                term_tok_indices.extend(tl)
            # Deduplicate
            term_tok_indices = sorted(set(term_tok_indices))
 
            # Get positions in the hidden_identity array
            positions = [idx_to_pos[ti] for ti in term_tok_indices if ti in idx_to_pos]
 
            if positions and hidden_identity_raw.shape[1] > 0:
                # Mean across all token positions for this term
                h = hidden_identity_raw[:, positions, :].mean(axis=1)  # (n_layers, hidden_dim)
                term_hidden[term] = h.astype(np.float32)
 
        items.append({
            "idx": idx,
            "hidden_final": data["hidden_final"].astype(np.float32),
            "hidden_context_mean": data["hidden_context_mean"].astype(np.float32),
            "term_hidden": term_hidden,
            "identities_present": [t.lower() for t in meta["identities_present"]],
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
 
    print(f"Loaded {len(items)} items")
 
    # Stats on term coverage
    n_with_two_terms = sum(1 for i in items if len(i["term_hidden"]) >= 2)
    print(f"Items with >=2 identity term hidden states: {n_with_two_terms}/{len(items)}")
 
    return items
 
 
def cosine_sim(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 1e-10 else 0.0
 
 
# ---------------------------------------------------------------------------
# Within-item delta computation
# ---------------------------------------------------------------------------
def compute_within_item_deltas(items: list[dict], n_layers: int):
    """For each item with two identity terms, compute h(term1) - h(term2).
 
    Convention: delta = h(stereotyped_term) - h(non_stereotyped_term)
    so positive projection means the stereotyped identity has larger value.
 
    Returns list of dicts with delta vectors and metadata.
    """
    deltas = []
 
    for item in items:
        if len(item["term_hidden"]) < 2:
            continue
 
        stereo_groups = item["stereotyped_groups"]
        terms = list(item["term_hidden"].keys())
 
        # Identify which term is the stereotyped target and which is not
        stereo_term = None
        non_stereo_term = None
 
        for t in terms:
            if t in stereo_groups:
                stereo_term = t
            else:
                if non_stereo_term is None:
                    non_stereo_term = t
 
        if stereo_term is None or non_stereo_term is None:
            # Try: stereotyped group might not exactly match term
            # E.g., stereotyped_groups=["gay"] but terms are ["gay", "lesbian"]
            # In this case, gay is stereotyped, lesbian is non-stereotyped
            for t in terms:
                if any(t == g for g in stereo_groups):
                    stereo_term = t
                    break
            for t in terms:
                if t != stereo_term:
                    non_stereo_term = t
                    break
 
        if stereo_term is None or non_stereo_term is None:
            continue
 
        h_stereo = item["term_hidden"][stereo_term]      # (n_layers, hidden_dim)
        h_non_stereo = item["term_hidden"][non_stereo_term]  # (n_layers, hidden_dim)
        delta = h_stereo - h_non_stereo                   # (n_layers, hidden_dim)
 
        deltas.append({
            "delta": delta,
            "stereo_term": stereo_term,
            "non_stereo_term": non_stereo_term,
            "context_condition": item["context_condition"],
            "question_polarity": item["question_polarity"],
            "alignment": item["alignment"],
            "question": item["question"],
            "stereotyped_groups": item["stereotyped_groups"],
            "correct": item["correct"],
            "predicted_letter": item["predicted_letter"],
            "answer_roles": item["answer_roles"],
            "hidden_final": item["hidden_final"],
        })
 
    print(f"Computed {len(deltas)} within-item deltas")
    return deltas
 
 
# ---------------------------------------------------------------------------
# Analysis 1: Identity direction from within-item deltas
# ---------------------------------------------------------------------------
def analysis_1_identity_direction(deltas: list[dict], n_layers: int):
    """Is there a consistent direction along which stereotyped identity tokens
    differ from non-stereotyped identity tokens?"""
    print("\n" + "=" * 60)
    print("  ANALYSIS 1: Within-item identity direction")
    print("=" * 60)
 
    results = {
        "mean_delta_norm": [],        # magnitude of mean delta per layer
        "mean_cosine_with_mean": [],  # how aligned are individual deltas with the mean?
        "delta_consistency": [],       # pairwise cosine among deltas
    }
 
    for layer in range(n_layers):
        layer_deltas = np.stack([d["delta"][layer] for d in deltas])  # (N, hidden_dim)
 
        # Mean delta direction
        mean_delta = layer_deltas.mean(axis=0)
        mean_norm = float(np.linalg.norm(mean_delta))
        results["mean_delta_norm"].append(mean_norm)
 
        # How aligned is each item's delta with the mean direction?
        if mean_norm > 1e-10:
            mean_dir = mean_delta / mean_norm
            cosines = []
            for d in layer_deltas:
                d_norm = np.linalg.norm(d)
                if d_norm > 1e-10:
                    cosines.append(float(np.dot(d, mean_dir) / d_norm))
            results["mean_cosine_with_mean"].append(float(np.mean(cosines)))
        else:
            results["mean_cosine_with_mean"].append(0.0)
 
        # Pairwise consistency (sample for efficiency)
        rng = np.random.RandomState(42)
        n_pairs = min(1000, len(layer_deltas) * (len(layer_deltas) - 1) // 2)
        pair_cosines = []
        for _ in range(n_pairs):
            i, j = rng.choice(len(layer_deltas), 2, replace=False)
            pair_cosines.append(cosine_sim(layer_deltas[i], layer_deltas[j]))
        results["delta_consistency"].append(float(np.mean(pair_cosines)))
 
        if (layer + 1) % 10 == 0:
            print(f"  Layer {layer:2d}: "
                  f"mean_delta_norm={mean_norm:.3f}  "
                  f"alignment_with_mean={results['mean_cosine_with_mean'][-1]:.3f}  "
                  f"pairwise_consistency={results['delta_consistency'][-1]:.3f}")
 
    peak_layer = int(np.argmax(results["mean_cosine_with_mean"]))
    print(f"\n  Peak alignment with mean direction: "
          f"{max(results['mean_cosine_with_mean']):.3f} at layer {peak_layer}")
    print(f"  Peak pairwise consistency: "
          f"{max(results['delta_consistency']):.3f} at layer {np.argmax(results['delta_consistency'])}")
 
    return results
 
 
# ---------------------------------------------------------------------------
# Analysis 2: Entanglement — ambig vs disambig
# ---------------------------------------------------------------------------
def analysis_2_entanglement(deltas: list[dict], n_layers: int):
    """Does the identity delta direction correlate with stereotype-selecting
    behavior differently in ambig vs disambig?"""
    print("\n" + "=" * 60)
    print("  ANALYSIS 2: Entanglement (ambig vs disambig)")
    print("=" * 60)
 
    ambig = [d for d in deltas if d["context_condition"] == "ambig"]
    disambig = [d for d in deltas if d["context_condition"] == "disambig"]
    print(f"  Ambiguous deltas: {len(ambig)}, Disambiguated: {len(disambig)}")
 
    results = {
        "delta_norm_ambig": [],
        "delta_norm_disambig": [],
        "identity_direction_cosine": [],  # cosine between ambig and disambig mean deltas
    }
 
    # Also: for each condition, compute the correlation between
    # delta magnitude and stereotype-selecting behavior
    results["stereo_selection_corr_ambig"] = []
    results["stereo_selection_corr_disambig"] = []
 
    for layer in range(n_layers):
        # Mean delta norms per condition
        ambig_norms = [float(np.linalg.norm(d["delta"][layer])) for d in ambig]
        disambig_norms = [float(np.linalg.norm(d["delta"][layer])) for d in disambig]
        results["delta_norm_ambig"].append(float(np.mean(ambig_norms)))
        results["delta_norm_disambig"].append(float(np.mean(disambig_norms)))
 
        # Mean delta direction per condition
        mean_ambig = np.mean(np.stack([d["delta"][layer] for d in ambig]), axis=0)
        mean_disambig = np.mean(np.stack([d["delta"][layer] for d in disambig]), axis=0)
        results["identity_direction_cosine"].append(cosine_sim(mean_ambig, mean_disambig))
 
        # Correlation: does delta magnitude predict stereotype selection?
        for condition, subset, key in [
            ("ambig", ambig, "stereo_selection_corr_ambig"),
            ("disambig", disambig, "stereo_selection_corr_disambig"),
        ]:
            norms = []
            labels = []  # 1 = selected stereotyped answer, 0 = didn't
            for d in subset:
                pred = d["predicted_letter"]
                role = d["answer_roles"].get(pred, "")
                norms.append(float(np.linalg.norm(d["delta"][layer])))
                labels.append(1 if role == "stereotyped_target" else 0)
 
            norms = np.array(norms)
            labels = np.array(labels)
            if norms.std() > 1e-10 and labels.std() > 0:
                corr = float(np.corrcoef(norms, labels)[0, 1])
            else:
                corr = 0.0
            results[key].append(corr)
 
        if (layer + 1) % 10 == 0:
            print(f"  Layer {layer:2d}: "
                  f"delta_norm ambig={results['delta_norm_ambig'][-1]:.3f} "
                  f"disambig={results['delta_norm_disambig'][-1]:.3f} | "
                  f"direction_cos={results['identity_direction_cosine'][-1]:.3f} | "
                  f"stereo_corr ambig={results['stereo_selection_corr_ambig'][-1]:.3f} "
                  f"disambig={results['stereo_selection_corr_disambig'][-1]:.3f}")
 
    return results
 
 
# ---------------------------------------------------------------------------
# Analysis 3: Error trials
# ---------------------------------------------------------------------------
def analysis_3_errors(deltas: list[dict], n_layers: int):
    """On disambig-conflicting items, do error trials have different
    identity delta patterns than correct trials?"""
    print("\n" + "=" * 60)
    print("  ANALYSIS 3: Error trial analysis")
    print("=" * 60)
 
    disambig_conf = [d for d in deltas
                     if d["context_condition"] == "disambig"
                     and d["alignment"] == "conflicting"]
 
    correct = [d for d in disambig_conf if d["correct"]]
    errors = [d for d in disambig_conf if not d["correct"]]
    print(f"  Disambig-conflicting: {len(disambig_conf)} "
          f"(correct={len(correct)}, errors={len(errors)})")
 
    if len(errors) < 10 or len(correct) < 10:
        print("  Too few items for reliable analysis")
        return {"skipped": True, "n_errors": len(errors), "n_correct": len(correct)}
 
    results = {
        "n_errors": len(errors),
        "n_correct": len(correct),
        "delta_norm_correct": [],
        "delta_norm_errors": [],
        "delta_projection_correct": [],  # projection of deltas onto overall identity direction
        "delta_projection_errors": [],
    }
 
    # Compute overall identity direction from ALL deltas for reference
    all_deltas_stack = {
        layer: np.stack([d["delta"][layer] for d in deltas])
        for layer in range(n_layers)
    }
 
    for layer in range(n_layers):
        # Identity direction from all items
        mean_delta = all_deltas_stack[layer].mean(axis=0)
        mean_norm = np.linalg.norm(mean_delta)
        if mean_norm > 1e-10:
            identity_dir = mean_delta / mean_norm
        else:
            identity_dir = np.zeros_like(mean_delta)
 
        # Delta norms
        correct_norms = [float(np.linalg.norm(d["delta"][layer])) for d in correct]
        error_norms = [float(np.linalg.norm(d["delta"][layer])) for d in errors]
        results["delta_norm_correct"].append(float(np.mean(correct_norms)))
        results["delta_norm_errors"].append(float(np.mean(error_norms)))
 
        # Projections onto identity direction
        correct_projs = [float(d["delta"][layer].astype(np.float32) @ identity_dir)
                        for d in correct]
        error_projs = [float(d["delta"][layer].astype(np.float32) @ identity_dir)
                      for d in errors]
        results["delta_projection_correct"].append(float(np.mean(correct_projs)))
        results["delta_projection_errors"].append(float(np.mean(error_projs)))
 
        if (layer + 1) % 10 == 0:
            print(f"  Layer {layer:2d}: "
                  f"norm correct={results['delta_norm_correct'][-1]:.3f} "
                  f"errors={results['delta_norm_errors'][-1]:.3f} | "
                  f"proj correct={results['delta_projection_correct'][-1]:.3f} "
                  f"errors={results['delta_projection_errors'][-1]:.3f}")
 
    return results
 
 
# ---------------------------------------------------------------------------
# Analysis 4: Group-level differences
# ---------------------------------------------------------------------------
def analysis_4_group_level(deltas: list[dict], n_layers: int):
    """Do identity deltas differ by which group is stereotyped?"""
    print("\n" + "=" * 60)
    print("  ANALYSIS 4: Group-level identity delta differences")
    print("=" * 60)
 
    groups = defaultdict(list)
    for d in deltas:
        for g in d["stereotyped_groups"]:
            if g in IDENTITY_TERMS:
                groups[g].append(d)
 
    print(f"  Groups: {', '.join(f'{g}: n={len(items)}' for g, items in groups.items())}")
 
    results = {"groups": {}}
 
    for group, group_deltas in groups.items():
        results["groups"][group] = {
            "n": len(group_deltas),
            "mean_delta_norm": [],
            "mean_alignment": [],
        }
 
        for layer in range(n_layers):
            # All deltas for reference direction
            all_layer_deltas = np.stack([d["delta"][layer] for d in deltas])
            mean_all = all_layer_deltas.mean(axis=0)
            mean_all_norm = np.linalg.norm(mean_all)
 
            # This group's deltas
            group_layer_deltas = np.stack([d["delta"][layer] for d in group_deltas])
            norms = [float(np.linalg.norm(d)) for d in group_layer_deltas]
            results["groups"][group]["mean_delta_norm"].append(float(np.mean(norms)))
 
            # Alignment with overall direction
            if mean_all_norm > 1e-10:
                mean_dir = mean_all / mean_all_norm
                alignments = [float(np.dot(d, mean_dir) / (np.linalg.norm(d) + 1e-10))
                             for d in group_layer_deltas]
                results["groups"][group]["mean_alignment"].append(float(np.mean(alignments)))
            else:
                results["groups"][group]["mean_alignment"].append(0.0)
 
    # Print summary at representative layers
    for layer in [10, 20, 30]:
        print(f"\n  Layer {layer}:")
        for group in sorted(results["groups"].keys()):
            info = results["groups"][group]
            print(f"    {group:12s}: norm={info['mean_delta_norm'][layer]:.3f}  "
                  f"alignment={info['mean_alignment'][layer]:.3f}")
 
    return results
 
 
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_all(a1, a2, a3, a4, n_layers, figures_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    figures_dir.mkdir(parents=True, exist_ok=True)
    layers = list(range(n_layers))
 
    # --- Figure 1: Identity direction consistency ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
 
    ax1.plot(layers, a1["mean_delta_norm"], color=COLORS["blue"], linewidth=2)
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("L2 norm", fontsize=12)
    ax1.set_title("Mean within-item identity delta magnitude", fontsize=12)
    ax1.grid(True, alpha=0.3)
 
    ax2.plot(layers, a1["mean_cosine_with_mean"],
             color=COLORS["blue"], linewidth=2, label="Alignment with mean")
    ax2.plot(layers, a1["delta_consistency"],
             color=COLORS["orange"], linewidth=2, label="Pairwise consistency")
    ax2.axhline(0, color=COLORS["gray"], linewidth=0.8, linestyle=":")
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("Cosine similarity", fontsize=12)
    ax2.set_title("Identity delta direction consistency", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
 
    fig.suptitle("Analysis 1: Within-item identity direction", fontsize=13)
    fig.tight_layout()
    fig.savefig(figures_dir / "a1_identity_direction.png", dpi=150)
    plt.close(fig)
    print("  Saved a1_identity_direction.png")
 
    # --- Figure 2: Ambig vs disambig ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
 
    ax1.plot(layers, a2["delta_norm_ambig"],
             color=COLORS["red"], linewidth=2, label="Ambiguous")
    ax1.plot(layers, a2["delta_norm_disambig"],
             color=COLORS["blue"], linewidth=2, label="Disambiguated")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean delta L2 norm")
    ax1.set_title("Identity delta magnitude by condition")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
 
    ax2.plot(layers, a2["identity_direction_cosine"],
             color=COLORS["purple"], linewidth=2)
    ax2.axhline(0, color=COLORS["gray"], linewidth=0.8, linestyle=":")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Cosine similarity")
    ax2.set_title("Ambig vs disambig direction similarity")
    ax2.grid(True, alpha=0.3)
 
    ax3.plot(layers, a2["stereo_selection_corr_ambig"],
             color=COLORS["red"], linewidth=2, label="Ambiguous")
    ax3.plot(layers, a2["stereo_selection_corr_disambig"],
             color=COLORS["blue"], linewidth=2, label="Disambiguated")
    ax3.axhline(0, color=COLORS["gray"], linewidth=0.8, linestyle=":")
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Pearson r")
    ax3.set_title("Delta norm ↔ stereotype selection")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
 
    fig.suptitle("Analysis 2: Ambiguous vs disambiguated geometry", fontsize=13)
    fig.tight_layout()
    fig.savefig(figures_dir / "a2_entanglement.png", dpi=150)
    plt.close(fig)
    print("  Saved a2_entanglement.png")
 
    # --- Figure 3: Error analysis ---
    if not a3.get("skipped"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
 
        ax1.plot(layers, a3["delta_norm_correct"],
                 color=COLORS["green"], linewidth=2, label="Correct")
        ax1.plot(layers, a3["delta_norm_errors"],
                 color=COLORS["red"], linewidth=2, label="Errors")
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Mean delta L2 norm")
        ax1.set_title("Identity delta magnitude: correct vs errors")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
 
        ax2.plot(layers, a3["delta_projection_correct"],
                 color=COLORS["green"], linewidth=2, label="Correct")
        ax2.plot(layers, a3["delta_projection_errors"],
                 color=COLORS["red"], linewidth=2, label="Errors")
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Mean projection")
        ax2.set_title("Projection onto identity direction: correct vs errors")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
 
        fig.suptitle("Analysis 3: Error trials (disambig-conflicting)", fontsize=13)
        fig.tight_layout()
        fig.savefig(figures_dir / "a3_error_analysis.png", dpi=150)
        plt.close(fig)
        print("  Saved a3_error_analysis.png")
 
    # --- Figure 4: Group-level ---
    group_colors = {
        "gay": COLORS["blue"],
        "lesbian": COLORS["purple"],
        "bisexual": COLORS["orange"],
        "pansexual": COLORS["green"],
        "straight": COLORS["gray"],
    }
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for group, info in a4["groups"].items():
        color = group_colors.get(group, COLORS["gray"])
        ax1.plot(layers, info["mean_delta_norm"],
                 color=color, linewidth=2, label=group)
        ax2.plot(layers, info["mean_alignment"],
                 color=color, linewidth=2, label=group)
 
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean delta L2 norm")
    ax1.set_title("Identity delta magnitude by stereotyped group")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
 
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Cosine with overall direction")
    ax2.set_title("Delta alignment with mean identity direction")
    ax2.axhline(0, color=COLORS["gray"], linewidth=0.8, linestyle=":")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
 
    fig.suptitle("Analysis 4: Group-level identity delta differences", fontsize=13)
    fig.tight_layout()
    fig.savefig(figures_dir / "a4_group_level.png", dpi=150)
    plt.close(fig)
    print("  Saved a4_group_level.png")
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation_dir", type=Path, default=ACTIVATION_DIR)
    parser.add_argument("--skip_plots", action="store_true")
    args = parser.parse_args()
 
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
 
    # Find behavioral results
    behavioral_candidates = sorted(BEHAVIORAL_DIR.glob("behavioral_results*.json"))
    if not behavioral_candidates:
        print("ERROR: No behavioral results found")
        return
    behavioral_path = behavioral_candidates[-1]
    print(f"Using behavioral results: {behavioral_path.name}")
 
    items = load_data(args.activation_dir, behavioral_path)
 
    n_layers = items[0]["hidden_final"].shape[0]
    print(f"Model: {n_layers} layers")
 
    # Compute within-item deltas
    deltas = compute_within_item_deltas(items, n_layers)
 
    # Run analyses
    a1 = analysis_1_identity_direction(deltas, n_layers)
    a2 = analysis_2_entanglement(deltas, n_layers)
    a3 = analysis_3_errors(deltas, n_layers)
    a4 = analysis_4_group_level(deltas, n_layers)
 
    # Save
    all_results = {
        "analysis_1": a1,
        "analysis_2": a2,
        "analysis_3": {k: v for k, v in a3.items() if k != "skipped"},
        "analysis_4": a4,
        "n_deltas": len(deltas),
    }
    with open(RESULTS_DIR / "analysis_results_v2.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved analysis_results_v2.json")
 
    if not args.skip_plots:
        print("\nGenerating figures...")
        plot_all(a1, a2, a3, a4, n_layers, FIGURES_DIR)
 
    print("\nDone.")
 
 
if __name__ == "__main__":
    main()
 