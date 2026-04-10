"""
analyze_cross_identity.py — Compare SO and GI identity directions
==================================================================
Loads activations from both Sexual Orientation and Gender Identity
(trans-targeted) subsets, computes within-item deltas for each, and
tests whether the identity directions share structure or are independent.

Analyses:
  1. GI internal structure: delta consistency, group-level breakdown
     (trans vs transgender_woman vs transgender_man if distinguishable)
  2. Cross-identity direction comparison: cosine between SO family
     directions and GI trans direction across layers
  3. Combined PCA: project SO and GI deltas into shared space
  4. Cross-identity probe generalization: train on SO, test on GI

Usage:
  python scripts/analyze_cross_identity.py
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SO_ACTIVATION_DIR = PROJECT_ROOT / "results" / "activations" / "so"
GI_ACTIVATION_DIR = PROJECT_ROOT / "results" / "activations" / "gi"
BEHAVIORAL_DIR = PROJECT_ROOT / "results" / "behavioral_pilot"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

SO_TERMS = {"gay", "lesbian", "bisexual", "pansexual", "straight"}
GI_TERMS = {"transgender", "transgendered", "transgender woman", "transgender man",
            "trans woman", "trans man", "trans", "cisgender", "cis",
            "nonbinary", "non-binary"}
TRANS_TERMS = {"transgender", "transgendered", "transgender woman", "transgender man",
               "trans woman", "trans man", "trans"}
CIS_TERMS = {"cisgender", "cis"}

COLORS = {
    "gay_lesbian": "#0072B2",
    "bi_pan": "#E69F00",
    "trans": "#D55E00",
    "gay": "#0072B2",
    "lesbian": "#CC79A7",
    "bisexual": "#E69F00",
    "pansexual": "#009E73",
    "gray": "#999999",
}


def cosine_sim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 1e-10 else 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_activations(activation_dir, behavioral_path, identity_terms_set):
    """Load activations and compute per-term hidden states."""
    # Try to load behavioral results; if not available, proceed without
    behav_by_idx = {}
    if behavioral_path and behavioral_path.exists():
        with open(behavioral_path) as f:
            behav_by_idx = {r["item_idx"]: r for r in json.load(f)}

    items = []
    for npz_path in sorted(activation_dir.glob("item_*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        meta = json.loads(str(data["metadata"]))
        idx = meta["item_idx"]
        behav = behav_by_idx.get(idx, {})

        term_to_tokens = defaultdict(list)
        for entry in meta["identity_terms_found"]:
            t = entry["term"].lower()
            if t in identity_terms_set:
                term_to_tokens[t].append(entry["token_indices"])

        all_identity_indices = meta["identity_token_indices"]
        hidden_identity_raw = data["hidden_identity"]
        idx_to_pos = {ti: p for p, ti in enumerate(all_identity_indices)}

        term_hidden = {}
        for term, token_lists in term_to_tokens.items():
            tok_indices = sorted(set(ti for tl in token_lists for ti in tl))
            positions = [idx_to_pos[ti] for ti in tok_indices if ti in idx_to_pos]
            if positions and hidden_identity_raw.shape[1] > 0:
                term_hidden[term] = hidden_identity_raw[:, positions, :].mean(axis=1).astype(np.float32)

        items.append({
            "idx": idx,
            "term_hidden": term_hidden,
            "hidden_final": data["hidden_final"].astype(np.float32),
            "stereotyped_groups": [g.lower() for g in meta["stereotyped_groups"]],
            "context_condition": meta["context_condition"],
            "question_polarity": meta["question_polarity"],
            "alignment": meta["alignment"],
            "question": meta["question"],
            "correct": behav.get("correct", False),
            "predicted_letter": behav.get("predicted_letter", ""),
            "answer_roles": meta["answer_roles"],
        })

    return items


def compute_gi_deltas(items, n_layers):
    """Compute within-item deltas for GI: h(trans_term) - h(cis_term)."""
    deltas = []
    for item in items:
        if len(item["term_hidden"]) < 2:
            continue

        terms = list(item["term_hidden"].keys())

        # Find trans and cis terms
        trans_term = None
        cis_term = None
        for t in terms:
            if t in TRANS_TERMS:
                trans_term = t
            elif t in CIS_TERMS:
                cis_term = t

        # If no explicit cis term, take any term that isn't trans
        if cis_term is None:
            for t in terms:
                if t not in TRANS_TERMS and t != trans_term:
                    cis_term = t
                    break

        if trans_term is None or cis_term is None:
            continue

        h_trans = item["term_hidden"][trans_term]
        h_cis = item["term_hidden"][cis_term]
        delta = h_trans - h_cis

        norms_mean = (np.linalg.norm(h_trans, axis=1, keepdims=True) +
                      np.linalg.norm(h_cis, axis=1, keepdims=True)) / 2
        norms_mean = np.maximum(norms_mean, 1e-10)
        delta_normed = delta / norms_mean

        deltas.append({
            "delta": delta,
            "delta_normed": delta_normed,
            "trans_term": trans_term,
            "cis_term": cis_term,
            "context_condition": item["context_condition"],
            "question_polarity": item["question_polarity"],
            "stereotyped_groups": item["stereotyped_groups"],
            "question": item["question"],
        })

    return deltas


def compute_so_deltas(items, n_layers):
    """Compute within-item deltas for SO (same as before)."""
    deltas = []
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
        delta = h_s - h_ns

        norms_mean = (np.linalg.norm(h_s, axis=1, keepdims=True) +
                      np.linalg.norm(h_ns, axis=1, keepdims=True)) / 2
        norms_mean = np.maximum(norms_mean, 1e-10)
        delta_normed = delta / norms_mean

        deltas.append({
            "delta": delta,
            "delta_normed": delta_normed,
            "stereo_term": stereo_term,
            "non_stereo_term": non_stereo_term,
            "stereo_group": stereo_groups[0] if stereo_groups else "",
            "context_condition": item["context_condition"],
            "question_polarity": item["question_polarity"],
            "question": item["question"],
        })

    return deltas


# ---------------------------------------------------------------------------
# Analysis 1: GI internal structure
# ---------------------------------------------------------------------------
def analysis_1_gi_internal(gi_deltas, n_layers):
    print("\n" + "=" * 60)
    print("  1. GI (trans) internal structure")
    print("=" * 60)

    print(f"  Total GI deltas: {len(gi_deltas)}")

    # Check trans term distribution
    from collections import Counter
    trans_terms = Counter(d["trans_term"] for d in gi_deltas)
    cis_terms = Counter(d["cis_term"] for d in gi_deltas)
    print(f"  Trans terms: {dict(trans_terms)}")
    print(f"  Cis terms: {dict(cis_terms)}")

    results = {"mean_delta_norm": [], "alignment_with_mean": [], "pairwise_consistency": []}

    for layer in range(n_layers):
        layer_deltas = np.stack([d["delta_normed"][layer] for d in gi_deltas])
        mean_delta = layer_deltas.mean(axis=0)
        mean_norm = float(np.linalg.norm(mean_delta))
        results["mean_delta_norm"].append(mean_norm)

        if mean_norm > 1e-10:
            mean_dir = mean_delta / mean_norm
            cosines = [float(np.dot(d, mean_dir) / (np.linalg.norm(d) + 1e-10))
                      for d in layer_deltas]
            results["alignment_with_mean"].append(float(np.mean(cosines)))
        else:
            results["alignment_with_mean"].append(0.0)

        rng = np.random.RandomState(42)
        n_pairs = min(1000, len(layer_deltas) * (len(layer_deltas) - 1) // 2)
        pair_cos = []
        for _ in range(n_pairs):
            i, j = rng.choice(len(layer_deltas), 2, replace=False)
            pair_cos.append(cosine_sim(layer_deltas[i], layer_deltas[j]))
        results["pairwise_consistency"].append(float(np.mean(pair_cos)))

        if (layer + 1) % 10 == 0:
            print(f"  Layer {layer:2d}: norm={mean_norm:.3f}  "
                  f"align={results['alignment_with_mean'][-1]:.3f}  "
                  f"pairwise={results['pairwise_consistency'][-1]:.3f}")

    return results


# ---------------------------------------------------------------------------
# Analysis 2: Cross-identity direction comparison
# ---------------------------------------------------------------------------
def analysis_2_cross_identity(so_deltas, gi_deltas, n_layers):
    print("\n" + "=" * 60)
    print("  2. Cross-identity direction comparison")
    print("=" * 60)

    # Compute SO family directions
    gl_deltas = [d for d in so_deltas if d["stereo_group"] in ["gay", "lesbian"]]
    bp_deltas = [d for d in so_deltas if d["stereo_group"] in ["bisexual", "pansexual"]]
    print(f"  SO gay/lesbian deltas: {len(gl_deltas)}")
    print(f"  SO bisexual/pansexual deltas: {len(bp_deltas)}")
    print(f"  GI trans deltas: {len(gi_deltas)}")

    # Also compute per-SO-group directions
    so_by_group = defaultdict(list)
    for d in so_deltas:
        so_by_group[d["stereo_group"]].append(d)

    results = {
        "trans_vs_gl": [],
        "trans_vs_bp": [],
        "trans_vs_gay": [],
        "trans_vs_lesbian": [],
        "trans_vs_bisexual": [],
        "trans_vs_pansexual": [],
        "gl_vs_bp": [],
    }

    for layer in range(n_layers):
        # Mean directions (normalized deltas)
        gl_dir = np.mean(np.stack([d["delta_normed"][layer] for d in gl_deltas]), axis=0)
        bp_dir = np.mean(np.stack([d["delta_normed"][layer] for d in bp_deltas]), axis=0)
        trans_dir = np.mean(np.stack([d["delta_normed"][layer] for d in gi_deltas]), axis=0)

        results["trans_vs_gl"].append(cosine_sim(trans_dir, gl_dir))
        results["trans_vs_bp"].append(cosine_sim(trans_dir, bp_dir))
        results["gl_vs_bp"].append(cosine_sim(gl_dir, bp_dir))

        # Per-group
        for group in ["gay", "lesbian", "bisexual", "pansexual"]:
            if so_by_group[group]:
                g_dir = np.mean(np.stack([d["delta_normed"][layer]
                                          for d in so_by_group[group]]), axis=0)
                results[f"trans_vs_{group}"].append(cosine_sim(trans_dir, g_dir))
            else:
                results[f"trans_vs_{group}"].append(0.0)

        if (layer + 1) % 10 == 0:
            print(f"  Layer {layer:2d}: "
                  f"trans↔GL={results['trans_vs_gl'][-1]:.3f}  "
                  f"trans↔BP={results['trans_vs_bp'][-1]:.3f}  "
                  f"GL↔BP={results['gl_vs_bp'][-1]:.3f}")

    return results


# ---------------------------------------------------------------------------
# Analysis 3: Combined PCA
# ---------------------------------------------------------------------------
def analysis_3_combined_pca(so_deltas, gi_deltas, n_layers, target_layers=None):
    print("\n" + "=" * 60)
    print("  3. Combined PCA visualization")
    print("=" * 60)

    from sklearn.decomposition import PCA

    if target_layers is None:
        target_layers = [10, 20, 30]

    results = {}
    for layer in target_layers:
        # Stack all normalized deltas
        so_normed = np.stack([d["delta_normed"][layer] for d in so_deltas])
        gi_normed = np.stack([d["delta_normed"][layer] for d in gi_deltas])
        all_normed = np.vstack([so_normed, gi_normed])

        # Labels
        so_labels = [d["stereo_group"] for d in so_deltas]
        gi_labels = ["trans"] * len(gi_deltas)
        all_labels = so_labels + gi_labels

        pca = PCA(n_components=5)
        coords = pca.fit_transform(all_normed)

        results[layer] = {
            "coords": coords,
            "labels": all_labels,
            "explained": pca.explained_variance_ratio_.tolist(),
            "n_so": len(so_deltas),
            "n_gi": len(gi_deltas),
        }

        print(f"  Layer {layer}: PC1={pca.explained_variance_ratio_[0]:.3f}  "
              f"PC2={pca.explained_variance_ratio_[1]:.3f}")

    return results


# ---------------------------------------------------------------------------
# Analysis 4: Cross-identity probe
# ---------------------------------------------------------------------------
def analysis_4_cross_probe(so_deltas, gi_deltas, n_layers, target_layers=None):
    print("\n" + "=" * 60)
    print("  4. Cross-identity probe generalization")
    print("=" * 60)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold

    if target_layers is None:
        target_layers = [10, 15, 20, 25, 30]

    # For SO: label = 1 if stereo_group in bisexual/pansexual (the stereotyped family)
    # For GI: label = 1 always (trans is always stereotyped target)
    # The probe question: can the SO direction distinguish trans from cis?
    # We need a different framing — train on SO to classify gay/lesbian vs bisexual/pansexual,
    # then test: where do trans items fall?

    results = {}

    for layer in target_layers:
        # SO features and labels
        so_features = np.stack([d["delta_normed"][layer] for d in so_deltas])
        so_labels = np.array([
            1 if d["stereo_group"] in ["bisexual", "pansexual"] else 0
            for d in so_deltas
        ])

        # GI features
        gi_features = np.stack([d["delta_normed"][layer] for d in gi_deltas])

        n_comp = min(50, so_features.shape[0] - 1, so_features.shape[1])
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_comp)),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ])

        # Cross-validated accuracy on SO
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        so_accs = []
        for train_idx, test_idx in skf.split(so_features, so_labels):
            pipe.fit(so_features[train_idx], so_labels[train_idx])
            so_accs.append(pipe.score(so_features[test_idx], so_labels[test_idx]))
        so_cv_acc = float(np.mean(so_accs))

        # Train on all SO, predict on GI
        pipe.fit(so_features, so_labels)
        gi_preds = pipe.predict(gi_features)
        gi_probs = pipe.predict_proba(gi_features)

        # What fraction of trans items are classified as bisexual/pansexual-like?
        frac_bp_like = float(gi_preds.mean())
        mean_bp_prob = float(gi_probs[:, 1].mean())

        # Also: train on all SO, apply DIM direction
        so_gl = so_features[so_labels == 0]
        so_bp = so_features[so_labels == 1]
        dim_dir = so_gl.mean(axis=0) - so_bp.mean(axis=0)
        dim_dir = dim_dir / (np.linalg.norm(dim_dir) + 1e-10)

        # Project GI onto this direction
        gi_projections = gi_features @ dim_dir
        so_gl_proj = so_gl @ dim_dir
        so_bp_proj = so_bp @ dim_dir

        results[layer] = {
            "so_cv_accuracy": so_cv_acc,
            "gi_frac_bp_like": frac_bp_like,
            "gi_mean_bp_prob": mean_bp_prob,
            "gi_mean_projection": float(gi_projections.mean()),
            "so_gl_mean_projection": float(so_gl_proj.mean()),
            "so_bp_mean_projection": float(so_bp_proj.mean()),
            "gi_proj_std": float(gi_projections.std()),
        }

        print(f"  Layer {layer:2d}: SO CV acc={so_cv_acc:.3f} | "
              f"GI classified as BP-like: {frac_bp_like:.1%} | "
              f"Projections: GL={so_gl_proj.mean():.3f}  BP={so_bp_proj.mean():.3f}  "
              f"Trans={gi_projections.mean():.3f}")

    return results


# ---------------------------------------------------------------------------
# Analysis 5: Permutation test for cross-identity cosines
# ---------------------------------------------------------------------------
def analysis_5_permutation(so_deltas, gi_deltas, n_layers, n_perm=5000):
    print("\n" + "=" * 60)
    print("  5. Permutation test: is trans direction closer to GL or BP?")
    print("=" * 60)

    gl_deltas = [d for d in so_deltas if d["stereo_group"] in ["gay", "lesbian"]]
    bp_deltas = [d for d in so_deltas if d["stereo_group"] in ["bisexual", "pansexual"]]

    target_layers = [10, 15, 20, 25, 30]
    results = {}

    for layer in target_layers:
        gl_dir = np.mean(np.stack([d["delta_normed"][layer] for d in gl_deltas]), axis=0)
        bp_dir = np.mean(np.stack([d["delta_normed"][layer] for d in bp_deltas]), axis=0)
        trans_dir = np.mean(np.stack([d["delta_normed"][layer] for d in gi_deltas]), axis=0)

        # Observed: cos(trans, GL) - cos(trans, BP)
        obs_diff = cosine_sim(trans_dir, gl_dir) - cosine_sim(trans_dir, bp_dir)

        # Permutation: shuffle SO items between GL and BP pools
        all_so = gl_deltas + bp_deltas
        n_gl = len(gl_deltas)
        rng = np.random.RandomState(42 + layer)
        perm_diffs = np.zeros(n_perm)

        for p in range(n_perm):
            perm_idx = rng.permutation(len(all_so))
            perm_gl = [all_so[i] for i in perm_idx[:n_gl]]
            perm_bp = [all_so[i] for i in perm_idx[n_gl:]]

            pgl = np.mean(np.stack([d["delta_normed"][layer] for d in perm_gl]), axis=0)
            pbp = np.mean(np.stack([d["delta_normed"][layer] for d in perm_bp]), axis=0)

            perm_diffs[p] = cosine_sim(trans_dir, pgl) - cosine_sim(trans_dir, pbp)

        p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff)))

        results[layer] = {
            "cos_trans_gl": cosine_sim(trans_dir, gl_dir),
            "cos_trans_bp": cosine_sim(trans_dir, bp_dir),
            "obs_diff": obs_diff,
            "p_value": p_value,
            "perm_diffs": perm_diffs.tolist(),
        }

        print(f"  Layer {layer:2d}: cos(trans,GL)={cosine_sim(trans_dir, gl_dir):.3f}  "
              f"cos(trans,BP)={cosine_sim(trans_dir, bp_dir):.3f}  "
              f"Δ={obs_diff:.3f}  p={p_value:.4f}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_all(a1, a2, a3, a4, a5, so_deltas, gi_deltas, n_layers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    layers = list(range(n_layers))

    ALL_COLORS = {
        "gay": "#0072B2", "lesbian": "#CC79A7",
        "bisexual": "#E69F00", "pansexual": "#009E73",
        "trans": "#D55E00",
    }
    ALL_MARKERS = {
        "gay": "o", "lesbian": "s", "bisexual": "^",
        "pansexual": "D", "trans": "P",
    }

    # --- Figure 1: GI internal consistency ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(layers, a1["mean_delta_norm"], color=ALL_COLORS["trans"], linewidth=2)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("L2 norm")
    ax1.set_title("GI (trans) mean delta magnitude")
    ax1.grid(True, alpha=0.3)

    ax2.plot(layers, a1["alignment_with_mean"], color=ALL_COLORS["trans"],
             linewidth=2, label="Alignment with mean")
    ax2.plot(layers, a1["pairwise_consistency"], color="#999999",
             linewidth=2, label="Pairwise consistency")
    ax2.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Cosine similarity")
    ax2.set_title("GI delta consistency")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Gender identity (trans) internal structure", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cross_1_gi_internal.png", dpi=150)
    plt.close(fig)
    print("  Saved cross_1_gi_internal.png")

    # --- Figure 2: Cross-identity cosines across layers ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(layers, a2["trans_vs_gl"], color=ALL_COLORS["gay"],
             linewidth=2.5, label="Trans ↔ Gay/Lesbian family")
    ax1.plot(layers, a2["trans_vs_bp"], color=ALL_COLORS["bisexual"],
             linewidth=2.5, label="Trans ↔ Bisexual/Pansexual family")
    ax1.plot(layers, a2["gl_vs_bp"], color="#999999",
             linewidth=2, linestyle="--", label="GL ↔ BP (reference)")
    ax1.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Cosine similarity", fontsize=12)
    ax1.set_title("Trans direction vs SO family directions", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.8, 0.8)

    # Per-group
    for group in ["gay", "lesbian", "bisexual", "pansexual"]:
        ax2.plot(layers, a2[f"trans_vs_{group}"],
                color=ALL_COLORS[group], linewidth=2, label=f"Trans ↔ {group}")
    ax2.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("Cosine similarity", fontsize=12)
    ax2.set_title("Trans direction vs individual SO groups", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.8, 0.8)

    fig.suptitle("Cross-identity direction comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cross_2_direction_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved cross_2_direction_comparison.png")

    # --- Figure 3: Combined PCA ---
    pca_layers = sorted(a3.keys())
    fig, axes = plt.subplots(1, len(pca_layers), figsize=(6 * len(pca_layers), 5))
    if len(pca_layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, pca_layers):
        info = a3[layer]
        coords = info["coords"]
        labels = info["labels"]
        ev = info["explained"]

        for group in ["gay", "lesbian", "bisexual", "pansexual", "trans"]:
            mask = [l == group for l in labels]
            if not any(mask):
                continue
            idx = np.where(mask)[0]
            ax.scatter(coords[idx, 0], coords[idx, 1],
                      c=ALL_COLORS[group], marker=ALL_MARKERS[group],
                      s=20, alpha=0.4, label=group, edgecolors="none")

        ax.set_xlabel(f"PC1 ({ev[0]:.1%})", fontsize=11)
        ax.set_ylabel(f"PC2 ({ev[1]:.1%})", fontsize=11)
        ax.set_title(f"Layer {layer}", fontsize=12)
        ax.legend(fontsize=8, markerscale=2)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Combined PCA: SO + GI identity deltas", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cross_3_combined_pca.png", dpi=150)
    plt.close(fig)
    print("  Saved cross_3_combined_pca.png")

    # --- Figure 4: Cross-probe projection distribution ---
    probe_layers = sorted(a4.keys())
    fig, axes = plt.subplots(1, len(probe_layers), figsize=(4 * len(probe_layers), 4))
    if len(probe_layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, probe_layers):
        info = a4[layer]
        # Bar chart: GL, BP, Trans projections
        positions = [0, 1, 2]
        values = [info["so_gl_mean_projection"],
                  info["so_bp_mean_projection"],
                  info["gi_mean_projection"]]
        colors = [ALL_COLORS["gay"], ALL_COLORS["bisexual"], ALL_COLORS["trans"]]
        labels = ["Gay/Les", "Bi/Pan", "Trans"]

        bars = ax.bar(positions, values, color=colors, edgecolor="white",
                      linewidth=1.5, width=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=10)
        ax.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
        ax.set_ylabel("Mean projection", fontsize=10)
        ax.set_title(f"Layer {layer}\nSO acc={info['so_cv_accuracy']:.2f}", fontsize=10)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Projection onto SO family-separation direction\n"
                 "(GL→positive, BP→negative; where does Trans fall?)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cross_4_probe_projection.png", dpi=150)
    plt.close(fig)
    print("  Saved cross_4_probe_projection.png")

    # --- Figure 5: Permutation test ---
    perm_layers = sorted(a5.keys())
    fig, axes = plt.subplots(1, len(perm_layers), figsize=(4 * len(perm_layers), 3.5))
    if len(perm_layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, perm_layers):
        info = a5[layer]
        perm_diffs = np.array(info["perm_diffs"])
        obs = info["obs_diff"]
        ax.hist(perm_diffs, bins=50, color="#CCCCCC", edgecolor="white", alpha=0.8)
        ax.axvline(obs, color=ALL_COLORS["trans"], linewidth=2.5,
                   label=f"Obs Δ={obs:.3f}")
        ax.set_xlabel("cos(trans,GL) − cos(trans,BP)")
        ax.set_title(f"L{layer} (p={info['p_value']:.4f})", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Permutation test: is trans closer to GL or BP?", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cross_5_permutation.png", dpi=150)
    plt.close(fig)
    print("  Saved cross_5_permutation.png")

    # --- Figure 6: Summary cosine matrix at representative layer ---
    layer = 20
    groups_all = ["gay", "lesbian", "bisexual", "pansexual", "trans"]

    # Compute all pairwise cosines
    all_dirs = {}
    for g in ["gay", "lesbian", "bisexual", "pansexual"]:
        g_items = [d for d in so_deltas if d["stereo_group"] == g]
        if g_items:
            all_dirs[g] = np.mean(np.stack([d["delta_normed"][layer] for d in g_items]), axis=0)
    all_dirs["trans"] = np.mean(np.stack([d["delta_normed"][layer] for d in gi_deltas]), axis=0)

    matrix = np.zeros((5, 5))
    for i, g1 in enumerate(groups_all):
        for j, g2 in enumerate(groups_all):
            if g1 in all_dirs and g2 in all_dirs:
                matrix[i, j] = cosine_sim(all_dirs[g1], all_dirs[g2])

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(matrix, vmin=-0.6, vmax=0.8, cmap="RdBu_r")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([g.capitalize() for g in groups_all], fontsize=11)
    ax.set_yticklabels([g.capitalize() for g in groups_all], fontsize=11)

    for i in range(5):
        for j in range(5):
            color = "white" if abs(matrix[i, j]) > 0.4 else "black"
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                   fontsize=11, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine similarity")
    ax.set_title(f"Identity direction pairwise cosine matrix (layer {layer})\n"
                 f"Normalized within-item deltas", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cross_6_cosine_matrix.png", dpi=150)
    plt.close(fig)
    print("  Saved cross_6_cosine_matrix.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, default=None, help="Optional model path to resolve model_id/run_dir.")
    parser.add_argument("--model_id", type=str, default=None, help="Override model id used for results/runs/<model_id>/")
    parser.add_argument("--run_date", type=str, default=None, help="Run date (YYYY-MM-DD). Defaults to newest for model_id.")
    parser.add_argument("--run_dir", type=Path, default=None, help="Explicit run directory override.")
    parser.add_argument("--skip_plots", action="store_true")
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

    global SO_ACTIVATION_DIR, GI_ACTIVATION_DIR, BEHAVIORAL_DIR, FIGURES_DIR, RESULTS_DIR
    SO_ACTIVATION_DIR = subdirs.activations_so_dir
    GI_ACTIVATION_DIR = subdirs.activations_gi_dir
    BEHAVIORAL_DIR = subdirs.behavioral_dir
    FIGURES_DIR = subdirs.figures_dir
    RESULTS_DIR = subdirs.analysis_dir

    print(f"Run: model_id={model_id}  run_date={run_date}")
    print(f"Run dir: {run_dir}")
    print(f"Activations (SO): {SO_ACTIVATION_DIR}")
    print(f"Activations (GI): {GI_ACTIVATION_DIR}")
    print(f"Behavioral dir: {BEHAVIORAL_DIR}")
    print(f"Analysis outputs: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Find behavioral results (SO only — GI doesn't have behavioral pilot)
    behavioral_candidates = sorted(BEHAVIORAL_DIR.glob("behavioral_results*.json"))
    behavioral_path = behavioral_candidates[-1] if behavioral_candidates else None
    if behavioral_path:
        print(f"SO behavioral: {behavioral_path.name}")

    # Load SO
    print("\nLoading SO activations...")
    so_items = load_activations(SO_ACTIVATION_DIR, behavioral_path,
                                SO_TERMS | {"straight"})
    print(f"  SO items: {len(so_items)}")

    # Load GI
    print("Loading GI activations...")
    gi_items = load_activations(GI_ACTIVATION_DIR, None,
                                GI_TERMS | CIS_TERMS | TRANS_TERMS | {"trans", "cis"})
    print(f"  GI items: {len(gi_items)}")

    n_layers = so_items[0]["hidden_final"].shape[0]

    # Compute deltas
    so_deltas = compute_so_deltas(so_items, n_layers)
    gi_deltas = compute_gi_deltas(gi_items, n_layers)
    print(f"\nSO deltas: {len(so_deltas)}, GI deltas: {len(gi_deltas)}")

    # Run analyses
    a1 = analysis_1_gi_internal(gi_deltas, n_layers)
    a2 = analysis_2_cross_identity(so_deltas, gi_deltas, n_layers)
    a3 = analysis_3_combined_pca(so_deltas, gi_deltas, n_layers)
    a4 = analysis_4_cross_probe(so_deltas, gi_deltas, n_layers)
    a5 = analysis_5_permutation(so_deltas, gi_deltas, n_layers)

    # Save
    save_results = {
        "gi_internal": a1,
        "cross_identity_cosines": {k: v for k, v in a2.items()},
        "pca_explained": {str(k): v["explained"] for k, v in a3.items()},
        "cross_probe": {str(k): v for k, v in a4.items()},
        "permutation": {str(k): {kk: vv for kk, vv in v.items() if kk != "perm_diffs"}
                       for k, v in a5.items()},
    }
    with open(RESULTS_DIR / "cross_identity_results.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nSaved cross_identity_results.json")

    if not args.skip_plots:
        print("\nGenerating figures...")
        plot_all(a1, a2, a3, a4, a5, so_deltas, gi_deltas, n_layers)

    print("\nDone.")


if __name__ == "__main__":
    main()