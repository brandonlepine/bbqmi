"""
analyze_fragmentation.py — Deep analysis of group-level representational fragmentation
========================================================================================
Deepens Analysis 4 finding: gay/lesbian and bisexual/pansexual identity terms
occupy qualitatively different directions in activation space.

Analyses:
  1. Group-specific DIM directions + pairwise cosine matrix
  2. Cross-group probe generalization (train on gay, test on bisexual, etc.)
  3. Norm-normalized deltas (control for residual stream scaling)
  4. PCA visualization of within-item deltas colored by group
  5. Permutation tests for alignment differences
  6. Item-level: does individual delta direction predict stereotype selection?

Usage:
  python scripts/analyze_fragmentation.py
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTIVATION_DIR = PROJECT_ROOT / "results" / "activations" / "so"
BEHAVIORAL_DIR = PROJECT_ROOT / "results" / "behavioral_pilot"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

IDENTITY_TERMS = {"gay", "lesbian", "bisexual", "pansexual", "straight"}

COLORS = {
    "gay": "#0072B2",
    "lesbian": "#CC79A7",
    "bisexual": "#E69F00",
    "pansexual": "#009E73",
    "straight": "#999999",
}

GROUP_MARKERS = {
    "gay": "o",
    "lesbian": "s",
    "bisexual": "^",
    "pansexual": "D",
}


# ---------------------------------------------------------------------------
# Data loading (same as v2)
# ---------------------------------------------------------------------------
def load_data(activation_dir, behavioral_path):
    with open(behavioral_path) as f:
        behavioral = json.load(f)
    behav_by_idx = {r["item_idx"]: r for r in behavioral}

    items = []
    for npz_path in sorted(activation_dir.glob("item_*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        meta = json.loads(str(data["metadata"]))
        idx = meta["item_idx"]
        behav = behav_by_idx.get(idx, {})

        term_to_tokens = defaultdict(list)
        for entry in meta["identity_terms_found"]:
            t = entry["term"].lower()
            if t in IDENTITY_TERMS:
                term_to_tokens[t].append(entry["token_indices"])

        all_identity_indices = meta["identity_token_indices"]
        hidden_identity_raw = data["hidden_identity"]
        n_layers = hidden_identity_raw.shape[0]
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


def compute_deltas(items, n_layers):
    """Compute within-item deltas: h(stereo_term) - h(non_stereo_term)."""
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

        # Raw delta
        delta = h_s - h_ns
        # Normalized delta (control for residual stream scaling)
        norms_mean = (np.linalg.norm(h_s, axis=1, keepdims=True) +
                      np.linalg.norm(h_ns, axis=1, keepdims=True)) / 2
        norms_mean = np.maximum(norms_mean, 1e-10)
        delta_normed = delta / norms_mean

        pred = item["predicted_letter"]
        pred_role = item["answer_roles"].get(pred, "")

        deltas.append({
            "delta": delta,
            "delta_normed": delta_normed,
            "h_stereo": h_s,
            "h_non_stereo": h_ns,
            "stereo_term": stereo_term,
            "non_stereo_term": non_stereo_term,
            "stereo_group": stereo_groups[0] if stereo_groups else "",
            "context_condition": item["context_condition"],
            "question_polarity": item["question_polarity"],
            "alignment": item["alignment"],
            "question": item["question"],
            "correct": item["correct"],
            "selected_stereotype": pred_role == "stereotyped_target",
        })
    return deltas


def cosine_sim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 1e-10 else 0.0


# ---------------------------------------------------------------------------
# Analysis 1: Group-specific DIM directions + pairwise cosine
# ---------------------------------------------------------------------------
def analysis_1_group_directions(deltas, n_layers):
    print("\n" + "=" * 60)
    print("  1. Group-specific DIM directions")
    print("=" * 60)

    groups = ["gay", "lesbian", "bisexual", "pansexual"]
    by_group = {g: [d for d in deltas if d["stereo_group"] == g] for g in groups}
    for g in groups:
        print(f"  {g}: n={len(by_group[g])}")

    # Compute mean delta direction per group per layer
    # Using normalized deltas to control for scaling
    group_directions = {}  # group -> (n_layers, hidden_dim)
    for g in groups:
        if not by_group[g]:
            continue
        stacked = np.stack([d["delta_normed"] for d in by_group[g]])  # (N, n_layers, hidden_dim)
        group_directions[g] = stacked.mean(axis=0)  # (n_layers, hidden_dim)

    # Pairwise cosine matrix per layer
    pair_cosines = {}  # (g1, g2) -> list of cosines per layer
    for g1, g2 in combinations(groups, 2):
        if g1 not in group_directions or g2 not in group_directions:
            continue
        cosines = []
        for layer in range(n_layers):
            cos = cosine_sim(group_directions[g1][layer], group_directions[g2][layer])
            cosines.append(cos)
        pair_cosines[(g1, g2)] = cosines

    # Print at key layers
    for layer in [5, 10, 15, 20, 25, 30, 35]:
        print(f"\n  Layer {layer} — pairwise cosine between group DIM directions:")
        for (g1, g2), cosines in sorted(pair_cosines.items()):
            print(f"    {g1:12s} ↔ {g2:12s}: {cosines[layer]:.3f}")

    return {"group_directions": group_directions, "pair_cosines": pair_cosines,
            "by_group": by_group, "groups": groups}


# ---------------------------------------------------------------------------
# Analysis 2: Cross-group probe generalization
# ---------------------------------------------------------------------------
def analysis_2_cross_probes(deltas, n_layers, target_layers=None):
    print("\n" + "=" * 60)
    print("  2. Cross-group probe generalization")
    print("=" * 60)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

    if target_layers is None:
        target_layers = [5, 10, 15, 20, 25, 30, 35]

    groups = ["gay", "lesbian", "bisexual", "pansexual"]
    by_group = {g: [d for d in deltas if d["stereo_group"] == g] for g in groups}

    # For each group, features = delta_normed at identity tokens,
    # label = 1 if model selected stereotype, 0 otherwise
    results = {}

    for layer in target_layers:
        matrix = np.full((len(groups), len(groups)), np.nan)

        for i, g_train in enumerate(groups):
            train_items = by_group[g_train]
            if len(train_items) < 20:
                continue

            X_train = np.stack([d["delta_normed"][layer] for d in train_items])
            y_train = np.array([int(d["selected_stereotype"]) for d in train_items])

            if len(np.unique(y_train)) < 2:
                continue

            n_comp = min(50, X_train.shape[0] - 1, X_train.shape[1])
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_comp)),
                ("clf", LogisticRegression(max_iter=1000, C=1.0)),
            ])
            pipe.fit(X_train, y_train)

            for j, g_test in enumerate(groups):
                test_items = by_group[g_test]
                if len(test_items) < 10:
                    continue
                X_test = np.stack([d["delta_normed"][layer] for d in test_items])
                y_test = np.array([int(d["selected_stereotype"]) for d in test_items])
                if len(np.unique(y_test)) < 2:
                    matrix[i, j] = pipe.score(X_test, y_test)
                else:
                    matrix[i, j] = pipe.score(X_test, y_test)

        results[layer] = {"matrix": matrix, "groups": groups}

        print(f"\n  Layer {layer}:")
        header = "  Train\\Test   " + "".join(f"{g:>12s}" for g in groups)
        print(header)
        for i, g in enumerate(groups):
            row = f"  {g:12s}  " + "".join(
                f"{matrix[i,j]:12.3f}" if not np.isnan(matrix[i, j]) else "         nan"
                for j in range(len(groups))
            )
            print(row)

    return results


# ---------------------------------------------------------------------------
# Analysis 3: Permutation test for alignment differences
# ---------------------------------------------------------------------------
def analysis_3_permutation_tests(deltas, n_layers, n_permutations=5000):
    print("\n" + "=" * 60)
    print("  3. Permutation tests for group alignment differences")
    print("=" * 60)

    groups = ["gay", "lesbian", "bisexual", "pansexual"]
    by_group = {g: [d for d in deltas if d["stereo_group"] == g] for g in groups}

    # Test: is the mean alignment of gay deltas significantly different
    # from bisexual deltas? (the key comparison)
    target_layers = [10, 15, 20, 25, 30]
    results = {}

    for layer in target_layers:
        # Compute overall mean direction
        all_deltas_layer = np.stack([d["delta_normed"][layer] for d in deltas])
        mean_dir = all_deltas_layer.mean(axis=0)
        mean_norm = np.linalg.norm(mean_dir)
        if mean_norm < 1e-10:
            continue
        mean_dir = mean_dir / mean_norm

        # Per-item alignment with mean direction
        alignments = {}
        for g in groups:
            g_deltas = np.stack([d["delta_normed"][layer] for d in by_group[g]])
            g_alignments = np.array([
                float(np.dot(d, mean_dir) / (np.linalg.norm(d) + 1e-10))
                for d in g_deltas
            ])
            alignments[g] = g_alignments

        # Test gay vs bisexual
        obs_diff = alignments["gay"].mean() - alignments["bisexual"].mean()

        # Permutation test
        combined = np.concatenate([alignments["gay"], alignments["bisexual"]])
        n_gay = len(alignments["gay"])
        rng = np.random.RandomState(42 + layer)
        perm_diffs = np.zeros(n_permutations)
        for p in range(n_permutations):
            perm = rng.permutation(len(combined))
            perm_diffs[p] = combined[perm[:n_gay]].mean() - combined[perm[n_gay:]].mean()

        p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff)))

        # Also test gay vs pansexual
        obs_diff_gp = alignments["gay"].mean() - alignments["pansexual"].mean()
        combined_gp = np.concatenate([alignments["gay"], alignments["pansexual"]])
        n_gay_gp = len(alignments["gay"])
        perm_diffs_gp = np.zeros(n_permutations)
        for p in range(n_permutations):
            perm = rng.permutation(len(combined_gp))
            perm_diffs_gp[p] = combined_gp[perm[:n_gay_gp]].mean() - combined_gp[perm[n_gay_gp:]].mean()
        p_value_gp = float(np.mean(np.abs(perm_diffs_gp) >= np.abs(obs_diff_gp)))

        # Bootstrap CI for each group's mean alignment
        bootstrap_cis = {}
        n_boot = 2000
        for g in groups:
            boot_means = np.zeros(n_boot)
            for b in range(n_boot):
                boot_sample = rng.choice(alignments[g], size=len(alignments[g]), replace=True)
                boot_means[b] = boot_sample.mean()
            ci_low = float(np.percentile(boot_means, 2.5))
            ci_high = float(np.percentile(boot_means, 97.5))
            bootstrap_cis[g] = (ci_low, ci_high)

        results[layer] = {
            "gay_mean": float(alignments["gay"].mean()),
            "bisexual_mean": float(alignments["bisexual"].mean()),
            "pansexual_mean": float(alignments["pansexual"].mean()),
            "lesbian_mean": float(alignments["lesbian"].mean()),
            "gay_vs_bisexual_diff": float(obs_diff),
            "gay_vs_bisexual_p": p_value,
            "gay_vs_pansexual_diff": float(obs_diff_gp),
            "gay_vs_pansexual_p": p_value_gp,
            "bootstrap_cis": {g: list(ci) for g, ci in bootstrap_cis.items()},
            "perm_diffs_gb": perm_diffs.tolist(),
        }

        print(f"\n  Layer {layer}:")
        for g in groups:
            ci = bootstrap_cis[g]
            print(f"    {g:12s}: alignment = {alignments[g].mean():.3f} "
                  f"[{ci[0]:.3f}, {ci[1]:.3f}]")
        print(f"    gay vs bisexual: Δ = {obs_diff:.3f}, p = {p_value:.4f}")
        print(f"    gay vs pansexual: Δ = {obs_diff_gp:.3f}, p = {p_value_gp:.4f}")

    return results


# ---------------------------------------------------------------------------
# Analysis 4: PCA visualization
# ---------------------------------------------------------------------------
def analysis_4_pca(deltas, n_layers, target_layers=None):
    print("\n" + "=" * 60)
    print("  4. PCA of within-item deltas")
    print("=" * 60)

    from sklearn.decomposition import PCA

    if target_layers is None:
        target_layers = [10, 20, 30]

    results = {}
    for layer in target_layers:
        all_normed = np.stack([d["delta_normed"][layer] for d in deltas])
        pca = PCA(n_components=5)
        coords = pca.fit_transform(all_normed)
        explained = pca.explained_variance_ratio_

        results[layer] = {
            "coords": coords,  # (N, 5)
            "explained": explained.tolist(),
            "groups": [d["stereo_group"] for d in deltas],
            "selected_stereotype": [d["selected_stereotype"] for d in deltas],
        }

        print(f"  Layer {layer}: explained variance = "
              f"PC1={explained[0]:.3f}, PC2={explained[1]:.3f}, "
              f"PC3={explained[2]:.3f}")

    return results


# ---------------------------------------------------------------------------
# Analysis 5: Item-level prediction
# ---------------------------------------------------------------------------
def analysis_5_item_level(deltas, a1_results, n_layers):
    """Does individual item delta direction predict stereotype selection?"""
    print("\n" + "=" * 60)
    print("  5. Item-level: delta direction predicts stereotype selection")
    print("=" * 60)

    groups = a1_results["groups"]
    group_dirs = a1_results["group_directions"]

    target_layers = [10, 15, 20, 25, 30]
    results = {}

    for layer in target_layers:
        # For each item, compute projection onto its own group's direction
        # and onto the "other family's" direction
        # gay/lesbian family vs bisexual/pansexual family

        # Compute family-level directions
        gl_deltas = [d["delta_normed"][layer] for d in deltas
                     if d["stereo_group"] in ["gay", "lesbian"]]
        bp_deltas = [d["delta_normed"][layer] for d in deltas
                     if d["stereo_group"] in ["bisexual", "pansexual"]]

        if not gl_deltas or not bp_deltas:
            continue

        gl_dir = np.mean(np.stack(gl_deltas), axis=0)
        gl_dir = gl_dir / (np.linalg.norm(gl_dir) + 1e-10)
        bp_dir = np.mean(np.stack(bp_deltas), axis=0)
        bp_dir = bp_dir / (np.linalg.norm(bp_dir) + 1e-10)

        cos_between_families = cosine_sim(gl_dir, bp_dir)

        # For each item: project onto own-family and other-family direction
        proj_own = []
        proj_other = []
        stereo_selected = []
        item_groups = []

        for d in deltas:
            delta = d["delta_normed"][layer]
            is_gl = d["stereo_group"] in ["gay", "lesbian"]

            if is_gl:
                proj_own.append(float(np.dot(delta, gl_dir)))
                proj_other.append(float(np.dot(delta, bp_dir)))
            else:
                proj_own.append(float(np.dot(delta, bp_dir)))
                proj_other.append(float(np.dot(delta, gl_dir)))

            stereo_selected.append(int(d["selected_stereotype"]))
            item_groups.append(d["stereo_group"])

        proj_own = np.array(proj_own)
        proj_other = np.array(proj_other)
        stereo_selected = np.array(stereo_selected)

        # Correlations
        corr_own = float(np.corrcoef(proj_own, stereo_selected)[0, 1])
        corr_other = float(np.corrcoef(proj_other, stereo_selected)[0, 1])

        # Split by family
        gl_mask = np.array([g in ["gay", "lesbian"] for g in item_groups])
        bp_mask = ~gl_mask

        corr_own_gl = float(np.corrcoef(proj_own[gl_mask], stereo_selected[gl_mask])[0, 1]) if gl_mask.sum() > 10 else np.nan
        corr_own_bp = float(np.corrcoef(proj_own[bp_mask], stereo_selected[bp_mask])[0, 1]) if bp_mask.sum() > 10 else np.nan

        results[layer] = {
            "cos_between_families": cos_between_families,
            "corr_own_direction": corr_own,
            "corr_other_direction": corr_other,
            "corr_own_gl_family": corr_own_gl,
            "corr_own_bp_family": corr_own_bp,
        }

        print(f"  Layer {layer}:")
        print(f"    Cosine between family directions: {cos_between_families:.3f}")
        print(f"    Projection → stereotype selection correlation:")
        print(f"      Own-family direction:   r = {corr_own:.3f}")
        print(f"      Other-family direction: r = {corr_other:.3f}")
        print(f"      GL items on GL dir:     r = {corr_own_gl:.3f}")
        print(f"      BP items on BP dir:     r = {corr_own_bp:.3f}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_all(a1, a2, a3, a4, a5, deltas, n_layers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    layers = list(range(n_layers))
    groups = ["gay", "lesbian", "bisexual", "pansexual"]

    # --- Figure 1: Pairwise cosine matrix across layers ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    pair_styles = {
        ("gay", "lesbian"): ("-", COLORS["gay"]),
        ("gay", "bisexual"): ("-", COLORS["bisexual"]),
        ("gay", "pansexual"): ("-", COLORS["pansexual"]),
        ("lesbian", "bisexual"): ("--", COLORS["bisexual"]),
        ("lesbian", "pansexual"): ("--", COLORS["pansexual"]),
        ("bisexual", "pansexual"): (":", COLORS["pansexual"]),
    }
    for (g1, g2), cosines in a1["pair_cosines"].items():
        style, color = pair_styles.get((g1, g2), ("-", "#999999"))
        ax.plot(layers, cosines, linestyle=style, color=color, linewidth=2,
                label=f"{g1} ↔ {g2}")
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cosine similarity", fontsize=12)
    ax.set_title("Pairwise cosine between group-specific identity directions\n"
                 "(normalized within-item deltas)", fontsize=12)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.6, 1.1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "frag_1_pairwise_cosine.png", dpi=150)
    plt.close(fig)
    print("  Saved frag_1_pairwise_cosine.png")

    # --- Figure 2: Permutation test visualization ---
    fig, axes = plt.subplots(1, len(a3), figsize=(5 * len(a3), 4))
    if len(a3) == 1:
        axes = [axes]
    for ax, (layer, info) in zip(axes, sorted(a3.items())):
        perm_diffs = np.array(info["perm_diffs_gb"])
        obs = info["gay_vs_bisexual_diff"]
        ax.hist(perm_diffs, bins=60, color="#CCCCCC", edgecolor="white", alpha=0.8)
        ax.axvline(obs, color=COLORS["gay"], linewidth=2.5,
                   label=f"Observed Δ = {obs:.3f}")
        ax.axvline(-obs, color=COLORS["gay"], linewidth=2.5, linestyle="--", alpha=0.5)
        ax.set_xlabel("Permuted difference (gay − bisexual)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"Layer {layer} (p = {info['gay_vs_bisexual_p']:.4f})", fontsize=11)
        ax.legend(fontsize=9)
    fig.suptitle("Permutation test: gay vs bisexual alignment difference", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "frag_2_permutation_test.png", dpi=150)
    plt.close(fig)
    print("  Saved frag_2_permutation_test.png")

    # --- Figure 3: Bootstrap CIs bar chart ---
    layer_for_bar = sorted(a3.keys())[len(a3) // 2]  # middle layer
    info = a3[layer_for_bar]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x_pos = np.arange(len(groups))
    means = [info[f"{g}_mean"] for g in groups]
    cis = [info["bootstrap_cis"][g] for g in groups]
    ci_low = [m - ci[0] for m, ci in zip(means, cis)]
    ci_high = [ci[1] - m for m, ci in zip(means, cis)]

    bars = ax.bar(x_pos, means, color=[COLORS[g] for g in groups],
                  edgecolor="white", linewidth=1.5, width=0.6)
    ax.errorbar(x_pos, means, yerr=[ci_low, ci_high],
                fmt="none", ecolor="black", capsize=5, linewidth=1.5)
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([g.capitalize() for g in groups], fontsize=12)
    ax.set_ylabel("Alignment with mean identity direction\n(cosine similarity)", fontsize=11)
    ax.set_title(f"Group-level alignment at layer {layer_for_bar}\n"
                 f"(95% bootstrap CI, n_boot=2000)", fontsize=12)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "frag_3_group_alignment_ci.png", dpi=150)
    plt.close(fig)
    print("  Saved frag_3_group_alignment_ci.png")

    # --- Figure 4: PCA scatter ---
    pca_layers = sorted(a4.keys())
    n_panels = len(pca_layers)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, layer in zip(axes, pca_layers):
        info = a4[layer]
        coords = info["coords"]
        grp_labels = info["groups"]
        ev = info["explained"]

        for g in groups:
            mask = [gl == g for gl in grp_labels]
            if not any(mask):
                continue
            idx = np.where(mask)[0]
            ax.scatter(coords[idx, 0], coords[idx, 1],
                      c=COLORS[g], marker=GROUP_MARKERS[g],
                      s=25, alpha=0.5, label=g, edgecolors="none")

        ax.set_xlabel(f"PC1 ({ev[0]:.1%} var)", fontsize=11)
        ax.set_ylabel(f"PC2 ({ev[1]:.1%} var)", fontsize=11)
        ax.set_title(f"Layer {layer}", fontsize=12)
        ax.legend(fontsize=9, markerscale=1.5)
        ax.grid(True, alpha=0.2)

    fig.suptitle("PCA of normalized within-item identity deltas", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "frag_4_pca_scatter.png", dpi=150)
    plt.close(fig)
    print("  Saved frag_4_pca_scatter.png")

    # --- Figure 5: Cross-group probe heatmaps ---
    probe_layers = sorted(a2.keys())
    n_panels = min(3, len(probe_layers))
    selected_layers = [probe_layers[i * len(probe_layers) // n_panels]
                      for i in range(n_panels)]

    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    for ax, layer in zip(axes, selected_layers):
        matrix = a2[layer]["matrix"]
        im = ax.imshow(matrix, vmin=0.3, vmax=0.85, cmap="RdYlBu_r")
        ax.set_xticks(range(len(groups)))
        ax.set_yticks(range(len(groups)))
        ax.set_xticklabels([g.capitalize() for g in groups], fontsize=10)
        ax.set_yticklabels([g.capitalize() for g in groups], fontsize=10)
        ax.set_xlabel("Test group", fontsize=10)
        ax.set_ylabel("Train group", fontsize=10)
        ax.set_title(f"Layer {layer}", fontsize=11)

        # Annotate cells
        for i in range(len(groups)):
            for j in range(len(groups)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = "white" if val > 0.65 or val < 0.4 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                           fontsize=9, color=color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Cross-group probe generalization\n"
                 "(train stereotype-selection probe on one group, test on another)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "frag_5_cross_probe.png", dpi=150)
    plt.close(fig)
    print("  Saved frag_5_cross_probe.png")

    # --- Figure 6: Family direction cosine across layers ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fam_cosines = [a5[l]["cos_between_families"] for l in sorted(a5.keys())]
    fam_layers = sorted(a5.keys())
    ax.plot(fam_layers, fam_cosines, color=COLORS["gay"], linewidth=2.5,
            marker="o", markersize=6)
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cosine similarity", fontsize=12)
    ax.set_title("Cosine between gay/lesbian family direction and\n"
                 "bisexual/pansexual family direction (normalized deltas)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "frag_6_family_cosine.png", dpi=150)
    plt.close(fig)
    print("  Saved frag_6_family_cosine.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation_dir", type=Path, default=ACTIVATION_DIR)
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

    global FIGURES_DIR, RESULTS_DIR, BEHAVIORAL_DIR
    FIGURES_DIR = subdirs.figures_dir
    RESULTS_DIR = subdirs.analysis_dir
    BEHAVIORAL_DIR = subdirs.behavioral_dir

    print(f"Run: model_id={model_id}  run_date={run_date}")
    print(f"Run dir: {run_dir}")
    print(f"Activations (SO): {subdirs.activations_so_dir}")
    print(f"Behavioral dir: {BEHAVIORAL_DIR}")
    print(f"Analysis outputs: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    behavioral_path = BEHAVIORAL_DIR / "behavioral_results.json"
    if not behavioral_path.exists():
        behavioral_candidates = sorted(BEHAVIORAL_DIR.glob("behavioral_results*.json"))
        if not behavioral_candidates:
            print(f"ERROR: No behavioral results in {BEHAVIORAL_DIR}")
            return
        behavioral_path = behavioral_candidates[-1]
    print(f"Using: {behavioral_path}")

    if args.activation_dir == ACTIVATION_DIR:
        args.activation_dir = subdirs.activations_so_dir

    items = load_data(args.activation_dir, behavioral_path)
    n_layers = items[0]["hidden_final"].shape[0]

    deltas = compute_deltas(items, n_layers)
    print(f"Computed {len(deltas)} within-item deltas")

    a1 = analysis_1_group_directions(deltas, n_layers)
    a2 = analysis_2_cross_probes(deltas, n_layers)
    a3 = analysis_3_permutation_tests(deltas, n_layers)
    a4 = analysis_4_pca(deltas, n_layers)
    a5 = analysis_5_item_level(deltas, a1, n_layers)

    # Save results (exclude large arrays)
    save_results = {
        "pair_cosines": {f"{g1}_{g2}": v for (g1, g2), v in a1["pair_cosines"].items()},
        "cross_probes": {str(k): {"matrix": v["matrix"].tolist(), "groups": v["groups"]}
                        for k, v in a2.items()},
        "permutation_tests": {str(k): {kk: vv for kk, vv in v.items() if kk != "perm_diffs_gb"}
                             for k, v in a3.items()},
        "pca_explained": {str(k): v["explained"] for k, v in a4.items()},
        "item_level": {str(k): v for k, v in a5.items()},
    }
    with open(RESULTS_DIR / "fragmentation_results.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved fragmentation_results.json")

    if not args.skip_plots:
        print("\nGenerating figures...")
        plot_all(a1, a2, a3, a4, a5, deltas, n_layers)

    print("\nDone.")


if __name__ == "__main__":
    main()