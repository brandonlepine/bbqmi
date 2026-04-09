from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from bbqmi.io import write_csv_dated, write_json_dated


def main() -> int:
    p = argparse.ArgumentParser(description="Fit a simple probe predicting group from hidden states.")
    p.add_argument("--hidden_states_npz", type=str, required=True, help="NPZ from 02_extract_hidden_states.py")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()

    data = np.load(Path(args.hidden_states_npz), allow_pickle=True)
    X = data["hidden_states"].astype(np.float32)
    group = data["group"].astype(object).tolist()

    # Encode group labels
    uniq = sorted(set(group))
    y = np.asarray([uniq.index(g) for g in group], dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    clf = LogisticRegression(max_iter=2000, n_jobs=None, multi_class="auto")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    metrics = {
        "n_examples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "labels": uniq,
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_f1_macro": float(f1_score(y_test, pred, average="macro")),
    }

    coef = clf.coef_
    coef_df = pd.DataFrame(coef, index=[f"group={g}" for g in uniq])

    write_json_dated(metrics, dir_path=Path("outputs"), stem="probe_metrics")
    write_csv_dated(coef_df.reset_index(names="class"), dir_path=Path("outputs"), stem="probe_coefficients")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

