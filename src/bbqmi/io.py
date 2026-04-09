from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bbqmi.dates import today_yyyy_mm_dd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dated_path(dir_path: str | Path, stem: str, suffix: str) -> Path:
    dir_p = ensure_dir(dir_path)
    return dir_p / f"{stem}_{today_yyyy_mm_dd()}{suffix}"


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_csv_dated(df: pd.DataFrame, dir_path: str | Path, stem: str) -> Path:
    out_path = dated_path(dir_path, stem=stem, suffix=".csv")
    df.to_csv(out_path, index=False)
    return out_path


def write_json_dated(obj: Any, dir_path: str | Path, stem: str) -> Path:
    out_path = dated_path(dir_path, stem=stem, suffix=".json")
    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def write_npz_dated(dir_path: str | Path, stem: str, **arrays: Any) -> Path:
    out_path = dated_path(dir_path, stem=stem, suffix=".npz")
    np.savez_compressed(out_path, **arrays)
    return out_path

