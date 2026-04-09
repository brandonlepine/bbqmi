from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class DatasetSchema:
    required_columns: tuple[str, ...] = ("id", "text", "group")
    optional_columns: tuple[str, ...] = ("target", "metadata_json")

    @property
    def all_columns(self) -> tuple[str, ...]:
        return self.required_columns + self.optional_columns


def require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")


def normalize_dataset(df: pd.DataFrame, schema: DatasetSchema | None = None) -> pd.DataFrame:
    schema = schema or DatasetSchema()
    require_columns(df, schema.required_columns)

    out = df.copy()
    out["id"] = out["id"].astype(str)
    out["text"] = out["text"].astype(str)
    out["group"] = out["group"].astype(str)

    for c in schema.optional_columns:
        if c not in out.columns:
            out[c] = ""
        else:
            out[c] = out[c].fillna("").astype(str)

    return out[list(schema.all_columns)]

