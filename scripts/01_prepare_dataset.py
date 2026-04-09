from __future__ import annotations

import argparse
from pathlib import Path

from bbqmi.io import read_csv, write_csv_dated
from bbqmi.schema import normalize_dataset


def main() -> int:
    p = argparse.ArgumentParser(description="Validate/normalize a bias analysis dataset CSV.")
    p.add_argument("--input_csv", type=str, required=True, help="Path to input CSV.")
    args = p.parse_args()

    df = read_csv(Path(args.input_csv))
    df2 = normalize_dataset(df)
    out_path = write_csv_dated(df2, dir_path=Path("data/processed"), stem="dataset_processed")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

