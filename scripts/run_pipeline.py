"""
run_pipeline.py — Orchestrate the BBQ MI pipeline end-to-end
============================================================

This script is a thin orchestrator that runs the repo's existing CLI scripts in
the correct order, wiring through the shared run directory convention:

  results/runs/<model_id>/<run_date>/
    activations/{so,gi}/...
    behavioral_pilot/...
    analysis/...
    figures/...

It is intended for RunPod/H100 (or any GPU box) so you can run one command per
model (or a list of models) to produce a complete run folder.

Typical usage (single model):
  python scripts/run_pipeline.py --device cuda --model_path /workspace/bbqmi/models/llama2-13b

Multi-model usage:
  python scripts/run_pipeline.py --device cuda \
    --model_path /workspace/bbqmi/models/llama2-13b \
    --model_path /workspace/bbqmi/models/llama2-7b
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence
from subprocess import CalledProcessError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class ModelSpec:
    model_path: Path
    model_id: str


def _today() -> str:
    return date.today().isoformat()


def _infer_model_id(model_path: Path) -> str:
    # Keep in sync with bbqmi.run_paths.get_model_id default behavior.
    return model_path.name


def _print_log_tail(log_path: Path, n_lines: int = 120) -> None:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        tail = lines[-n_lines:] if len(lines) > n_lines else lines
        if tail:
            print("\n--- pipeline.log (tail) ---", file=sys.stderr)
            print("\n".join(tail), file=sys.stderr)
            print("--- end tail ---\n", file=sys.stderr)
    except Exception as e:
        print(f"(Could not read log tail from {log_path}: {e})", file=sys.stderr)


def _run_cmd(*, cmd: Sequence[str], log_path: Path | None, dry_run: bool) -> None:
    cmd_str = " ".join(cmd)
    print(f"\n$ {cmd_str}", flush=True)
    if dry_run:
        return
    if log_path is None:
        subprocess.run(cmd, check=True)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n$ {cmd_str}\n")
        f.flush()
        try:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
        except CalledProcessError as e:
            f.flush()
            _print_log_tail(log_path)
            raise e


def _stimuli_paths_for_date(run_date: str) -> tuple[Path, Path, Path]:
    so = DATA_PROCESSED / f"stimuli_so_{run_date}.json"
    gi = DATA_PROCESSED / f"stimuli_gi_{run_date}.json"
    summary = DATA_PROCESSED / f"stimuli_summary_{run_date}.txt"
    return so, gi, summary


def _ensure_stimuli(*, run_date: str, bbq_dir: Path | None, log_path: Path | None) -> tuple[Path, Path]:
    so_path, gi_path, _ = _stimuli_paths_for_date(run_date)
    if so_path.exists() and gi_path.exists():
        print(f"Stimuli already exist for {run_date}: {so_path.name}, {gi_path.name}")
        return so_path, gi_path

    cmd = [sys.executable, str(SCRIPTS_DIR / "prepare_stimuli.py"), "--run_date", run_date]
    if bbq_dir is not None:
        cmd.extend(["--bbq_dir", str(bbq_dir)])
    _run_cmd(cmd=cmd, log_path=log_path, dry_run=False)

    if not so_path.exists() or not gi_path.exists():
        raise FileNotFoundError(
            "prepare_stimuli.py did not produce expected outputs: "
            f"{so_path} and {gi_path}"
        )
    return so_path, gi_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Orchestrate BBQ MI pipeline end-to-end.")

    # Model(s)
    p.add_argument(
        "--model_path",
        type=Path,
        action="append",
        required=True,
        help="Path to a local HF model directory. Pass multiple times for multi-model runs.",
    )
    p.add_argument(
        "--model_id",
        type=str,
        action="append",
        default=None,
        help="Optional model id override. If provided multiple times, must align with --model_path order.",
    )

    # Shared run config
    p.add_argument("--device", type=str, default="cuda", choices=["mps", "cuda", "cpu", "auto"])
    p.add_argument("--run_date", type=str, default=None, help="Run date (YYYY-MM-DD). Defaults to today.")
    p.add_argument("--bbq_dir", type=Path, default=None, help="Optional override for BBQ data/ directory.")

    # Scale knobs passed through
    p.add_argument("--max_items", type=int, default=None, help="Limit number of items for quick tests.")
    p.add_argument("--alpha", type=float, default=14.0, help="Intervention strength for ablation analyses.")
    p.add_argument("--target_layer", type=int, default=20, help="Intervention layer for ablation analyses.")
    p.add_argument("--attn_max_items", type=int, default=200, help="Max items for attention-only causal analysis.")

    # Pipeline selection
    p.add_argument("--skip_behavioral", action="store_true")
    p.add_argument("--skip_activations", action="store_true")
    p.add_argument("--skip_analysis", action="store_true")
    p.add_argument("--skip_causal", action="store_true", help="Skip scripts/causal_analysis.py")
    p.add_argument("--skip_head_ablation", action="store_true", help="Skip scripts/ablate_heads.py")
    p.add_argument("--skip_backfire", action="store_true", help="Skip scripts/analyze_backfire.py")

    p.add_argument("--include_gi", action="store_true", help="Also extract GI activations and run GI analyses.")
    p.add_argument("--skip_gi_deep", action="store_true", help="Skip scripts/analyze_gi_deep.py (if include_gi).")
    p.add_argument("--skip_cross_identity", action="store_true", help="Skip scripts/analyze_cross_identity.py (if include_gi).")

    # Logging
    p.add_argument("--no_logs", action="store_true", help="Do not tee subprocess output into run_dir logs.")
    p.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    p.add_argument(
        "--skip_stimuli",
        action="store_true",
        help="Do not run prepare_stimuli.py; require dated stimuli files already exist in data/processed/.",
    )

    return p.parse_args()


def _build_model_specs(model_paths: list[Path], model_ids: list[str] | None) -> list[ModelSpec]:
    if model_ids is None:
        return [ModelSpec(model_path=mp, model_id=_infer_model_id(mp)) for mp in model_paths]
    if len(model_ids) != len(model_paths):
        raise ValueError("If you pass --model_id, you must pass it once per --model_path (same order).")
    return [ModelSpec(model_path=mp, model_id=mid) for mp, mid in zip(model_paths, model_ids)]


def main() -> int:
    args = _parse_args()
    run_date = args.run_date or _today()
    specs = _build_model_specs(args.model_path, args.model_id)

    # Prepare stimuli once (shared across models for this run_date)
    global_log = None if args.no_logs else (PROJECT_ROOT / "results" / "pipeline_logs" / f"pipeline_stimuli_{run_date}.log")
    so_stimuli, gi_stimuli, _ = _stimuli_paths_for_date(run_date)
    if args.skip_stimuli:
        if not so_stimuli.exists() or not gi_stimuli.exists():
            raise FileNotFoundError(
                f"--skip_stimuli was set, but expected stimuli files are missing: "
                f"{so_stimuli} and/or {gi_stimuli}"
            )
        print(f"Using existing stimuli for {run_date}: {so_stimuli.name}, {gi_stimuli.name}")
    elif args.dry_run:
        print(
            f"[dry_run] Would ensure stimuli exist for {run_date}: "
            f"{so_stimuli.name}, {gi_stimuli.name}"
        )
    else:
        so_stimuli, gi_stimuli = _ensure_stimuli(run_date=run_date, bbq_dir=args.bbq_dir, log_path=global_log)

    for spec in specs:
        print("\n" + "=" * 90)
        print(f"MODEL: {spec.model_id}")
        print(f"PATH:  {spec.model_path}")
        print(f"DATE:  {run_date}")
        print("=" * 90)

        # Per-model log goes into the run directory itself.
        run_dir = PROJECT_ROOT / "results" / "runs" / spec.model_id / run_date
        log_path = None if args.no_logs else (run_dir / "pipeline.log")

        # 1) Behavioral pilot (SO)
        if not args.skip_behavioral:
            cmd = [
                sys.executable,
                str(SCRIPTS_DIR / "behavioral_pilot.py"),
                "--device",
                args.device,
                "--model_path",
                str(spec.model_path),
                "--model_id",
                spec.model_id,
                "--run_date",
                run_date,
                "--run_dir",
                str(run_dir),
                "--stimuli_json",
                str(so_stimuli),
            ]
            if args.max_items is not None:
                cmd.extend(["--max_items", str(args.max_items)])
            _run_cmd(cmd=cmd, log_path=log_path, dry_run=args.dry_run)

        # 2) Activation extraction (SO, optional GI)
        if not args.skip_activations:
            cmd = [
                sys.executable,
                str(SCRIPTS_DIR / "extract_activations.py"),
                "--device",
                args.device,
                "--model_path",
                str(spec.model_path),
                "--model_id",
                spec.model_id,
                "--run_date",
                run_date,
                "--run_dir",
                str(run_dir),
                "--stimuli",
                so_stimuli.name,
                "--output_subdir",
                "so",
            ]
            if args.max_items is not None:
                cmd.extend(["--max_items", str(args.max_items)])
            _run_cmd(cmd=cmd, log_path=log_path, dry_run=args.dry_run)

            if args.include_gi:
                cmd = [
                    sys.executable,
                    str(SCRIPTS_DIR / "extract_activations.py"),
                    "--device",
                    args.device,
                    "--model_path",
                    str(spec.model_path),
                    "--model_id",
                    spec.model_id,
                    "--run_date",
                    run_date,
                    "--run_dir",
                    str(run_dir),
                    "--stimuli",
                    gi_stimuli.name,
                    "--output_subdir",
                    "gi",
                ]
                if args.max_items is not None:
                    cmd.extend(["--max_items", str(args.max_items)])
                _run_cmd(cmd=cmd, log_path=log_path, dry_run=args.dry_run)

        # 3) Analyses (SO + optional GI)
        if not args.skip_analysis:
            # SO analyses (no model needed)
            _run_cmd(
                cmd=[
                    sys.executable,
                    str(SCRIPTS_DIR / "analyze_representations.py"),
                    "--model_path",
                    str(spec.model_path),
                    "--model_id",
                    spec.model_id,
                    "--run_date",
                    run_date,
                    "--run_dir",
                    str(run_dir),
                ],
                log_path=log_path,
                dry_run=args.dry_run,
            )
            _run_cmd(
                cmd=[
                    sys.executable,
                    str(SCRIPTS_DIR / "analyze_fragmentation.py"),
                    "--model_path",
                    str(spec.model_path),
                    "--model_id",
                    spec.model_id,
                    "--run_date",
                    run_date,
                    "--run_dir",
                    str(run_dir),
                ],
                log_path=log_path,
                dry_run=args.dry_run,
            )
            _run_cmd(
                cmd=[
                    sys.executable,
                    str(SCRIPTS_DIR / "analyze_decomposition.py"),
                    "--decomp_only",
                    "--model_path",
                    str(spec.model_path),
                    "--model_id",
                    spec.model_id,
                    "--run_date",
                    run_date,
                    "--run_dir",
                    str(run_dir),
                ],
                log_path=log_path,
                dry_run=args.dry_run,
            )

            # SO analyses (requires model)
            _run_cmd(
                cmd=[
                    sys.executable,
                    str(SCRIPTS_DIR / "analyze_decomposition.py"),
                    "--ablation_only",
                    "--device",
                    args.device,
                    "--model_path",
                    str(spec.model_path),
                    "--model_id",
                    spec.model_id,
                    "--run_date",
                    run_date,
                    "--run_dir",
                    str(run_dir),
                    "--alpha",
                    str(args.alpha),
                    "--target_layer",
                    str(args.target_layer),
                ]
                + (["--max_items", str(args.max_items)] if args.max_items is not None else []),
                log_path=log_path,
                dry_run=args.dry_run,
            )

            if not args.skip_backfire:
                _run_cmd(
                    cmd=[
                        sys.executable,
                        str(SCRIPTS_DIR / "analyze_backfire.py"),
                        "--model_path",
                        str(spec.model_path),
                        "--model_id",
                        spec.model_id,
                        "--run_date",
                        run_date,
                        "--run_dir",
                        str(run_dir),
                    ],
                    log_path=log_path,
                    dry_run=args.dry_run,
                )

            if not args.skip_causal:
                _run_cmd(
                    cmd=[
                        sys.executable,
                        str(SCRIPTS_DIR / "causal_analysis.py"),
                        "--analysis",
                        "all",
                        "--device",
                        args.device,
                        "--model_path",
                        str(spec.model_path),
                        "--model_id",
                        spec.model_id,
                        "--run_date",
                        run_date,
                        "--run_dir",
                        str(run_dir),
                        "--alpha",
                        str(args.alpha),
                        "--target_layer",
                        str(args.target_layer),
                        "--attn_max_items",
                        str(args.attn_max_items),
                    ]
                    + (["--max_items", str(args.max_items)] if args.max_items is not None else []),
                    log_path=log_path,
                    dry_run=args.dry_run,
                )

            if not args.skip_head_ablation:
                _run_cmd(
                    cmd=[
                        sys.executable,
                        str(SCRIPTS_DIR / "ablate_heads.py"),
                        "--device",
                        args.device,
                        "--model_path",
                        str(spec.model_path),
                        "--model_id",
                        spec.model_id,
                        "--run_date",
                        run_date,
                        "--run_dir",
                        str(run_dir),
                    ]
                    + (["--max_items", str(args.max_items)] if args.max_items is not None else []),
                    log_path=log_path,
                    dry_run=args.dry_run,
                )

            # GI analyses
            if args.include_gi and not args.skip_cross_identity:
                _run_cmd(
                    cmd=[
                        sys.executable,
                        str(SCRIPTS_DIR / "analyze_cross_identity.py"),
                        "--model_path",
                        str(spec.model_path),
                        "--model_id",
                        spec.model_id,
                        "--run_date",
                        run_date,
                        "--run_dir",
                        str(run_dir),
                    ],
                    log_path=log_path,
                    dry_run=args.dry_run,
                )
            if args.include_gi and not args.skip_gi_deep:
                _run_cmd(
                    cmd=[
                        sys.executable,
                        str(SCRIPTS_DIR / "analyze_gi_deep.py"),
                        "--analysis",
                        "all",
                        "--device",
                        args.device,
                        "--model_path",
                        str(spec.model_path),
                        "--model_id",
                        spec.model_id,
                        "--run_date",
                        run_date,
                        "--run_dir",
                        str(run_dir),
                        "--alpha",
                        str(args.alpha),
                        "--target_layer",
                        str(args.target_layer),
                    ]
                    + (["--max_items", str(args.max_items)] if args.max_items is not None else []),
                    log_path=log_path,
                    dry_run=args.dry_run,
                )

        print(f"\nFinished model {spec.model_id}. Run dir: {run_dir}")

    print("\nAll requested models completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

