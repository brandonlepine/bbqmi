from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


def _sanitize_segment(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-") or "run"


def get_model_id(*, model_path: Path | None, model_id_arg: str | None) -> str:
    if model_id_arg:
        return _sanitize_segment(model_id_arg)
    if model_path is None:
        return "unknown-model"
    return _sanitize_segment(model_path.name)


def get_run_date(run_date_arg: str | None) -> str:
    return run_date_arg or date.today().isoformat()


def get_run_dir(*, project_root: Path, model_id: str, run_date: str) -> Path:
    return project_root / "results" / "runs" / model_id / run_date


def newest_run_dir(*, project_root: Path, model_id: str) -> Path | None:
    base = project_root / "results" / "runs" / model_id
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        return None
    # Prefer lexicographic sort (YYYY-MM-DD), fallback to mtime if unexpected names
    try:
        return sorted(candidates, key=lambda p: p.name)[-1]
    except Exception:
        return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def resolve_run_dir(
    *,
    project_root: Path,
    run_dir_arg: Path | None,
    model_path: Path | None,
    model_id_arg: str | None,
    run_date_arg: str | None,
    must_exist: bool = False,
) -> tuple[Path, str, str]:
    model_id = get_model_id(model_path=model_path, model_id_arg=model_id_arg)
    run_date = get_run_date(run_date_arg)

    if run_dir_arg is not None:
        run_dir = Path(run_dir_arg)
    else:
        run_dir = get_run_dir(project_root=project_root, model_id=model_id, run_date=run_date)
        if not run_dir.exists():
            fallback = newest_run_dir(project_root=project_root, model_id=model_id)
            if fallback is not None:
                run_dir = fallback
                run_date = run_dir.name

    if must_exist and not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    return run_dir, model_id, run_date


@dataclass(frozen=True)
class RunSubdirs:
    run_dir: Path
    analysis_dir: Path
    figures_dir: Path
    behavioral_dir: Path
    activations_dir: Path
    activations_so_dir: Path
    activations_gi_dir: Path


def ensure_run_subdirs(run_dir: Path) -> RunSubdirs:
    analysis_dir = run_dir / "analysis"
    figures_dir = run_dir / "figures"
    behavioral_dir = run_dir / "behavioral_pilot"
    activations_dir = run_dir / "activations"
    activations_so_dir = activations_dir / "so"
    activations_gi_dir = activations_dir / "gi"

    for p in [analysis_dir, figures_dir, behavioral_dir, activations_so_dir, activations_gi_dir]:
        p.mkdir(parents=True, exist_ok=True)

    return RunSubdirs(
        run_dir=run_dir,
        analysis_dir=analysis_dir,
        figures_dir=figures_dir,
        behavioral_dir=behavioral_dir,
        activations_dir=activations_dir,
        activations_so_dir=activations_so_dir,
        activations_gi_dir=activations_gi_dir,
    )


def update_run_metadata(*, run_dir: Path, step: str, payload: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "metadata.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    else:
        existing = {}

    steps = existing.get("steps")
    if not isinstance(steps, dict):
        steps = {}
    steps[step] = payload
    existing["steps"] = steps

    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

