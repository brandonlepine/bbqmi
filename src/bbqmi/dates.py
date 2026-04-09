from __future__ import annotations

from datetime import date


def today_yyyy_mm_dd() -> str:
    return date.today().isoformat()

