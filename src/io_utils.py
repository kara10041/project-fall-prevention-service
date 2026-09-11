from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


def read_csv_flexible(path: str | Path) -> pd.DataFrame:
    """UTF-8/CP949 CSV를 자동 판별해 읽는다."""
    path = Path(path)
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError(f"CSV 인코딩을 판별하지 못했습니다: {path}\n" + "\n".join(errors))


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"JSON으로 변환할 수 없는 타입: {type(value).__name__}")


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
