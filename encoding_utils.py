# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


MOJIBAKE_MARKERS = ["\u00d0", "\u00d1", "\u00c2", "\u0420", "\u0421\u0453", "\u0421\u201a", "\u0420\u00b0"]
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251", "latin1")


def _mojibake_score(text: str) -> int:
    return (
        sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
        + text.count("?") * 2
        + sum(1 for char in text if ord(char) < 32 and char not in "\t\n\r")
    )


def _recover_cp1252_mojibake(text: str) -> str | None:
    raw = bytearray()
    for char in text:
        codepoint = ord(char)
        if codepoint <= 255:
            raw.append(codepoint)
            continue
        try:
            encoded = char.encode("cp1252")
        except Exception:
            return None
        if len(encoded) != 1:
            return None
        raw.extend(encoded)
    try:
        return raw.decode("utf-8")
    except Exception:
        return None


def fix_mojibake(text: str) -> str:
    if not isinstance(text, str):
        return text
    markers = MOJIBAKE_MARKERS
    if not any(marker in text for marker in markers):
        return text
    candidates = [text]
    for enc in ["latin1", "cp1252"]:
        try:
            candidates.append(text.encode(enc).decode("utf-8"))
        except Exception:
            pass
    recovered = _recover_cp1252_mojibake(text)
    if recovered is not None:
        candidates.append(recovered)
    if candidates:
        # выбрать вариант с меньшим количеством mojibake-маркеров
        def score(s):
            return _mojibake_score(s)

        best = min(candidates, key=score)
        if score(best) <= score(text):
            return best
    return text


def clean_dataframe_text(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with mojibake repaired in text columns and column names."""
    cleaned = df.copy()
    cleaned.columns = [fix_mojibake(str(column)) for column in cleaned.columns]
    object_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        cleaned[column] = cleaned[column].map(lambda value: fix_mojibake(value) if isinstance(value, str) else value)
    return cleaned


def safe_read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read CSV with common encodings and repair mojibake in text fields."""
    path = Path(path)
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=encoding, **kwargs)
            return clean_dataframe_text(df)
        except UnicodeDecodeError as exc:
            last_error = exc
        except Exception:
            raise
    if last_error is not None:
        raise last_error
    return clean_dataframe_text(pd.read_csv(path, **kwargs))
