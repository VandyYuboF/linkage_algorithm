#!/usr/bin/env python3
"""Split a "full name" column into first_name / last_name for preprocessing.

This repo's linkage pipeline expects `first_name` and `last_name` columns.
Some sites export only a single combined name field (e.g., `Patient`,
`patient_name`, `full_name`). This utility converts such inputs into a
linkage-ready CSV without discarding any other columns.

Behavior (heuristic)
--------------------
The splitter is intentionally conservative and deterministic:

1) If the name contains a comma ("LAST, FIRST ..."), treat the part before the
   comma as last_name and the first token after the comma as first_name.
2) Otherwise ("FIRST ... LAST"), treat the first token as first_name and the
   last token as last_name.
3) Common suffixes (jr, sr, ii, iii, iv, v) are removed when determining
   last_name.

The original full-name string is preserved (default output column: `full_name`).

Example
-------
python name_preprocess_split_full_name.py \
  --input litholink.csv \
  --full-name-col Patient \
  --out litholink.with_split_names.csv
"""

from __future__ import annotations

import argparse
import re
from typing import Tuple

import pandas as pd


_SPACE_RE = re.compile(r"\s+")
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _clean_name(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = _SPACE_RE.sub(" ", s)
    return s


def _strip_suffix(tokens: list[str]) -> list[str]:
    if not tokens:
        return tokens
    last = tokens[-1].lower().strip(".")
    if last in _SUFFIXES:
        return tokens[:-1]
    return tokens


def split_full_name(name: str) -> Tuple[str | None, str | None]:
    """Return (first_name, last_name) using the heuristic described above."""
    name = _clean_name(name)
    if not name:
        return None, None

    # Format: "LAST, FIRST MIDDLE"
    if "," in name:
        last_part, rest = name.split(",", 1)
        last = _clean_name(last_part) or None
        rest = _clean_name(rest)
        rest_tokens = _strip_suffix([t for t in rest.split(" ") if t])
        first = rest_tokens[0] if rest_tokens else None
        return first, last

    # Format: "FIRST MIDDLE LAST"
    tokens = _strip_suffix([t for t in name.split(" ") if t])
    if len(tokens) == 1:
        return tokens[0], None
    return tokens[0], tokens[-1]


def add_split_name_columns(
    df: pd.DataFrame,
    *,
    full_name_col: str,
    out_first_col: str = "first_name",
    out_last_col: str = "last_name",
    out_full_col: str = "full_name",
) -> pd.DataFrame:
    if full_name_col not in df.columns:
        raise ValueError(f"Input does not contain full-name column '{full_name_col}'. Present columns: {list(df.columns)}")

    out = df.copy()
    # Preserve original full name (optionally under a standardized column)
    if out_full_col != full_name_col:
        out[out_full_col] = out[full_name_col]

    parsed = out[full_name_col].apply(split_full_name)
    out[out_first_col] = parsed.apply(lambda x: x[0])
    out[out_last_col] = parsed.apply(lambda x: x[1])
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Split a full name column into first_name and last_name.")
    p.add_argument("--input", required=True, help="Input CSV path")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument(
        "--full-name-col",
        required=True,
        help="Column containing the full name (e.g., Patient, patient_name, full_name)",
    )
    p.add_argument("--out-first-col", default="first_name", help="Output first-name column (default: first_name)")
    p.add_argument("--out-last-col", default="last_name", help="Output last-name column (default: last_name)")
    p.add_argument("--out-full-col", default="full_name", help="Output preserved full-name column (default: full_name)")

    args = p.parse_args(argv)

    df = pd.read_csv(args.input)
    out = add_split_name_columns(
        df,
        full_name_col=args.full_name_col,
        out_first_col=args.out_first_col,
        out_last_col=args.out_last_col,
        out_full_col=args.out_full_col,
    )
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out):,} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
