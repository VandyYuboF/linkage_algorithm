#!/usr/bin/env python3
"""
Record linkage utility for cross-site patient matching.

Features
--------
- Blocking to reduce comparisons (default: ['dob', 'sex'])
- Jaro–Winkler string matching on first/last name
- Exact matching on DOB/sex
- Tunable thresholds and rules
- Light preprocessing (lowercasing, trimming, punctuation removal)
- Returns a scored pairs DataFrame and can write to CSV

Usage (CLI)
-----------
python linkage_algorithm.py \
  --left left.csv --right right.csv \
  --left-id person_id --right-id person_id \
  --out matches.csv

Or import and call `run_linkage(...)`.

Columns expected in each dataset (configurable via kwargs):
- first_name, last_name, dob (YYYY-MM-DD), sex
"""

from __future__ import annotations
import argparse
import logging
import re
from typing import Iterable, List, Mapping, Optional, Tuple

import pandas as pd
import recordlinkage


# -----------------------------
# Helpers
# -----------------------------
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)

def _normalize_name(s: pd.Series) -> pd.Series:
    """Lowercase, strip, remove punctuation and extra spaces."""
    s = s.astype(str).str.lower().str.strip()
    s = s.apply(lambda x: _PUNCT_RE.sub("", x))
    s = s.str.replace(r"\s+", " ", regex=True)
    return s

def _normalize_sex(s: pd.Series) -> pd.Series:
    """Map sex to {F, M, O, U} where possible."""
    m = {
        "female": "F", "f": "F", "woman": "F",
        "male": "M", "m": "M", "man": "M",
        "other": "O", "o": "O",
        "unknown": "U", "u": "U", "undifferentiated": "U",
    }
    s = s.astype(str).str.lower().str.strip()
    return s.map(m).fillna(s.str.upper())


def _prep(
    df: pd.DataFrame,
    colmap: Mapping[str, str],
    id_col: str,
) -> pd.DataFrame:
    """
    Standardize required columns: first_name, last_name, dob, sex, plus an ID.
    Returns a copy with standardized names.
    """
    req = ["first_name", "last_name", "dob", "sex"]
    missing = [k for k in req if k not in colmap]
    if missing:
        raise ValueError(f"Missing required column mappings: {missing}")

    out = df.copy()
    # Rename to standard names
    out = out.rename(columns={colmap[k]: k for k in colmap})
    if id_col not in out.columns:
        raise ValueError(f"ID column '{id_col}' not found in dataframe")

    # Normalize fields
    out["first_name"] = _normalize_name(out["first_name"].fillna(""))
    out["last_name"] = _normalize_name(out["last_name"].fillna(""))
    # Ensure dob is ISO-like string for exact match blocking
    out["dob"] = pd.to_datetime(out["dob"], errors="coerce").dt.date.astype(str)
    out["sex"] = _normalize_sex(out["sex"].fillna("U"))
    return out


# -----------------------------
# Core function
# -----------------------------
def run_linkage(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    *,
    left_id: str,
    right_id: str,
    left_colmap: Mapping[str, str],
    right_colmap: Mapping[str, str],
    block_on: Iterable[str] = ("dob", "sex"),
    jw_threshold_first: float = 0.85,
    jw_threshold_last: float = 0.85,
    min_rule_sum: int = 3,
    return_all_features: bool = False,
) -> pd.DataFrame:
    """
    Perform record linkage between two datasets.

    Parameters
    ----------
    df_left, df_right : DataFrames with identifiable columns.
    left_id, right_id : Unique identifier column names for left/right.
    left_colmap, right_colmap : dicts mapping standard keys to your columns,
        e.g. {'first_name': 'fname', 'last_name': 'lname', 'dob': 'birth_date', 'sex': 'sex'}
    block_on : fields to block on (subset of ['dob','sex'])
    jw_threshold_first, jw_threshold_last : Jaro–Winkler thresholds for names
    min_rule_sum : require at least this many rules to be satisfied (e.g., 3 of 4)
    return_all_features : if True, returns all candidate pairs with feature values and sum;
                          if False, returns only pairs passing the min_rule_sum filter.

    Returns
    -------
    DataFrame with columns:
      - left_index  (left_id)
      - right_index (right_id)
      - first_name, last_name, dob, sex (feature flags or similarities per recordlinkage)
      - sum (integer count of satisfied rules)
    """
    # Prepare data
    L = _prep(df_left, left_colmap, left_id).set_index(left_id, drop=False)
    R = _prep(df_right, right_colmap, right_id).set_index(right_id, drop=False)

    # Indexing (blocking) to reduce comparisons
    indexer = recordlinkage.Index()
    for key in block_on:
        if key not in ("dob", "sex"):
            raise ValueError(f"Unsupported block key '{key}'. Use 'dob' and/or 'sex'.")
    if block_on:
        indexer.block(list(block_on))
    else:
        # Warning: full comparison can be very slow on big data
        indexer.full()

    candidate_links = indexer.index(L, R)

    # Define comparisons
    compare = recordlinkage.Compare()

    compare.string(
        "first_name", "first_name",
        method="jarowinkler", threshold=jw_threshold_first, label="first_name"
    )
    compare.string(
        "last_name", "last_name",
        method="jarowinkler", threshold=jw_threshold_last, label="last_name"
    )
    compare.exact("dob", "dob", label="dob")
    compare.exact("sex", "sex", label="sex")

    # Compute feature matrix
    features = compare.compute(candidate_links, L, R)

    # Sum satisfied rules (recordlinkage encodes booleans as {0,1})
    features["sum"] = features.sum(axis=1)

    if return_all_features:
        out = features.reset_index()
    else:
        out = features.loc[features["sum"] >= min_rule_sum].reset_index()

    # Replace generic names with the actual ID column names for clarity
    out = out.rename(columns={"level_0": "left_index", "level_1": "right_index"})
    out = out.rename(columns={"left_index": left_id, "right_index": right_id})

    return out


# -----------------------------
# CLI
# -----------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run record linkage between two CSVs.")
    p.add_argument("--left", required=True, help="Path to left CSV")
    p.add_argument("--right", required=True, help="Path to right CSV")
    p.add_argument("--out", required=True, help="Path to write matches CSV")
    p.add_argument("--left-id", required=True, help="Unique ID column in left CSV")
    p.add_argument("--right-id", required=True, help="Unique ID column in right CSV")

    # Column mappings (defaults assume identical names)
    p.add_argument("--left-first", default="first_name")
    p.add_argument("--left-last", default="last_name")
    p.add_argument("--left-dob", default="dob")
    p.add_argument("--left-sex", default="sex")
    p.add_argument("--right-first", default="first_name")
    p.add_argument("--right-last", default="last_name")
    p.add_argument("--right-dob", default="dob")
    p.add_argument("--right-sex", default="sex")

    # Matching config
    p.add_argument("--block-on", nargs="*", default=["dob", "sex"], choices=["dob", "sex", "NONE"],
                   help="Fields to block on (use 'NONE' for full comparison)")
    p.add_argument("--jw-first", type=float, default=0.85, help="Jaro–Winkler threshold for first name")
    p.add_argument("--jw-last", type=float, default=0.85, help="Jaro–Winkler threshold for last name")
    p.add_argument("--min-sum", type=int, default=3, help="Minimum number of rules satisfied to keep a pair")
    p.add_argument("--return-all", action="store_true", help="Return all candidate pairs (not just passing filter)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()

def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    left = pd.read_csv(args.left, dtype=str)
    right = pd.read_csv(args.right, dtype=str)

    left_map = {
        "first_name": args.left_first,
        "last_name": args.left_last,
        "dob": args.left_dob,
        "sex": args.left_sex,
    }
    right_map = {
        "first_name": args.right_first,
        "last_name": args.right_last,
        "dob": args.right_dob,
        "sex": args.right_sex,
    }

    block_on = [] if (args.block_on == ["NONE"]) else args.block_on

    out = run_linkage(
        left, right,
        left_id=args.left_id,
        right_id=args.right_id,
        left_colmap=left_map,
        right_colmap=right_map,
        block_on=block_on,
        jw_threshold_first=args.jw_first,
        jw_threshold_last=args.jw_last,
        min_rule_sum=args.min_sum,
        return_all_features=args.return_all,
    )

    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out):,} rows to {args.out}")

if __name__ == "__main__":
    main()
