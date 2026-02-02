#!/usr/bin/env python3
"""Add absolute DOB difference (days) to the Litholink *unmatched* output.

This script is a *post-processing* helper. It takes:
  - matches.csv (from linkage_algorithm.py)
  - litholink.unmatched.csv (from filter_matches.py)
  - the original left/right tables (so we can pull DOB values)

For each unmatched RIGHT record, we look for any candidate pairs in matches.csv
that involve that RIGHT id, compute the absolute DOB difference in days for each
pair, and then report the *minimum* absolute difference.

If no candidate pairs exist for an unmatched RIGHT record, the DOB difference is
left as NA.

Typical usage
-------------
1) Run linkage with DOB removed from blocking (example: block on sex only):

    python linkage_algorithm.py \
      --left local.csv --right litholink.csv \
      --left-id USDHubID --right-id PatientID \
      --block-on sex \
      --out matches.csv

2) Run filter_matches.py to produce litholink.unmatched.csv.

3) Add DOB diff to the unmatched output:

    python add_dob_diff_to_unmatched.py \
      --matches matches.csv \
      --unmatched litholink.unmatched.csv \
      --left local.csv --right litholink.csv \
      --left-id USDHubID --right-id PatientID \
      --left-dob-col dob --right-dob-col DOB \
      --out litholink.unmatched.with_dob_diff.csv
"""

from __future__ import annotations

import argparse
import pandas as pd


def _to_datetime(series: pd.Series) -> pd.Series:
    """Parse a date-like series into pandas datetime (date precision)."""
    s = series.astype("string").str.strip()
    # pandas >=2.0 uses a strict consistent parser; infer_datetime_format is deprecated.
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.floor("D")


def add_min_abs_dob_diff(
    *,
    matches_df: pd.DataFrame,
    unmatched_df: pd.DataFrame,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_id: str,
    right_id: str,
    left_dob_col: str,
    right_dob_col: str,
    out_col: str = "min_abs_dob_diff_days",
    out_best_left_id_col: str = "closest_left_id",
    out_num_cands_col: str = "num_candidate_pairs",
) -> pd.DataFrame:
    """Return unmatched_df with DOB-diff annotation columns added."""

    # Validate required columns
    for c in [left_id, right_id]:
        if c not in matches_df.columns:
            raise ValueError(f"matches.csv missing required column '{c}'. Present: {list(matches_df.columns)}")
    if right_id not in unmatched_df.columns:
        raise ValueError(f"unmatched file missing required column '{right_id}'. Present: {list(unmatched_df.columns)}")
    if left_id not in left_df.columns:
        raise ValueError(f"left file missing required column '{left_id}'. Present: {list(left_df.columns)}")
    if right_id not in right_df.columns:
        raise ValueError(f"right file missing required column '{right_id}'. Present: {list(right_df.columns)}")
    if left_dob_col not in left_df.columns:
        raise ValueError(f"left file missing DOB column '{left_dob_col}'. Present: {list(left_df.columns)}")
    if right_dob_col not in right_df.columns:
        raise ValueError(f"right file missing DOB column '{right_dob_col}'. Present: {list(right_df.columns)}")

    # Pull DOB values keyed by ID
    left_dob = left_df[[left_id, left_dob_col]].copy()
    left_dob[left_id] = left_dob[left_id].astype("string")
    left_dob["__dob_left_dt"] = _to_datetime(left_dob[left_dob_col])

    right_dob = right_df[[right_id, right_dob_col]].copy()
    right_dob[right_id] = right_dob[right_id].astype("string")
    right_dob["__dob_right_dt"] = _to_datetime(right_dob[right_dob_col])

    # Candidate pairs (from linkage_algorithm.py)
    pairs = matches_df[[left_id, right_id]].copy()
    pairs[left_id] = pairs[left_id].astype("string")
    pairs[right_id] = pairs[right_id].astype("string")

    pairs = pairs.merge(left_dob[[left_id, "__dob_left_dt"]], how="left", on=left_id)
    pairs = pairs.merge(right_dob[[right_id, "__dob_right_dt"]], how="left", on=right_id)

    # Compute abs difference in days for each candidate pair
    pairs["__abs_diff_days"] = (
        pairs["__dob_left_dt"] - pairs["__dob_right_dt"]
    ).abs().dt.days

    # For each RIGHT id, find the min abs diff and the corresponding LEFT id
    # (ties broken by choosing the first after sort).
    pairs_sorted = pairs.sort_values(by=[right_id, "__abs_diff_days", left_id], ascending=[True, True, True])
    best = pairs_sorted.dropna(subset=["__abs_diff_days"]).drop_duplicates(subset=[right_id], keep="first")
    best = best[[right_id, left_id, "__abs_diff_days"]].rename(
        columns={left_id: out_best_left_id_col, "__abs_diff_days": out_col}
    )

    # Count candidates per RIGHT id (even if DOB is missing)
    cand_counts = pairs.groupby(right_id, dropna=False).size().reset_index(name=out_num_cands_col)

    out = unmatched_df.copy()
    out[right_id] = out[right_id].astype("string")
    out = out.merge(cand_counts, how="left", on=right_id)
    out = out.merge(best, how="left", on=right_id)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Annotate Litholink unmatched output with minimum absolute DOB difference (days).")
    p.add_argument("--matches", required=True, help="matches.csv from linkage_algorithm.py")
    p.add_argument("--unmatched", required=True, help="litholink.unmatched.csv from filter_matches.py")
    p.add_argument("--left", required=True, help="Left table (e.g., local.csv)")
    p.add_argument("--right", required=True, help="Right table (e.g., litholink.csv)")
    p.add_argument("--left-id", required=True, help="Left ID column name (e.g., USDHubID)")
    p.add_argument("--right-id", required=True, help="Right ID column name (e.g., PatientID)")
    p.add_argument("--left-dob-col", default="dob", help="DOB column name in left table (default: dob)")
    p.add_argument("--right-dob-col", default="DOB", help="DOB column name in right table (default: DOB)")
    p.add_argument("--out", required=True, help="Output path for unmatched with DOB diff added")
    p.add_argument("--out-col", default="min_abs_dob_diff_days", help="Output column name for min abs DOB diff (days)")
    p.add_argument("--out-best-left-id-col", default="closest_left_id", help="Output column name for the closest left id")
    p.add_argument("--out-num-cands-col", default="num_candidate_pairs", help="Output column name for candidate-pair count")

    args = p.parse_args(argv)

    matches_df = pd.read_csv(args.matches)
    unmatched_df = pd.read_csv(args.unmatched)
    left_df = pd.read_csv(args.left)
    right_df = pd.read_csv(args.right)

    out = add_min_abs_dob_diff(
        matches_df=matches_df,
        unmatched_df=unmatched_df,
        left_df=left_df,
        right_df=right_df,
        left_id=args.left_id,
        right_id=args.right_id,
        left_dob_col=args.left_dob_col,
        right_dob_col=args.right_dob_col,
        out_col=args.out_col,
        out_best_left_id_col=args.out_best_left_id_col,
        out_num_cands_col=args.out_num_cands_col,
    )
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out):,} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
