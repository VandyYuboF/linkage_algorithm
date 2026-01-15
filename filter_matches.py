#!/usr/bin/env python3
"""Filter linkage matches and attach Litholink columns.

This script is intentionally narrow and deterministic:

Input:
  - matches.csv produced by linkage_algorithm.py, with columns:
      <left_id>, <right_id>, first_name, last_name, dob, sex, sum
  - litholink.csv (the RIGHT table), containing <right_id> plus any extra columns.

Output:
  1) filtered matches:
     - keep all 4-field matches (first_name=1,last_name=1,dob=1,sex=1)
     - keep 3-field matches of type (first_name + dob + sex), i.e.
          first_name=1,dob=1,sex=1,last_name=0
     - BUT: if a Litholink record (<right_id>) has any 4-field match,
            DO NOT return any 3-field matches for that <right_id>.
  2) merged file = filtered matches joined with litholink.csv on <right_id>
     (so you retain Litholink extra columns).

Example:
  python filter_matches.py \
    --matches matches.csv \
    --litholink litholink.csv \
    --left-id person_id \
    --right-id PatientID \
    --out-filtered matches.filtered.csv \
    --out-merged matches.filtered.with_litholink.csv
"""

from __future__ import annotations

import argparse
import sys
import pandas as pd


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}. Present columns: {list(df.columns)}")


def filter_matches(
    matches_df: pd.DataFrame,
    *,
    left_id: str,
    right_id: str,
) -> pd.DataFrame:
    """Apply the filtering policy described in the module docstring."""
    required = [left_id, right_id, "first_name", "last_name", "dob", "sex", "sum"]
    _require_cols(matches_df, required, "matches.csv")

    # normalize the indicator columns to integers 0/1 (tolerate bool/float/str)
    df = matches_df.copy()
    for c in ["first_name", "last_name", "dob", "sex", "sum"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Define match types
    is_4field = (df["first_name"] == 1) & (df["last_name"] == 1) & (df["dob"] == 1) & (df["sex"] == 1)
    is_3field_first = (df["first_name"] == 1) & (df["dob"] == 1) & (df["sex"] == 1) & (df["last_name"] == 0)

    df = df[is_4field | is_3field_first].copy()

    # If right_id has any 4-field match, drop its 3-field-first matches
    right_has_4 = df.loc[is_4field.reindex(df.index, fill_value=False), right_id].dropna().unique()
    if len(right_has_4) > 0:
        mask_drop_3 = df[right_id].isin(right_has_4) & is_3field_first.reindex(df.index, fill_value=False)
        df = df[~mask_drop_3].copy()

    # Sort for readability
    df = df.sort_values(by=[right_id, left_id, "sum"], ascending=[True, True, False])
    return df.reset_index(drop=True)


def merge_with_litholink(
    filtered_matches: pd.DataFrame,
    litholink_df: pd.DataFrame,
    *,
    right_id: str,
) -> pd.DataFrame:
    if right_id not in litholink_df.columns:
        raise ValueError(f"litholink.csv does not contain right-id column '{right_id}'. Present columns: {list(litholink_df.columns)}")
    merged = filtered_matches.merge(litholink_df, how="left", on=right_id, suffixes=("", "_litholink"))
    merged = merged.drop(['Physician', 'Patient', 'PatientID', 'DOB', 'SampleID', 'CystineSampleID', 'first_name' ,'last_name' ,'dob' ,'sex' ,'sum'], axis = 1)
    return merged

def find_unmatched_litholink(
    litholink_df: pd.DataFrame,
    filtered_matches: pd.DataFrame,
    *,
    right_id: str,
) -> pd.DataFrame:
    """Return Litholink records that do not appear in filtered matches."""
    if right_id not in litholink_df.columns:
        raise ValueError(
            f"litholink.csv does not contain column '{right_id}'. "
            f"Present columns: {list(litholink_df.columns)}"
        )

    matched_ids = filtered_matches[right_id].dropna().unique()
    unmatched = litholink_df[~litholink_df[right_id].isin(matched_ids)].copy()
    return unmatched.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Filter matches and attach Litholink columns.")
    p.add_argument("--matches", required=True, help="Path to matches.csv produced by linkage_algorithm.py")
    p.add_argument("--litholink", required=True, help="Path to litholink.csv (RIGHT table) to attach extra columns")
    p.add_argument("--left-id", required=True, help="Left ID column name (e.g., person_id)")
    p.add_argument("--right-id", required=True, help="Right ID column name (e.g., PatientID)")
    p.add_argument("--out-filtered", required=True, help="Output path for filtered matches CSV")
    p.add_argument("--out-merged", required=True, help="Output path for filtered+merged CSV (matches + litholink columns)")
    p.add_argument("--out-unmatched-litholink", required=True, help="Output path for Litholink records with no match in target dataset")

    args = p.parse_args(argv)

    matches_df = pd.read_csv(args.matches)
    litho_df = pd.read_csv(args.litholink)

    filtered = filter_matches(matches_df, left_id=args.left_id, right_id=args.right_id)
    filtered.to_csv(args.out_filtered, index=False)

    merged = merge_with_litholink(filtered, litho_df, right_id=args.right_id)
    merged.to_csv(args.out_merged, index=False)

    unmatched_litho = find_unmatched_litholink(litho_df, filtered, right_id=args.right_id)
    unmatched_litho.to_csv(args.out_unmatched_litholink, index=False)

    print(f"Wrote {len(filtered):,} rows to {args.out_filtered}")
    print(f"Wrote {len(merged):,} rows to {args.out_merged}")
    print(f"Wrote {len(unmatched_litho):,} rows to {args.out_unmatched_litholink}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
