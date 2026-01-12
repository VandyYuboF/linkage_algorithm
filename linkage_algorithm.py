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
from rapidfuzz.distance import JaroWinkler


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


def _parse_litholink_patient_name(
    df: pd.DataFrame,
    *,
    name_col: str = "patient_name",
    out_first: str = "__parsed_first_name",
    out_last: str = "__parsed_last_name",
) -> pd.DataFrame:
    """
    Parse Litholink-style combined name field.

    In common Litholink exports, the combined name field is stored as:

        Patient (or patient_name) = "last_name, first_name middle_name"

    Middle name is optional. This function extracts:
      - out_last:  text before the first comma
      - out_first: first token after the first comma

    Notes
    -----
    - Handles missing middle name (e.g., "Smith, John")
    - Handles extra whitespace
    - If the comma is missing, the extracted fields will be NA
    """
    out = df.copy()
    s = out[name_col].astype("string")

    parts = s.str.split(",", n=1, expand=True)
    first_part = parts[0].fillna("").str.strip()
    rest_part = parts[1].fillna("").str.strip() if parts.shape[1] > 1 else ""

    # Litholink format is typically: "LAST, FIRST MIDDLE".
    out[out_last] = first_part
    out[out_first] = (
        rest_part.astype("string")
        .str.split(r"\s+", n=1, expand=True)[0]
        .fillna("")
        .str.strip()
    )

    # Convert empty strings to NA for downstream normalization
    out[out_first] = out[out_first].replace("", pd.NA)
    out[out_last] = out[out_last].replace("", pd.NA)
    return out


def _parse_dob_series_to_iso(dob: pd.Series) -> pd.Series:
    """Parse DOB into ISO-like 'YYYY-MM-DD' strings.

    Handles common exports such as:
      - M/D/YY or MM/DD/YY (e.g., 1/1/80)
      - M/D/YYYY or MM/DD/YYYY
      - already-ISO strings

    Falls back to dateutil parsing only when needed.
    """
    s = dob.astype("string").str.strip()
    sample = s.dropna().head(50)
    fmt = None
    if len(sample):
        if sample.str.match(r"^\d{1,2}/\d{1,2}/\d{2}$").all():
            fmt = "%m/%d/%y"
        elif sample.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$").all():
            fmt = "%m/%d/%Y"

    if fmt is not None:
        dt = pd.to_datetime(s, format=fmt, errors="coerce")
    else:
        dt = pd.to_datetime(s, errors="coerce")

    return dt.dt.date.astype(str)


def _autofix_colmap(df: pd.DataFrame, colmap: Mapping[str, str]) -> Mapping[str, str]:
    """If a mapped column name does not exist, try common alternatives."""
    alternatives = {
        "first_name": ["first_name", "FirstName", "FIRST_NAME", "fname", "FNAME"],
        "last_name": ["last_name", "LastName", "LAST_NAME", "lname", "LNAME"],
        "dob": ["dob", "DOB", "Dob", "birth_date", "BirthDate", "BIRTH_DATE"],
        "sex": ["sex", "Sex", "SEX", "gender", "Gender", "GENDER"],
    }

    fixed = dict(colmap)
    for k, mapped in list(fixed.items()):
        if mapped in df.columns:
            continue
        for alt in alternatives.get(k, []):
            if alt in df.columns:
                fixed[k] = alt
                break
    return fixed


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

    # Normalize fields (defensive: guarantee required columns exist)
    for col in ["first_name", "last_name", "dob", "sex"]:
        if col not in out.columns:
            out[col] = pd.NA

    out["first_name"] = _normalize_name(out["first_name"].fillna(""))
    out["last_name"] = _normalize_name(out["last_name"].fillna(""))
    # Ensure DOB is ISO-like string for exact match blocking
    out["dob"] = _parse_dob_series_to_iso(out["dob"])
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
    try_three_field: Optional[bool] = None,
    three_field_min_rule_sum: int = 3,
    try_firstname_fallback: bool = False,
    firstname_fallback_min_rule_sum: int = 3,
    # Backwards-compatible alias (deprecated):
    fallback_three_field: Optional[bool] = None,
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
    try_three_field : if True, perform a second-pass fallback match for LEFT records
                      that had no passing matches in the first pass. The fallback
                      ignores first_name and matches on (last_name, dob, sex).
                      If None (default), will use `fallback_three_field` if provided,
                      otherwise defaults to False.
    three_field_min_rule_sum : minimum rules to keep a 3-field fallback pair (max is 3).
    try_firstname_fallback : if True, perform an additional fallback pass to handle last-name changes.
                      This pass ignores last_name and matches on (first_name, dob, sex).
                      Recommended only when last-name changes are expected.
    firstname_fallback_min_rule_sum : minimum rules to keep a (first_name, dob, sex) fallback pair (max is 3).
    fallback_three_field : deprecated alias for `try_three_field`.

    Returns
    -------
    DataFrame with columns:
      - left_index  (left_id)
      - right_index (right_id)
      - first_name, last_name, dob, sex (feature flags or similarities per recordlinkage)
      - sum (integer count of satisfied rules)
      - match_strategy (only present when try_three_field=True): {'4field','3field'}
    """
    # Resolve backwards-compatible parameter name
    if try_three_field is None:
        try_three_field = bool(fallback_three_field) if fallback_three_field is not None else False

    # Make CLI defaults more forgiving (e.g., Litholink uses 'DOB'/'Gender').
    left_colmap = _autofix_colmap(df_left, left_colmap)
    right_colmap = _autofix_colmap(df_right, right_colmap)

    # ------------------------------------------------------------------
    # Litholink convenience: extract first/last name from a combined field
    # ------------------------------------------------------------------
    # Some Litholink exports store patient name as a single column:
    #   Patient (or patient_name) = "first_name, last_name middle_name"
    # Middle name is optional. If the provided right_colmap refers to missing
    # first/last name columns and patient_name exists, we parse it and rewrite
    # right_colmap to use the parsed columns.
    r_first = right_colmap.get("first_name")
    r_last = right_colmap.get("last_name")
    name_col_candidates = ["Patient", "patient_name"]
    name_col = next((c for c in name_col_candidates if c in df_right.columns), None)

    if (
        name_col is not None
        and (r_first is None or r_first not in df_right.columns or r_last is None or r_last not in df_right.columns)
    ):
        df_right = _parse_litholink_patient_name(df_right, name_col=name_col)
        right_colmap = dict(right_colmap)
        right_colmap["first_name"] = "__parsed_first_name"
        right_colmap["last_name"] = "__parsed_last_name"

    # Prepare data
    L = _prep(df_left, left_colmap, left_id).set_index(left_id, drop=False)
    R = _prep(df_right, right_colmap, right_id).set_index(right_id, drop=False)

    def _compute(
        Lsub: pd.DataFrame,
        Rsub: pd.DataFrame,
        *,
        include_first_name: bool,
        include_last_name: bool,
        jw_first: float,
        jw_last: float,
        block_keys: Iterable[str],
    ) -> pd.DataFrame:
        # Candidate generation via exact blocking on requested keys.
        block_keys = list(block_keys)
        for key in block_keys:
            if key not in ("dob", "sex"):
                raise ValueError(f"Unsupported block key '{key}'. Use 'dob' and/or 'sex'.")

        left_cols = [left_id, "first_name", "last_name", "dob", "sex"]
        right_cols = [right_id, "first_name", "last_name", "dob", "sex"]

        Lc = Lsub[left_cols].copy()
        Rc = Rsub[right_cols].copy()

        if block_keys:
            cand = Lc.merge(
                Rc,
                how="inner",
                on=block_keys,
                suffixes=("_l", "_r"),
            )
        else:
            # Full Cartesian product (can be very slow on large data).
            Lc["__tmp"] = 1
            Rc["__tmp"] = 1
            cand = Lc.merge(Rc, on="__tmp", suffixes=("_l", "_r")).drop(columns=["__tmp"])

        if cand.empty:
            # Return empty with expected columns
            base_cols = [left_id, right_id]
            feat_cols = ([]
                        + (["first_name"] if include_first_name else [])
                        + (["last_name"] if include_last_name else [])
                        + ["dob", "sex", "sum"])
            return pd.DataFrame(columns=base_cols + feat_cols)

        # Name similarity (Jaro-Winkler returns [0,1])
        if include_first_name:
            fn_sim = [JaroWinkler.similarity(a, b) for a, b in zip(cand["first_name_l"], cand["first_name_r"])]
            cand["first_name"] = (pd.Series(fn_sim, index=cand.index) >= jw_first).astype(int)

        if include_last_name:
            ln_sim = [JaroWinkler.similarity(a, b) for a, b in zip(cand["last_name_l"], cand["last_name_r"])]
            cand["last_name"] = (pd.Series(ln_sim, index=cand.index) >= jw_last).astype(int)

        # Exact matches (if blocked on dob/sex, they are guaranteed equal; still compute flags for scoring)
        cand["dob"] = 1 if "dob" in block_keys else (cand["dob_l"] == cand["dob_r"]).astype(int)
        cand["sex"] = 1 if "sex" in block_keys else (cand["sex_l"] == cand["sex_r"]).astype(int)

        feat_cols = ([]
                    + (["first_name"] if include_first_name else [])
                    + (["last_name"] if include_last_name else [])
                    + ["dob", "sex"])
        cand["sum"] = cand[feat_cols].sum(axis=1)

        # Return normalized feature frame
        out = cand[[left_id, right_id] + feat_cols + ["sum"]].copy()
        out = out.rename(columns={left_id: "level_0", right_id: "level_1"})
        return out

    block_keys = tuple(k for k in block_on if k != "NONE")

    # Pass 1: 4-field (includes first_name)
    pass1 = _compute(
        L, R,
        include_first_name=True,
        include_last_name=True,
        jw_first=jw_threshold_first,
        jw_last=jw_threshold_last,
        block_keys=block_keys,
    )

    if return_all_features:
        out1 = pass1
    else:
        out1 = pass1.loc[pass1["sum"] >= min_rule_sum].copy()

    if try_three_field:
        # Identify records that have no passing matches from pass 1.
        # We run the fallback for BOTH sides:
        #   - unmatched LEFT vs all RIGHT
        #   - all LEFT vs unmatched RIGHT
        matched_left_ids = set(out1["level_0"].astype(str).tolist()) if len(out1) else set()
        matched_right_ids = set(out1["level_1"].astype(str).tolist()) if len(out1) else set()

        all_left_ids = set(L.index.astype(str).tolist())
        all_right_ids = set(R.index.astype(str).tolist())

        unmatched_left_ids = sorted(all_left_ids - matched_left_ids)
        unmatched_right_ids = sorted(all_right_ids - matched_right_ids)

        out2_parts = []

        if unmatched_left_ids:
            L_unmatched = L.loc[unmatched_left_ids]
            pass2_left = _compute(
                L_unmatched, R,
                include_first_name=False,
                include_last_name=True,
                jw_first=jw_threshold_first,  # unused, kept for signature consistency
                jw_last=jw_threshold_last,
                block_keys=block_keys,
            )
            out2_parts.append(pass2_left)

        if unmatched_right_ids:
            R_unmatched = R.loc[unmatched_right_ids]
            pass2_right = _compute(
                L, R_unmatched,
                include_first_name=False,
                include_last_name=True,
                jw_first=jw_threshold_first,  # unused
                jw_last=jw_threshold_last,
                block_keys=block_keys,
            )
            out2_parts.append(pass2_right)

        if out2_parts:
            pass2 = pd.concat(out2_parts, ignore_index=True)
            if return_all_features:
                out2 = pass2
            else:
                out2 = pass2.loc[pass2["sum"] >= three_field_min_rule_sum].copy()

            # Tag strategies
            out1["match_strategy"] = "4field"
            out2["match_strategy"] = "3field"

            # Combine, deduplicate preferring 4field
            out = pd.concat([out1, out2], ignore_index=True)
            out["__strategy_rank"] = out["match_strategy"].map({"4field": 0, "3field": 1, "3field_first": 2}).fillna(9)
            out = out.sort_values(["__strategy_rank"]).drop_duplicates(subset=["level_0", "level_1"], keep="first")
            out = out.drop(columns=["__strategy_rank"])

            # Optional Pass 3: handle last-name changes by matching on (first_name, dob, sex)
            if try_firstname_fallback:
                matched_left_ids2 = set(out["level_0"].astype(str).tolist()) if len(out) else set()
                matched_right_ids2 = set(out["level_1"].astype(str).tolist()) if len(out) else set()

                unmatched_left_ids2 = sorted(all_left_ids - matched_left_ids2)
                unmatched_right_ids2 = sorted(all_right_ids - matched_right_ids2)

                out3_parts = []

                if unmatched_left_ids2:
                    L_unmatched2 = L.loc[unmatched_left_ids2]
                    pass3_left = _compute(
                        L_unmatched2, R,
                        include_first_name=True,
                        include_last_name=False,
                        jw_first=jw_threshold_first,
                        jw_last=jw_threshold_last,  # unused
                        block_keys=block_keys,
                    )
                    out3_parts.append(pass3_left)

                if unmatched_right_ids2:
                    R_unmatched2 = R.loc[unmatched_right_ids2]
                    pass3_right = _compute(
                        L, R_unmatched2,
                        include_first_name=True,
                        include_last_name=False,
                        jw_first=jw_threshold_first,
                        jw_last=jw_threshold_last,  # unused
                        block_keys=block_keys,
                    )
                    out3_parts.append(pass3_right)

                if out3_parts:
                    pass3 = pd.concat(out3_parts, ignore_index=True)
                    if return_all_features:
                        out3 = pass3
                    else:
                        out3 = pass3.loc[pass3["sum"] >= firstname_fallback_min_rule_sum].copy()

                    out3["match_strategy"] = "3field_first"

                    out = pd.concat([out, out3], ignore_index=True)
                    out["__strategy_rank"] = out["match_strategy"].map({"4field": 0, "3field": 1, "3field_first": 2}).fillna(9)
                    out = out.sort_values(["__strategy_rank"]).drop_duplicates(subset=["level_0", "level_1"], keep="first")
                    out = out.drop(columns=["__strategy_rank"])
        else:
            out1["match_strategy"] = "4field"
            out = out1
    else:
        out = out1

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
    p.add_argument("--fallback-3field", action="store_true",
                   help="If enabled, retry unmatched LEFT records with 3-field match (last_name, dob, sex) ignoring first_name")
    p.add_argument("--fallback-min-sum", type=int, default=3,
                   help="Minimum number of rules satisfied to keep a (last_name, dob, sex) fallback pair (max is 3)")
    p.add_argument("--fallback-firstname", action="store_true",
                   help="If enabled, an additional fallback pass matches on (first_name, dob, sex) ignoring last_name (helps when last name changes)")
    p.add_argument("--fallback-firstname-min-sum", type=int, default=3,
                   help="Minimum number of rules satisfied to keep a (first_name, dob, sex) fallback pair (max is 3)")
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
        try_three_field=args.fallback_3field,
        three_field_min_rule_sum=args.fallback_min_sum,
        try_firstname_fallback=args.fallback_firstname,
        firstname_fallback_min_rule_sum=args.fallback_firstname_min_sum,
        return_all_features=args.return_all,
    )

    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out):,} rows to {args.out}")

if __name__ == "__main__":
    main()
