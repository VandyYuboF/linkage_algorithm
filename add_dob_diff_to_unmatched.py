
import argparse
import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher

def _parse_dob(series: pd.Series, fmt: str) -> pd.Series:
    # Deterministic parse; invalid -> NaT
    return pd.to_datetime(series, format=fmt, errors="coerce")

def _norm_sex(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().lower()
    if s in {"m", "male", "man"}:
        return "M"
    if s in {"f", "female", "woman"}:
        return "F"
    return s.upper()

_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}

def _clean_token(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\-']", "", s)
    return s.strip().lower()

def _split_full_name(full_name: str):
    '''
    Supports:
      - "Last,First Middle" (middle may be "A" or "A.")
      - "First Middle Last"
    Returns (first, last) lowercased tokens (may be empty).
    '''
    if full_name is None or (isinstance(full_name, float) and np.isnan(full_name)):
        return "", ""
    s = str(full_name).strip()
    s = re.sub(r"\s+", " ", s).strip().strip(",")

    if not s:
        return "", ""

    if "," in s:
        # Last, First Middle
        last_part, rest = s.split(",", 1)
        last = _clean_token(last_part)
        rest = re.sub(r"\s+", " ", rest.strip())
        toks = [t for t in rest.split(" ") if t]
        if not toks:
            return "", last
        first = _clean_token(toks[0])
        return first, last

    # First Middle Last
    toks = [t for t in re.split(r"\s+", s) if t]
    # drop suffix if present
    if toks and _clean_token(toks[-1]) in _SUFFIXES:
        toks = toks[:-1]
    if len(toks) == 1:
        return _clean_token(toks[0]), ""
    first = _clean_token(toks[0])
    last = _clean_token(toks[-1])
    return first, last

def _norm_name(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\-\'\s]", "", s)
    return s.strip()

def _fuzzy_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def _generate_pairs_from_two_files(local: pd.DataFrame,
                                   lith: pd.DataFrame,
                                   left_id: str,
                                   left_first: str,
                                   left_last: str,
                                   left_sex: str,
                                   left_dob: str,
                                   right_id: str,
                                   right_full_name: str,
                                   right_sex: str,
                                   right_dob: str,
                                   name_threshold: float):
    # Prepare local columns
    l = local[[left_id, left_first, left_last, left_sex, left_dob]].copy()
    l = l.rename(columns={
        left_id: "USDHubID",
        left_first: "local_first_name",
        left_last: "local_last_name",
        left_sex: "local_sex",
        left_dob: "local_dob",
    })
    l["local_sex_norm"] = l["local_sex"].map(_norm_sex)
    l["local_first_norm"] = l["local_first_name"].map(_norm_name)
    l["local_last_norm"] = l["local_last_name"].map(_norm_name)

    r = lith.copy()
    first_last = r[right_full_name].map(_split_full_name)
    r["_right_first"] = [x[0] for x in first_last]
    r["_right_last"] = [x[1] for x in first_last]
    r["right_first_norm"] = r["_right_first"].map(_norm_name)
    r["right_last_norm"] = r["_right_last"].map(_norm_name)
    r["right_sex_norm"] = r[right_sex].map(_norm_sex)

    out_rows = []
    for sex_val in sorted(set(r["right_sex_norm"].dropna().unique()) | set(l["local_sex_norm"].dropna().unique())):
        rr = r[r["right_sex_norm"] == sex_val].copy()
        ll = l[l["local_sex_norm"] == sex_val].copy()
        if rr.empty or ll.empty:
            continue

        rr["_k"] = 1
        ll["_k"] = 1
        cross = rr.merge(ll, on="_k", how="inner").drop(columns=["_k"])

        keep_idx = []
        for i, row in cross.iterrows():
            fr = _fuzzy_ratio(row["right_first_norm"], row["local_first_norm"])
            lr = _fuzzy_ratio(row["right_last_norm"], row["local_last_norm"])
            if fr >= name_threshold and lr >= name_threshold:
                keep_idx.append(i)

        if keep_idx:
            out_rows.append(cross.loc[keep_idx])

    if not out_rows:
        return pd.DataFrame(columns=list(lith.columns) + [
            "USDHubID","local_last_name","local_first_name","local_sex","local_dob"
        ])

    out = pd.concat(out_rows, ignore_index=True)

    # Remove helper norm columns (keep original lith columns + local columns)
    for c in ["local_first_norm","local_last_norm","local_sex_norm","right_first_norm","right_last_norm","right_sex_norm","_right_first","_right_last"]:
        if c in out.columns:
            out = out.drop(columns=[c])
    return out

def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Generate Litholink–Local candidate pairs for review, adding local identifiers + abs DOB difference (days). "
            "Two modes:\n"
            "  (A) Pipeline mode: provide --matches and --unmatched.\n"
            "  (B) Two-file mode: omit --matches/--unmatched; fuzzy match names between --left and --right."
        )
    )

    ap.add_argument("--matches", default=None, help="Output from linkage_algorithm.py (e.g., matches.csv)")
    ap.add_argument("--unmatched", default=None, help="Output from filter_matches.py (e.g., litholink.unmatched.csv)")

    ap.add_argument("--left", required=True, help="Local CSV (e.g., local.csv)")
    ap.add_argument("--right", required=True, help="Litholink CSV (e.g., litholink.csv)")

    ap.add_argument("--left-id", required=True, help="Local patient ID column (e.g., USDHubID)")
    ap.add_argument("--right-id", required=True, help="Litholink patient ID column (e.g., PatientID)")

    ap.add_argument("--left-first-col", default="first_name")
    ap.add_argument("--left-last-col", default="last_name")
    ap.add_argument("--left-sex-col", default="sex")
    ap.add_argument("--left-dob-col", default="dob")

    ap.add_argument("--right-full-name-col", default="Patient")
    ap.add_argument("--right-sex-col", default="Gender")
    ap.add_argument("--right-dob-col", default="DOB")

    ap.add_argument("--dob-format", default="%m/%d/%Y")
    ap.add_argument("--name-threshold", type=float, default=0.90)
    ap.add_argument("--include-zero", action="store_true", help="Include pairs where abs_dob_diff_days == 0")
    ap.add_argument("--out", required=True)

    args = ap.parse_args()

    local = pd.read_csv(args.left, dtype=str)
    lith = pd.read_csv(args.right, dtype=str)

    for c in [args.left_id, args.left_first_col, args.left_last_col, args.left_sex_col, args.left_dob_col]:
        if c not in local.columns:
            raise SystemExit(f"[ERROR] Missing column in LEFT file: {c}")

    for c in [args.right_id, args.right_full_name_col, args.right_sex_col, args.right_dob_col]:
        if c not in lith.columns:
            raise SystemExit(f"[ERROR] Missing column in RIGHT file: {c}")

    if args.matches and args.unmatched:
        matches = pd.read_csv(args.matches, dtype=str)
        unmatched = pd.read_csv(args.unmatched, dtype=str)

        if args.right_id not in matches.columns:
            raise SystemExit(f"[ERROR] right-id column '{args.right_id}' not found in matches file.")
        if args.right_id not in unmatched.columns:
            raise SystemExit(f"[ERROR] right-id column '{args.right_id}' not found in unmatched file.")
        if args.left_id not in matches.columns:
            raise SystemExit(
                f"[ERROR] left-id column '{args.left_id}' not found in matches file. "
                "Ensure linkage_algorithm.py writes the local ID column into matches.csv."
            )

        pairs = matches.merge(unmatched[[args.right_id]], how="inner", on=args.right_id)

        if args.left_id != "USDHubID":
            pairs = pairs.rename(columns={args.left_id: "USDHubID"})
        else:
            pairs = pairs.copy()

        left_min = local[[args.left_id, args.left_first_col, args.left_last_col, args.left_sex_col, args.left_dob_col]].copy()
        left_min = left_min.rename(columns={
            args.left_id: "USDHubID",
            args.left_first_col: "local_first_name",
            args.left_last_col: "local_last_name",
            args.left_sex_col: "local_sex",
            args.left_dob_col: "local_dob",
        })
        pairs = pairs.merge(left_min, how="left", on="USDHubID")

        right_min = lith[[args.right_id, args.right_sex_col, args.right_dob_col]].copy()
        right_min = right_min.rename(columns={
            args.right_sex_col: "right_sex",
            args.right_dob_col: "right_dob",
        })
        pairs = pairs.merge(right_min, how="left", on=args.right_id)

        # sex exact match (normalized)
        pairs["local_sex_norm"] = pairs["local_sex"].map(_norm_sex)
        pairs["right_sex_norm"] = pairs["right_sex"].map(_norm_sex)
        pairs = pairs[pairs["local_sex_norm"] == pairs["right_sex_norm"]].copy()

        pairs = pairs.merge(unmatched, on=args.right_id, how="left", suffixes=("", "_unmatched"))
        base_cols = list(unmatched.columns)

    else:
        pairs = _generate_pairs_from_two_files(
            local=local,
            lith=lith,
            left_id=args.left_id,
            left_first=args.left_first_col,
            left_last=args.left_last_col,
            left_sex=args.left_sex_col,
            left_dob=args.left_dob_col,
            right_id=args.right_id,
            right_full_name=args.right_full_name_col,
            right_sex=args.right_sex_col,
            right_dob=args.right_dob_col,
            name_threshold=args.name_threshold,
        )

        # Use all litholink columns as base in two-file mode
        base_cols = list(lith.columns)

        # Ensure right_sex/right_dob exist for downstream
        if "right_sex" not in pairs.columns:
            pairs["right_sex"] = pairs[args.right_sex_col]
        if "right_dob" not in pairs.columns:
            pairs["right_dob"] = pairs[args.right_dob_col]

        # sex exact match again for safety
        pairs["local_sex_norm"] = pairs["local_sex"].map(_norm_sex)
        pairs["right_sex_norm"] = pairs["right_sex"].map(_norm_sex)
        pairs = pairs[pairs["local_sex_norm"] == pairs["right_sex_norm"]].copy()

    pairs["_ld"] = _parse_dob(pairs["local_dob"], args.dob_format)
    pairs["_rd"] = _parse_dob(pairs["right_dob"], args.dob_format)
    pairs["abs_dob_diff_days"] = (pairs["_ld"] - pairs["_rd"]).abs().dt.days

    # default keep only != 0
    if not args.include_zero:
        pairs = pairs[pairs["abs_dob_diff_days"].fillna(0) != 0].copy()

    out_cols = base_cols + [
        "USDHubID",
        "local_last_name",
        "local_first_name",
        "local_sex",
        "local_dob",
        "abs_dob_diff_days",
    ]
    out_cols = [c for c in out_cols if c in pairs.columns]

    out = pairs[out_cols].drop_duplicates()
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows to {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
