# Linkage Algorithm (Cross-Site)

This repository provides a small, configurable CLI tool to link patient records across sites using:

- Blocking (default: `dob` + `sex`)
- Jaro–Winkler string similarity for names
- Exact matches for DOB/sex
- A simple rule score (`sum`) that counts how many fields matched

It also includes a second script to post-filter matches into:
- **4-field matches**: first + last + DOB + sex
- **3-field matches (first + DOB + sex)** with a guardrail:
  - If a Litholink record already has any 4-field match, we do **not** emit any 3-field matches for that Litholink ID.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 1) Generate candidate matches (matches.csv)

Minimal usage (the pattern you described):

```bash
python linkage_algorithm.py \
  --left vumc.csv --right litholink.csv \
  --left-id USDHubID --right-id PatientID \
  --min-sum 3 \
  --out matches.csv
```

### Required input columns (defaults)

By default, both files are expected to have:

- `first_name`
- `last_name`
- `dob` (any parseable date; internally normalized to `YYYY-MM-DD`)
- `sex`

If your columns have different names, map them:

```bash
python linkage_algorithm.py \
  --left vumc.csv --right litholink.csv \
  --left-id USDHubID --right-id PatientID \
  --left-first fname --left-last lname --left-dob birth_date --left-sex gender \
  --right-first first --right-last last --right-dob dob --right-sex sex \
  --min-sum 3 \
  --out matches.csv
```

### Litholink combined-name convenience (`Patient` / `patient_name`)

Many Litholink exports store the combined patient name in a single column:

- `Patient` (or `patient_name`) like: `LAST, FIRST MIDDLE`

If the RIGHT table does **not** have usable first/last name columns, the linkage script will automatically:
1) detect `Patient` or `patient_name`
2) parse first/last into internal columns
3) proceed with matching using those parsed fields

In this case you typically do **not** need to specify `--right-first/--right-last`.

### Key scoring/tuning flags

- `--min-sum` (default `3`): minimum number of satisfied rules to keep a pair
- `--jw-first`, `--jw-last` (default `0.85`): Jaro–Winkler thresholds for first/last name
- `--block-on dob sex` (default): reduce comparisons by only comparing within same DOB+sex
- `--block-on NONE`: full cartesian comparison (can be very slow)

Optional fallback passes (off by default):
- `--fallback-3field`: retry unmatched records using a **(last_name + dob + sex)** rule set
- `--fallback-firstname`: retry unmatched records using **(first_name + dob + sex)** (helpful when last name changes)

---

## 2) Filter matches and attach Litholink columns

This script takes:
- `matches.csv` from step (1)
- the original Litholink file (`litholink.csv`) to attach extra columns

It outputs:
- `matches.filtered.csv`: only 4-field matches and 3-field (first+dob+sex) matches, with the guardrail described above
- `matches.filtered.with_litholink.csv`: the filtered matches joined back to Litholink columns on `PatientID`, this file only includes `USDHubID` and the rest of columns in litholink.csv, except: [`Physician`, `Patient`, `PatientID`, `DOB`, `SampleID`, `CystineSampleID`]

Example:

```bash
python filter_matches.py \
  --matches matches.csv \
  --litholink litholink.csv \
  --left-id USDHubID \
  --right-id PatientID \
  --out-filtered matches.filtered.csv \
  --out-merged matches.filtered.with_litholink.csv
```

---

## Output schema (matches.csv)

`matches.csv` contains (at minimum):

- `<left-id>` (e.g., `USDHubID`)
- `<right-id>` (e.g., `PatientID`)
- `first_name`, `last_name`, `dob`, `sex` (0/1 match flags)
- `sum` (integer count of matched fields)

If fallback modes are enabled, an additional `match_strategy` column may appear (e.g., `4field`, `3field`, `3field_first`).

---

## Example

Use `vumc.csv` and `litholink,csv` as example, running step 1) Generate candidate matches (get `matcheds.csv`) and 2) Filter matches and attach Litholink columns (get `matches.filtered.csv` and `matches.filtered.with_litholink.csv`)

---

## Programmatic use

```python
from linkage_algorithm import run_linkage

out = run_linkage(
    df_left, df_right,
    left_id="person_id", right_id="PatientID",
    left_colmap={"first_name": "fname", "last_name": "lname", "dob": "birth_date", "sex": "gender"},
    right_colmap={"first_name": "first", "last_name": "last", "dob": "dob", "sex": "sex"},
    min_rule_sum=3,
)
```
