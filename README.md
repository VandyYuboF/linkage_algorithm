# Linkage Algorithm (Cross-Site)

### What this tool does (at a glance)

This pipeline links patient records between a local dataset (e.g., VUMC) and a Litholink dataset using conservative name, date-of-birth (dob), and sex matching rules. In the first step, it generates candidate matches based on agreement across identifying fields. In the second step, it filters those candidates to retain only high-confidence matches, attaches Litholink attributes, and separates records into two final outputs: one file containing successfully matched, de-identified Litholink records ready to be sent to USDHub, and another file containing unmatched Litholink records that may require manual review or further investigation.

```bash
Step 1
Local CSV  ──┐
             ├─► linkage_algorithm.py ──► matches.csv
Litholink  ──┘

Step 2
matches.csv + litholink.csv
            └─► filter_matches.py
                  ├─► matches.filtered.with_litholink.csv  (SEND TO USDHUB)
                  └─► litholink.unmatched.csv              (REVIEW)
```

This repository provides a small, configurable CLI tool to link patient records across sites using:

- Blocking (default: `dob` + `sex`)
- Jaro–Winkler string similarity for names
- Exact matches for DOB/sex
- A simple rule score (`sum`) that counts how many fields matched

It also includes a second script to post-filter matches into:
- **4-field matches**: first + last + DOB + sex
- **3-field matches (first + DOB + sex)** with a guardrail:
  - If a Litholink record already has any 4-field match, we do **not** emit any 3-field matches for that Litholink ID.

Data note: This pipeline operates on local files only and does not transmit data externally.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Not sure where to start?
- If you just need results, follow **Quick Start**
- If you need to adjust matching logic, read **Step-by-step guide**
- If you need to adjust in input and ouput, read **Optional utilities**  
- If you are integrating into code, see **Programmatic use**

---

## Quick Start:

If you just want to run the linkage with default settings:

1. Put your local file and Litholink file in this folder. 

**Input requirements (Quick Start)**

- The local file is expected to contain:
  - `USDHubID`
  - `first_name`
  - `last_name`
  - `sex` (recommended format: 'M' for male, 'F' for female)
  - `dob` (recommended format: mm/dd/yyyy)

- The Litholink file is expected to contain:
  - `PatientID` (or another unique patient identifier)
  - `Patient` (full patient name)
  - `Gender` (recommended format: 'M' for male, 'F' for female)
  - `DOB` (recommended format: mm/dd/yyyy)

If your files use different identifier column names, specify them using `--left-id` and `--right-id`.  
You do **not** need to rename your CSV columns.
  
2. Rename them to `local.csv` and `litholink.csv`

3. Run:

```bash
python linkage_algorithm.py \
  --left local.csv --right litholink.csv \
  --left-id USDHubID --right-id PatientID \
  --min-sum 3 \
  --out matches.csv

python filter_matches.py \
  --matches matches.csv \
  --litholink litholink.csv \
  --left-id USDHubID \
  --right-id PatientID \
  --out-filtered matches.filtered.csv \
  --out-merged matches.filtered.with_litholink.csv \
  --out-unmatched-litholink litholink.unmatched.csv
```

4. After running the code, check:

- **`matches.filtered.with_litholink.csv`** should **not be empty**
- **`litholink.unmatched.csv`** should usually contain some rows
- If both files are empty, double-check:
  - Column names
  - ID mappings (`--left-id`, `--right-id`)
  - Date formats in DOB columns

5. After completing the above steps, review the following files:

- **`matches.filtered.with_litholink.csv`**

  Contains all successfully matched, de-identified Litholink records.

  This file can be sent directly to **USDHub**:
  - CHOP (for PEDSnet sites)
  - VUMC (for STAR sites)

- **`litholink.unmatched.csv`**

  Contains Litholink records that could not be matched to any local record. These records may require manual chart review or additional linkage work.

  This file should be shared only with your site’s USDHub PI and research coordinator and must not leave your institution.

---

## Step-by-step guide (with explanations)

### Step 1: Generate candidate matches (matches.csv)

Minimal usage:

```bash
python linkage_algorithm.py \
  --left local.csv --right litholink.csv \
  --left-id USDHubID --right-id PatientID \
  --min-sum 3 \
  --out matches.csv
```

#### Required input columns (defaults)

By default, both files are expected to have:

- `first_name`
- `last_name`
- `dob` (any parseable date; internally normalized to `YYYY-MM-DD`)
- `sex`

If your columns have different names, map them:

```bash
python linkage_algorithm.py \
  --left local.csv --right litholink.csv \
  --left-id USDHubID --right-id PatientID \
  --left-first fname --left-last lname --left-dob birth_date --left-sex gender \
  --right-first first --right-last last --right-dob dob --right-sex sex \
  --min-sum 3 \
  --out matches.csv
```

#### Litholink combined-name convenience (`Patient` / `patient_name`)

Many Litholink exports store the combined patient name in a single column:

- `Patient` (or `patient_name`) like: `LAST, FIRST MIDDLE`

If the RIGHT table does **not** have usable first/last name columns, the linkage script will automatically:
1) detect `Patient` or `patient_name`
2) parse first/last into internal columns
3) proceed with matching using those parsed fields

In this case you typically do **not** need to specify `--right-first/--right-last`.

#### Key scoring/tuning flags

- `--min-sum` (default `3`): minimum number of satisfied rules to keep a pair
- `--jw-first`, `--jw-last` (default `0.85`): Jaro–Winkler thresholds for first/last name
- `--block-on dob sex` (default): reduce comparisons by only comparing within same DOB+sex
- `--block-on NONE`: full cartesian comparison (can be very slow)

Optional fallback passes (off by default):
- `--fallback-3field`: retry unmatched records using a **(last_name + dob + sex)** rule set
- `--fallback-firstname`: retry unmatched records using **(first_name + dob + sex)** (helpful when last name changes)

### Step 2: Filter matches and attach Litholink columns

This script takes:
- `matches.csv` from step (1)
- the original Litholink file (`litholink.csv`) to attach extra columns

It outputs:
- `matches.filtered.csv`: intermediate filtered linkage results (primarily for auditing and debugging)
- `matches.filtered.with_litholink.csv`: the filtered matches joined back to Litholink columns on `PatientID`, this file only includes `USDHubID` and the rest of columns in litholink.csv, except: [`Physician`, `Patient`, `PatientID`, `DOB`, `SampleID`, `CystineSampleID`]
- `litholink.unmatched.csv`: contains all records in `litholink.csv` that could not be matched to any record in `local.csv` after filtering.

Example:

```bash
python filter_matches.py \
  --matches matches.csv \
  --litholink litholink.csv \
  --left-id USDHubID \
  --right-id PatientID \
  --out-filtered matches.filtered.csv \
  --out-merged matches.filtered.with_litholink.csv \
  --out-unmatched-litholink litholink.unmatched.csv
```

### Output schema (matches.csv)

`matches.csv` contains (at minimum):

- `<left-id>` (e.g., `USDHubID`)
- `<right-id>` (e.g., `PatientID`)
- `first_name`, `last_name`, `dob`, `sex` (0/1 match flags)
- `sum` (integer count of matched fields)

If fallback modes are enabled, an additional `match_strategy` column may appear (e.g., `4field`, `3field`, `3field_first`).

---

## Optional utilities

### 1) Add DOB difference for unmatched Litholink candidates

`add_dob_diff_to_unmatched.py` takes:
- `matches.csv` from `linkage_algorithm.py` (candidate pairs where first/last names fuzzy matched per the linkage step)
- `litholink.unmatched.csv` from `filter_matches.py`
- the original `local.csv` and `litholink.csv`

It outputs **all** Litholink–Local pairs (for unmatched Litholink IDs) where **sex matches exactly**, and adds:
- local patient info: `USDHubID`, `local_last_name`, `local_first_name`, `local_sex`, `local_dob`
- `abs_dob_diff_days`

By default it keeps both DOB-equal and DOB-different pairs; add `--dob-diff-only` to keep only `abs_dob_diff_days > 0`.

Example:
```bash
python add_dob_diff_to_unmatched.py
  --matches matches.csv
  --unmatched litholink.unmatched.csv
  --left local.csv --right litholink.csv
  --left-id USDHubID --right-id PatientID
  --left-first-col first_name
  --left-last-col last_name
  --left-sex-col sex
  --right-sex-col Gender
  --left-dob-col dob
  --right-dob-col DOB
  --dob-format "%m/%d/%Y"
  --out litholink.unmatched.dob_review_pairs.csv
```

### 2) Split a full-name column into first_name / last_name

If your input file has a single full-name field (e.g., `Patient`), you can generate
`first_name` and `last_name` columns (while keeping all original columns):

```bash
python name_preprocess_split_full_name.py \
  --input litholink.csv \
  --full-name-col Patient \
  --out litholink.with_split_names.csv
```

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
