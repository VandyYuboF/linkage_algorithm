# Linkage Algorithm (Cross-Site)

A small, configurable tool to link patient records across sites using blocking + Jaro–Winkler + exact matches.

## Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

## Run (CSV → CSV)
python linkage_algorithm.py \
  --left vumc.csv --right litholink.csv \
  --left-id person_id --right-id person_id \
  --out matches.csv

# If your column names differ:
  --left-first fname --left-last lname --left-dob birth_date --left-sex gender \
  --right-first first --right-last last --right-dob dob --right-sex sex

# Tuning
  --jw-first 0.90 --jw-last 0.90 --min-sum 3
  --block-on dob sex      # default
  --block-on NONE         # full comparison (careful: slow)

## Programmatic use
from linkage_algorithm import run_linkage
df = run_linkage(df_left, df_right,
                 left_id="person_id", right_id="person_id",
                 left_colmap={"first_name":"fname","last_name":"lname","dob":"birth_date","sex":"gender"},
                 right_colmap={"first_name":"first","last_name":"last","dob":"dob","sex":"sex"})
