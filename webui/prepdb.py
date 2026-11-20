"""
Build the web UI's SQLite database from the display set and submodel linkages.

    python webui/prepdb.py [--data-dir DIR] [--db DB]

Inputs (DIR, default ~/Documents/dolphin_web_data):
    results_display_set.csv   display_set.py
    submodel_dtw.csv          greedy_seriation_prep.py

Output (default DIR/web.db), rebuilt from scratch on every run:
    display_set   one row per results_id (unique index), indexed by submodel_id
    linkages      model_i, model_j, single, complete, average, medoid
                  (unique index on model_i, model_j)
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

DATA_DIR = Path.home() / "Documents" / "dolphin_web_data"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--db", type=Path)
    args = parser.parse_args()
    db = args.db or args.data_dir / "web.db"

    display = pd.read_csv(args.data_dir / "results_display_set.csv")
    linkages = pd.read_csv(args.data_dir / "submodel_dtw.csv")

    db.unlink(missing_ok=True)
    with sqlite3.connect(db) as con:
        display.to_sql("display_set", con, index=False)
        linkages.to_sql("linkages", con, index=False)
        con.execute("CREATE UNIQUE INDEX display_set_results_id ON display_set (results_id)")
        con.execute("CREATE INDEX display_set_submodel_id ON display_set (submodel_id)")
        con.execute("CREATE UNIQUE INDEX linkages_pair ON linkages (model_i, model_j)")
    con.close()
    print(f"display_set: {len(display)} rows, linkages: {len(linkages)} rows -> {db}")


if __name__ == "__main__":
    main()
