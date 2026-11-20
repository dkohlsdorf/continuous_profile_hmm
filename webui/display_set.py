"""
Build the web display set from a find results csv and download its
preprocessing outputs (spectrograms, region_spectrograms, embeddings) from
the webui_preprocessing Drive folder written by preprocessing.py.

    python webui/display_set.py [--results RESULTS_CSV] [--display DISPLAY_CSV]
                                [--run-folder RUN_FOLDER_ID] [--out-dir OUT_DIR]

The display set keeps, per submodel, the N_INSTANCES hits with the highest
mean_llr_score among those classified with confidence > TH_CLASS; submodels
with fewer than 5 such hits are dropped. It is written to DISPLAY_CSV.

If --run-folder is given, files land in OUT_DIR/<subfolder>/<results_id>.<ext>;
files already on disk are skipped, so the script can be rerun after an
interruption. Auth: see gdrive_helper.py.
"""

import argparse
import heapq
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from gdrive_helper import OUTPUT_FOLDER_NAME, SUBFOLDERS, download, find_child, get_drive_service, list_children, thread_service

DATA_DIR = Path.home() / "Documents" / "dolphin_web_data"
N_INSTANCES = 15
TH_CLASS = 0.8


def make_display_set(df):
    selection = {submodel: [] for submodel in set(df['submodel_id'])}
    for _, row in df.iterrows():
        if row['classification_confidence'] > TH_CLASS:
            motif = row['submodel_id']
            if (len(selection[motif]) < N_INSTANCES) or (row['mean_llr_score'] > selection[motif][0][0]):
                if len(selection[motif]) < N_INSTANCES:
                    heapq.heappush(selection[motif], (row['mean_llr_score'], row['results_id']))
                else:
                    heapq.heappushpop(selection[motif], (row['mean_llr_score'], row['results_id']))
    ids = {results_id for heap in selection.values() if len(heap) > 4 for _, results_id in heap}
    return df[df['results_id'].isin(ids)]


def fetch(job):
    file_id, path = job
    tmp = path.with_suffix(path.suffix + ".part")
    download(thread_service(), file_id, tmp)
    tmp.rename(path)


def download_display_set(run_folder, ids, out_dir, workers):
    service = get_drive_service()
    out_id = find_child(service, run_folder, OUTPUT_FOLDER_NAME)
    jobs = []
    for name in SUBFOLDERS:
        (out_dir / name).mkdir(parents=True, exist_ok=True)
        files = list_children(service, find_child(service, out_id, name))
        wanted = [f for f in files if f["name"].rsplit(".", 1)[0] in ids]
        print(f"{name}: {len(wanted)} of {len(ids)} found on Drive")
        jobs += [(f["id"], out_dir / name / f["name"]) for f in wanted
                 if not (out_dir / name / f["name"]).exists()]
    print(f"downloading {len(jobs)} files")
    with ThreadPoolExecutor(workers) as pool:
        for i, _ in enumerate(pool.map(fetch, jobs), start=1):
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=DATA_DIR / "results.csv")
    parser.add_argument("--display", type=Path, default=DATA_DIR / "results_display_set.csv")
    parser.add_argument("--run-folder", help="Drive id of the motif_extend_find_<timestamp> folder; omit to skip download")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    df = make_display_set(pd.read_csv(args.results))
    df.to_csv(args.display, index=False)
    print(f"{len(df)} hits from {df['submodel_id'].nunique()} submodels -> {args.display}")

    if args.run_folder:
        download_display_set(args.run_folder, set(df["results_id"].astype(str)), args.out_dir, args.workers)


if __name__ == "__main__":
    main()
