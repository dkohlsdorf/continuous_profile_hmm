"""
Distances for greedy seriation of the web display set.

    python webui/greedy_seriation_prep.py [--data-dir DIR] [--warping-band 5] [--norm path]

Inputs (DIR, default ~/Documents/dolphin_web_data):
    results_display_set.csv     display set (display_set.py)
    embeddings/<results_id>.pkl (frames, dim) arrays

Outputs (DIR):
    results_dtw.csv     result_id_anchor, result_id_comparison, dtw
                        every ordered pair of distinct results; both
                        directions carry the same (symmetric) distance
    submodel_dtw.csv    model_i, model_j, single, complete, average, medoid
                        every ordered pair of distinct submodels; linkages
                        over the result-level DTW distances between their
                        members -- min, max, mean, and the distance between
                        the two submodels' medoids (member with the smallest
                        summed DTW distance to the rest of its submodel)

DTW is lib_phmm's Sakoe-Chiba banded dtw, same defaults as the DTW medoid
filter (warping_band 5, norm 'path').
"""

import argparse
import pickle as pkl
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lib_phmm.hierarchical_kmedian_dtw as hkd  # noqa: E402

DATA_DIR = Path.home() / "Documents" / "dolphin_web_data"
LINKAGES = ("single", "complete", "average", "medoid")


def pairwise_dtw(seqs, warping_band, norm):
    """
    Symmetric (n, n) DTW matrix, zero diagonal. hkd.nearest against a single
    "medoid" is a threaded (GIL-released) one-vs-many dtw, so row i computes
    seqs[i] against seqs[i+1:] in parallel.
    """
    n = len(seqs)
    D = np.zeros((n, n))
    for i in range(n - 1):
        _, dists = hkd.nearest(seqs[i + 1:], [seqs[i]], warping_band, norm)
        D[i, i + 1:] = D[i + 1:, i] = dists
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}")
    return D


def submodel_linkages(D, submodels):
    groups = {m: np.flatnonzero(submodels == m) for m in np.unique(submodels)}
    medoid = {m: idx[np.argmin(D[np.ix_(idx, idx)].sum(axis=1))] for m, idx in groups.items()}
    rows = []
    for mi, a in groups.items():
        for mj, b in groups.items():
            if mi == mj:
                continue
            block = D[np.ix_(a, b)]
            rows.append((mi, mj, block.min(), block.max(), block.mean(), D[medoid[mi], medoid[mj]]))
    return pd.DataFrame(rows, columns=["model_i", "model_j", *LINKAGES])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--warping-band", type=int, default=5)
    parser.add_argument("--norm", default="path", choices=("path", "length", "none"))
    args = parser.parse_args()

    df = pd.read_csv(args.data_dir / "results_display_set.csv").sort_values("results_id")
    ids = df["results_id"].to_numpy()
    seqs = []
    for results_id in ids:
        with open(args.data_dir / "embeddings" / f"{results_id}.pkl", "rb") as f:
            seqs.append(np.asarray(pkl.load(f), dtype=np.float64))
    print(f"{len(seqs)} results, {df['submodel_id'].nunique()} submodels")

    D = pairwise_dtw(seqs, args.warping_band, args.norm)

    anchor, comparison = np.nonzero(~np.eye(len(ids), dtype=bool))
    out = args.data_dir / "results_dtw.csv"
    pd.DataFrame({"result_id_anchor": ids[anchor], "result_id_comparison": ids[comparison],
                  "dtw": D[anchor, comparison]}).to_csv(out, index=False)
    print(f"{len(anchor)} result pairs -> {out}")

    out = args.data_dir / "submodel_dtw.csv"
    links = submodel_linkages(D, df["submodel_id"].to_numpy())
    links.to_csv(out, index=False)
    print(f"{len(links)} submodel pairs -> {out}")


if __name__ == "__main__":
    main()
