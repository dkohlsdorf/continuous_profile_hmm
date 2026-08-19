"""
motif_discovery.py

Standalone motif-discovery pipeline for one long recording with
pre-annotated candidate regions (e.g. L2.csv/L2.wav, where L2.csv has
`starts,stops` sample offsets into L2.wav bounding each full motif
candidate):

  1. embed every candidate (no NOISE-classification filtering -- the
     candidate boundaries are already given, so every analysis window
     inside [start, stop) is kept)
  2. thin the candidate pool with DTW hierarchical k-medoid filtering
     down to --keep-fraction of the original candidates
  3. run the same BIC-driven match-state / sub-model sweep used by
     processing.py to fit a profile HMM on the filtered pool

Unlike processing.py's --motif-mode-filtered (which decodes every
candidate against its parent *file's* full embedding sequence so the
flank states can absorb the surrounding non-motif audio), there is no
broader file context here to model -- each candidate's own embedding
sequence stands in for both the exemplar material (compressed into
per-column Gaussians) and the sequence that gets Viterbi-decoded /
scored against the final model.

The fitted ProfileHMM is a pybind11 object and is not picklable
directly, so `model.pkl` stores the reconstructable record
(exemplar embeddings/classifications + selected n_match_states)
instead -- rebuild the live model with:

    lib_phmm.phmm_utils.make_hmm(
        record['exemplar_embeddings'],
        record['exemplar_classifications'],
        record['n_match_states'],
    )

Usage:
    python motif_discovery.py L2.csv L2.wav output_dir
"""
import argparse
import datetime
import json
import random
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from lib_phmm.config import CONFIG
from lib_phmm.model import whisper_model_v2, whisper_processor
from lib_phmm.signals import load_filtered_waveform, process, classify
from lib_phmm.phmm_utils import sweep_one_state_count, make_hmm, decode_all, best_fit
from lib_phmm.metrics import build_msa_from_paths, compute_all_metrics, metrics_to_dataframe
import lib_phmm.hierarchical_kmedian_dtw as hkd


MAX_MATCH  = 5
MAX_STATES = 32
MIN_STATES = 3


def load_candidates(csv_path, wav_path, model, processor):
    """
    Every row in csv_path is (start, stop) sample offsets into wav_path
    bounding one full motif candidate. No NOISE-classification
    filtering is applied within a candidate -- every analysis window in
    [start, stop) is embedded and kept.
    """
    df = pd.read_csv(csv_path)
    waveform, sr = load_filtered_waveform(wav_path)
    n_samples = waveform.shape[1]
    window = CONFIG['window_size_samples']

    candidates = []
    skipped = 0
    for start, stop in zip(df['starts'], df['stops']):
        start, stop = int(start), min(int(stop), n_samples)
        if stop - start < window:
            skipped += 1
            continue
        clip = waveform[:, start:stop]
        results = list(process(clip, model, processor))
        embeddings = np.array([r['embeddings'] for r in results])
        embeddings = embeddings.reshape((embeddings.shape[0], embeddings.shape[2]))
        classifications = [classify(r['classifications']) for r in results]
        # region embeddings and the sequence to decode are the same array
        # here -- a candidate already IS the whole unit of interest, with
        # no surrounding file context for the flank states to model.
        candidates.append((embeddings, embeddings, classifications))

    if skipped:
        print(f"Skipped {skipped}/{len(df)} candidates shorter than one analysis window ({window} samples)")
    return candidates


def filter_by_keep_fraction(candidates, keep_fraction, warping_band, epochs,
                             n_samples=500, max_search_iters=15, seed=None):
    """
    hierarchical_kmedian_dtw's kmedoids() is threshold-driven, not
    count-driven, so --keep-fraction is hit by bisecting the threshold,
    expressed as a percentile of sampled pairwise DTW distances: a
    higher threshold makes branches stop splitting sooner -> fewer
    medoids kept, a lower threshold -> more medoids kept.

    kmedoids' internal anchor/second-seed choice is randomized
    (unseeded, on the C++ side), so the count-vs-threshold relationship
    is only approximately monotonic -- this keeps the closest-to-target
    result seen across the search rather than assuming the last
    bisection step is the best one.
    """
    dataset = [[[float(v) for v in frame] for frame in region_embeddings]
               for region_embeddings, _, _ in candidates]
    n = len(dataset)
    target = max(1, round(n * keep_fraction))

    dm = hkd.DistanceManager(dataset, warping_band)

    rng = random.Random(seed)
    max_pairs = n * (n - 1) // 2
    n_samples = min(n_samples, max_pairs)
    pairs = set()
    while len(pairs) < n_samples:
        i, j = rng.randrange(n), rng.randrange(n)
        if i != j:
            pairs.add((min(i, j), max(i, j)))
    sampled_distances = sorted(dm.distance(i, j) for i, j in pairs)

    lo, hi = 0.0, 100.0
    best_ids, best_gap, trace = None, float('inf'), []
    for _ in range(max_search_iters):
        mid = (lo + hi) / 2
        threshold = float(np.percentile(sampled_distances, mid))
        medoid_ids = dm.kmedoids([], epochs, threshold)
        gap = len(medoid_ids) - target
        trace.append({"percentile": mid, "threshold": threshold, "n_kept": len(medoid_ids)})

        if abs(gap) < abs(best_gap):
            best_ids, best_gap = medoid_ids, gap
        if gap == 0:
            break
        if len(medoid_ids) > target:
            lo = mid   # too many kept -> raise the threshold -> more merging
        else:
            hi = mid   # too few kept -> lower the threshold -> less merging

    kept = [candidates[i] for i in best_ids]
    return kept, trace


def sweep(candidates):
    D = candidates[0][0].shape[1]
    n_frames_total = sum(len(embedding) for embedding, _, _ in candidates)
    nested_results = Parallel(n_jobs=-1, backend="loky", verbose=10)(
        delayed(sweep_one_state_count)(n_match_states, candidates, MAX_MATCH, D, n_frames_total)
        for n_match_states in range(MIN_STATES, MAX_STATES)
    )
    return [r for sublist in nested_results for r in sublist]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="CSV with starts,stops sample offsets of full motif candidates")
    parser.add_argument("wav_path", help="wav file the CSV offsets index into")
    parser.add_argument("output_path", help="output directory for model.pkl and statistics")
    parser.add_argument("--keep-fraction", type=float, default=0.10, help="target fraction of candidates to keep after DTW medoid filtering")
    parser.add_argument("--dtw-warping-band", type=int, default=5, help="Sakoe-Chiba warping band for DTW distance")
    parser.add_argument("--dtw-epochs", type=int, default=20, help="max k-medoids refinement epochs per split")
    parser.add_argument("--dtw-threshold-samples", type=int, default=500, help="random candidate pairs sampled to estimate the filtering threshold")
    parser.add_argument("--well-fit-threshold", type=float, default=34, help="per-sequence normalized Viterbi score below which a candidate counts as well-fit")
    args = parser.parse_args()

    dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("==========================================")
    print("Motif discovery: embedding candidates       ")
    print("==========================================")
    candidates_path = output_path / "candidates.pkl"
    if candidates_path.exists():
        with open(candidates_path, "rb") as f:
            candidates = pkl.load(f)
        print(f"Loaded cached candidate embeddings from: {candidates_path}")
    else:
        model      = whisper_model_v2()
        processor  = whisper_processor()
        candidates = load_candidates(args.csv_path, args.wav_path, model, processor)
        with open(candidates_path, "wb") as f:
            pkl.dump(candidates, f)
    print(f"{len(candidates)} candidates embedded")

    print("==========================================")
    print("DTW hierarchical k-medoid filtering        ")
    print("==========================================")
    filtered, trace = filter_by_keep_fraction(
        candidates,
        keep_fraction=args.keep_fraction,
        warping_band=args.dtw_warping_band,
        epochs=args.dtw_epochs,
        n_samples=args.dtw_threshold_samples,
    )
    target = round(len(candidates) * args.keep_fraction)
    print(f"DTW medoid filter: {len(candidates)} candidates -> {len(filtered)} medoids "
          f"(target {args.keep_fraction:.0%} = {target})")

    print("==========================================")
    print("Parameter sweep HMM                        ")
    print("==========================================")
    results = sweep(filtered)
    best = min(results, key=lambda r: r["bic"])
    hmm, n_states = make_hmm(best["exemplar_embeddings"], best["exemplar_classifications"], best["n_match_states"])
    n_models = len(hmm.pdf)
    print(f"selected n_match_states={best['n_match_states']}, n_sub_models={best['n_sub_models']}, BIC={best['bic']:.1f}")

    print("==========================================")
    print("Scoring against the full candidate pool    ")
    print("==========================================")
    scores_norm, paths, raw_scores = decode_all(candidates, hmm)
    well_fits, well_fits_scores = best_fit(scores_norm, args.well_fit_threshold)
    total_fit = sum(scores_norm)

    msa = build_msa_from_paths(candidates, paths, n_models, n_states)
    metrics = compute_all_metrics(msa, len(candidates))
    metrics_df = metrics_to_dataframe(metrics)
    metrics_df.insert(0, 'run', dt)
    metrics_df.insert(1, 'n_candidates_total', len(candidates))
    metrics_df.insert(2, 'n_candidates_filtered', len(filtered))
    metrics_df.insert(3, 'n_match_states', best['n_match_states'])
    metrics_df.insert(4, 'n_sub_models', best['n_sub_models'])
    metrics_df.insert(5, 'bic', best['bic'])
    metrics_df.insert(6, 'mean_fit', total_fit / len(candidates))
    metrics_df.insert(7, 'well_fit_count', len(well_fits))
    metrics_df.insert(8, 'well_fit_rate', len(well_fits) / len(candidates) * 100)
    metrics_df.to_csv(output_path / "metrics.csv", index=False)
    print(metrics_df.to_string(index=False))

    with open(output_path / "filter_trace.json", "w") as f:
        json.dump(trace, f, indent=2)

    model_record = {
        "exemplar_embeddings": best["exemplar_embeddings"],
        "exemplar_classifications": best["exemplar_classifications"],
        "n_match_states": best["n_match_states"],
        "n_sub_models": best["n_sub_models"],
        "bic": best["bic"],
        "csv_path": str(args.csv_path),
        "wav_path": str(args.wav_path),
        "keep_fraction": args.keep_fraction,
        "n_candidates_total": len(candidates),
        "n_candidates_filtered": len(filtered),
        "run": dt,
    }
    with open(output_path / "model.pkl", "wb") as f:
        pkl.dump(model_record, f)
    print(f"Saved model to {output_path / 'model.pkl'}")
    print("(ProfileHMM is a pybind11 object and isn't picklable -- rebuild it with "
          "lib_phmm.phmm_utils.make_hmm(record['exemplar_embeddings'], "
          "record['exemplar_classifications'], record['n_match_states']))")

    print("==========================================")
    print("Done")
    print("==========================================")
