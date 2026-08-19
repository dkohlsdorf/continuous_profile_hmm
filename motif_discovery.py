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

Because there's no surrounding noise for the model to learn N/C flank
dwell from, estimate_transitions() would otherwise collapse nn/cc to a
1-2 frame dwell (see lib_phmm/phmm_utils.py's _smoothed_with_prior
docstring) -- --flank-dwell-frames/--flank-alpha instead give N/C a
prior expected dwell on the order of a real candidate length.

The fitted ProfileHMM (a pybind11 object) is pickled directly --
Gaussian/FlankTransitions/ProfileHMM all carry __getstate__/__setstate__
(see profile_hmm_bindings.cpp), so `pkl.load()` on the output file
hands back a live, ready-to-decode model:

    with open(output_dir/"phmm_l2.pkl", "rb") as f:
        record = pkl.load(f)
    score, path = lib_phmm.profile_hmm.viterbi(sequence, record["hmm"])

Usage:
    python motif_discovery.py L2.csv L2.wav output_dir
    python motif_discovery.py L2.csv L2.wav output_dir --model-name phmm_l2.pkl
"""
import argparse
import datetime
import json
import random
import time
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


def load_candidates(csv_path, wav_path, model, processor, verbose=False, log_every=1):
    """
    Every row in csv_path is (start, stop) sample offsets into wav_path
    bounding one full motif candidate. No NOISE-classification
    filtering is applied within a candidate -- every analysis window in
    [start, stop) is embedded and kept.

    verbose logs progress every log_every candidates -- each candidate
    is a real (multi-second) Whisper forward pass over its own windows,
    so this loop is comparatively slow per item and worth a tighter
    default log interval than decode_all()'s.
    """
    df = pd.read_csv(csv_path)
    waveform, sr = load_filtered_waveform(wav_path)
    n_samples = waveform.shape[1]
    window = CONFIG['window_size_samples']
    n_rows = len(df)

    candidates = []
    skipped = 0
    start_time = time.time() if verbose else None
    for i, (start, stop) in enumerate(zip(df['starts'], df['stops'])):
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

        if verbose and ((i + 1) % log_every == 0 or i + 1 == n_rows):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            eta = (n_rows - (i + 1)) / rate if rate > 0 else float('inf')
            print(f"  embedded {i + 1}/{n_rows} ({(i + 1) / n_rows * 100:.1f}%) "
                  f"elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta:.1f}s")

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


def sweep(candidates, flank_dwell_frames=None, flank_alpha=500.0):
    D = candidates[0][0].shape[1]
    n_frames_total = sum(len(embedding) for embedding, _, _ in candidates)
    nested_results = Parallel(n_jobs=-1, backend="loky", verbose=10)(
        delayed(sweep_one_state_count)(n_match_states, candidates, MAX_MATCH, D, n_frames_total,
                                        flank_dwell_frames=flank_dwell_frames, flank_alpha=flank_alpha)
        for n_match_states in range(MIN_STATES, MAX_STATES)
    )
    return [r for sublist in nested_results for r in sublist]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="CSV with starts,stops sample offsets of full motif candidates")
    parser.add_argument("wav_path", help="wav file the CSV offsets index into")
    parser.add_argument("output_path", help="output directory for the model pickle and statistics")
    parser.add_argument("--model-name", default=None, help="filename for the saved model pickle, written under output_path (default: phmm_<wav stem>.pkl)")
    parser.add_argument("--keep-fraction", type=float, default=0.10, help="target fraction of candidates to keep after DTW medoid filtering")
    parser.add_argument("--dtw-warping-band", type=int, default=5, help="Sakoe-Chiba warping band for DTW distance")
    parser.add_argument("--dtw-epochs", type=int, default=20, help="max k-medoids refinement epochs per split")
    parser.add_argument("--dtw-threshold-samples", type=int, default=500, help="random candidate pairs sampled to estimate the filtering threshold")
    parser.add_argument("--well-fit-threshold", type=float, default=34, help="per-sequence normalized Viterbi score below which a candidate counts as well-fit")
    parser.add_argument("--verbose", action="store_true", help="log progress while decoding the full candidate pool against the final model")
    parser.add_argument("--flank-dwell-frames", type=float, default=None, help="target expected N/C flank dwell length in frames -- default: mean candidate length (printed at startup), since candidates carry little/no real NOISE for nn/cc to learn from otherwise")
    parser.add_argument("--flank-alpha", type=float, default=500.0, help="pseudocount weight for the flank dwell prior -- higher pulls closer to --flank-dwell-frames, lower lets any real observed NOISE counts matter more")
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
        candidates = load_candidates(args.csv_path, args.wav_path, model, processor, verbose=args.verbose)
        with open(candidates_path, "wb") as f:
            pkl.dump(candidates, f)
    print(f"{len(candidates)} candidates embedded")

    lengths = [len(embedding) for embedding, _, _ in candidates]
    mean_len, median_len, max_len = float(np.mean(lengths)), float(np.median(lengths)), int(np.max(lengths))
    print(f"candidate lengths (frames): mean={mean_len:.1f} median={median_len:.1f} max={max_len}")

    flank_dwell_frames = args.flank_dwell_frames if args.flank_dwell_frames is not None else mean_len
    print(f"flank dwell target: {flank_dwell_frames:.1f} frames (flank_alpha={args.flank_alpha})")

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
    results = sweep(filtered, flank_dwell_frames=flank_dwell_frames, flank_alpha=args.flank_alpha)
    best = min(results, key=lambda r: r["bic"])
    hmm, n_states = make_hmm(best["exemplar_embeddings"], best["exemplar_classifications"], best["n_match_states"],
                              flank_dwell_frames=flank_dwell_frames, flank_alpha=args.flank_alpha)
    n_models = len(hmm.pdf)
    print(f"selected n_match_states={best['n_match_states']}, n_sub_models={best['n_sub_models']}, BIC={best['bic']:.1f}")

    print("==========================================")
    print("Scoring against the full candidate pool    ")
    print("==========================================")
    scores_norm, paths, raw_scores = decode_all(candidates, hmm, verbose=args.verbose)
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

    model_name = args.model_name or f"phmm_{Path(args.wav_path).stem.lower()}.pkl"
    model_path = output_path / model_name
    model_record = {
        "hmm": hmm,
        "n_states": n_states,
        "n_models": n_models,
        "n_match_states": best["n_match_states"],
        "n_sub_models": best["n_sub_models"],
        "bic": best["bic"],
        "csv_path": str(args.csv_path),
        "wav_path": str(args.wav_path),
        "keep_fraction": args.keep_fraction,
        "n_candidates_total": len(candidates),
        "n_candidates_filtered": len(filtered),
        "flank_dwell_frames": flank_dwell_frames,
        "flank_alpha": args.flank_alpha,
        "run": dt,
    }
    with open(model_path, "wb") as f:
        pkl.dump(model_record, f)
    print(f"Saved model to {model_path}")

    print("==========================================")
    print("Done")
    print("==========================================")
