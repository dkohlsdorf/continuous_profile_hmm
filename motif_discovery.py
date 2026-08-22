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
  4. decode every candidate (the full, unfiltered pool) against the
     final model and write, per candidate, which sub-model it matched

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

If that model pickle already exists in output_dir, it's loaded instead
of re-running DTW filtering + the sweep (delete it, or pass a different
--model-name, to force a recompute). Candidate embeddings are cached
the same way, in candidates.pkl.

Besides the model pickle and metrics.csv, this also writes:
  - submodel_assignments.csv: one row per candidate (start, stop,
    which sub-model its Viterbi path matched most, its normalized fit
    score) -- submodel_id is -1 if the path visited no match state.
  - submodel_<n>.wav: one file per sub-model id, the raw audio of every
    candidate assigned to it concatenated back to back.

Usage:
    python motif_discovery.py L2.csv L2.wav output_dir
    python motif_discovery.py L2.csv L2.wav output_dir --model-name phmm_l2.pkl
"""
import argparse
import datetime
import json
import random
import time
import wave
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from joblib import Parallel, delayed

from lib_phmm.config import CONFIG
from lib_phmm.model import whisper_model_v2, whisper_processor
from lib_phmm.signals import load_filtered_waveform, process, classify
from lib_phmm.phmm_utils import sweep_one_state_count, make_hmm, decode_all, best_fit
from lib_phmm.metrics import build_msa_from_paths, compute_all_metrics, metrics_to_dataframe
import lib_phmm.profile_hmm as phmm
import lib_phmm.hierarchical_kmedian_dtw as hkd


MAX_MATCH  = 5
MAX_STATES = 32
MIN_STATES = 3


def wav_num_samples(wav_path):
    """Frame count straight from the WAV header -- no decode, so this is
    fast even on a multi-hour file (unlike load_filtered_waveform)."""
    with wave.open(str(wav_path), 'rb') as f:
        return f.getnframes()


def filtered_spans(csv_path, n_samples):
    """
    (start, stop) sample-offset spans from csv_path that survive the
    same short-candidate filter load_candidates() applies, in the same
    order -- so this regenerates the span list load_candidates() would
    have produced without needing candidates.pkl to store it, as long as
    it's called with the same n_samples (from wav_num_samples on the
    same wav_path) both times.
    """
    df = pd.read_csv(csv_path)
    window = CONFIG['window_size_samples']
    spans, skipped = [], 0
    for start, stop in zip(df['starts'], df['stops']):
        start, stop = int(start), min(int(stop), n_samples)
        if stop - start < window:
            skipped += 1
            continue
        spans.append((start, stop))

    if skipped:
        print(f"Skipped {skipped}/{len(df)} candidates shorter than one analysis window ({window} samples)")
    return spans


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
    waveform, sr = load_filtered_waveform(wav_path)
    spans = filtered_spans(csv_path, waveform.shape[1])
    n_rows = len(spans)

    candidates = []
    start_time = time.time() if verbose else None
    for i, (start, stop) in enumerate(spans):
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


def sweep(candidates, max_match=MAX_MATCH, flank_dwell_frames=None, flank_alpha=500.0):
    """
    max_match caps how many sub-models sweep_one_state_count() may
    greedily add (BIC then picks how many of those to actually keep).
    Cost is roughly O(max_match * candidates^2) per n_match_states value
    -- see sweep_one_state_count()/filter_by_keep_fraction()'s
    docstrings -- so raising it much past the default scales the whole
    sweep accordingly, not just this one knob.
    """
    D = candidates[0][0].shape[1]
    n_frames_total = sum(len(embedding) for embedding, _, _ in candidates)
    nested_results = Parallel(n_jobs=-1, backend="loky", verbose=10)(
        delayed(sweep_one_state_count)(n_match_states, candidates, max_match, D, n_frames_total,
                                        flank_dwell_frames=flank_dwell_frames, flank_alpha=flank_alpha)
        for n_match_states in range(MIN_STATES, MAX_STATES)
    )
    return [r for sublist in nested_results for r in sublist]


def assign_submodels(paths, n_states):
    """
    For each Viterbi path, the sub-model index (n) most represented
    among its match-state steps (majority vote across the path), or -1
    if the path visited no match state at all (pure N/B/E/C/J -- no
    well-fit candidate should normally hit this, but a badly-fit one
    could).
    """
    assignments = []
    for path in paths:
        counts = {}
        for step in path:
            if step.state < phmm.MATCH_STATE:
                continue
            idx = step.state - phmm.MATCH_STATE
            n = idx // n_states
            counts[n] = counts.get(n, 0) + 1
        assignments.append(max(counts, key=counts.get) if counts else -1)
    return assignments


def write_submodel_audio(wav_path, spans, submodel_ids, n_models, output_path):
    """
    One submodel_<n>.wav per sub-model id: the raw audio of every
    candidate assigned to it (per assign_submodels), concatenated back
    to back in candidate order. Re-loads/filters the source wav rather
    than reusing anything from load_candidates(), since candidates may
    have come from the candidates.pkl cache this run.
    """
    waveform, sr = load_filtered_waveform(wav_path)
    n_samples = waveform.shape[1]
    for n in range(n_models):
        idxs = [i for i, sid in enumerate(submodel_ids) if sid == n]
        if not idxs:
            print(f"  submodel {n}: no candidates assigned, skipping audio export")
            continue
        clips = [waveform[:, spans[i][0]:min(spans[i][1], n_samples)] for i in idxs]
        concatenated = torch.cat(clips, dim=1)
        out_file = output_path / f"submodel_{n}.wav"
        torchaudio.save(str(out_file), concatenated, sr)
        print(f"  submodel {n}: {len(idxs)} candidates -> {out_file}")

    n_unassigned = sum(1 for sid in submodel_ids if sid == -1)
    if n_unassigned:
        print(f"  {n_unassigned} candidates had no match-state visits (submodel_id=-1), excluded from audio export")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="CSV with starts,stops sample offsets of full motif candidates")
    parser.add_argument("wav_path", help="wav file the CSV offsets index into")
    parser.add_argument("output_path", help="output directory for the model pickle and statistics")
    parser.add_argument("--model-name", default=None, help="filename for the saved model pickle, written under output_path (default: phmm_<wav stem>.pkl). If it already exists, it's loaded instead of recomputing DTW filtering + the sweep.")
    parser.add_argument("--keep-fraction", type=float, default=0.10, help="target fraction of candidates to keep after DTW medoid filtering")
    parser.add_argument("--max-match", type=int, default=MAX_MATCH, help=f"max sub-models the greedy sweep may try (BIC picks how many to keep) -- default {MAX_MATCH}. Cost is roughly O(max_match * filtered_candidates^2) per n_match_states value, so raising this much scales the whole sweep, not just the sub-model cap")
    parser.add_argument("--dtw-warping-band", type=int, default=5, help="Sakoe-Chiba warping band for DTW distance")
    parser.add_argument("--dtw-epochs", type=int, default=20, help="max k-medoids refinement epochs per split")
    parser.add_argument("--dtw-threshold-samples", type=int, default=500, help="random candidate pairs sampled to estimate the filtering threshold")
    parser.add_argument("--well-fit-threshold", type=float, default=34, help="per-sequence normalized Viterbi score below which a candidate counts as well-fit")
    parser.add_argument("--verbose", action="store_true", help="log progress while embedding candidates and while decoding the full candidate pool against the final model")
    parser.add_argument("--flank-dwell-frames", type=float, default=None, help="target expected N/C flank dwell length in frames -- default: mean candidate length (printed at startup), since candidates carry little/no real NOISE for nn/cc to learn from otherwise")
    parser.add_argument("--flank-alpha", type=float, default=500.0, help="pseudocount weight for the flank dwell prior -- higher pulls closer to --flank-dwell-frames, lower lets any real observed NOISE counts matter more")
    args = parser.parse_args()

    dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model_name = args.model_name or f"phmm_{Path(args.wav_path).stem.lower()}.pkl"
    model_path = output_path / model_name
    model_was_cached = model_path.exists()

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

    # spans are derived from csv_path/wav_path rather than cached in
    # candidates.pkl -- they're fully determined by the two, so this stays
    # correct on a cache hit without widening what gets pickled there.
    spans = filtered_spans(args.csv_path, wav_num_samples(args.wav_path))
    assert len(spans) == len(candidates), (
        f"spans ({len(spans)}) don't match cached candidates ({len(candidates)}) -- "
        f"csv_path/wav_path must have changed since candidates.pkl was written; "
        f"delete {candidates_path} and rerun"
    )

    lengths = [len(embedding) for embedding, _, _ in candidates]
    mean_len, median_len, max_len = float(np.mean(lengths)), float(np.median(lengths)), int(np.max(lengths))
    print(f"candidate lengths (frames): mean={mean_len:.1f} median={median_len:.1f} max={max_len}")

    flank_dwell_frames = args.flank_dwell_frames if args.flank_dwell_frames is not None else mean_len
    print(f"flank dwell target: {flank_dwell_frames:.1f} frames (flank_alpha={args.flank_alpha})")

    if model_was_cached:
        print("==========================================")
        print("Model already exists -- loading, not recomputing")
        print("==========================================")
        with open(model_path, "rb") as f:
            model_record = pkl.load(f)
        hmm      = model_record["hmm"]
        n_states = model_record["n_states"]
        n_models = model_record["n_models"]
        best = {
            "n_match_states": model_record["n_match_states"],
            "n_sub_models": model_record["n_sub_models"],
            "bic": model_record["bic"],
        }
        n_filtered = model_record.get("n_candidates_filtered")
        trace = None
        print(f"Loaded model from: {model_path} "
              f"(n_match_states={best['n_match_states']}, n_sub_models={best['n_sub_models']}, BIC={best['bic']:.1f})")
    else:
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
        n_filtered = len(filtered)

        print("==========================================")
        print("Parameter sweep HMM                        ")
        print("==========================================")
        print(f"max_match={args.max_match} (cost scales ~O(max_match * candidates^2) per n_match_states value)")
        results = sweep(filtered, max_match=args.max_match, flank_dwell_frames=flank_dwell_frames, flank_alpha=args.flank_alpha)
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
    metrics_df.insert(2, 'n_candidates_filtered', n_filtered)
    metrics_df.insert(3, 'n_match_states', best['n_match_states'])
    metrics_df.insert(4, 'n_sub_models', best['n_sub_models'])
    metrics_df.insert(5, 'bic', best['bic'])
    metrics_df.insert(6, 'mean_fit', total_fit / len(candidates))
    metrics_df.insert(7, 'well_fit_count', len(well_fits))
    metrics_df.insert(8, 'well_fit_rate', len(well_fits) / len(candidates) * 100)
    metrics_df.to_csv(output_path / "metrics.csv", index=False)
    print(metrics_df.to_string(index=False))

    if trace is not None:
        with open(output_path / "filter_trace.json", "w") as f:
            json.dump(trace, f, indent=2)

    print("==========================================")
    print("Sub-model assignments                      ")
    print("==========================================")
    submodel_ids = assign_submodels(paths, n_states)
    assignments_df = pd.DataFrame({
        "candidate_id": range(len(candidates)),
        "start": [s for s, _ in spans],
        "stop": [e for _, e in spans],
        "submodel_id": submodel_ids,
        "score_norm": scores_norm,
    })
    assignments_path = output_path / "submodel_assignments.csv"
    assignments_df.to_csv(assignments_path, index=False)
    print(f"Saved sub-model assignments to {assignments_path}")

    print("==========================================")
    print("Sub-model audio export                     ")
    print("==========================================")
    write_submodel_audio(args.wav_path, spans, submodel_ids, n_models, output_path)

    if not model_was_cached:
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
            "n_candidates_filtered": n_filtered,
            "max_match": args.max_match,
            "flank_dwell_frames": flank_dwell_frames,
            "flank_alpha": args.flank_alpha,
            "run": dt,
        }
        with open(model_path, "wb") as f:
            pkl.dump(model_record, f)
        print(f"Saved model to {model_path}")
    else:
        print(f"Model at {model_path} was loaded from cache, not re-saved")

    print("==========================================")
    print("Done")
    print("==========================================")
