"""
motif_discovery.py

Two subcommands:

  train -- fit a profile HMM motif model from CSV-bounded candidate
  regions in one recording (e.g. L2.csv/L2.wav, where L2.csv has
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

  find -- given a model `train` produced, search a (potentially large,
  unbounded) recording for occurrences of its motifs: embed the whole
  file continuously, Viterbi-decode it in one pass, and read off every
  B..E span the path visits as one motif hit, labeled by which
  sub-model it matched. This falls out of the profile HMM's own
  topology for free -- the E->J->B loop-back already exists so a
  single decode can walk through any number of motif occurrences
  separated by background (N/J/C), so "search" is not a new algorithm,
  just reading segments back off a path the same way
  lib_phmm.visualization.extract_match_wav already does for plotting.

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

Besides the model pickle and metrics.csv, `train` also writes:
  - submodel_assignments.csv: one row per candidate (start, stop,
    which sub-model its Viterbi path matched most, its normalized fit
    score) -- submodel_id is -1 if the path visited no match state.
  - submodel_<n>.wav: one file per sub-model id, the raw audio of every
    candidate assigned to it concatenated back to back.

`find` writes:
  - motif_hits.csv: one row per detected motif occurrence (which
    sub-model, start/stop in samples and seconds, length in frames).
  - submodel_<n>_hits.wav: one file per sub-model id, the raw audio of
    every hit for it concatenated back to back.
  - embeddings.pkl: cached whole-file embedding, since embedding a
    large recording end to end is the slow part here (no candidates.pkl
    equivalent to reuse -- this is a continuous scan, not per-candidate
    slices) and you'll often want to try more than one model against
    the same target file.

Usage:
    python motif_discovery.py train L2.csv L2.wav output_dir
    python motif_discovery.py train L2.csv L2.wav output_dir --model-name phmm_l2.pkl
    python motif_discovery.py find output_dir/phmm_l2.pkl target.wav search_dir --verbose
"""
import argparse
import datetime
import json
import math
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
from lib_phmm.phmm_utils import sweep_one_state_count, score_all_as_submodels, make_hmm, decode_all, best_fit, dwell_to_prior_mean
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


def sweep_all_medoids(candidates, flank_dwell_frames=None, flank_alpha=500.0):
    """
    --all-medoids counterpart to sweep(): every filtered candidate
    becomes a sub-model unconditionally (score_all_as_submodels()), so
    this is one make_hmm()+decode_all() per n_match_states value instead
    of sweep()'s O(max_match * candidates^2) greedy search per value --
    --max-match is unused here. Only n_match_states is still BIC-picked.
    """
    D = candidates[0][0].shape[1]
    n_frames_total = sum(len(embedding) for embedding, _, _ in candidates)
    return Parallel(n_jobs=-1, backend="loky", verbose=10)(
        delayed(score_all_as_submodels)(n_match_states, candidates, D, n_frames_total,
                                         flank_dwell_frames=flank_dwell_frames, flank_alpha=flank_alpha)
        for n_match_states in range(MIN_STATES, MAX_STATES)
    )


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


def write_submodel_audio(wav_path, spans, submodel_ids, n_models, output_path, suffix=""):
    """
    One submodel_<n><suffix>.wav per sub-model id: the raw audio of
    every (span, submodel_id) pair assigned to it, concatenated back to
    back in span order. Used for both training's whole-candidate spans
    (assign_submodels()) and search's motif-hit spans
    (extract_motif_segments()) -- same shape either way: a list of
    (start, stop) sample spans and a parallel list of submodel ids.
    Re-loads/filters the source wav rather than reusing an
    already-loaded one, since the caller's embeddings may have come
    from a pkl cache this run.
    """
    waveform, sr = load_filtered_waveform(wav_path)
    n_samples = waveform.shape[1]
    for n in range(n_models):
        idxs = [i for i, sid in enumerate(submodel_ids) if sid == n]
        if not idxs:
            print(f"  submodel {n}: nothing assigned, skipping audio export")
            continue
        clips = [waveform[:, spans[i][0]:min(spans[i][1], n_samples)] for i in idxs]
        concatenated = torch.cat(clips, dim=1)
        out_file = output_path / f"submodel_{n}{suffix}.wav"
        torchaudio.save(str(out_file), concatenated, sr)
        print(f"  submodel {n}: {len(idxs)} -> {out_file}")

    n_unassigned = sum(1 for sid in submodel_ids if sid == -1)
    if n_unassigned:
        print(f"  {n_unassigned} had no match-state visits (submodel_id=-1), excluded from audio export")


def embed_full_file(wav_path, model, processor, verbose=False, log_every=None):
    """
    Embed an entire (potentially long) recording via the same sliding
    window as load_candidates(), but continuously across the whole file
    instead of per-candidate slices -- for `find`, not `train`. Returns
    (embeddings, step_samples): an (n_windows, D) array and the sample
    hop between windows, needed to convert frame indices in a Viterbi
    path back to sample offsets.
    """
    waveform, sr = load_filtered_waveform(wav_path)
    step_samples = CONFIG['window_size_samples'] // CONFIG['step_denominator']
    n_expected = -(-waveform.shape[1] // step_samples)  # ceil division

    if verbose and log_every is None:
        log_every = max(1, n_expected // 20)

    raw_embeddings = []
    start_time = time.time() if verbose else None
    for i, result in enumerate(process(waveform, model, processor)):
        raw_embeddings.append(result['embeddings'])
        if verbose and ((i + 1) % log_every == 0 or i + 1 == n_expected):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            eta = (n_expected - (i + 1)) / rate if rate > 0 else float('inf')
            print(f"  embedded {i + 1}/{n_expected} windows ({(i + 1) / n_expected * 100:.1f}%) "
                  f"elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta:.1f}s")

    embeddings = np.array(raw_embeddings)
    embeddings = embeddings.reshape((embeddings.shape[0], embeddings.shape[2]))
    return embeddings, step_samples


def extract_motif_segments(path, n_states):
    """
    Walk a Viterbi path and return (start_frame, stop_frame, submodel)
    for each contiguous B..E domain -- one entry per detected motif
    occurrence. Frame indices are into the decoded embedding sequence;
    multiply by embed_full_file()'s step_samples to get sample offsets.

    Segment-tracking logic adapted from
    lib_phmm.visualization.extract_match_wav, stripped of the
    audio-reconstruction machinery that function also does for
    plotting -- not needed here, this only wants the spans.
    """
    segments = []
    seg_start, seg_stop, seg_model = None, None, None
    for step in path:
        if step.state == phmm.B:
            seg_start, seg_stop, seg_model = None, None, None
        elif step.state == phmm.E:
            if seg_start is not None:
                segments.append((seg_start, seg_stop, seg_model))
            seg_start, seg_stop, seg_model = None, None, None
        elif step.state >= phmm.MATCH_STATE:
            idx = step.state - phmm.MATCH_STATE
            n = idx // n_states
            if seg_start is None:
                seg_start = step.time
                seg_model = n
            seg_stop = step.time
    return segments


def run_train(args):
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
        if args.all_medoids:
            print(f"all_medoids: every one of {n_filtered} filtered candidates becomes a sub-model (--max-match ignored)")
            results = sweep_all_medoids(filtered, flank_dwell_frames=flank_dwell_frames, flank_alpha=args.flank_alpha)
        else:
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
            "all_medoids": args.all_medoids,
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


def run_find(args):
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("==========================================")
    print("Motif search: loading model                ")
    print("==========================================")
    with open(args.model_path, "rb") as f:
        model_record = pkl.load(f)
    hmm      = model_record["hmm"]
    n_states = model_record["n_states"]
    n_models = model_record["n_models"]
    print(f"Loaded model from {args.model_path} "
          f"(n_match_states={model_record['n_match_states']}, n_sub_models={model_record['n_sub_models']})")

    # `ec`/`ej` (E -> C "done, background forever" vs E -> J "loop back,
    # keep scanning for another occurrence") were estimated from training
    # candidates, each of which contains exactly one motif occurrence by
    # construction -- estimate_transitions() only ever observes the "one
    # motif then exit" event (ec_hits_total += 1, unconditionally, once
    # per exemplar) and never an ej event, so ec comes out of training
    # near 1 regardless of the data. C has no transition back out of it
    # in this topology, so left as trained, Viterbi would exit to C after
    # the *first* hit here and silently absorb every later occurrence
    # into one long C run instead of finding it. Override for search:
    # keep the model looping (E->J->B) so multiple hits in one long file
    # actually get found.
    hmm.trans.ej = math.log2(args.repeat_prob)
    hmm.trans.ec = math.log2(1.0 - args.repeat_prob)

    # jj/jb (J's own self-transition -- how long the model tolerates
    # dwelling in J, i.e. background *between* two motif occurrences,
    # before giving up and trying B anyway) are hardcoded constants in
    # phmm_utils.py (DEFAULT_JJ=0.01), never estimated from data --
    # mean dwell ~1 frame. That's fine once ej routes the path into J at
    # all, but the "memory" of having just exited a match decays almost
    # immediately (~6.6 bits/frame), so any real gap of more than a
    # couple of frames between occurrences already loses the next hit
    # before it starts. Override with a gap tolerance sized for this
    # search, same dwell_to_prior_mean() math train's --flank-dwell-frames
    # uses for N/C.
    jj = dwell_to_prior_mean(args.gap_dwell_frames)
    hmm.trans.jj = math.log2(jj)
    hmm.trans.jb = math.log2(1.0 - jj)

    print("==========================================")
    print("Motif search: embedding target file        ")
    print("==========================================")
    embeddings_path = output_path / "embeddings.pkl"
    if embeddings_path.exists():
        with open(embeddings_path, "rb") as f:
            embeddings, step_samples = pkl.load(f)
        print(f"Loaded cached embeddings from: {embeddings_path}")
    else:
        whisper_model = whisper_model_v2()
        processor     = whisper_processor()
        embeddings, step_samples = embed_full_file(args.wav_path, whisper_model, processor, verbose=args.verbose)
        with open(embeddings_path, "wb") as f:
            pkl.dump((embeddings, step_samples), f)
    print(f"{len(embeddings)} windows embedded ({step_samples} samples/window hop)")

    print("==========================================")
    print("Motif search: decoding                     ")
    print("==========================================")
    score, path = phmm.viterbi(embeddings, hmm)
    print(f"Viterbi score: {score:.1f}")

    segments = extract_motif_segments(path, n_states)
    print(f"Found {len(segments)} motif occurrences")

    hit_spans = [(start * step_samples, (stop + 1) * step_samples) for start, stop, _ in segments]
    hit_submodel_ids = [model for _, _, model in segments]

    print("==========================================")
    print("Motif search: writing hits                 ")
    print("==========================================")
    hits_df = pd.DataFrame({
        "hit_id": range(len(segments)),
        "submodel_id": hit_submodel_ids,
        "start_sample": [s for s, _ in hit_spans],
        "stop_sample": [e for _, e in hit_spans],
        "start_time_s": [s / CONFIG['sampling_rate'] for s, _ in hit_spans],
        "stop_time_s": [e / CONFIG['sampling_rate'] for _, e in hit_spans],
        "n_frames": [stop - start + 1 for start, stop, _ in segments],
    })
    hits_path = output_path / "motif_hits.csv"
    hits_df.to_csv(hits_path, index=False)
    print(f"Saved {len(segments)} hits to {hits_path}")

    print("==========================================")
    print("Motif search: per-submodel hit audio       ")
    print("==========================================")
    if segments:
        write_submodel_audio(args.wav_path, hit_spans, hit_submodel_ids, n_models, output_path, suffix="_hits")
    else:
        print("  no hits, nothing to export")

    print("==========================================")
    print("Done")
    print("==========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train", help="fit a profile HMM motif model from CSV-bounded candidate regions")
    train_parser.add_argument("csv_path", help="CSV with starts,stops sample offsets of full motif candidates")
    train_parser.add_argument("wav_path", help="wav file the CSV offsets index into")
    train_parser.add_argument("output_path", help="output directory for the model pickle and statistics")
    train_parser.add_argument("--model-name", default=None, help="filename for the saved model pickle, written under output_path (default: phmm_<wav stem>.pkl). If it already exists, it's loaded instead of recomputing DTW filtering + the sweep.")
    train_parser.add_argument("--keep-fraction", type=float, default=0.10, help="target fraction of candidates to keep after DTW medoid filtering")
    train_parser.add_argument("--max-match", type=int, default=MAX_MATCH, help=f"max sub-models the greedy sweep may try (BIC picks how many to keep) -- default {MAX_MATCH}. Cost is roughly O(max_match * filtered_candidates^2) per n_match_states value, so raising this much scales the whole sweep, not just the sub-model cap. Ignored if --all-medoids is set.")
    train_parser.add_argument("--all-medoids", action="store_true", help="skip greedy exemplar selection -- every DTW-filtered candidate becomes a sub-model unconditionally (n_sub_models = n_candidates_filtered). Only n_match_states is still BIC-picked. Much cheaper than raising --max-match to cover the whole filtered pool, since it skips the O(max_match * candidates^2) search entirely.")
    train_parser.add_argument("--dtw-warping-band", type=int, default=5, help="Sakoe-Chiba warping band for DTW distance")
    train_parser.add_argument("--dtw-epochs", type=int, default=20, help="max k-medoids refinement epochs per split")
    train_parser.add_argument("--dtw-threshold-samples", type=int, default=500, help="random candidate pairs sampled to estimate the filtering threshold")
    train_parser.add_argument("--well-fit-threshold", type=float, default=34, help="per-sequence normalized Viterbi score below which a candidate counts as well-fit")
    train_parser.add_argument("--verbose", action="store_true", help="log progress while embedding candidates and while decoding the full candidate pool against the final model")
    train_parser.add_argument("--flank-dwell-frames", type=float, default=None, help="target expected N/C flank dwell length in frames -- default: mean candidate length (printed at startup), since candidates carry little/no real NOISE for nn/cc to learn from otherwise")
    train_parser.add_argument("--flank-alpha", type=float, default=500.0, help="pseudocount weight for the flank dwell prior -- higher pulls closer to --flank-dwell-frames, lower lets any real observed NOISE counts matter more")

    find_parser = subparsers.add_parser("find", help="search a fitted model's motifs in a (potentially large) recording")
    find_parser.add_argument("model_path", help="model pickle produced by `train` (e.g. output_dir/phmm_l2.pkl)")
    find_parser.add_argument("wav_path", help="recording to search for motifs")
    find_parser.add_argument("output_path", help="output directory for motif_hits.csv, embeddings.pkl cache, and submodel_<n>_hits.wav")
    find_parser.add_argument("--verbose", action="store_true", help="log progress while embedding the target file")
    find_parser.add_argument("--repeat-prob", type=float, default=0.999, help="probability of looping back (E->J->B) to keep scanning for another occurrence after a hit, instead of exiting to background forever (E->C, a dead end in this topology) -- overrides the trained ec/ej, which come from single-occurrence training candidates and are unusable for scanning a long file as-is (see run_find()'s comment). Default strongly favors continuing to scan.")
    find_parser.add_argument("--gap-dwell-frames", type=float, default=100.0, help="expected background gap (in frames) the model should tolerate between two motif occurrences before giving up on finding another one nearby -- overrides jj/jb, hardcoded elsewhere to a ~1-frame dwell (fine within training, unusable for scanning a file with real gaps between hits). Generous default: underestimating this risks missing real hits, overestimating mainly costs a bit of false-positive risk once inside a real match.")

    args = parser.parse_args()
    if args.mode == "train":
        run_train(args)
    else:
        run_find(args)
