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
    sub-model, start/stop in samples and seconds, length in frames,
    mean_emission_score/mean_noise_score/mean_llr_score,
    classification/classification_confidence -- see
    extract_motif_segments()). mean_noise_score/mean_llr_score are only
    populated when --noise-wav is given; classification/
    classification_confidence come from the Whisper model's own
    BURST/ECHO/UP/DOWN/NOISE classifier head (majority vote + agreement
    fraction over the hit's matched frames), always populated.
  - submodel_<n>_hits.wav: one file per sub-model id, the raw audio of
    every hit for it concatenated back to back, with --hit-gap-seconds
    of silence between consecutive hits (default 0.5s) so occurrence
    boundaries are audible/visible instead of every hit running
    seamlessly into the next. Any submodel_*_hits.wav left in
    output_path from a prior find run against the same output_path is
    cleared first, so re-running (e.g. against cached embeddings.pkl
    with different --noise-* settings) doesn't mix stale clips from the
    old run with fresh ones.
  - embeddings.pkl: cached (embeddings, classifications, step_samples)
    for the whole file, since embedding a large recording end to end is
    the slow part here (no candidates.pkl equivalent to reuse -- this is
    a continuous scan, not per-candidate slices) and you'll often want
    to try more than one model against the same target file. A cache
    written before classifications were added is a 2-tuple and won't
    unpack against this -- delete it to force a recompute.

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
from collections import Counter
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

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

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


def write_submodel_audio(wav_path, spans, submodel_ids, n_models, output_path, suffix="", gap_seconds=0.0):
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

    gap_seconds > 0 inserts that much digital silence between
    consecutive clips (none before the first or after the last) so
    where one hit ends and the next begins is audible/visible in a
    waveform view, instead of every occurrence running seamlessly into
    the next. Default 0.0 keeps clips back to back, unchanged.
    """
    waveform, sr = load_filtered_waveform(wav_path)
    n_samples = waveform.shape[1]
    gap_samples = int(round(gap_seconds * sr))
    gap = torch.zeros((waveform.shape[0], gap_samples)) if gap_samples > 0 else None
    for n in range(n_models):
        idxs = [i for i, sid in enumerate(submodel_ids) if sid == n]
        if not idxs:
            print(f"  submodel {n}: nothing assigned, skipping audio export")
            continue
        clips = [waveform[:, spans[i][0]:min(spans[i][1], n_samples)] for i in idxs]
        if gap is not None:
            interleaved = [clips[0]]
            for clip in clips[1:]:
                interleaved.append(gap)
                interleaved.append(clip)
            clips = interleaved
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
    (embeddings, classifications, step_samples): an (n_windows, D) array,
    a parallel list of per-window BURST/ECHO/UP/DOWN/NOISE labels (see
    lib_phmm.signals.classify()/LABEL_MAP), and the sample hop between
    windows, needed to convert frame indices in a Viterbi path back to
    sample offsets.

    model.forward() already computes both outputs from the same pass
    over each window (see WDPWhisperEmbModelV2.forward() in
    lib_phmm/model.py) -- classifications used to be dropped on the
    floor here, so re-deriving them for an already-embedded recording
    meant re-running the model over its audio all over again. Capturing
    them alongside embeddings on this same pass costs nothing extra.
    """
    waveform, sr = load_filtered_waveform(wav_path)
    step_samples = CONFIG['window_size_samples'] // CONFIG['step_denominator']
    n_expected = -(-waveform.shape[1] // step_samples)  # ceil division

    if verbose and log_every is None:
        log_every = max(1, n_expected // 20)

    raw_embeddings = []
    classifications = []
    start_time = time.time() if verbose else None
    for i, result in enumerate(process(waveform, model, processor)):
        raw_embeddings.append(result['embeddings'])
        classifications.append(classify(result['classifications']))
        if verbose and ((i + 1) % log_every == 0 or i + 1 == n_expected):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            eta = (n_expected - (i + 1)) / rate if rate > 0 else float('inf')
            print(f"  embedded {i + 1}/{n_expected} windows ({(i + 1) / n_expected * 100:.1f}%) "
                  f"elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta:.1f}s")

    embeddings = np.array(raw_embeddings)
    embeddings = embeddings.reshape((embeddings.shape[0], embeddings.shape[2]))
    return embeddings, classifications, step_samples


def extract_motif_segments(path, hmm, n_states, embeddings, noise_pdf=None, classifications=None):
    """
    Walk a Viterbi path and return (start_frame, stop_frame, submodel,
    mean_score, mean_noise_score, mean_llr_score, majority_class,
    majority_fraction) for each contiguous B..E domain -- one entry per
    detected motif occurrence. Frame indices are into the decoded
    embedding sequence; multiply by embed_full_file()'s step_samples to
    get sample offsets.

    mean_score is the mean per-frame emission log2-likelihood
    (hmm.pdf[n][k].ll(embeddings[time]), same conversion viterbi() uses
    internally) over the segment's matched frames -- a match-quality
    signal that's independent of apply_search_transitions()'s ec/ej/jj/jb
    overrides, so it's useful for filtering false positives that the
    loop-back bias let through on transition-prior cheapness alone
    rather than genuine acoustic similarity.

    mean_noise_score/mean_llr_score are only populated when noise_pdf is
    given (the --noise-wav mixture model, see build_noise_model()) --
    None otherwise. mean_noise_score is the same matched frames' mean
    log2-likelihood under noise_pdf instead of the match states.
    mean_llr_score is mean_score - mean_noise_score: since both are
    log2-likelihoods, their difference is the log2 likelihood ratio
    motif-vs-noise (log(P_match/P_noise), base 2) -- positive means the
    segment fits the motif model better than background even after
    accounting for how plausible that stretch of audio is as noise,
    which mean_score alone can't tell you (a low mean_score can still
    beat an even-lower-likelihood noise stretch, or a high mean_score
    can be unremarkable if that region is generically easy to score high
    under noise too).

    majority_class/majority_fraction are only populated when
    classifications is given (embed_full_file()'s per-window
    BURST/ECHO/UP/DOWN/NOISE labels, aligned frame-for-frame with
    embeddings) -- None otherwise. majority_class is the most common
    label among the segment's matched frames (collections.Counter over
    classifications[step.time] for each matched frame); majority_fraction
    is that count divided by the segment's frame count, i.e. how
    internally consistent the segment's classification is (1.0 = every
    frame agreed).

    Segment-tracking logic adapted from
    lib_phmm.visualization.extract_match_wav, stripped of the
    audio-reconstruction machinery that function also does for
    plotting -- not needed here, this only wants the spans (+ scores).
    """
    segments = []
    seg_start, seg_stop, seg_model, seg_scores, seg_noise_scores, seg_classes = None, None, None, [], [], []
    for step in path:
        if step.state == phmm.B:
            seg_start, seg_stop, seg_model, seg_scores, seg_noise_scores, seg_classes = None, None, None, [], [], []
        elif step.state == phmm.E:
            if seg_start is not None:
                mean_score = sum(seg_scores) / len(seg_scores)
                if noise_pdf is not None:
                    mean_noise_score = sum(seg_noise_scores) / len(seg_noise_scores)
                    mean_llr_score = mean_score - mean_noise_score
                else:
                    mean_noise_score, mean_llr_score = None, None
                if classifications is not None:
                    label, count = Counter(seg_classes).most_common(1)[0]
                    majority_class, majority_fraction = label, count / len(seg_classes)
                else:
                    majority_class, majority_fraction = None, None
                segments.append((seg_start, seg_stop, seg_model, mean_score, mean_noise_score, mean_llr_score,
                                  majority_class, majority_fraction))
            seg_start, seg_stop, seg_model, seg_scores, seg_noise_scores, seg_classes = None, None, None, [], [], []
        elif step.state >= phmm.MATCH_STATE:
            idx = step.state - phmm.MATCH_STATE
            n, k = idx // n_states, idx % n_states
            if seg_start is None:
                seg_start = step.time
                seg_model = n
            seg_stop = step.time
            seg_scores.append(hmm.pdf[n][k].ll(embeddings[step.time]) / math.log(2.0))
            if noise_pdf is not None:
                seg_noise_scores.append(noise_pdf.ll(embeddings[step.time]) / math.log(2.0))
            if classifications is not None:
                seg_classes.append(classifications[step.time])
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


def apply_search_transitions(hmm, repeat_prob, gap_dwell_frames):
    """
    Mutate a loaded model's flank transitions in place for `find`'s
    "many occurrences in one long file" decode -- both overrides are
    read-write pybind11 fields, no retraining needed.

    `ec`/`ej` (E -> C "done, background forever" vs E -> J "loop back,
    keep scanning for another occurrence") were estimated from training
    candidates, each of which contains exactly one motif occurrence by
    construction -- estimate_transitions() only ever observes the "one
    motif then exit" event (ec_hits_total += 1, unconditionally, once
    per exemplar) and never an ej event, so ec comes out of training
    near 1 regardless of the data. C has no transition back out of it
    in this topology, so left as trained, Viterbi would exit to C after
    the *first* hit here and silently absorb every later occurrence
    into one long C run instead of finding it.

    jj/jb (J's own self-transition -- how long the model tolerates
    dwelling in J, i.e. background *between* two motif occurrences,
    before giving up and trying B anyway) are hardcoded constants in
    phmm_utils.py (DEFAULT_JJ=0.01), never estimated from data -- mean
    dwell ~1 frame. That's fine once ej routes the path into J at all,
    but the "memory" of having just exited a match decays almost
    immediately (~6.6 bits/frame), so any real gap of more than a
    couple of frames between occurrences already loses the next hit
    before it starts. gap_dwell_frames sizes a gap tolerance for this
    search, same dwell_to_prior_mean() math train's --flank-dwell-frames
    uses for N/C.
    """
    hmm.trans.ej = math.log2(repeat_prob)
    hmm.trans.ec = math.log2(1.0 - repeat_prob)
    jj = dwell_to_prior_mean(gap_dwell_frames)
    hmm.trans.jj = math.log2(jj)
    hmm.trans.jb = math.log2(1.0 - jj)


def build_noise_model(noise_wav_path, output_path, n_components=None, bayesian=False,
                       var_floor=0.1, var_scale=1.0, verbose=False, seed=None):
    """
    Ad-hoc "background" emission model for find --noise-wav: a
    MixtureModel fit to the whole file's embeddings. Find-only, never
    touches train/processing.py or the saved model -- see viterbi()'s
    optional noise_pdf argument (profile_hmm.hpp) for why giving N/J/C a
    real emission model helps: without it they're silent/transition-only,
    so a frame only has to be cheaper than the transition cost to get
    pulled into a match run, with no competing "this looks like
    background" hypothesis scored against the same data.

    n_components=None (default): single Gaussian, no EM -- mean/var
    computed directly, weight 1 (log_weight=0.0). This is exactly a
    1-component MixtureModel, verified to score identically to a bare
    Gaussian.

    n_components=k, bayesian=False (default when k given):
    sklearn.mixture.GaussianMixture(n_components=k, covariance_type='diag',
    init_params='k-means++') -- exactly k components.

    n_components=k, bayesian=True: sklearn.mixture.BayesianGaussianMixture
    with k as an upper bound, not a target -- its Dirichlet process
    weight-concentration prior drives unneeded components' weights
    toward ~0 on its own (verified: capping at 6 on 2 true clusters left
    4 components at ~0 weight), so you don't have to search for the
    right k yourself. Components whose weight is ~0 (<1e-3) are dropped
    entirely rather than kept as dead weight in the mixture.

    Both GaussianMixture and BayesianGaussianMixture expose the same
    post-fit shape (means_/covariances_/weights_), so both paths share
    the same extraction code below. sklearn's weights_ are raw
    probabilities either way, not log -- MixtureModel expects
    log_weights, so these get np.log()'d before construction.

    var_floor (matches compress()'s default in lib_phmm/compression.py)
    guards against a near-zero-variance dimension blowing up
    Gaussian.ll()'s division, for every path.

    var_scale > 1.0 inflates every fitted covariance (after the floor is
    applied) by this factor -- a cheap knob for "the noise pdf is too
    narrow, real background keeps out-scoring it as motif" without
    needing to collect more/broader noise recordings. Widening the
    Gaussians raises the noise pdf's likelihood over a larger region of
    embedding space, so more background frames score competitively
    against match states in viterbi(). Trade-off: too high starts
    swallowing real motif frames into background too. Default 1.0 keeps
    today's fit untouched.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_path / "noise_embeddings.pkl"
    if embeddings_path.exists():
        with open(embeddings_path, "rb") as f:
            embeddings, _ = pkl.load(f)
        print(f"Loaded cached noise embeddings from: {embeddings_path}")
    else:
        whisper_model = whisper_model_v2()
        processor     = whisper_processor()
        # noise model only fits a Gaussian/GMM over embeddings -- per-window
        # classifications aren't needed here (unlike find_in_file's
        # embeddings.pkl), so they're computed (free, same forward pass)
        # but not cached.
        embeddings, _, step_samples = embed_full_file(noise_wav_path, whisper_model, processor, verbose=verbose)
        with open(embeddings_path, "wb") as f:
            pkl.dump((embeddings, step_samples), f)
    print(f"Noise model fit from {len(embeddings)} windows of {noise_wav_path}")

    if n_components is None:
        mean = embeddings.mean(axis=0)
        var = np.maximum(embeddings.var(axis=0), var_floor) * var_scale
        scale_note = f", var_scale={var_scale}" if var_scale != 1.0 else ""
        print(f"Noise model: single Gaussian (no EM){scale_note}")
        return phmm.MixtureModel([phmm.Gaussian(mean, var)], [0.0])

    if bayesian:
        gmm = BayesianGaussianMixture(n_components=n_components, covariance_type='diag',
                                       init_params='k-means++',
                                       weight_concentration_prior_type='dirichlet_process',
                                       random_state=seed)
    else:
        gmm = GaussianMixture(n_components=n_components, covariance_type='diag',
                               init_params='k-means++', random_state=seed)
    gmm.fit(embeddings)

    active = gmm.weights_ > 1e-3
    components = [
        phmm.Gaussian(gmm.means_[i], np.maximum(gmm.covariances_[i], var_floor) * var_scale)
        for i in range(n_components) if active[i]
    ]
    log_weights = np.log(gmm.weights_[active]).tolist()  # weights_ are raw probabilities, not log
    kind = f"Bayesian GMM (Dirichlet process, cap k={n_components})" if bayesian else f"GMM (exactly k={n_components})"
    scale_note = f", var_scale={var_scale}" if var_scale != 1.0 else ""
    print(f"Noise model: {kind}, diag covariance, k-means++ init{scale_note}, "
          f"{active.sum()}/{n_components} active components, weights={[f'{w:.3f}' for w in gmm.weights_[active]]}")
    return phmm.MixtureModel(components, log_weights)


def find_in_file(wav_path, output_path, hmm, n_states, n_models, noise_pdf=None, verbose=False, hit_gap_seconds=0.5):
    """
    Search one recording for hmm's motifs: embed it (cached to
    embeddings.pkl under output_path), Viterbi-decode in one pass, and
    write motif_hits.csv + submodel_<n>_hits.wav under output_path,
    clearing any submodel_*_hits.wav left there from a prior run first.
    Returns the segments found (see extract_motif_segments()).
    """
    output_path.mkdir(parents=True, exist_ok=True)

    print("==========================================")
    print("Motif search: embedding target file        ")
    print("==========================================")
    embeddings_path = output_path / "embeddings.pkl"
    if embeddings_path.exists():
        with open(embeddings_path, "rb") as f:
            embeddings, classifications, step_samples = pkl.load(f)
        print(f"Loaded cached embeddings from: {embeddings_path}")
    else:
        whisper_model = whisper_model_v2()
        processor     = whisper_processor()
        embeddings, classifications, step_samples = embed_full_file(wav_path, whisper_model, processor, verbose=verbose)
        with open(embeddings_path, "wb") as f:
            pkl.dump((embeddings, classifications, step_samples), f)
    print(f"{len(embeddings)} windows embedded ({step_samples} samples/window hop)")

    print("==========================================")
    print("Motif search: decoding                     ")
    print("==========================================")
    score, path = phmm.viterbi(embeddings, hmm, noise_pdf)
    print(f"Viterbi score: {score:.1f}")

    segments = extract_motif_segments(path, hmm, n_states, embeddings, noise_pdf=noise_pdf,
                                       classifications=classifications)
    print(f"Found {len(segments)} motif occurrences")

    hit_spans = [(start * step_samples, (stop + 1) * step_samples) for start, stop, _, _, _, _, _, _ in segments]
    hit_submodel_ids = [model for _, _, model, _, _, _, _, _ in segments]
    hit_scores = [score for _, _, _, score, _, _, _, _ in segments]
    hit_noise_scores = [noise_score for _, _, _, _, noise_score, _, _, _ in segments]
    hit_llr_scores = [llr_score for _, _, _, _, _, llr_score, _, _ in segments]
    hit_classes = [cls for _, _, _, _, _, _, cls, _ in segments]
    hit_class_confidences = [conf for _, _, _, _, _, _, _, conf in segments]

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
        "n_frames": [stop - start + 1 for start, stop, _, _, _, _, _, _ in segments],
        "mean_emission_score": hit_scores,
        "mean_noise_score": hit_noise_scores,
        "mean_llr_score": hit_llr_scores,
        "classification": hit_classes,
        "classification_confidence": hit_class_confidences,
    })
    hits_path = output_path / "motif_hits.csv"
    hits_df.to_csv(hits_path, index=False)
    print(f"Saved {len(segments)} hits to {hits_path}")

    print("==========================================")
    print("Motif search: per-submodel hit audio       ")
    print("==========================================")
    # Clear any submodel_<n>_hits.wav left from a prior run in this same
    # output_path before writing this run's clips -- e.g. re-running find
    # against cached embeddings.pkl with a different --noise-var-scale
    # would otherwise mix stale and fresh clips under the same names/ids
    # with no way to tell them apart. This run's clips are kept (not
    # deleted after writing).
    stale_hit_wavs = sorted(output_path.glob("submodel_*_hits.wav"))
    for f in stale_hit_wavs:
        f.unlink()
    if stale_hit_wavs:
        print(f"  cleared {len(stale_hit_wavs)} submodel_*_hits.wav file(s) from a prior run")
    if segments:
        write_submodel_audio(wav_path, hit_spans, hit_submodel_ids, n_models, output_path,
                              suffix="_hits", gap_seconds=hit_gap_seconds)
    else:
        print("  no hits, nothing to export")

    return segments


def run_find(args):
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

    apply_search_transitions(hmm, args.repeat_prob, args.gap_dwell_frames)

    output_path = Path(args.output_path)
    wav_path = Path(args.wav_path)

    noise_pdf = None
    if args.noise_wav is not None:
        if args.noise_bayesian and args.noise_components is None:
            raise SystemExit("--noise-bayesian needs --noise-components (used as the "
                              "upper bound on components, not an exact count)")
        print("==========================================")
        print("Motif search: building noise model         ")
        print("==========================================")
        noise_pdf = build_noise_model(Path(args.noise_wav), output_path,
                                       n_components=args.noise_components, bayesian=args.noise_bayesian,
                                       var_scale=args.noise_var_scale, verbose=args.verbose)

    if wav_path.is_dir():
        wav_files = sorted(wav_path.glob("*.wav"))
        print(f"{args.wav_path} is a directory -- found {len(wav_files)} wav files")
        if not wav_files:
            print("Nothing to search, exiting")
            return
        # one motif_hits.csv/embeddings.pkl/submodel_<n>_hits.wav set per
        # wav file, namespaced under output_path/<wav stem>/ so files
        # can't collide with each other's output. noise_pdf (if any) is
        # built once above and reused across every file.
        for i, wav_file in enumerate(wav_files):
            print("==========================================")
            print(f"[{i + 1}/{len(wav_files)}] {wav_file.name}")
            print("==========================================")
            find_in_file(wav_file, output_path / wav_file.stem, hmm, n_states, n_models,
                         noise_pdf=noise_pdf, verbose=args.verbose, hit_gap_seconds=args.hit_gap_seconds)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        find_in_file(wav_path, output_path, hmm, n_states, n_models, noise_pdf=noise_pdf,
                     verbose=args.verbose, hit_gap_seconds=args.hit_gap_seconds)

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
    find_parser.add_argument("wav_path", help="recording to search for motifs, or a directory of .wav files to search each of independently (one output_path/<wav stem>/ subfolder per file)")
    find_parser.add_argument("output_path", help="output directory for motif_hits.csv, embeddings.pkl cache, and submodel_<n>_hits.wav (any submodel_*_hits.wav left here from a prior find run against this output_path is cleared before this run's clips are written)")
    find_parser.add_argument("--verbose", action="store_true", help="log progress while embedding the target file")
    find_parser.add_argument("--repeat-prob", type=float, default=0.999, help="probability of looping back (E->J->B) to keep scanning for another occurrence after a hit, instead of exiting to background forever (E->C, a dead end in this topology) -- overrides the trained ec/ej, which come from single-occurrence training candidates and are unusable for scanning a long file as-is (see run_find()'s comment). Default strongly favors continuing to scan.")
    find_parser.add_argument("--gap-dwell-frames", type=float, default=100.0, help="expected background gap (in frames) the model should tolerate between two motif occurrences before giving up on finding another one nearby -- overrides jj/jb, hardcoded elsewhere to a ~1-frame dwell (fine within training, unusable for scanning a file with real gaps between hits). Generous default: underestimating this risks missing real hits, overestimating mainly costs a bit of false-positive risk once inside a real match.")
    find_parser.add_argument("--noise-wav", default=None, help="optional recording of background/non-motif audio. If given, N/J/C (background states) get a real emission model -- a MixtureModel fit to this file's embeddings (see --noise-components) -- competing against match states for each frame, instead of being silent/transition-only. This is what actually fixes segments bleeding into surrounding noise (widened start/stop spans), which --repeat-prob/--gap-dwell-frames don't touch at all (those only control whether to look for another occurrence, not how tightly one occurrence's span is drawn). Ad hoc for this run only -- never touches train/processing.py or the saved model. Omit to keep today's behavior exactly.")
    find_parser.add_argument("--noise-components", type=int, default=None, help="number of Gaussian components (k) for the --noise-wav mixture model. Omit for a single Gaussian fit directly (mean/var, no EM) -- the default. Otherwise fits sklearn.mixture.GaussianMixture(n_components=k) with exactly k components, or -- with --noise-bayesian -- BayesianGaussianMixture using k as an upper bound instead of an exact count. Ignored if --noise-wav isn't set.")
    find_parser.add_argument("--noise-bayesian", action="store_true", help="use sklearn.mixture.BayesianGaussianMixture (Dirichlet process prior) instead of a plain GaussianMixture for --noise-components -- its own weight-concentration prior drives unneeded components' weights toward ~0 on its own, so --noise-components acts as an upper bound rather than a count you have to get exactly right. Requires --noise-components (used as that upper bound).")
    find_parser.add_argument("--noise-var-scale", type=float, default=1.0, help="multiply every fitted noise-model covariance (after the var_floor clamp) by this factor. >1 widens the noise Gaussians so background scores competitively against match states over a larger region of embedding space -- a quick way to fight motif segments bleeding into noise without collecting more/broader --noise-wav recordings. Too high starts eating real motif frames into background instead. Default 1.0 (no change). Ignored if --noise-wav isn't set.")
    find_parser.add_argument("--hit-gap-seconds", type=float, default=0.5, help="digital silence (in seconds) inserted between consecutive hits when concatenating each submodel_<n>_hits.wav clip, so occurrence boundaries are audible/visible in a waveform view instead of every hit running seamlessly into the next. 0 disables the gap (old back-to-back behavior).")

    args = parser.parse_args()
    if args.mode == "train":
        run_train(args)
    else:
        run_find(args)
