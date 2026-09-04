import numpy as np
import itertools
import random
import time
import lib_phmm.profile_hmm as phmm
import lib_phmm.hierarchical_kmedian_dtw as hkd
from lib_phmm.compression import *


DEFAULT_JJ = 0.01
DEFAULT_JB = 1.0 - DEFAULT_JJ


def _smoothed(self_ct, exit_ct, alpha):
    return (self_ct + alpha) / (self_ct + exit_ct + 2 * alpha)


def _smoothed_with_prior(self_ct, exit_ct, alpha, prior_mean):
    """
    Beta(alpha*prior_mean, alpha*(1-prior_mean)) conjugate-prior version
    of _smoothed(): _smoothed(ct, ct', a) is exactly this with
    prior_mean=0.5, i.e. a Beta(a,a) prior centered at 0.5. That means
    with no observed self-transitions (self_ct=0, the usual case when
    there's no real background/noise to learn from), _smoothed() can
    never push the estimate above 0.5 no matter how large alpha gets --
    it just interpolates towards its own 0.5 prior. This generalizes to
    an arbitrary prior mean so a target self-transition probability (and
    therefore a target expected dwell length, see dwell_to_prior_mean())
    can be encoded directly instead.
    """
    return (self_ct + alpha * prior_mean) / (self_ct + exit_ct + alpha)


def dwell_to_prior_mean(frames):
    """
    Self-transition probability p whose geometric-distribution mean
    dwell time (in frames) equals `frames`: mean = 1 / (1 - p), so
    p = 1 - 1/frames.
    """
    return 1.0 - 1.0 / frames


def estimate_transitions(alignments, n_match_states,
                          self_alpha=1.0, entry_alpha=0.2, fork_alpha=1.0,
                          flank_dwell_frames=None, flank_alpha=500.0,
                          fold_internal_gaps=True):
    """
    alignments:     list of per-sequence alignments (each a sequence of
                     match-state indices, -1 marks a no-call frame)
    n_match_states: number of match states k = 0..n_match_states-1,
                     shared by every sub-HMM

    Flank transitions (nn/nb/ec/ej/jj/jb/cc) are pooled across all
    alignments into one shared set. Match dwell (mm) and entry (b_to_hmm)
    are estimated per alignment, so mm/b_to_hmm come back as one row per
    input sequence -- ready for p.FlankTransitions(**result) with one
    sub-HMM per sequence.

    flank_dwell_frames: if given, nn/cc (the N/C flank self-transitions)
    are instead estimated with a prior centered on the self-transition
    probability implying this expected dwell length in frames (pseudocount
    weight flank_alpha), rather than the default symmetric self_alpha
    prior. Use this when the input sequences are already tightly-bounded
    candidates with little or no real NOISE-labeled background for nn/cc
    to learn from, but the flank states should still be able to absorb a
    stretch on the order of a typical/maximal candidate length instead of
    collapsing to a 1-2 frame dwell.
    """
    n_self_total, n_exit_total = 0, 0
    c_self_total, c_exit_total = 0, 0
    ec_hits_total, ej_hits_total = 0, 0

    mm_rows, b_to_hmm_rows = [], []

    for alignment in alignments:
        runs = [(v, len(list(g))) for v, g in itertools.groupby(alignment)]

        # ---- leading run: N dwell. Exit to B is always observed ----
        if runs and runs[0][0] == -1:
            n_self, n_exit = runs[0][1] - 1, 1
            runs = runs[1:]
        else:
            n_self, n_exit = 0, 1
        n_self_total += n_self
        n_exit_total += n_exit

        # ---- trailing run: C dwell. Exit is NOT observed -> censored ----
        if runs and runs[-1][0] == -1:
            c_self, c_exit = runs[-1][1] - 1, 0
            runs = runs[:-1]
        else:
            c_self, c_exit = 0, 0
        c_self_total += c_self
        c_exit_total += c_exit

        # ---- E fired exactly once per sequence -> ec/ej ----
        ec_hits_total += 1

        # ---- resolve interior -1 gaps, then compute match dwell -> mm ----
        resolved = []
        for value, length in runs:
            if value == -1:
                if fold_internal_gaps and resolved:
                    resolved.extend([resolved[-1]] * length)
            else:
                resolved.extend([value] * length)

        dwell = {k: 0 for k in range(n_match_states)}
        for value, length in itertools.groupby(resolved):
            dwell[value] += len(list(length))

        mm = []
        for k in range(n_match_states):
            d = dwell[k]
            mm.append(0.5 if d == 0 else _smoothed(d - 1, 1, self_alpha))
        mm_rows.append(mm)

        observed_entry = resolved[0] if resolved else 0
        b_to_hmm = [
            (1.0 if k == observed_entry else 0.0) + entry_alpha
            for k in range(n_match_states)
        ]
        total = sum(b_to_hmm)
        b_to_hmm_rows.append([v / total for v in b_to_hmm])

    if flank_dwell_frames is not None:
        prior_mean = dwell_to_prior_mean(flank_dwell_frames)
        nn = _smoothed_with_prior(n_self_total, n_exit_total, flank_alpha, prior_mean)
        cc = _smoothed_with_prior(c_self_total, c_exit_total, flank_alpha, prior_mean)
    else:
        nn = _smoothed(n_self_total, n_exit_total, self_alpha)
        cc = _smoothed(c_self_total, c_exit_total, self_alpha)
    nb = 1.0 - nn
    ec = _smoothed(ec_hits_total, ej_hits_total, fork_alpha)
    ej = 1.0 - ec
    jj, jb = DEFAULT_JJ, DEFAULT_JB

    return {
        "nn": nn, "nb": nb,
        "ec": ec, "ej": ej,
        "jj": jj, "jb": jb,
        "cc": cc,
        "b_to_hmm": b_to_hmm_rows,
        "mm": mm_rows,
    }


def make_hmm(sequences, classifications_list, max_switchpoints=12,
             flank_dwell_frames=None, flank_alpha=500.0):
    """
    Build one ProfileHMM with a shared flank (N/B/E/C/J) and one
    match-state sub-HMM per input sequence -- pdf[n] is sequence n's own
    compressed Gaussians. Every sub-HMM is padded to the same
    n_match_states (the widest any sequence produced), since the C++
    Viterbi assumes all sub-HMMs share one match-state count.

    flank_dwell_frames/flank_alpha are forwarded to estimate_transitions()
    -- see its docstring.
    """
    compressed = []
    for sequence, classifications in zip(sequences, classifications_list):
        profile    = distance_profile(sequence)
        points     = top_k_switchpoints(profile, max_switchpoints)
        a, mu, std = compress(np.array(sequence), points)
        path       = weave_path(a, classifications)
        compressed.append((mu, std, path))

    n_match_states = max(len(mu) for mu, std, path in compressed) + 1  # +1 for M0 sentinel
    dummy_mean = np.zeros_like(compressed[0][0][0])
    dummy_var  = np.ones_like(compressed[0][0][0])

    pdf, alignments = [], []
    for mu, std, path in compressed:
        states = [phmm.Gaussian(dummy_mean, dummy_var)] + [phmm.Gaussian(c, s) for c, s in zip(mu, std)]
        while len(states) < n_match_states:
            states.append(phmm.Gaussian(dummy_mean, dummy_var * 1e6))  # padding: never a good match
        pdf.append(states)
        alignments.append([x if x == -1 else x + 1 for x in path])     # M0 sentinel occupies slot 0

    transition_dict = estimate_transitions(
        alignments, n_match_states=n_match_states,
        flank_dwell_frames=flank_dwell_frames, flank_alpha=flank_alpha,
    )

    trans = phmm.FlankTransitions(
        nn=transition_dict['nn'], nb=transition_dict['nb'],
        ec=transition_dict['ec'], ej=transition_dict['ej'],
        jj=transition_dict['jj'], jb=transition_dict['jb'],
        cc=transition_dict['cc'],
        b_to_hmm=transition_dict['b_to_hmm'],
        mm=transition_dict['mm'],
    )
    hmm = phmm.ProfileHMM(pdf, trans)
    return hmm, n_match_states


def _to_dtw_dataset(candidates):
    return [[[float(v) for v in frame] for frame in region_embeddings]
            for region_embeddings, _, _ in candidates]


def filter_candidates_by_medoid(candidates, warping_band=5, epochs=20, threshold=1.0):
    """
    Cluster candidate motif regions by DTW distance (divisive hierarchical
    k-medoids -- see lib_phmm/hierarchical_kmedian_dtw.hpp) and keep only
    the medoid of each resulting leaf cluster. sweep_one_state_count()
    rescores every candidate against the full candidate pool each round
    (O(candidates^2) per round), so shrinking the pool here cuts that cost
    quadratically. Oversplitting (threshold too tight) just leaves
    redundant candidates for the greedy/BIC sweep to filter out downstream;
    undersplitting (threshold too loose) can permanently merge two distinct
    motifs into one medoid, so prefer erring tight.
    """
    dataset = _to_dtw_dataset(candidates)
    dm = hkd.DistanceManager(dataset, warping_band)
    medoid_ids = dm.kmedoids([], epochs, threshold)
    return [candidates[i] for i in medoid_ids]


def estimate_dtw_threshold(candidates, warping_band=5, n_samples=500, percentile=99, seed=None):
    """
    Default --dtw-threshold: DTW distances are normalized by warp-path
    length (see dtw() in hierarchical_kmedian_dtw.hpp), so this percentile
    is comparable across candidate pools with differently-sized regions.
    Sampled pairs are random, so most are distinct-motif pairs rather than
    near-duplicates -- the 99th percentile therefore sits near the top of
    that "distinct" distribution, i.e. a lenient default (few splits) that
    only merges near-exact duplicates. Pass a lower --dtw-threshold
    explicitly for the tighter clustering filter_candidates_by_medoid
    otherwise prefers.
    """
    dataset = _to_dtw_dataset(candidates)
    n = len(dataset)
    max_pairs = n * (n - 1) // 2
    n_samples = min(n_samples, max_pairs)

    rng = random.Random(seed)
    pairs = set()
    while len(pairs) < n_samples:
        i, j = rng.randrange(n), rng.randrange(n)
        if i != j:
            pairs.add((min(i, j), max(i, j)))

    distances = [hkd.dtw(dataset[i], dataset[j], warping_band) for i, j in pairs]
    return float(np.percentile(distances, percentile))


def sweep_one_state_count(n_match_states, embeddings, max_match, dim, n_frames_total,
                           flank_dwell_frames=None, flank_alpha=500.0, require_full_match=False):
    best_embeddings = []
    best_classifications = []
    done = set()
    local_results = []

    for i in range(max_match):
        best_hmm = None
        best_total = float('-inf')
        best_embedding = None
        best_classification = None
        best_embedding_id = None
        best_raw_total = None
        for embedding_id, (embedding, _, classifications) in enumerate(embeddings):
            if embedding_id in done:
                continue
            hmm, n_states = make_hmm(best_embeddings + [embedding], best_classifications + [classifications], n_match_states,
                                      flank_dwell_frames=flank_dwell_frames, flank_alpha=flank_alpha)
            scores_norm, paths, raw_scores = decode_all(embeddings, hmm, require_full_match=require_full_match)
            total_fit = sum(scores_norm)
            if best_total < total_fit:
                best_total = total_fit
                best_hmm = (hmm, n_states)
                best_embedding = embedding
                best_classification = classifications
                best_embedding_id = embedding_id
                best_raw_total = sum(raw_scores)

        if best_embedding_id is None:
            break

        done.add(best_embedding_id)
        best_embeddings.append(best_embedding)
        best_classifications.append(best_classification)

        n_sub_models = len(best_embeddings)
        ll_nats = best_raw_total * np.log(2)
        k = count_params(n_sub_models, n_match_states, dim)
        bic = k * np.log(n_frames_total) - 2 * ll_nats

        local_results.append({
            "n_match_states": n_match_states,
            "n_sub_models": n_sub_models,
            "bic": bic,
            "exemplar_embeddings": list(best_embeddings),
            "exemplar_classifications": list(best_classifications),
        })

    return local_results


def score_all_as_submodels(n_match_states, embeddings, dim, n_frames_total,
                            flank_dwell_frames=None, flank_alpha=500.0, require_full_match=False):
    """
    Like sweep_one_state_count(), but skips the greedy exemplar
    selection entirely: every sequence in `embeddings` becomes its own
    sub-model, unconditionally. sweep_one_state_count() spends
    O(max_match * candidates^2) deciding *which* candidates to add as
    sub-models -- pointless work once max_match is large enough that
    the answer is always "all of them" anyway (e.g. after DTW medoid
    filtering already thinned the pool to the set you want). This does
    one make_hmm() + one decode_all() per n_match_states value instead,
    same BIC formula, same result shape as sweep_one_state_count()'s.
    """
    all_embeddings = [embedding for embedding, _, _ in embeddings]
    all_classifications = [classifications for _, _, classifications in embeddings]

    hmm, n_states = make_hmm(all_embeddings, all_classifications, n_match_states,
                              flank_dwell_frames=flank_dwell_frames, flank_alpha=flank_alpha)
    scores_norm, paths, raw_scores = decode_all(embeddings, hmm, require_full_match=require_full_match)

    n_sub_models = len(all_embeddings)
    ll_nats = sum(raw_scores) * np.log(2)
    k = count_params(n_sub_models, n_match_states, dim)
    bic = k * np.log(n_frames_total) - 2 * ll_nats

    return {
        "n_match_states": n_match_states,
        "n_sub_models": n_sub_models,
        "bic": bic,
        "exemplar_embeddings": all_embeddings,
        "exemplar_classifications": all_classifications,
    }


def decode_all(embeddings, hmm, verbose=False, log_every=None, require_full_match=False):
    """
    verbose logs progress every log_every sequences (default: ~20 log
    lines spread across the pool, regardless of its size). Off by
    default since this is called on every greedy candidate trial inside
    sweep_one_state_count() -- turn it on only for a one-off decode over
    a large pool, e.g. motif_discovery.py's final full-pool scoring pass.

    require_full_match forwards to phmm.viterbi() -- see its docstring.
    Off by default, matching the sweep's own scoring (which never passed
    it before this option existed).
    """
    n_total = len(embeddings)
    if verbose and log_every is None:
        log_every = max(1, n_total // 20)

    scores = []
    paths  = []
    start = time.time() if verbose else None
    for i, (_, embedding, classifications) in enumerate(embeddings):
        score, path = phmm.viterbi(embedding, hmm, None, require_full_match)
        scores.append(score)
        paths.append(path)
        if verbose and ((i + 1) % log_every == 0 or i + 1 == n_total):
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            eta = (n_total - (i + 1)) / rate if rate > 0 else float('inf')
            print(f"  decoded {i + 1}/{n_total} ({(i + 1) / n_total * 100:.1f}%) "
                  f"elapsed={elapsed:.1f}s rate={rate:.1f}/s eta={eta:.1f}s")

    lengths = [len(embedding) for _, embedding, _ in embeddings]
    scores_norm = [score / n for score, n in zip(scores, lengths)]
    return scores_norm, paths, scores


def best_fit(scores_norm, threshold=30):
    n = len(scores_norm)
    well_fits = [i for i in range(n) if scores_norm[i] < threshold]
    well_fits_scores = [scores_norm[i] for i in range(n) if scores_norm[i] < threshold]
    return well_fits, well_fits_scores


def count_params(n_sub_models, n_states, dim):
    # per sub-model: n_states Gaussians (mean + variance per dim),
    # entry distribution over n_states (n_states - 1 free),
    # self-transition dwell prob per column (n_states free)
    per_model = n_states * 2 * dim + (n_states - 1) + n_states
    shared_flank = 4   # nn, ec, jj, cc -- nb/ej/jb are each 1 - one of these
    return shared_flank + n_sub_models * per_model
