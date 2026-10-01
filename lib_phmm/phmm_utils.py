import numpy as np
import itertools
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
                          fold_internal_gaps=True, groups=None):
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

    groups: optional sub-HMM index per alignment. Alignments sharing a
    group (e.g. every member of one cluster, aligned to its medoid's
    states) pool their dwell and entry counts into that group's single
    mm/b_to_hmm row, so rows come back one per group instead of one per
    alignment. With one alignment per group this is exactly the
    per-alignment estimate.

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

    if groups is None:
        groups = range(len(alignments))
    n_groups = max(groups) + 1 if len(alignments) else 0
    # per group and state: frames dwelt beyond the first (self) and visits (exit)
    self_ct = np.zeros((n_groups, n_match_states))
    visit_ct = np.zeros((n_groups, n_match_states))
    entry_ct = np.zeros((n_groups, n_match_states))

    for alignment, group in zip(alignments, groups):
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

        for k in range(n_match_states):
            if dwell[k] > 0:
                self_ct[group, k] += dwell[k] - 1
                visit_ct[group, k] += 1

        entry_ct[group, resolved[0] if resolved else 0] += 1

    mm_rows = [[0.5 if visit_ct[g, k] == 0 else _smoothed(self_ct[g, k], visit_ct[g, k], self_alpha)
                for k in range(n_match_states)]
               for g in range(n_groups)]
    b_to_hmm_rows = []
    for g in range(n_groups):
        b_to_hmm = entry_ct[g] + entry_alpha   # Dirichlet-smoothed counts; one alignment = one-hot + alpha
        b_to_hmm_rows.append((b_to_hmm / b_to_hmm.sum()).tolist())

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


VARIANCE_MODES = ('medoid', 'shared', 'cluster')
VAR_FLOOR = 0.1   # same floor compress() applies


def _diag_gauss_ll(frames, means, variances):
    """(T, D) frames against K diagonal Gaussians (K, D) -> (T, K) log-likelihoods in nats."""
    diff = frames[:, None, :] - means[None, :, :]
    return -0.5 * ((diff ** 2 / variances[None]).sum(-1) + np.log(2 * np.pi * variances).sum(-1)[None])


def align_to_states(frames, means, variances):
    """
    Full-match left-to-right alignment of `frames` to K states: start in
    state 0, end in state K-1, every frame either stays or moves on by one,
    no transition costs (a DTW against the state sequence). Returns the
    state index of every frame, or None if there are fewer frames than
    states.
    """
    ll = _diag_gauss_ll(frames, means, variances)
    T, K = ll.shape
    if T < K:
        return None
    W = np.full((T, K), -np.inf)
    W[0, 0] = ll[0, 0]
    moved = np.zeros((T, K), dtype=bool)
    for t in range(1, T):
        from_prev = np.r_[-np.inf, W[t - 1, :-1]]
        moved[t] = from_prev > W[t - 1]
        W[t] = ll[t] + np.maximum(W[t - 1], from_prev)
    k = K - 1
    states = [k]
    for t in range(T - 1, 0, -1):
        k -= int(moved[t, k])
        states.append(k)
    return np.array(states[::-1])


def _pool_members(compressed, members, variance, max_members, shrinkage):
    """
    Align every cluster member to its medoid's states and re-estimate the
    emissions from all of them (see make_hmm()). Returns the new
    [(means, variances)] per sub-model and, per sub-model, the extra
    alignments (1-based states, -1 for NOISE frames) for estimate_transitions().
    """
    per_model = []   # (frames per state, member alignments) for each sub-model
    for (mu, var, path, a, sequence), cluster in zip(compressed, members):
        mu, var = np.asarray(mu), np.asarray(var)
        frames_by_state = [[np.asarray(sequence)[a == k]] for k in range(len(mu))]   # medoid's own frames
        extra_paths = []
        if variance == 'cluster' and max_members is not None:
            cluster = cluster[:max_members]                                       # nearest to the medoid first
        for embedding, _, classifications in cluster:
            frames = np.asarray(embedding)
            states = align_to_states(frames, mu, var)
            if states is None:
                continue
            for k in range(len(mu)):
                frames_by_state[k].append(frames[states == k])
            extra_paths.append([-1 if c == 'NOISE' else int(s) + 1 for s, c in zip(states, classifications)])
        per_model.append(([np.concatenate(f) for f in frames_by_state], extra_paths))

    # one diagonal variance for every state: deviations of all member frames from their medoid state's mean
    sq_sum, n_frames = 0.0, 0
    for (mu, _, _, _, _), (state_frames, _) in zip(compressed, per_model):
        for k, x in enumerate(state_frames):
            sq_sum = sq_sum + ((x - np.asarray(mu[k])) ** 2).sum(axis=0)
            n_frames += len(x)
    shared_var = np.maximum(sq_sum / max(1, n_frames), VAR_FLOOR)

    emissions = []
    for (mu, _, _, _, _), (state_frames, _) in zip(compressed, per_model):
        if variance == 'shared':
            emissions.append(([np.asarray(m) for m in mu], [shared_var] * len(mu)))
            continue
        means, variances = [], []
        for k, x in enumerate(state_frames):
            # shrink towards the medoid's mean and the shared variance by `shrinkage` pseudo-frames
            n = len(x)
            mean = (x.sum(axis=0) + shrinkage * np.asarray(mu[k])) / (n + shrinkage)
            leaf_var = ((x - mean) ** 2).mean(axis=0) if n else shared_var
            means.append(mean)
            variances.append(np.maximum((n * leaf_var + shrinkage * shared_var) / (n + shrinkage), VAR_FLOOR))
        emissions.append((means, variances))
    return emissions, [paths for _, paths in per_model]


def make_hmm(sequences, classifications_list, max_switchpoints=12,
             flank_dwell_frames=None, flank_alpha=500.0,
             members=None, variance='medoid', max_members=5, shrinkage=10.0):
    """
    Build one ProfileHMM with a shared flank (N/B/E/C/J) and one
    match-state sub-HMM per input sequence -- pdf[n] is sequence n's own
    compressed Gaussians. Every sub-HMM is padded to the same
    n_match_states (the widest any sequence produced), since the C++
    Viterbi assumes all sub-HMMs share one match-state count. Each
    sub-HMM's own last real state is passed along as last_match, so
    require_full_match exits there instead of at the padded end.

    flank_dwell_frames/flank_alpha are forwarded to estimate_transitions()
    -- see its docstring.

    variance picks how emissions are estimated:
      'medoid'  (default) each sub-HMM from its own sequence only, as above.
      'shared'  states and means still come from the sequence (the medoid),
                but every state of every sub-HMM gets one diagonal variance,
                pooled from all cluster members aligned to their medoid's
                states -- same width everywhere, so cluster size can't make
                one sub-HMM broader than another.
      'cluster' each state's mean and variance from its medoid's frames plus
                its members' aligned frames (the `max_members` nearest),
                shrunk towards the medoid's mean and the shared variance by
                `shrinkage` pseudo-frames so small clusters stay close to
                the medoid.
    Both pooled modes need `members` (per sequence, the other candidates
    of its cluster as (embedding, full, classifications), nearest first),
    align each member once to its medoid's states (align_to_states(), no
    re-estimation loop), and also pool the members' alignments into that
    sub-HMM's dwell (mm) and entry (b_to_hmm) estimates and the shared
    flank estimates.
    """
    if variance not in VARIANCE_MODES:
        raise ValueError(f"variance must be one of {VARIANCE_MODES}, got {variance!r}")
    if variance != 'medoid' and members is None:
        raise ValueError(f"variance={variance!r} needs the cluster members of every sequence")

    compressed = []
    for sequence, classifications in zip(sequences, classifications_list):
        profile    = distance_profile(sequence)
        points     = top_k_switchpoints(profile, max_switchpoints)
        a, mu, std = compress(np.array(sequence), points)
        path       = weave_path(a, classifications)
        compressed.append((mu, std, path, a, sequence))

    extra_paths = [[] for _ in compressed]
    if variance != 'medoid':
        emissions, extra_paths = _pool_members(compressed, members, variance, max_members, shrinkage)
        compressed = [(mu, std, path, a, sequence) for (mu, std), (_, _, path, a, sequence) in zip(emissions, compressed)]

    n_match_states = max(len(mu) for mu, _, _, _, _ in compressed) + 1  # +1 for M0 sentinel
    dummy_mean = np.zeros_like(compressed[0][0][0])
    dummy_var  = np.ones_like(compressed[0][0][0])

    pdf, alignments, groups = [], [], []
    for n, (mu, std, path, _, _) in enumerate(compressed):
        states = [phmm.Gaussian(dummy_mean, dummy_var)] + [phmm.Gaussian(c, s) for c, s in zip(mu, std)]
        while len(states) < n_match_states:
            states.append(phmm.Gaussian(dummy_mean, dummy_var * 1e6))  # padding: never a good match
        pdf.append(states)
        alignments.append([x if x == -1 else x + 1 for x in path])     # M0 sentinel occupies slot 0
        alignments.extend(extra_paths[n])                               # members: already 1-based
        groups.extend([n] * (1 + len(extra_paths[n])))

    transition_dict = estimate_transitions(
        alignments, n_match_states=n_match_states,
        flank_dwell_frames=flank_dwell_frames, flank_alpha=flank_alpha,
        groups=groups,
    )

    trans = phmm.FlankTransitions(
        nn=transition_dict['nn'], nb=transition_dict['nb'],
        ec=transition_dict['ec'], ej=transition_dict['ej'],
        jj=transition_dict['jj'], jb=transition_dict['jb'],
        cc=transition_dict['cc'],
        b_to_hmm=transition_dict['b_to_hmm'],
        mm=transition_dict['mm'],
    )
    last_match = [len(mu) for mu, _, _, _, _ in compressed]            # M0 sentinel shifts real states to 1..len(mu)
    hmm = phmm.ProfileHMM(pdf, trans, last_match)
    return hmm, n_match_states


def _zscore_pool(candidates, eps=1e-8):
    frames = np.concatenate([np.asarray(region) for region, _, _ in candidates], axis=0)
    mean, std = frames.mean(axis=0), frames.std(axis=0) + eps
    return [((np.asarray(region) - mean) / std, full, cls) for region, full, cls in candidates]


def _to_dtw_dataset(candidates):
    return [[[float(v) for v in frame] for frame in region_embeddings]
            for region_embeddings, _, _ in candidates]


def filter_candidates_by_medoid(candidates, warping_band=5, epochs=20, restarts=10, tolerance=0.05,
                                norm="path", zscore=False, return_members=False):
    """
    Cluster candidate motif regions by DTW distance (divisive hierarchical
    k-medoids -- see lib_phmm/hierarchical_kmedian_dtw.hpp) and keep only
    the medoid of each resulting leaf cluster. sweep_one_state_count()
    rescores every candidate against the full candidate pool each round
    (O(candidates^2) per round), so shrinking the pool here cuts that cost
    quadratically. Splitting is density-driven and stops on its own (a
    branch only recurses while its children are tighter around their
    medoid than it is), so there's no distance/count threshold to tune --
    restarts/tolerance instead control how readily a branch keeps
    splitting: more restarts try more random splits per level and keep
    the densest (costs more, rarely hurts quality), a higher tolerance
    lets a child up to that fraction less dense than its parent still
    count as an improvement (more medoids, less tight). Defaults match
    hierarchical_kmedian_dtw.hpp's own defaults.

    DTW runs on the raw (uncompressed) candidate frames -- compression to
    match states only happens afterwards, in make_hmm(), on the surviving
    medoids -- so warping_band is in frames. norm picks how the summed DTW
    cost becomes a distance: "path" divides by warp-path length (default),
    "length" by n + m (fixed per pair, so it doesn't reward longer warp
    paths), "none" keeps the raw sum.

    zscore=True standardizes each embedding dimension by its mean/std over
    every frame in the pool before DTW, so the squared-Euclidean frame cost
    isn't dominated by a few high-variance dimensions. Only the distances
    change -- the returned medoids are the original, unscaled candidates.

    return_members=True also returns, per medoid, the other candidates of
    its leaf cluster, nearest (by the same DTW distance) first -- what
    make_hmm()'s pooled variance modes estimate from. Candidates the tree
    used as a split anchor/sample end up in no leaf (see kmedoids_leaves());
    each of those joins its nearest medoid's cluster.
    """
    dataset = _to_dtw_dataset(_zscore_pool(candidates) if zscore else candidates)
    dm = hkd.DistanceManager(dataset, warping_band, norm)
    if not return_members:
        medoid_ids = dm.kmedoids([], epochs, restarts, tolerance)
        return [candidates[i] for i in medoid_ids]

    leaves = dm.kmedoids_leaves([], epochs, restarts, tolerance)
    medoid_ids = sorted(leaves)
    clusters = {m: [i for i in leaves[m] if i != m] for m in medoid_ids}
    covered = {i for leaf in leaves.values() for i in leaf}
    for i in range(len(candidates)):
        if i not in covered:
            clusters[min(medoid_ids, key=lambda m: dm.distance(i, m))].append(i)
    members = [[candidates[i] for i in sorted(clusters[m], key=lambda i: dm.distance(i, m))]
               for m in medoid_ids]
    return [candidates[m] for m in medoid_ids], members


def sweep_one_state_count(n_match_states, embeddings, max_match, dim, n_frames_total,
                           flank_dwell_frames=None, flank_alpha=500.0, require_full_match=False,
                           members=None, variance='medoid', max_members=5, shrinkage=10.0):
    # members/variance/max_members/shrinkage: see make_hmm() -- members is
    # parallel to `embeddings` (each candidate's cluster)
    best_embeddings = []
    best_classifications = []
    best_members = []
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
                                      flank_dwell_frames=flank_dwell_frames, flank_alpha=flank_alpha,
                                      members=None if members is None else best_members + [members[embedding_id]],
                                      variance=variance, max_members=max_members, shrinkage=shrinkage)
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
        if members is not None:
            best_members.append(members[best_embedding_id])

        n_sub_models = len(best_embeddings)
        ll_nats = best_raw_total * np.log(2)
        k = count_params(n_sub_models, n_match_states, dim, variance)
        bic = k * np.log(n_frames_total) - 2 * ll_nats

        local_results.append({
            "n_match_states": n_match_states,
            "n_sub_models": n_sub_models,
            "bic": bic,
            "exemplar_embeddings": list(best_embeddings),
            "exemplar_classifications": list(best_classifications),
            "exemplar_members": list(best_members) if members is not None else None,
        })

    return local_results


def score_all_as_submodels(n_match_states, embeddings, dim, n_frames_total,
                            flank_dwell_frames=None, flank_alpha=500.0, require_full_match=False,
                            members=None, variance='medoid', max_members=5, shrinkage=10.0):
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
                              flank_dwell_frames=flank_dwell_frames, flank_alpha=flank_alpha,
                              members=members, variance=variance, max_members=max_members, shrinkage=shrinkage)
    scores_norm, paths, raw_scores = decode_all(embeddings, hmm, require_full_match=require_full_match)

    n_sub_models = len(all_embeddings)
    ll_nats = sum(raw_scores) * np.log(2)
    k = count_params(n_sub_models, n_match_states, dim, variance)
    bic = k * np.log(n_frames_total) - 2 * ll_nats

    return {
        "n_match_states": n_match_states,
        "n_sub_models": n_sub_models,
        "bic": bic,
        "exemplar_embeddings": all_embeddings,
        "exemplar_classifications": all_classifications,
        "exemplar_members": members,
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


def count_params(n_sub_models, n_states, dim, variance='medoid'):
    # per sub-model: n_states Gaussians (mean + variance per dim; with
    # variance='shared' one variance vector is shared by every state instead),
    # entry distribution over n_states (n_states - 1 free),
    # self-transition dwell prob per column (n_states free)
    per_state_var = 0 if variance == 'shared' else dim
    per_model = n_states * (dim + per_state_var) + (n_states - 1) + n_states
    shared_flank = 4   # nn, ec, jj, cc -- nb/ej/jb are each 1 - one of these
    shared_var = dim if variance == 'shared' else 0
    return shared_flank + shared_var + n_sub_models * per_model
