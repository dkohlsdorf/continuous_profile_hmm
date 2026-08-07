import numpy as np
import lib_phmm.profile_hmm as phmm
from lib_phmm.compression import *


DEFAULT_JJ = 0.01
DEFAULT_JB = 1.0 - DEFAULT_JJ


def _smoothed(self_ct, exit_ct, alpha):
    return (self_ct + alpha) / (self_ct + exit_ct + 2 * alpha)


def estimate_transitions(alignments, n_match_states,
                          self_alpha=1.0, entry_alpha=0.2, fork_alpha=1.0,
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

    nn = _smoothed(n_self_total, n_exit_total, self_alpha)
    nb = 1.0 - nn
    cc = _smoothed(c_self_total, c_exit_total, self_alpha)
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


def make_hmm(sequences, classifications_list, max_switchpoints=12):
    """
    Build one ProfileHMM with a shared flank (N/B/E/C/J) and one
    match-state sub-HMM per input sequence -- pdf[n] is sequence n's own
    compressed Gaussians. Every sub-HMM is padded to the same
    n_match_states (the widest any sequence produced), since the C++
    Viterbi assumes all sub-HMMs share one match-state count.
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

    transition_dict = estimate_transitions(alignments, n_match_states=n_match_states)

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


def sweep_one_state_count(n_match_states, embeddings, max_match, dim, n_frames_total):
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
            hmm, n_states = make_hmm(best_embeddings + [embedding], best_classifications + [classifications], n_match_states)
            scores_norm, paths, raw_scores = decode_all(embeddings, hmm)
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


def decode_all(embeddings, hmm):
    scores = []
    paths  = []
    for _, embedding, classifications in embeddings:
        score, path = phmm.viterbi(embedding, hmm)
        scores.append(score)
        paths.append(path)
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
