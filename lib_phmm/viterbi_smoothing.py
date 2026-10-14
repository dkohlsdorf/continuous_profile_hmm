"""
Viterbi smoothing of the per-frame classifier output before candidate mining.

The classifier's argmax flickers: a single NOISE frame splits a whistle, and
bounded_regions()/find_non_noise_spans() cut a candidate at the first NOISE
frame. Here the classifier's softmax probabilities are the emissions of a
small HMM over its own labels (NOISE, UP, DOWN, ECHO, BURST), decoded with
Viterbi, and NOISE gaps the decode still leaves inside a call are closed
afterwards.

Transitions have two levels: staying in the same label and UP <-> DOWN get
weight `stay_weight`, every other transition (to and from NOISE, between
classes) weight 1, each row normalised. A whistle turning from UP to DOWN is
as cheap as continuing, so the decode follows the classifier's direction and
only removes short excursions to NOISE or another class.

The classifier is sometimes confident a few frames inside a whistle are
NOISE, which no transition setting overrides, so every NOISE gap of at most
`max_gap` frames between two signal runs is filled afterwards; a filled frame
gets the non-NOISE label the classifier gives most probability. Inside the
curated L2 whistles such gaps are 1-6 frames long and never longer.

Defaults are the best setting of analysis_notebooks/ClassifierSmoothing.ipynb
(H 300 + gap closing 6): on 5 whistle-rich recordings, non-NOISE runs too
short to become candidates drop from 87% to 44% and the share of whistle
frames inside a candidate rises from 51% to 84%. The cost: whistles less than
max_gap frames apart, or a whistle and an adjacent click train, become one
candidate.
"""
import numpy as np

from lib_phmm.signals import LABEL_MAP

LABELS = [LABEL_MAP[i] for i in range(len(LABEL_MAP))]
NOISE = LABELS.index('NOISE')
UP = LABELS.index('UP')
DOWN = LABELS.index('DOWN')

STAY_WEIGHT = 300
MAX_GAP_FRAMES = 6


def transitions(stay_weight=STAY_WEIGHT):
    weights = np.ones((len(LABELS), len(LABELS)))
    np.fill_diagonal(weights, stay_weight)
    weights[UP, DOWN] = weights[DOWN, UP] = stay_weight
    return weights / weights.sum(axis=1, keepdims=True)


def viterbi(log_emissions, log_transitions):
    T, S = log_emissions.shape
    back = np.empty((T, S), dtype=np.int8)
    score = np.log(np.full(S, 1 / S)) + log_emissions[0]
    for t in range(1, T):
        candidates = score[:, None] + log_transitions
        back[t] = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + log_emissions[t]
    path = np.empty(T, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def close_gaps(path, probs, max_gap=MAX_GAP_FRAMES):
    path = path.copy()
    edges = np.flatnonzero(np.diff(np.r_[0, (path != NOISE).astype(np.int8), 0]))
    starts, stops = edges[::2], edges[1::2]
    signal_probs = probs.copy()
    signal_probs[:, NOISE] = -np.inf
    for gap_start, gap_stop in zip(stops[:-1], starts[1:]):
        if gap_stop - gap_start <= max_gap:
            path[gap_start:gap_stop] = signal_probs[gap_start:gap_stop].argmax(axis=1)
    return path


def smooth(probs, stay_weight=STAY_WEIGHT, max_gap=MAX_GAP_FRAMES):
    """
    probs: (n_frames, n_labels) classifier softmax in LABEL_MAP order.
    Returns the smoothed per-frame labels (same strings as classify()).
    """
    probs = np.asarray(probs)
    with np.errstate(divide='ignore'):
        path = viterbi(np.log(probs), np.log(transitions(stay_weight)))
    return [LABELS[i] for i in close_gaps(path, probs, max_gap)]
