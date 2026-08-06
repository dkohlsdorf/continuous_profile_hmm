"""
Alignment quality metrics for profile-HMM Viterbi decodings.

dolphin_msa's lib_dolphin/metrics.py scores a DTW-built multiple sequence
alignment stored as `msa[column][seq_idx] -> list of embeddings` (empty
list = gap). build_msa_from_paths() below turns a set of PHMM Viterbi
paths into that same shape -- one column per (sub-model, match state)
pair -- so the metrics themselves are ported as-is and stay comparable
across the two alignment methods.
"""
import numpy as np
import pandas as pd
import lib_phmm.profile_hmm as phmm


def build_msa_from_paths(embeddings, paths, n_models, n_states):
    """
    Convert per-sequence Viterbi paths into an MSA-style column structure.

    embeddings: the same list passed to decode_all, i.e. (id, embedding, classifications)
    paths:      per-sequence Pred lists, as returned by decode_all / phmm.viterbi
    n_models:   number of sub-HMMs (n_sub_models)
    n_states:   n_match_states used to build the HMM (state 0 is the M0
                sentinel and is never visited by Viterbi, so it's skipped)

    Returns msa: list of (n_models * (n_states - 1)) columns, each a list
    of length len(embeddings) holding that sequence's embeddings at this
    (model, match-state) column -- empty where the sequence never visited it.
    """
    n_sequences = len(embeddings)
    n_columns = n_models * (n_states - 1)
    msa = [[[] for _ in range(n_sequences)] for _ in range(n_columns)]

    for seq_idx, path in enumerate(paths):
        _, embedding, _ = embeddings[seq_idx]
        for step in path:
            if step.state < phmm.MATCH_STATE:
                continue
            idx = step.state - phmm.MATCH_STATE
            n, k = idx // n_states, idx % n_states
            if k == 0:
                continue  # M0 sentinel, never a real match
            col = n * (n_states - 1) + (k - 1)
            msa[col][seq_idx].append(embedding[step.time])

    return msa


def gap_open_extend_ratio(msa, n_sequences):
    """
    Gap Open / Extend Ratio

    Controls fragmentation vs long gaps. Helps see if gaps explode or vanish.

    Args:
        msa: Multiple sequence alignment (list of columns, each containing embeddings)
        n_sequences: Number of sequences in the alignment

    Returns:
        Dict with ratio, gap counts
    """
    gap_opens = 0
    gap_extends = 0
    total_gaps = 0

    in_gap = [False] * n_sequences

    for column in msa:
        for seq_idx in range(n_sequences):
            has_gap = seq_idx >= len(column) or len(column[seq_idx]) == 0

            if has_gap:
                total_gaps += 1
                if not in_gap[seq_idx]:
                    gap_opens += 1
                    in_gap[seq_idx] = True
                else:
                    gap_extends += 1
            else:
                in_gap[seq_idx] = False

    ratio = gap_opens / gap_extends if gap_extends > 0 else float('inf')

    return {
        'ratio': ratio,
        'gap_opens': gap_opens,
        'gap_extends': gap_extends,
        'total_gaps': total_gaps
    }


def column_cleanliness(msa):
    """
    Column Cleanliness

    Tells you if embeddings cluster tightly -> too many gaps = noisy columns.
    Measures variance within columns to assess alignment quality.

    Args:
        msa: Multiple sequence alignment

    Returns:
        Dict with variance stats, gap percentage, and column counts
    """
    variances = []
    gap_count = 0
    total_positions = 0

    clean_threshold = 0.1
    noisy_threshold = 1.0
    clean_columns = 0
    noisy_columns = 0

    for column in msa:
        embeddings = []
        for seq_embeddings in column:
            if len(seq_embeddings) > 0:
                emb = np.array(seq_embeddings).mean(axis=0)
                embeddings.append(emb)
            else:
                gap_count += 1

        total_positions += len(column)

        if len(embeddings) >= 2:
            embeddings = np.array(embeddings)
            col_variance = np.var(embeddings, axis=0).mean()
            variances.append(col_variance)

            if col_variance < clean_threshold:
                clean_columns += 1
            elif col_variance > noisy_threshold:
                noisy_columns += 1

    mean_var = np.mean(variances) if len(variances) > 0 else 0.0
    median_var = np.median(variances) if len(variances) > 0 else 0.0
    gap_pct = (gap_count / total_positions * 100) if total_positions > 0 else 0.0

    total_cols = len(msa)
    percentage_clean = (clean_columns / total_cols * 100) if total_cols > 0 else 0.0
    percentage_noise = (noisy_columns / total_cols * 100) if total_cols > 0 else 0.0

    return {
        'mean_variance': mean_var,
        'median_variance': median_var,
        'gap_percentage': gap_pct,
        'percentage_clean': percentage_clean,
        'percentage_noise': percentage_noise,
        'total_columns': total_cols
    }


def gap_statistics_per_sequence(msa, n_sequences):
    """
    Compute gap statistics for each sequence in the MSA.

    Useful for comparing alignment quality across different datasets.

    Args:
        msa: Multiple sequence alignment (list of columns, each containing embeddings)
        n_sequences: Number of sequences in the alignment

    Returns:
        Dict with per-sequence gap statistics and overall gap distribution
    """
    sequence_gap_counts = [0 for _ in range(n_sequences)]
    sequence_total_positions = [0 for _ in range(n_sequences)]
    max_consecutive_gaps = [0 for _ in range(n_sequences)]
    current_gap_run = [0 for _ in range(n_sequences)]

    for column in msa:
        for seq_idx in range(n_sequences):
            has_gap = seq_idx >= len(column) or len(column[seq_idx]) == 0
            sequence_total_positions[seq_idx] += 1

            if has_gap:
                sequence_gap_counts[seq_idx] += 1
                current_gap_run[seq_idx] += 1
                max_consecutive_gaps[seq_idx] = max(max_consecutive_gaps[seq_idx], current_gap_run[seq_idx])
            else:
                current_gap_run[seq_idx] = 0

    gap_percentages = []
    for seq_idx in range(n_sequences):
        if sequence_total_positions[seq_idx] > 0:
            gap_pct = (sequence_gap_counts[seq_idx] / sequence_total_positions[seq_idx]) * 100
            gap_percentages.append(gap_pct)
        else:
            gap_percentages.append(0.0)

    gap_percentages = np.array(gap_percentages)

    return {
        'mean_gap_percentage': float(np.mean(gap_percentages)),
        'median_gap_percentage': float(np.median(gap_percentages)),
        'std_gap_percentage': float(np.std(gap_percentages)),
        'min_gap_percentage': float(np.min(gap_percentages)),
        'max_gap_percentage': float(np.max(gap_percentages)),
        'mean_max_consecutive_gaps': float(np.mean(max_consecutive_gaps)),
        'max_consecutive_gaps': int(np.max(max_consecutive_gaps)),
        'total_gap_positions': int(np.sum(sequence_gap_counts)),
        'total_positions': int(np.sum(sequence_total_positions))
    }


def compute_all_metrics(msa, n_sequences):
    """
    Compute all MSA quality metrics.

    Args:
        msa: Multiple sequence alignment
        n_sequences: Number of sequences in the alignment

    Returns:
        Dict containing all metric results
    """
    return {
        'gap_open_extend_ratio': gap_open_extend_ratio(msa, n_sequences),
        'column_cleanliness': column_cleanliness(msa),
        'gap_statistics': gap_statistics_per_sequence(msa, n_sequences)
    }


def alignment_metrics(embeddings, paths, n_models, n_states):
    """
    Convenience wrapper: build the MSA column structure from Viterbi paths
    and score it in one call.
    """
    msa = build_msa_from_paths(embeddings, paths, n_models, n_states)
    return compute_all_metrics(msa, len(embeddings))


def metrics_to_dataframe(metrics):
    """
    Flatten compute_all_metrics()'s output into a one-row DataFrame, using
    the same column names as dolphin_msa's evaluate.py so runs from either
    pipeline can be compared with the same tooling.
    """
    data = {
        'gap_ratio': metrics['gap_open_extend_ratio']['ratio'],
        'gap_opens': metrics['gap_open_extend_ratio']['gap_opens'],
        'gap_extends': metrics['gap_open_extend_ratio']['gap_extends'],
        'total_gaps': metrics['gap_open_extend_ratio']['total_gaps'],
        'mean_variance': metrics['column_cleanliness']['mean_variance'],
        'median_variance': metrics['column_cleanliness']['median_variance'],
        'gap_percentage': metrics['column_cleanliness']['gap_percentage'],
        'percentage_clean': metrics['column_cleanliness']['percentage_clean'],
        'percentage_noise': metrics['column_cleanliness']['percentage_noise'],
        'total_columns': metrics['column_cleanliness']['total_columns'],
        'mean_gap_pct': metrics['gap_statistics']['mean_gap_percentage'],
        'median_gap_pct': metrics['gap_statistics']['median_gap_percentage'],
        'std_gap_pct': metrics['gap_statistics']['std_gap_percentage'],
        'min_gap_pct': metrics['gap_statistics']['min_gap_percentage'],
        'max_gap_pct': metrics['gap_statistics']['max_gap_percentage'],
        'mean_max_consecutive': metrics['gap_statistics']['mean_max_consecutive_gaps'],
        'max_consecutive': metrics['gap_statistics']['max_consecutive_gaps'],
        'total_gap_positions': metrics['gap_statistics']['total_gap_positions'],
        'total_positions': metrics['gap_statistics']['total_positions'],
    }
    return pd.DataFrame([data])
