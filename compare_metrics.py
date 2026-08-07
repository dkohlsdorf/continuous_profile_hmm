"""
Compare Profile HMM alignment runs.

Loads metrics.csv (written by processing.py) from two or more run output
folders and reports each metric's percentage lift relative to a baseline
(the first folder given on the command line). Mirrors the comparison
logic in dolphin_msa/evaluate.py so runs from either pipeline read the
same way.

Usage:
  python compare_metrics.py <run_folder_baseline> <run_folder2> [<run_folder3> ...]
"""
import numpy as np
import pandas as pd
from sys import argv


# metrics where a decrease from baseline is an improvement
LOWER_IS_BETTER = [
    'bic',
    'gap_ratio', 'gap_opens', 'gap_extends', 'total_gaps', 'gap_percentage',
    'mean_variance', 'median_variance', 'percentage_noise',
    'mean_gap_pct', 'median_gap_pct', 'std_gap_pct', 'max_gap_pct',
    'mean_max_consecutive', 'max_consecutive',
]

# metrics where an increase from baseline is an improvement
HIGHER_IS_BETTER = [
    'mean_fit', 'well_fit_rate', 'percentage_clean',
]

KEY_METRICS = [
    'bic', 'mean_fit', 'well_fit_count', 'well_fit_rate',
    'n_match_states', 'n_sub_models',
    'gap_ratio', 'gap_opens', 'gap_extends', 'total_gaps', 'gap_percentage',
    'mean_variance', 'median_variance', 'percentage_clean', 'percentage_noise', 'total_columns',
    'mean_gap_pct', 'median_gap_pct', 'std_gap_pct', 'min_gap_pct', 'max_gap_pct',
    'mean_max_consecutive', 'max_consecutive', 'total_gap_positions', 'total_positions',
]


def load_metrics(run_folder):
    path = f'{run_folder}/metrics.csv'
    df = pd.read_csv(path)
    if len(df) != 1:
        print(f"WARNING: expected exactly 1 row in {path}, found {len(df)} -- using the first row")
    return df.iloc[0].to_dict()


def compute_comparison_table(folder_metrics):
    folder_names = list(folder_metrics.keys())
    baseline_name = folder_names[0]
    baseline = folder_metrics[baseline_name]

    rows = []
    for metric in KEY_METRICS:
        if metric not in baseline:
            continue

        row = {'metric': metric, baseline_name: baseline[metric]}

        for folder_name in folder_names[1:]:
            metrics = folder_metrics[folder_name]
            if metric not in metrics:
                row[folder_name] = np.nan
                row[f'{folder_name}_lift%'] = np.nan
                continue

            value = metrics[metric]
            baseline_value = baseline[metric]

            if baseline_value != 0:
                lift = ((value - baseline_value) / abs(baseline_value)) * 100
            else:
                lift = 0.0 if value == 0 else np.nan

            row[folder_name] = value
            row[f'{folder_name}_lift%'] = lift

        rows.append(row)

    return pd.DataFrame(rows)


def interpret_metric_lift(metric, lift, value, baseline_value):
    if np.isnan(lift):
        return f"**{metric}**: N/A (baseline = 0.0, current = {value:.4f}) - o Not comparable"

    abs_lift = abs(lift)

    if metric in LOWER_IS_BETTER:
        is_positive = lift < 0
        direction = "decreased" if lift < 0 else "increased"
    elif metric in HIGHER_IS_BETTER:
        is_positive = lift > 0
        direction = "increased" if lift > 0 else "decreased"
    else:
        direction = "increased" if lift > 0 else "decreased"
        is_positive = None

    if abs_lift < 1:
        magnitude = "negligible"
    elif abs_lift < 5:
        magnitude = "small"
    elif abs_lift < 15:
        magnitude = "moderate"
    elif abs_lift < 30:
        magnitude = "substantial"
    else:
        magnitude = "significant"

    parts = [f"**{metric}**: {direction} by {abs_lift:.2f}% ({baseline_value:.4f} -> {value:.4f})"]
    if is_positive is True:
        parts.append(f"+ {magnitude.capitalize()} improvement")
    elif is_positive is False:
        parts.append(f"- {magnitude.capitalize()} degradation")
    else:
        parts.append(f"o {magnitude.capitalize()} change")

    return " - ".join(parts)


def generate_report(comparison_df, folder_metrics):
    folder_names = list(folder_metrics.keys())
    baseline_name = folder_names[0]

    lines = ["# Profile HMM Alignment Comparison Report\n"]
    lines.append(f"**Baseline Run**: `{baseline_name}`\n")
    lines.append(f"**Runs Compared**: {len(folder_names)}\n")

    lines.append("## Comparison Table\n")
    lines.append(comparison_df.to_markdown(index=False))
    lines.append("\n")

    lines.append("## Run Interpretations\n")
    for folder_name in folder_names[1:]:
        lines.append(f"### {folder_name} vs {baseline_name}\n")

        interpretations = []
        for _, row in comparison_df.iterrows():
            metric = row['metric']
            baseline_value = row[baseline_name]
            value = row[folder_name]
            lift = row[f'{folder_name}_lift%']
            interpretations.append(interpret_metric_lift(metric, lift, value, baseline_value))

        improvements = [i for i in interpretations if i.split(' - ')[-1].startswith('+')]
        degradations = [i for i in interpretations if i.split(' - ')[-1].startswith('-')]
        neutral = [i for i in interpretations if i not in improvements and i not in degradations]

        if improvements:
            lines.append("#### Improvements\n")
            lines += [f"- {i}" for i in improvements]
            lines.append("")
        if degradations:
            lines.append("#### Degradations\n")
            lines += [f"- {i}" for i in degradations]
            lines.append("")
        if neutral:
            lines.append("#### Neutral / Informational\n")
            lines += [f"- {i}" for i in neutral]
            lines.append("")

        n_imp, n_deg = len(improvements), len(degradations)
        lines.append("#### Overall Assessment\n")
        if n_imp > n_deg:
            lines.append(f"**Net Positive**: {n_imp} improvements vs {n_deg} degradations\n")
        elif n_deg > n_imp:
            lines.append(f"**Net Negative**: {n_deg} degradations vs {n_imp} improvements\n")
        else:
            lines.append(f"**Mixed Results**: {n_imp} improvements, {n_deg} degradations\n")

    lines.append("## Metric Descriptions\n")
    lines.append("- **bic**: Bayesian Information Criterion for the selected model (lower = better fit for its complexity)")
    lines.append("- **mean_fit**: Mean per-sequence normalized Viterbi log2-score (higher, i.e. less negative = better fit)")
    lines.append("- **well_fit_count / well_fit_rate**: Sequences (and %) scoring below the fit threshold (higher rate = better)")
    lines.append("- **n_match_states / n_sub_models**: Selected model complexity (informational)")
    lines.append("- **gap_ratio**: Gap opens / gap extends (lower = more efficient gapping)")
    lines.append("- **gap_opens / gap_extends / total_gaps**: Raw gap counts across all columns (lower = denser alignment)")
    lines.append("- **gap_percentage**: % of column positions that are gaps (lower = denser alignment)")
    lines.append("- **mean_variance / median_variance**: Column variance (lower = tighter/cleaner columns)")
    lines.append("- **percentage_clean / percentage_noise**: % of low- vs high-variance columns (higher clean / lower noise = better)")
    lines.append("- **total_columns**: Alignment length in match-state columns (informational)")
    lines.append("- **mean/median/std/min/max_gap_pct**: Per-sequence gap % distribution (lower = better, except min/std which are informational)")
    lines.append("- **mean_max_consecutive / max_consecutive**: Longest gap runs per sequence (lower = fewer long dropouts)")
    lines.append("- **total_gap_positions / total_positions**: Raw totals behind the gap percentages (informational)\n")

    return "\n".join(lines)


if __name__ == '__main__':
    if len(argv) < 3:
        print("Usage: python compare_metrics.py <baseline_run_folder> <run_folder2> [<run_folder3> ...]")
        print("\nFirst folder is treated as the baseline; all others are compared against it.")
        exit(1)

    run_folders = argv[1:]

    print(f"Comparing {len(run_folders)} run(s)")
    print("=" * 70)

    folder_metrics = {}
    for folder in run_folders:
        try:
            folder_metrics[folder] = load_metrics(folder)
        except FileNotFoundError:
            print(f"ERROR: no metrics.csv found in {folder} -- skipping")

    if len(folder_metrics) < 2:
        print("Need at least 2 runs with metrics.csv to compare.")
        exit(1)

    comparison_df = compute_comparison_table(folder_metrics)

    baseline_folder = list(folder_metrics.keys())[0]
    comparison_output = f'{baseline_folder}/comparison_table.csv'
    comparison_df.to_csv(comparison_output, index=False)
    print(f"Saved comparison table to: {comparison_output}")

    report_output = f'{baseline_folder}/comparison_report.md'
    with open(report_output, 'w') as f:
        f.write(generate_report(comparison_df, folder_metrics))
    print(f"Saved comparison report to: {report_output}")

    print("\n" + comparison_df.to_string(index=False))
    print("\nNote: lift% shows percentage change from baseline (first folder)")
    print("  Positive lift = increase, Negative lift = decrease")
    print("=" * 70)
