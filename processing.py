import os
import argparse
import datetime
import pickle as pkl
from lib_phmm.config import CONFIG
from lib_phmm.model import *
from lib_phmm.signals import *
from lib_phmm.phmm_utils import *
from lib_phmm.visualization import *
from lib_phmm.metrics import *

from pathlib import Path
from joblib import Parallel, delayed


MAX_MATCH  = 5
MAX_STATES = 32
MIN_STATES = 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motif-mode", action="store_true", help="split each file into bounded motif regions instead of one whole-file sequence")
    parser.add_argument("--motif-mode-filtered", action="store_true", help="like --motif-mode, but first cluster candidate regions by DTW distance (hierarchical k-medoids) and only pass the resulting medoids into the HMM sweep")
    parser.add_argument("--dtw-warping-band", type=int, default=5, help="Sakoe-Chiba warping band for DTW distance, used by --motif-mode-filtered")
    parser.add_argument("--dtw-epochs", type=int, default=20, help="max k-medoids refinement epochs per split, used by --motif-mode-filtered")
    parser.add_argument("--dtw-threshold", type=float, default=None, help="stop splitting a branch once its average intra-cluster DTW distance drops below this -- default: estimated from --dtw-threshold-percentile of sampled candidate pair distances")
    parser.add_argument("--dtw-threshold-percentile", type=float, default=99, help="when --dtw-threshold is not given, estimate it as this percentile of DTW distances over --dtw-threshold-samples random candidate pairs")
    parser.add_argument("--dtw-threshold-samples", type=int, default=500, help="number of random candidate pairs to sample when estimating --dtw-threshold")
    parser.add_argument("--path", default="../audio/aggression", help="directory containing the input .wav files")
    parser.add_argument("--output-path", default=None, help="directory for cached embeddings/results and plots (default: <path>/output)")
    args = parser.parse_args()

    dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    path = args.path
    output_path = args.output_path if args.output_path is not None else f'{path}/output'
    Path(output_path).mkdir(parents=True, exist_ok=True)

    print("==========================================")
    print("Profile HMM based local alignment         ")
    print("by Daniel Kohlsdorf                       ")
    print("==========================================")
    print(f"Config: {CONFIG}")
    print("==========================================")

    embeddings_path = f'{output_path}/embeddings.pkl'
    results_path    = f'{output_path}/results.pkl'

    files = [f'{path}/{file}' for file in os.listdir(path) if file.endswith('.wav')]

    print("Embedddings")
    embeddings = None
    if os.path.exists(embeddings_path):
        with open(embeddings_path, "rb") as f:
            embeddings = pkl.load(f)
        if len(embeddings) != len(files):
            print(f"Cached embeddings ({len(embeddings)}) don't match {len(files)} wav files found -- recomputing")
            embeddings = None
        else:
            print(f"Loaded cached embeddings from: {embeddings_path}")

    if embeddings is None:
        model      = whisper_model_v2()
        processor  = whisper_processor()
        embeddings = []
        for file in files:
            embeddings.append(load_file(file, model, processor, motif_mode=args.motif_mode or args.motif_mode_filtered))
        with open(embeddings_path, "wb") as f:
            pkl.dump(embeddings, f)
    print("==========================================")

    candidates = []
    for sequence, full_embedding, _ in embeddings:
        for region_classifications, region_embeddings in sequence:
            candidates.append((region_embeddings, full_embedding, region_classifications))

    if args.motif_mode_filtered:
        threshold = args.dtw_threshold
        if threshold is None:
            threshold = estimate_dtw_threshold(
                candidates,
                warping_band=args.dtw_warping_band,
                n_samples=args.dtw_threshold_samples,
                percentile=args.dtw_threshold_percentile,
            )
            print(f"Estimated --dtw-threshold: {threshold} "
                  f"({args.dtw_threshold_percentile}th percentile over {args.dtw_threshold_samples} sampled candidate pairs)")

        n_before = len(candidates)
        candidates = filter_candidates_by_medoid(
            candidates,
            warping_band=args.dtw_warping_band,
            epochs=args.dtw_epochs,
            threshold=threshold,
        )
        print(f"DTW medoid filter: {n_before} candidates -> {len(candidates)} medoids")

    print("Parameter Sweep HMM")
    results = None
    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            results = pkl.load(f)
        print(f"Loaded cached sweep results from: {results_path}")

    if results is None:
        D = embeddings[0][1].shape[1]
        n_frames_total = sum(len(embedding) for _, embedding, _ in embeddings)
        nested_results = Parallel(n_jobs=-1, backend="loky", verbose=10)(
            delayed(sweep_one_state_count)(n_match_states, candidates, MAX_MATCH, D, n_frames_total)
            for n_match_states in range(MIN_STATES, MAX_STATES)
        )
        results = [r for sublist in nested_results for r in sublist]
        with open(results_path, "wb") as f:
            pkl.dump(results, f)

    best = min(results, key=lambda r: r["bic"])
    hmm, n_states = make_hmm(best["exemplar_embeddings"], best["exemplar_classifications"], best["n_match_states"])
    scores_norm, paths, raw_scores = decode_all(embeddings, hmm)
    well_fits, well_fits_scores = best_fit(scores_norm, 34)
    total_fit = sum(scores_norm)
    
    print(f"selected n_match_states={best['n_match_states']}, n_sub_models={best['n_sub_models']}, BIC={best['bic']:.1f}")
    n_models = len(hmm.pdf)

    print("==========================================")
    print("Alignment Quality Metrics")
    msa = build_msa_from_paths(embeddings, paths, n_models, n_states)
    metrics = compute_all_metrics(msa, len(embeddings))
    metrics_df = metrics_to_dataframe(metrics)
    metrics_df.insert(0, 'run', dt)
    metrics_df.insert(1, 'n_sequences', len(embeddings))
    metrics_df.insert(2, 'n_match_states', best['n_match_states'])
    metrics_df.insert(3, 'n_sub_models', best['n_sub_models'])
    metrics_df.insert(4, 'bic', best['bic'])
    metrics_df.insert(5, 'mean_fit', total_fit / len(embeddings))
    metrics_df.insert(6, 'well_fit_count', len(well_fits))
    metrics_df.insert(7, 'well_fit_rate', len(well_fits) / len(embeddings) * 100)
    metrics_df.to_csv(f'{output_path}/metrics.csv', index=False)
    print(metrics_df.to_string(index=False))
    print("==========================================")

    print("Plot results")
    all_segments = {}
    for i in range(0, len(files)):
        waveform, sr           = load_filtered_waveform(files[i])
        gapped, segments, _    = extract_match_wav(paths[i], waveform, n_models, n_states)
        all_segments[i]        = segments
        torchaudio.save(f'{output_path}/aligned_{i}.wav', gapped, sr)

    fig, axes = plot_aligned_spectrograms(list(range(len(files))), files, paths, n_states, n_models)
    plt.savefig(f'{output_path}/aligned_plots.png')
    plt.close()
    print("==========================================") 

    
