import os
import datetime
import pickle as pkl
from lib_phmm.config import CONFIG
from lib_phmm.model import *
from lib_phmm.signals import *
from lib_phmm.phmm_utils import *
from lib_phmm.visualization import *
from pathlib import Path


MAX_MATCH  = 5
MAX_STATES = 32
MIN_STATES = 8


if __name__ == "__main__":
    dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    path = '../audio/aggression'
    output_path = f'{path}/output_{dt}'
    Path(output_path).mkdir(parents=True, exist_ok=True)

    print("==========================================")
    print("Profile HMM based local alignment         ")
    print("by Daniel Kohlsdorf                       ")
    print("==========================================")
    print(f"Config: {CONFIG}")
    print("==========================================")
    
    model     = whisper_model_v2()
    processor = whisper_processor()

    print("Embedddings")
    files = [f'{path}/{file}' for file in os.listdir(path) if file.endswith('.wav')]
    embeddings = [load_file(file, model, processor) for file in files]
    with open(f"{output_path}/embeddings.pkl", "wb") as f:
        pkl.dump(embeddings, f)
    print("==========================================")

    print("Parameter Sweep HMM")
    D = len(embeddings[0][0][0])
    n_frames_total = sum(len(embedding) for _, embedding, _ in embeddings)    
    nested_results = Parallel(n_jobs=-1, backend="loky", verbose=10)(
        delayed(sweep_one_state_count)(n_match_states, embeddings, MAX_MATCH, D, n_frames_total)
        for n_match_states in range(MIN_STATES, MAX_STATES)
    )
    results = [r for sublist in nested_results for r in sublist]
    with open(f"{output_path}/results.pkl", "wb") as f:
        pkl.dump(results, f)
    
    best = min(results, key=lambda r: r["bic"])
    hmm, n_states = make_hmm(best["exemplar_embeddings"], best["exemplar_classifications"], best["n_match_states"])
    scores_norm, paths, raw_scores = decode_all(embeddings, hmm)
    well_fits, well_fits_scores = best_fit(scores_norm, 34)
    total_fit = sum(scores_norm)
    
    print(f"selected n_match_states={best['n_match_states']}, n_sub_models={best['n_sub_models']}, BIC={best['bic']:.1f}")
    n_models = len(hmm.pdf)

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

    
