import torch
import matplotlib.pyplot as plt
import lib_phmm.profile_hmm as phmm

from lib_phmm.config import CONFIG
from lib_phmm.signals import *


def extract_match_wav(path, waveform, n_models, n_states, frames_per_column=2,
                       front_frames=2, back_frames=2,
                       step_samples=CONFIG['window_size_samples'] // CONFIG['step_denominator']):
    """
    Fixed-width aligned export. Each slot is exactly `step_samples` long --
    the embedding hop, not the (overlapping) analysis window -- so
    consecutive real slots splice together as genuinely contiguous,
    non-overlapping audio instead of duplicating the ~50% overlap between
    neighboring windows, which was producing a discontinuity (audible
    click / spectrogram streak) at every join.

    Also returns `segments`: one (start_time, stop_time, model_number)
    tuple per domain (B..E span) in the original path, so you know which
    sub-HMM each stretch of the recording matched.

    Also returns `real_spans`: one (start_sample, stop_sample, model)
    tuple per real (non-silence-padded) column slot in the exported audio,
    so callers can highlight only the actual matched regions instead of
    the whole fixed-width model block.
    """
    def window_audio(t):
        start = t * step_samples
        return waveform[:, start:start + step_samples]

    silence = torch.zeros((waveform.shape[0], step_samples))

    def fixed_slot(times, n_slots):
        times = times[:n_slots] + [None] * (n_slots - len(times))
        clips = [window_audio(t) if t is not None else silence for t in times]
        return clips, [t is not None for t in times]

    front_ts, back_ts = [], []
    col_ts = [[[] for _ in range(n_states)] for _ in range(n_models)]
    segments = []
    seg_start, seg_stop, seg_model = None, None, None
    entered = False
    for step in path:
        if step.state == phmm.N and not entered:
            front_ts.append(step.time)
        elif step.state == phmm.B:
            entered = True
            seg_start, seg_stop, seg_model = None, None, None
        elif step.state == phmm.C:
            back_ts.append(step.time)
        elif step.state == phmm.E:
            if seg_start is not None:
                segments.append((seg_start, seg_stop, seg_model))
            seg_start, seg_stop, seg_model = None, None, None
        elif step.state >= phmm.MATCH_STATE:
            idx = step.state - phmm.MATCH_STATE
            n   = idx // n_states
            k   = idx % n_states
            col_ts[n][k].append(step.time)
            if seg_start is None:
                seg_start = step.time
                seg_model = n
            seg_stop = step.time

    clips, real_spans = [], []
    offset = 0

    front_clips, _ = fixed_slot(front_ts, front_frames)
    clips += front_clips
    offset += len(front_clips) * step_samples

    for n in range(n_models):
        for k in range(n_states):
            col_clips, col_real = fixed_slot(col_ts[n][k], frames_per_column)
            for clip, is_real in zip(col_clips, col_real):
                clips.append(clip)
                if is_real and k != 0:
                    real_spans.append((offset, offset + step_samples, n))
                offset += step_samples

    back_clips, _ = fixed_slot(back_ts, back_frames)
    clips += back_clips

    return torch.cat(clips, dim=1), segments, real_spans


def plot_aligned_spectrograms(indices, files, paths, n_states, n_models,
                               frames_per_column=2, front_frames=2, back_frames=2,
                               step_samples=CONFIG['window_size_samples'] // CONFIG['step_denominator'],
                               figsize_per_row=2.5, cmap_name='tab10'):
    """
    Stack one spectrogram per file (given by `indices` into `files`/`paths`)
    of the fixed-width aligned export from extract_match_wav, shading only
    the real (non-silence-padded) matched regions by which sub-HMM they
    belong to -- same color means the same sub-HMM in every row.
    """
    cmap = plt.get_cmap(cmap_name, n_models)
    fig, axes = plt.subplots(len(indices), 1, figsize=(14, figsize_per_row * len(indices)), sharex=False)
    if len(indices) == 1:
        axes = [axes]

    for ax, i in zip(axes, indices):
        waveform, sr = load_filtered_waveform(files[i])
        gapped, _, real_spans = extract_match_wav(paths[i], waveform, n_models, n_states,
                                                    frames_per_column, front_frames, back_frames,
                                                    step_samples)
        signal = gapped[0].numpy()
        ax.specgram(signal, Fs=sr, NFFT=1024, noverlap=512, cmap='gray')

        for start, stop, model in real_spans:
            ax.axvspan(start / sr, stop / sr, color=cmap(model), alpha=0.35, lw=0)

        ax.set_ylabel(f'file {i}')
        ax.set_yticks([])

    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(n), alpha=0.35) for n in range(n_models)]
    fig.legend(handles, [f'model {n}' for n in range(n_models)], loc='upper right')
    axes[-1].set_xlabel('time (s)')
    fig.tight_layout()
    return fig, axes
