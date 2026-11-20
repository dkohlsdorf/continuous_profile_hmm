"""
Preprocess one extend-find run for the web UI.

For every recording subfolder of RUN_FOLDER (a motif_extend_find_<timestamp>
Drive folder written by serverless/runpod_worker/handler.py), each row of its
motif_hits.csv not classified NOISE becomes one globally numbered result:

    webui_preprocessing/spectrograms/<results_id>.png         the hit only
    webui_preprocessing/region_spectrograms/<results_id>.png  hit +- CONTEXT_S
    webui_preprocessing/embeddings/<results_id>.pkl           (frames, dim) array
    webui_preprocessing/results.csv                           one row per results_id
    webui_preprocessing/done.txt                              finished stems

results_id is one counter across all recordings. Only one recording's audio,
embeddings and outputs are on local disk at a time: everything is uploaded and
deleted before the next recording starts.

Restarts: stems in done.txt are skipped. results.csv rows are written before a
recording's files are uploaded, so the next id is always above every id ever
handed out; rows of a recording that didn't finish are dropped on restart (its
already-uploaded files are left unreferenced, never overwritten by name).

Sample positions in motif_hits.csv are in the embedding pickle's sample_rate
(the handler resamples to the model's training rate), not the recording's
native rate, so hits are cut from the native audio in seconds.

Usage:
    python webui/preprocessing.py RUN_FOLDER_ID CACHE_FOLDER_ID \
        --wav-folders ID [ID ...] [--embedding-model whisper]

Auth: same env vars as the handler (GOOGLE_REFRESH_TOKEN + GOOGLE_CLIENT_ID +
GOOGLE_CLIENT_SECRET, or GOOGLE_SERVICE_ACCOUNT_JSON).
"""

import argparse
import base64
import io
import json
import os
import pickle as pkl
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload, MediaIoBaseUpload

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from motif_discovery import load_embeddings_cache  # noqa: E402

OUTPUT_FOLDER_NAME = "webui_preprocessing"
SUBFOLDERS = ("spectrograms", "region_spectrograms", "embeddings")
CACHE_VERSIONS = ("v2", "v3")  # same lookup order as the handler: v3 is a superset of v2
AUDIO_EXTENSIONS = (".wav", ".m4a")
FOLDER_MIME = "application/vnd.google-apps.folder"

CONTEXT_S = 0.5
# Same STFT as the Viterbi notebooks' plot_example(); one fixed dB range
# (relative to the region's peak) for every image, and the hit image is a
# column slice of its region image, so the two share one normalisation.
N_FFT = 512
HOP = 128
FMAX_HZ = 30000
DB_RANGE = 60


# ---------------------------------------------------------------- local processing

def to_mono_wav(src, dst):
    """Channel 0 only, native rate -- same ffmpeg call as the handler's convert_to_mono_first_channel()."""
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src,
                    "-af", "pan=mono|c0=c0", dst], check=True)


def spectrogram_db(x, sr):
    S = librosa.amplitude_to_db(np.abs(librosa.stft(x.astype(np.float32), n_fft=N_FFT, hop_length=HOP)),
                                ref=np.max)
    n_bins = int(np.searchsorted(librosa.fft_frequencies(sr=sr, n_fft=N_FFT), FMAX_HZ, side="right"))
    return S[:n_bins]


def save_png(S, path):
    plt.imsave(path, S[::-1], cmap="gray_r", vmin=-DB_RANGE, vmax=0)


def process_recording(hits, embeddings_path, mono_wav, first_id, out_dir):
    """
    Write <id>.png / region <id>.png / <id>.pkl for every row of hits into
    out_dir's SUBFOLDERS, ids counting up from first_id. Returns the
    results.csv rows (without the recording-level columns).
    """
    embeddings, _, step_samples, emb_sr, _ = load_embeddings_cache(embeddings_path)
    info = sf.info(mono_wav)
    sr, n_samples = info.samplerate, info.frames
    rows = []
    for results_id, hit in enumerate(hits.itertuples(index=False), start=first_id):
        t0, t1 = hit.start_sample / emb_sr, hit.stop_sample / emb_sr
        r0, r1 = max(0.0, t0 - CONTEXT_S), min(n_samples / sr, t1 + CONTEXT_S)
        x, _ = sf.read(mono_wav, start=int(r0 * sr), stop=int(r1 * sr))
        S = spectrogram_db(x, sr)
        c0, c1 = round((t0 - r0) * sr / HOP), round((t1 - r0) * sr / HOP)
        save_png(S, out_dir / "region_spectrograms" / f"{results_id}.png")
        save_png(S[:, c0:c1 + 1], out_dir / "spectrograms" / f"{results_id}.png")
        with open(out_dir / "embeddings" / f"{results_id}.pkl", "wb") as f:
            pkl.dump(embeddings[hit.start_sample // step_samples: hit.stop_sample // step_samples], f)
        rows.append({
            "results_id": results_id, "hit_id": hit.hit_id, "submodel_id": hit.submodel_id,
            "start_sample": hit.start_sample, "stop_sample": hit.stop_sample, "sample_rate": emb_sr,
            "start_time_s": t0, "stop_time_s": t1, "n_frames": hit.n_frames,
            "region_start_s": r0, "region_offset_s": t0 - r0,
            "classification": hit.classification,
            "classification_confidence": hit.classification_confidence,
            "whistle_direction": hit.whistle_direction,
            "mean_emission_score": hit.mean_emission_score, "mean_noise_score": hit.mean_noise_score,
            "mean_llr_score": hit.mean_llr_score,
        })
    return rows


# ---------------------------------------------------------------- Google Drive

def get_drive_service():
    """Same credentials as the handler's get_drive_service()."""
    if os.environ.get("GOOGLE_REFRESH_TOKEN"):
        from google.oauth2.credentials import Credentials
        credentials = Credentials(
            token=None, refresh_token=os.environ["GOOGLE_REFRESH_TOKEN"],
            token_uri="https://oauth2.googleapis.com/token",
            client_id=os.environ["GOOGLE_CLIENT_ID"], client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
            scopes=["https://www.googleapis.com/auth/drive"])
    else:
        creds = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"].strip()
        creds_data = json.loads(base64.b64decode(creds + "=" * (-len(creds) % 4)))
        credentials = service_account.Credentials.from_service_account_info(
            creds_data, scopes=["https://www.googleapis.com/auth/drive"])
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def list_children(service, folder_id, query=""):
    q = f"'{folder_id}' in parents and trashed = false" + (f" and {query}" if query else "")
    files, token = [], None
    while True:
        res = service.files().list(q=q, fields="nextPageToken, files(id, name, mimeType)", pageSize=1000,
                                   pageToken=token, supportsAllDrives=True,
                                   includeItemsFromAllDrives=True).execute()
        files += res.get("files", [])
        token = res.get("nextPageToken")
        if not token:
            return files


def find_child(service, folder_id, name):
    escaped = name.replace("\\", "\\\\").replace("'", "\\'")
    found = list_children(service, folder_id, f"name = '{escaped}'")
    return found[0]["id"] if found else None


def find_or_create_folder(service, parent_id, name):
    return find_child(service, parent_id, name) or service.files().create(
        body={"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]},
        fields="id", supportsAllDrives=True).execute()["id"]


def download(service, file_id, path):
    with open(path, "wb") as f:
        downloader = MediaIoBaseDownload(f, service.files().get_media(fileId=file_id, supportsAllDrives=True),
                                         chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk(num_retries=5)


def read_text(service, file_id):
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, service.files().get_media(fileId=file_id, supportsAllDrives=True))
    done = False
    while not done:
        _, done = downloader.next_chunk(num_retries=5)
    return buf.getvalue().decode()


def write_text(service, folder_id, file_id, name, text, mime):
    """Create or overwrite a small text file; returns its id."""
    media = MediaIoBaseUpload(io.BytesIO(text.encode()), mimetype=mime)
    if file_id:
        service.files().update(fileId=file_id, media_body=media, supportsAllDrives=True).execute()
        return file_id
    return service.files().create(body={"name": name, "parents": [folder_id]}, media_body=media,
                                  fields="id", supportsAllDrives=True).execute()["id"]


_thread_local = threading.local()


def upload(folder_id, path, mime):
    # googleapiclient services aren't thread-safe: one per upload thread.
    if not hasattr(_thread_local, "service"):
        _thread_local.service = get_drive_service()
    _thread_local.service.files().create(
        body={"name": path.name, "parents": [folder_id]},
        media_body=MediaFileUpload(str(path), mimetype=mime),
        fields="id", supportsAllDrives=True).execute(num_retries=5)


# ---------------------------------------------------------------- main loop

def cache_names(stem, embedding_model):
    suffix = "" if embedding_model == "whisper" else f".{embedding_model}"
    return [f"{stem}.{version}{suffix}.pkl" for version in CACHE_VERSIONS]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_folder", help="Drive id of the motif_extend_find_<timestamp> folder")
    parser.add_argument("cache_folder", help="Drive id of motif_embeddings_cache")
    parser.add_argument("--wav-folders", nargs="+", required=True, help="Drive ids of the recording folders")
    parser.add_argument("--embedding-model", default="whisper")
    parser.add_argument("--workers", type=int, default=8, help="parallel uploads")
    args = parser.parse_args()

    service = get_drive_service()
    out_id = find_or_create_folder(service, args.run_folder, OUTPUT_FOLDER_NAME)
    sub_ids = {name: find_or_create_folder(service, out_id, name) for name in SUBFOLDERS}

    results_file = find_child(service, out_id, "results.csv")
    done_file = find_child(service, out_id, "done.txt")
    done = set(read_text(service, done_file).splitlines()) if done_file else set()
    results = pd.read_csv(io.StringIO(read_text(service, results_file))) if results_file else pd.DataFrame()
    next_id = int(results["results_id"].max()) + 1 if len(results) else 1
    if len(results):
        results = results[results["stem"].isin(done)]  # drop rows of an unfinished recording

    # Same stem in several places: keep the copy the handler embedded -- earliest
    # folder in --wav-folders order, then alphabetically first name.
    audio = {}
    for folder in args.wav_folders:
        for f in sorted(list_children(service, folder, f"mimeType != '{FOLDER_MIME}'"), key=lambda f: f["name"]):
            if f["name"].lower().endswith(AUDIO_EXTENSIONS):
                audio.setdefault(Path(f["name"]).stem, f)
    recordings = sorted((f for f in list_children(service, args.run_folder, f"mimeType = '{FOLDER_MIME}'")
                         if f["name"] != OUTPUT_FOLDER_NAME), key=lambda f: f["name"])
    print(f"{len(recordings)} recordings, {len(done)} already done, next results_id {next_id}")

    failed = []
    for i, rec in enumerate(recordings, start=1):
        stem = rec["name"]
        if stem in done:
            continue
        print(f"[{i}/{len(recordings)}] {stem}")
        work = Path(tempfile.mkdtemp(prefix="webui_"))
        try:
            hits_id = find_child(service, rec["id"], "motif_hits.csv")
            cache_id = next(filter(None, (find_child(service, args.cache_folder, n)
                                          for n in cache_names(stem, args.embedding_model))), None)
            if not hits_id or not cache_id or stem not in audio:
                raise FileNotFoundError(f"motif_hits.csv: {bool(hits_id)}, embeddings: {bool(cache_id)}, "
                                        f"audio: {stem in audio}")
            hits = pd.read_csv(io.StringIO(read_text(service, hits_id)))
            hits = hits[hits["classification"] != "NOISE"]
            download(service, cache_id, work / "embeddings.pkl")
            raw = work / audio[stem]["name"]
            download(service, audio[stem]["id"], raw)
            to_mono_wav(str(raw), str(work / "mono.wav"))
            raw.unlink()

            out = work / "out"
            for name in SUBFOLDERS:
                (out / name).mkdir(parents=True)
            rows = pd.DataFrame(process_recording(hits, work / "embeddings.pkl", work / "mono.wav", next_id, out))
            if len(rows):
                rows.insert(1, "stem", stem)
                rows.insert(2, "wav_file", audio[stem]["name"])
                rows.insert(3, "wav_file_id", audio[stem]["id"])
                rows["embedding_model"] = args.embedding_model
                next_id = int(rows["results_id"].max()) + 1
                # rows first, then files: the next id stays above every uploaded file even after a crash
                results_file = write_text(service, out_id, results_file, "results.csv",
                                          pd.concat([results, rows]).to_csv(index=False), "text/csv")
                jobs = [(sub_ids[name], p, "image/png" if name != "embeddings" else "application/octet-stream")
                        for name in SUBFOLDERS for p in sorted((out / name).iterdir())]
                with ThreadPoolExecutor(args.workers) as pool:
                    list(pool.map(lambda job: upload(*job), jobs))
                results = pd.concat([results, rows])
            done.add(stem)
            done_file = write_text(service, out_id, done_file, "done.txt", "\n".join(sorted(done)) + "\n",
                                   "text/plain")
            print(f"  {len(rows)} hits -> results_id up to {next_id - 1}")
        except Exception as err:
            print(f"  FAILED: {err!r}")
            failed.append(stem)
        finally:
            shutil.rmtree(work)

    print(f"\ndone: {len(done)}/{len(recordings)} recordings, {len(results)} results"
          + (f"; failed: {', '.join(failed)}" if failed else ""))


if __name__ == "__main__":
    main()
