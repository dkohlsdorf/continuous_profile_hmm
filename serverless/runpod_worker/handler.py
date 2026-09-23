"""
RunPod Serverless Handler for motif `find`

Receives a Google Drive folder of recordings, and for each one: downloads
it, converts it to a single channel (the first) with ffmpeg, runs
motif_discovery.py find against the packaged L2 motif model with the
packaged noise recording as --noise-wav, uploads that recording's
motif_hits.csv + submodel_<n>_hits.wav clips to its own subfolder in
Google Drive, then deletes every local file for that recording before
moving on to the next -- so local disk stays bounded regardless of how
many recordings are in the input folder, instead of needing all of them
downloaded at once.

Google Drive auth mirrors dolphin_msa/serverless/runpod_worker/handler.py
exactly (get_drive_service() below) -- same OAuth-refresh-token /
service-account fallback, same environment variables.

Environment Variables Required:
    Option A - OAuth (recommended for personal Gmail):
        GOOGLE_REFRESH_TOKEN: OAuth refresh token
        GOOGLE_CLIENT_ID: OAuth client ID
        GOOGLE_CLIENT_SECRET: OAuth client secret

    Option B - Service Account (for Workspace with Shared Drives):
        GOOGLE_SERVICE_ACCOUNT_JSON: Base64-encoded service account JSON credentials

Input:
    {
        "input": {
            "gdrive_folder_id": "your-google-drive-folder-id",
            "output_gdrive_folder_id": "folder-id-for-output",  # optional, defaults to gdrive_folder_id
            "noise_components": 6,      # optional, default 6 (upper bound on noise GMM components)
            "noise_var_scale": 1.0,     # optional, default 1.0 -- see the note at its parse site;
                                        # values >1 flatly handicap the noise model by
                                        # D/2*log2(scale) bits and break mean_llr_score
            "hit_gap_seconds": 0.5,     # optional, default 0.5 (silence gap between hits in submodel_<n>_hits.wav)
            "require_full_match": false,           # optional, default false -- disallow partial matches
                                                    # (entering/exiting a submodel's match-state chain at
                                                    # an interior state). See motif_discovery.py's
                                                    # --require-full-match / phmm.viterbi()'s docstring.
            "use_embeddings_cache": true,          # optional, default true (see Embeddings cache below)
            "embeddings_cache_folder_id": "..."    # optional, defaults to a "motif_embeddings_cache"
                                                   # folder under the output parent, created on demand
            "pvl_file_id": "..."        # optional, default none -- a Drive file id for a PVL .xlsx
                                        # shotlog (see Annotation below). Downloaded once per job; if
                                        # omitted, annotation is simply skipped, find is unaffected.
        }
    }

Extend mode:
    Set "mode": "extend" to run a different, staged pipeline instead of the
    find job above -- see handle_extend()'s docstring for the full stage
    breakdown (embed -> build candidates -> train a new model -> find with
    it). Same output_gdrive_folder_id/noise_*/require_full_match inputs
    apply (the noise_*/require_full_match ones only matter once the
    pipeline reaches its find stage), plus:
        "mode": "extend",
        "gdrive_folder_id": "..." or ["...", "..."],  # one folder id, or a
                                    # list of them -- every file across all
                                    # of them feeds the same shared
                                    # candidates pool/model (not a separate
                                    # pipeline per folder). output_gdrive_
                                    # folder_id anchors the state/output
                                    # folders if given, else the first
                                    # listed input folder does.
        "pvl_file_id": "..." or ["...", null, "..."],  # optional. One id
                                    # shared by every folder, or a list
                                    # matched positionally to
                                    # gdrive_folder_id (use null for a
                                    # folder with no PVL) -- since folders
                                    # can span different datasets/years
                                    # with different shotlogs. Only matters
                                    # once the pipeline reaches its find
                                    # stage; a length mismatch against
                                    # gdrive_folder_id is a hard error, not
                                    # a silent misalignment.
        "min_region_frames": 10,    # optional, default 10 -- shortest
                                    # contiguous non-NOISE run (in frames)
                                    # to keep as a new training candidate
        "all_medoids": true,        # optional, default true (NOT
                                    # train-candidates' own CLI default) --
                                    # every DTW-filtered candidate becomes
                                    # its own sub-model directly, skipping
                                    # the greedy O(max_match * candidates^2)
                                    # sub-model search. Set false to use
                                    # that search instead; see
                                    # run_train_candidates()'s docstring for
                                    # why growing-pool extend defaults the
                                    # other way from train-candidates itself.
        "dtw_restarts": 10,          # optional, default: train-candidates'
                                    # own CLI default (10) -- random splits
                                    # tried per k-medoids level, keeping the
                                    # densest. See hierarchical_kmedian_dtw
                                    # .hpp's KMEDOIDS_RESTARTS_DEFAULT.
        "dtw_density_tolerance": 0.05,  # optional, default: train-candidates'
                                    # own CLI default (0.05) -- how much less
                                    # dense than its parent a branch may
                                    # still be and keep splitting (higher ->
                                    # more, finer medoids). See
                                    # DENSITY_TOLERANCE_DEFAULT there.
        "dtw_no_path_norm": False,  # optional, default False -- True uses
                                    # the raw summed DTW cost for medoid
                                    # filtering instead of dividing by the
                                    # warp-path length.
        "min_match_states": 3       # optional, default: train-candidates'
                                    # own CLI default (MIN_STATES=3) -- floor
                                    # for the BIC n_match_states sweep. With
                                    # all_medoids=True especially, a large
                                    # sub-model pool can make BIC floor
                                    # n_match_states at the sweep's minimum
                                    # regardless of true motif complexity
                                    # (count_params()'s penalty scales with
                                    # n_sub_models * n_match_states); raise
                                    # this to force richer models. See
                                    # run_train_candidates()'s docstring.
    The candidates-building stage also uploads l2_<timestamp>.csv/.wav into
    the "motif_extend_state" folder -- that cycle's newly-mined regions, in
    the same starts,stops-into-one-wav format train's own csv_path/wav_path
    input uses. These accumulate (one dated pair per cycle) rather than
    being overwritten, unlike candidates.pkl/phmm_extended.pkl.
    Call repeatedly (one job submission per stage) until the response's
    "stage" field reports "find" -- each call only advances the pipeline
    by whichever single stage isn't done yet, so a multi-hour DTW filter +
    BIC sweep (the training stage) is never bundled into the same job as
    embedding or searching a whole folder. State (the growing
    candidates.pkl, then phmm_extended.pkl) lives in a persistent
    "motif_extend_state" Drive folder under the output parent, alongside
    the existing "motif_embeddings_cache" folder both this mode and find
    share.

Annotation:
    When pvl_file_id is given, each recording's motif_hits.csv is merged
    into the matching rows of the PVL via annotate.py, producing
    annotated_motif_hits.csv alongside it (uploaded automatically -- it's
    just another file in that recording's output_path by the time results
    upload). Which PVL rows match a recording is decided by an encounter
    number read from the recording's OWN FILENAME (encounter_from_filename()
    -- same "E<number>_..." convention as annotate.py's key(), e.g.
    "E1234_dolphins.wav" -> encounter 1234): if your recordings aren't named
    that way, or an encounter isn't in the PVL, that one file's annotation
    is skipped (logged, not fatal) -- find's own results for it still
    upload normally.

Combined pvl db/csv:
    At the end of a run (both find's own handler() and extend mode's find
    stage -- see build_and_upload_pvl_db()), every annotated_motif_hits.csv
    this run actually produced is pulled back down and combined via
    pvl_feature_extractor.py's own clean()/context_features()/
    actors_features()/add_path() into pvl.db + pvl_annotated_audio.csv,
    plus hotcoded.out + hotcoding_meta.json from that same module's
    hotcode_df() (a one-hot per annotated event of its own context/actor
    flags and the motif cluster ids in the window before it -- see
    hotcode_df()'s own docstring), all uploaded into this run's own output
    folder. Best-effort: a run with no PVL configured, or where every
    file's annotation was skipped, simply has nothing to combine and
    uploads none of these files -- logged, not fatal, find/extend's own
    results are unaffected either way.

Embeddings cache:
    Whisper-decoding a recording dominates find's runtime, and the result
    depends only on the audio -- not on any --noise-* setting -- so it's
    cached to Google Drive as <recording stem>.pkl and reused whenever the
    same recording is processed again. That makes re-running a whole folder
    to sweep noise_var_scale / noise_components cheap after the first pass.
    The cache folder deliberately sits next to the timestamped result
    folders rather than inside one, so it survives across runs. Set
    use_embeddings_cache=false to force a fresh decode (e.g. after changing
    the embedding model itself, which the cache can't detect).

    The packaged --noise-wav's own embeddings are cached the same way, as
    a single NOISE_EMBEDDINGS_CACHE_NAME entry in this same Drive folder
    (see restore_noise_embeddings()/save_noise_embeddings()) -- it's a
    fixed, never-changing asset, so this decode happens once ever across
    every job/container from here on, not once per job or once per
    recording. Bump NOISE_EMBEDDINGS_CACHE_VERSION if the packaged noise
    wav itself, or what gets written into noise_embeddings.pkl, ever
    changes.

Output:
    {
        "status": "success",
        "output_folder_id": "created-output-folder-id",
        "output_folder_name": "motif_find_2026-08-31_12-30-45",
        "files_found": 14,
        "files_processed": 14,
        "files_failed": 0,
        "embeddings_cache_folder_id": "cache-folder-id",
        "results": [
            {"name": "...", "status": "success", "hits": 235, "uploaded": 22,
             "output_folder_id": "...", "embeddings_from_cache": false, "embeddings_cached": true},
            ...
        ]
    }
"""

import os
import json
import base64
import binascii
import shutil
import sqlite3
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import runpod

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

import pvl_feature_extractor

MOTIF_DISCOVERY_SCRIPT = "/app/motif_discovery.py"
ANNOTATE_SCRIPT = "/app/annotate.py"
APP_DIR = "/app"
MODEL_PATH = "/app/motif/l2/phmm_l2.pkl"
NOISE_WAV_PATH = "/app/assets/naive_noise.wav"
BAKED_CANDIDATES_PATH = "/app/motif/l2/candidates.pkl"
TRAINING_METADATA_PATH = "/app/motif/l2/training_metadata.json"

# Persistent (NOT per-run/timestamped) Drive folder holding one "extend" job
# family's staged state -- candidates.pkl, then phmm_extended.pkl once
# trained -- so repeated job submissions pick up where the last one left
# off (see handle_extend()). Created on demand under the output parent
# folder, same convention as EMBEDDINGS_CACHE_FOLDER_NAME below.
EXTEND_STATE_FOLDER_NAME = "motif_extend_state"
EXTEND_CANDIDATES_NAME = "candidates.pkl"
EXTEND_MODEL_NAME = "phmm_extended.pkl"
EXTEND_TRAINING_METADATA_NAME = "training_metadata.json"

# motif_hits.csv + submodel_<n>_hits.wav are this run's deliverables.
# embeddings.pkl and noise_embeddings.pkl both go to the persistent Drive
# cache folder instead (see EMBEDDINGS_CACHE_FOLDER_NAME/
# NOISE_EMBEDDINGS_CACHE_NAME, restore_noise_embeddings()/
# save_noise_embeddings()), not into the timestamped results folder -- a
# results folder is per-run, so a copy there could never be found again by
# a later run looking to skip decoding.
UPLOAD_SKIP_NAMES = {"embeddings.pkl", "noise_embeddings.pkl"}

# Persistent (NOT per-run/timestamped) Drive folder holding one
# <recording stem>.pkl per already-embedded recording, so a later run over
# the same recordings skips the Whisper decode entirely -- by far the
# slowest part of find. Created on demand under the output parent folder.
EMBEDDINGS_CACHE_FOLDER_NAME = "motif_embeddings_cache"

# Bump whenever anything changes what the stored embedding MEANS -- the
# pickle's shape, the audio preprocessing, or the embedding model itself.
# Entries are keyed <stem>.<version>.pkl, so a bump makes older ones simply
# invisible rather than silently reused or blowing up on unpack.
#   v1 -> v2: pickle gained sample_rate (3-tuple -> 4-tuple) AND search
#             audio is now resampled to TRAINING_SAMPLE_RATE, so v1
#             embeddings describe different audio at a different rate.
EMBEDDINGS_CACHE_VERSION = "v2"

# The noise wav is baked into the image and identical for every file, but
# build_noise_model() caches its embeddings under each run's output_path --
# which is a fresh temp dir per recording here, so the same ~105s clip would
# otherwise be re-embedded once per recording. Stashing it at this
# container-local path after the first file and copying it back in for
# every subsequent one keeps that decode to once per job.
NOISE_EMBEDDINGS_CACHE_PATH = "/app/work/noise_embeddings.pkl"

# Drive-persisted counterpart of NOISE_EMBEDDINGS_CACHE_PATH -- lives in the
# same motif_embeddings_cache folder find's own per-recording embeddings
# use (see EMBEDDINGS_CACHE_FOLDER_NAME), so it survives across separate
# job submissions/cold containers too, not just within one already-warm
# job. Versioned the same way as EMBEDDINGS_CACHE_VERSION: bump this if
# what build_noise_model() writes into noise_embeddings.pkl ever changes
# shape, or if the packaged noise wav / its target sample rate changes --
# a bump makes the old Drive entry simply invisible rather than silently
# reused.
NOISE_EMBEDDINGS_CACHE_VERSION = "v1"
NOISE_EMBEDDINGS_CACHE_NAME = f"noise_embeddings.{NOISE_EMBEDDINGS_CACHE_VERSION}.pkl"



def get_drive_service():
    """Initialize Google Drive API service using OAuth or service account credentials."""

    # Option A: OAuth refresh token (for personal Gmail accounts)
    refresh_token = os.environ.get('GOOGLE_REFRESH_TOKEN')
    client_id = os.environ.get('GOOGLE_CLIENT_ID')
    client_secret = os.environ.get('GOOGLE_CLIENT_SECRET')

    if refresh_token and client_id and client_secret:
        print("Using OAuth credentials (personal account)")
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request

        credentials = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri='https://oauth2.googleapis.com/token',
            client_id=client_id,
            client_secret=client_secret,
            scopes=['https://www.googleapis.com/auth/drive']
        )

        # Refresh to get access token
        credentials.refresh(Request())
        return build('drive', 'v3', credentials=credentials)

    # Option B: Service account (for Workspace with Shared Drives)
    creds_json = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON')
    if creds_json:
        print("Using service account credentials")
        # Clean and decode base64 credentials
        # 1. Remove any whitespace/newlines that may have been introduced
        creds_clean = creds_json.strip().replace('\n', '').replace('\r', '').replace(' ', '')

        # 2. Handle URL-safe base64 (replace - and _ with + and /)
        creds_clean = creds_clean.replace('-', '+').replace('_', '/')

        # 3. Add padding if needed (base64 strings must have length divisible by 4)
        padding_needed = (4 - len(creds_clean) % 4) % 4
        creds_padded = creds_clean + '=' * padding_needed

        try:
            creds_data = json.loads(base64.b64decode(creds_padded))
        except (json.JSONDecodeError, binascii.Error) as e:
            raise ValueError(f"Invalid GOOGLE_SERVICE_ACCOUNT_JSON: {e}. Re-encode with: cat creds.json | base64 | tr -d '\\n'")

        credentials = service_account.Credentials.from_service_account_info(
            creds_data,
            scopes=['https://www.googleapis.com/auth/drive']
        )

        return build('drive', 'v3', credentials=credentials)

    raise ValueError(
        "No Google credentials configured. Set either:\n"
        "  - GOOGLE_REFRESH_TOKEN + GOOGLE_CLIENT_ID + GOOGLE_CLIENT_SECRET (for personal Gmail)\n"
        "  - GOOGLE_SERVICE_ACCOUNT_JSON (for Workspace with Shared Drives)"
    )


AUDIO_EXTENSIONS = ('.wav', '.m4a')


def list_audio_files(service, folder_id):
    """
    List (not download) every WAV/M4A file in a Google Drive folder --
    m4a is fine as input since convert_to_mono_first_channel()'s ffmpeg
    step reads whatever container/codec the source is and always writes
    a wav out, regardless of what came in.

    Split out from downloading so the caller can process (download, find,
    upload, delete) one file at a time instead of pulling the whole folder
    to disk up front.
    """
    name_clauses = " or ".join(f"name contains '{ext}'" for ext in AUDIO_EXTENSIONS)
    query = (
        f"'{folder_id}' in parents and "
        f"(mimeType='audio/wav' or mimeType='audio/x-wav' or "
        f"mimeType='audio/mp4' or mimeType='audio/x-m4a' or mimeType='audio/m4a' or "
        f"{name_clauses}) and trashed=false"
    )
    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType)",
        pageSize=1000,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()

    # The Drive query's "name contains '<ext>'" clauses can match a
    # Google-native file (Doc/Sheet/Slides/etc.) whose name merely contains
    # e.g. ".wav" as a substring; those have no binary content and 403 on
    # download (only export), so exclude any Google Workspace mimeType and
    # require the name to actually end in one of AUDIO_EXTENSIONS.
    return [
        f for f in results.get('files', [])
        if not f.get('mimeType', '').startswith('application/vnd.google-apps')
        and f['name'].lower().endswith(AUDIO_EXTENSIONS)
    ]


def download_file(service, file_id, local_path):
    """Download a single Drive file to local_path."""
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with open(local_path, 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()


def create_output_folder(service, parent_folder_id, folder_name):
    """Create a new folder in Google Drive (supports Shared Drives)."""
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_folder_id]
    }

    folder = service.files().create(
        body=file_metadata,
        fields='id',
        supportsAllDrives=True
    ).execute()

    return folder.get('id')


def _escape_drive_query_literal(value):
    """Escape a value for use inside a single-quoted Drive query literal."""
    return value.replace('\\', '\\\\').replace("'", "\\'")


def find_folder(service, parent_folder_id, folder_name):
    """Return the id of a non-trashed child folder by name, or None."""
    query = (
        f"'{parent_folder_id}' in parents and "
        f"name = '{_escape_drive_query_literal(folder_name)}' and "
        f"mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    results = service.files().list(
        q=query, fields="files(id, name)", pageSize=1,
        supportsAllDrives=True, includeItemsFromAllDrives=True
    ).execute().get('files', [])
    return results[0]['id'] if results else None


def find_or_create_folder(service, parent_folder_id, folder_name):
    """find_folder(), creating the folder if it isn't there yet."""
    existing = find_folder(service, parent_folder_id, folder_name)
    if existing:
        return existing
    return create_output_folder(service, parent_folder_id, folder_name)


def find_file(service, folder_id, file_name):
    """Return the id of a non-trashed file by exact name in a folder, or None."""
    query = (
        f"'{folder_id}' in parents and "
        f"name = '{_escape_drive_query_literal(file_name)}' and trashed = false"
    )
    results = service.files().list(
        q=query, fields="files(id, name)", pageSize=1,
        supportsAllDrives=True, includeItemsFromAllDrives=True
    ).execute().get('files', [])
    return results[0]['id'] if results else None


def upload_file(service, folder_id, local_path, name, mime_type='application/octet-stream'):
    """Upload one local file into a Drive folder under the given name."""
    media = MediaFileUpload(local_path, mimetype=mime_type, resumable=True)
    service.files().create(
        body={'name': name, 'parents': [folder_id]},
        media_body=media,
        fields='id',
        supportsAllDrives=True
    ).execute()


def upload_folder_contents(service, folder_id, local_folder, skip_names=None):
    """
    Upload every file directly inside local_folder to a Google Drive folder,
    skipping any filename in skip_names (e.g. embeddings.pkl).
    """
    skip_names = skip_names or set()
    uploaded_count = 0

    for filename in sorted(os.listdir(local_folder)):
        if filename in skip_names:
            continue
        local_path = os.path.join(local_folder, filename)

        if not os.path.isfile(local_path):
            continue

        print(f"    Uploading: {filename}")

        if filename.endswith('.wav'):
            mime_type = 'audio/wav'
        elif filename.endswith('.csv'):
            mime_type = 'text/csv'
        elif filename.endswith('.pkl'):
            mime_type = 'application/octet-stream'
        elif filename.endswith('.json'):
            mime_type = 'application/json'
        else:
            mime_type = 'application/octet-stream'

        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }

        media = MediaFileUpload(local_path, mimetype=mime_type, resumable=True)

        service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id',
            supportsAllDrives=True
        ).execute()

        uploaded_count += 1

    return uploaded_count


def convert_to_mono_first_channel(input_path, output_path):
    """
    ffmpeg: take only the first channel (not an averaged downmix) as a mono
    wav -- pan=mono|c0=c0 selects input channel 0 explicitly, unlike -ac 1
    which mixes every channel together.

    Deliberately does NOT resample: find does that itself, to whatever rate
    the model records as its training rate (training_metadata.json next to
    the model), so the target rate isn't duplicated as a constant here.
    """
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', input_path,
        '-af', 'pan=mono|c0=c0',
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mono conversion failed for {input_path}:\n{result.stderr}")


def run_find(wav_path, output_path, noise_components, noise_var_scale, hit_gap_seconds,
             require_full_match=False, model_path=MODEL_PATH):
    """
    Run motif_discovery.py find for one wav file against a model + the
    packaged noise recording, streaming its output live. cwd=APP_DIR because
    CONFIG['checkpoint_path'] (lib_phmm/config.py) is a path relative to cwd,
    not to motif_discovery.py's location.

    model_path defaults to the packaged L2 model; handle_extend() overrides
    it with a downloaded phmm_extended.pkl once one exists.

    require_full_match forwards to find's --require-full-match (see there
    and phmm.viterbi()'s docstring) -- disallows partial matches (entering
    or exiting a submodel's match-state chain at an interior state). Off
    by default, matching find's own default.
    """
    cmd = [
        'python', '-u', MOTIF_DISCOVERY_SCRIPT, 'find',
        model_path, wav_path, output_path,
        '--noise-wav', NOISE_WAV_PATH,
        '--noise-components', str(noise_components),
        '--noise-bayesian',
        '--noise-var-scale', str(noise_var_scale),
        '--hit-gap-seconds', str(hit_gap_seconds),
        '--verbose',
    ]
    if require_full_match:
        cmd.append('--require-full-match')
    process = subprocess.Popen(
        cmd, cwd=APP_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in process.stdout:
        print(f"    {line}", end='', flush=True)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"motif_discovery.py find failed with return code {process.returncode}")


def encounter_from_filename(stem):
    """
    "E1234_whatever" -> 1234 -- same convention as annotate.py's own key(),
    duplicated (not imported) rather than shared, since annotate.py runs as
    its own subprocess here, same as motif_discovery.py; handler.py never
    imports either script's internals.
    """
    return int(stem.split('_')[0][1:])


def run_annotate(output_path, pvl_path, encounter):
    """
    Best-effort: merge this recording's motif_hits.csv into the matching
    rows of the job's PVL spreadsheet via annotate.py, writing
    annotated_motif_hits.csv into output_path -- picked up automatically by
    the upload_folder_contents() call that follows in process_one_file().
    Never raises: an encounter that isn't in this PVL, or any other
    annotate.py failure, must not fail a file that find already succeeded
    on -- it just means no annotated_motif_hits.csv gets uploaded for it.
    """
    cmd = ['python', '-u', ANNOTATE_SCRIPT, output_path, pvl_path, '--encounter', str(encounter)]
    result = subprocess.run(cmd, cwd=APP_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        last_line = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        print(f"  (annotation skipped: {last_line})")
    else:
        print(f"  {result.stdout.strip()}")


ANNOTATED_CSV_NAME = "annotated_motif_hits.csv"
PVL_DB_NAME = "pvl.db"
PVL_CSV_NAME = "pvl_annotated_audio.csv"
HOTCODED_NAME = "hotcoded.out"
HOTCODING_META_NAME = "hotcoding_meta.json"


def build_and_upload_pvl_db(service, results, run_output_folder_id):
    """
    Best-effort, called once at the end of a find/extend job (see handler()/
    handle_extend()): pull every ANNOTATED_CSV_NAME this job actually
    produced (only files whose encounter matched a PVL row get one -- see
    run_annotate()), combine them via pvl_feature_extractor.py's own
    clean()/context_features()/actors_features()/add_path(), and upload the
    result as PVL_DB_NAME + PVL_CSV_NAME, plus HOTCODED_NAME +
    HOTCODING_META_NAME from that same module's hotcode_df(), into this
    run's own output folder. A job with no PVL configured (or where every
    file's annotation was skipped) simply has nothing to combine -- logged,
    not an error, since find/extend's own results are unaffected either way.

    add_path() only ever needs each recording's filename (never its audio
    content) for the audio_path column, so this reconstructs that mapping
    from each result's own already-known "name" instead of re-downloading
    every original recording just to satisfy pvl_feature_extractor.py's own
    glob-over-a-directory-of-audio-files CLI entry point.
    """
    candidates = [
        (Path(r["name"]).stem, r["name"], r["output_folder_id"])
        for r in results
        if r.get("status") == "success" and r.get("output_folder_id")
    ]
    if not candidates:
        return

    workdir = tempfile.mkdtemp(prefix='motif_pvl_db_')
    try:
        local_paths = []
        audio_map = {}
        for stem, name, output_folder_id in candidates:
            try:
                csv_id = find_file(service, output_folder_id, ANNOTATED_CSV_NAME)
                if not csv_id:
                    continue
                dest_dir = os.path.join(workdir, stem)
                os.makedirs(dest_dir, exist_ok=True)
                dest = os.path.join(dest_dir, ANNOTATED_CSV_NAME)
                download_file(service, csv_id, dest)
                local_paths.append(dest)
                audio_map[stem] = name
            except Exception as e:
                print(f"  (pvl db: skipping {name}, couldn't fetch {ANNOTATED_CSV_NAME}: {e})")

        if not local_paths:
            print(f"\nNo {ANNOTATED_CSV_NAME} files produced this run -- skipping combined pvl db/csv")
            return

        print(f"\nBuilding combined pvl db/csv from {len(local_paths)} {ANNOTATED_CSV_NAME} file(s)")
        dataframes = [pvl_feature_extractor.add_path(pd.read_csv(f), f, audio_map) for f in local_paths]
        df = pvl_feature_extractor.clean(pd.concat(dataframes).reset_index())
        df = pvl_feature_extractor.context_features(df)
        df = pvl_feature_extractor.actors_features(df)

        db_local = os.path.join(workdir, PVL_DB_NAME)
        csv_local = os.path.join(workdir, PVL_CSV_NAME)
        conn = sqlite3.connect(db_local)
        df.to_sql("pvl", conn, if_exists="replace", index=False)
        conn.close()
        df.to_csv(csv_local, index=False)

        print(f"Uploading {PVL_DB_NAME} and {PVL_CSV_NAME} to this run's output folder")
        upload_file(service, run_output_folder_id, db_local, PVL_DB_NAME)
        upload_file(service, run_output_folder_id, csv_local, PVL_CSV_NAME, mime_type='text/csv')

        try:
            hotcoded, hotcoding_meta = pvl_feature_extractor.hotcode_df(df)
            hotcoded_local = os.path.join(workdir, HOTCODED_NAME)
            hotcoding_meta_local = os.path.join(workdir, HOTCODING_META_NAME)
            np.savetxt(hotcoded_local, hotcoded)
            with open(hotcoding_meta_local, 'w') as f:
                json.dump(hotcoding_meta, f)

            print(f"Uploading {HOTCODED_NAME} and {HOTCODING_META_NAME} to this run's output folder")
            upload_file(service, run_output_folder_id, hotcoded_local, HOTCODED_NAME)
            upload_file(service, run_output_folder_id, hotcoding_meta_local, HOTCODING_META_NAME,
                        mime_type='application/json')
        except Exception as e:
            import traceback
            print(f"WARNING: building/uploading hotcoded features failed "
                  f"(pvl db/csv upload is unaffected): {e}\n{traceback.format_exc()}")
    except Exception as e:
        import traceback
        print(f"WARNING: building/uploading combined pvl db/csv failed "
              f"(job results are unaffected): {e}\n{traceback.format_exc()}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _as_folder_id_list(value):
    """gdrive_folder_id accepts either one folder id or a list of them (extend mode only, see handle_extend())."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _as_pvl_id_list(value, n):
    """
    pvl_file_id (extend mode only) accepts: omitted (no PVL for any
    folder), one id (the same PVL shared across every gdrive_folder_id --
    e.g. one shotlog covering several encounter folders from the same
    year), or a list matched positionally to gdrive_folder_id (one PVL --
    or null for "none" -- per folder, for folders spanning different
    datasets/years). Always returns a list of length n; raises ValueError
    on a length mismatch rather than guessing an alignment.
    """
    if value is None:
        return [None] * n
    if isinstance(value, str):
        return [value] * n
    pvl_ids = list(value)
    if len(pvl_ids) != n:
        raise ValueError(f"pvl_file_id has {len(pvl_ids)} entries but gdrive_folder_id has {n} -- "
                          f"give one pvl_file_id per folder (or a single id to share across all)")
    return pvl_ids


def read_training_sample_rate(metadata_path):
    """
    Same convention as motif_discovery.py's load_training_sample_rate(),
    read directly here since handler.py never imports either script's
    internals (both only ever run as subprocesses -- see run_annotate()'s
    docstring for why).
    """
    with open(metadata_path) as f:
        return int(json.load(f)['sample_rate'])


def _stream_subprocess(cmd, step_name):
    """Shared plumbing for the extend-pipeline subprocess wrappers below --
    same streaming/error-checking shape as run_find()."""
    process = subprocess.Popen(
        cmd, cwd=APP_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in process.stdout:
        print(f"    {line}", end='', flush=True)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"motif_discovery.py {step_name} failed with return code {process.returncode}")


def run_embed(wav_path, output_path, resample_hz):
    """motif_discovery.py embed for one file -- find's embedding step with no model needed yet."""
    _stream_subprocess([
        'python', '-u', MOTIF_DISCOVERY_SCRIPT, 'embed',
        wav_path, output_path,
        '--resample-hz', str(resample_hz),
        '--verbose',
    ], 'embed')


def run_extend_candidates(baked_candidates_path, embeddings_dir, output_candidates_path, min_region_frames):
    """motif_discovery.py extend-candidates -- combine the baked L2 candidates with new ones mined from embeddings_dir."""
    _stream_subprocess([
        'python', '-u', MOTIF_DISCOVERY_SCRIPT, 'extend-candidates',
        baked_candidates_path, embeddings_dir, output_candidates_path,
        '--min-region-frames', str(min_region_frames),
    ], 'extend-candidates')


def run_train_candidates(candidates_path, output_path, sample_rate, all_medoids=True,
                          dtw_restarts=None, dtw_density_tolerance=None, min_match_states=None,
                          dtw_no_path_norm=False):
    """
    motif_discovery.py train-candidates -- fit a new model directly from a
    candidates.pkl. all_medoids defaults on here (unlike train-candidates'
    own CLI default): extend's whole point is a pool that keeps growing
    every training cycle (baked L2 candidates + everything mined since), so
    the greedy O(max_match * candidates^2) sub-model search train-candidates
    otherwise runs is exactly the cost that scales worst as that pool grows
    -- --all-medoids instead makes every DTW-filtered candidate its own
    sub-model directly, BIC-picking only n_match_states.

    dtw_restarts/dtw_density_tolerance forward to train-candidates'
    --dtw-restarts/--dtw-density-tolerance (see hierarchical_kmedian_dtw.hpp's
    KMEDOIDS_RESTARTS_DEFAULT/DENSITY_TOLERANCE_DEFAULT for what they
    control) -- None (the default) omits the flag entirely so
    motif_discovery.py's own CLI default applies, unchanged from before
    these existed.

    min_match_states forwards to train-candidates' --min-match-states (see
    motif_discovery.py's sweep()/sweep_all_medoids() docstrings): raises the
    floor of the BIC n_match_states sweep. Matters especially with
    all_medoids=True, since count_params()'s BIC penalty scales with
    n_sub_models * n_match_states -- a large all-medoids sub-model pool can
    make BIC floor n_match_states at the sweep's minimum regardless of how
    many segments the true motif shape actually needs. None (the default)
    omits the flag, same as the others above.
    """
    cmd = [
        'python', '-u', MOTIF_DISCOVERY_SCRIPT, 'train-candidates',
        candidates_path, output_path,
        '--sample-rate', str(sample_rate),
        '--model-name', EXTEND_MODEL_NAME,
        '--verbose',
    ]
    if all_medoids:
        cmd.append('--all-medoids')
    if dtw_restarts is not None:
        cmd += ['--dtw-restarts', str(dtw_restarts)]
    if dtw_density_tolerance is not None:
        cmd += ['--dtw-density-tolerance', str(dtw_density_tolerance)]
    if min_match_states is not None:
        cmd += ['--min-match-states', str(min_match_states)]
    if dtw_no_path_norm:
        cmd.append('--dtw-no-path-norm')
    _stream_subprocess(cmd, 'train-candidates')


def count_hits(motif_hits_csv):
    """Row count of motif_hits.csv minus its header, or 0 if it's missing/empty."""
    if not os.path.exists(motif_hits_csv):
        return 0
    with open(motif_hits_csv) as f:
        return max(0, sum(1 for _ in f) - 1)


def restore_noise_embeddings(output_path, service=None, embeddings_cache_folder_id=None):
    """
    Seed output_path/noise_embeddings.pkl so build_noise_model() loads it
    instead of re-embedding the packaged, never-changing noise wav from
    scratch. Two layers, cheapest first:
      1. The job-local NOISE_EMBEDDINGS_CACHE_PATH, if an earlier file in
         this same job already produced one.
      2. The persistent Drive cache (embeddings_cache_folder_id, the same
         motif_embeddings_cache folder find's own per-recording embeddings
         use), if a *prior job* already cached one there -- this is what
         makes the very first file of a cold container skip re-embedding
         too. Found-on-Drive also re-seeds the local cache, so later files
         in this job hit layer 1 instead of Drive again.

    service/embeddings_cache_folder_id are optional -- omit either to skip
    straight to a local-only lookup (e.g. a caller with no Drive context).

    Best-effort throughout: this is purely a speedup, so a broken/
    unwritable/unreachable cache must never take down the actual run --
    worst case find re-embeds the noise.
    """
    try:
        if os.path.exists(NOISE_EMBEDDINGS_CACHE_PATH):
            shutil.copy2(NOISE_EMBEDDINGS_CACHE_PATH, os.path.join(output_path, 'noise_embeddings.pkl'))
            print("  Reusing noise embeddings from this job's earlier file")
            return
    except OSError as e:
        print(f"  (local noise embeddings cache unreadable, trying Drive: {e})")

    if service is None or embeddings_cache_folder_id is None:
        return
    try:
        cached_id = find_file(service, embeddings_cache_folder_id, NOISE_EMBEDDINGS_CACHE_NAME)
        if not cached_id:
            return
        print(f"  Using cached noise embeddings from Drive: {NOISE_EMBEDDINGS_CACHE_NAME}")
        dest = os.path.join(output_path, 'noise_embeddings.pkl')
        download_file(service, cached_id, dest)
        os.makedirs(os.path.dirname(NOISE_EMBEDDINGS_CACHE_PATH), exist_ok=True)
        shutil.copy2(dest, NOISE_EMBEDDINGS_CACHE_PATH)
    except Exception as e:
        print(f"  (Drive noise embeddings cache fetch failed, re-embedding: {e})")


def save_noise_embeddings(output_path, service=None, embeddings_cache_folder_id=None):
    """
    Stash this run's noise_embeddings.pkl (produced because restore_noise_
    embeddings() found nothing cached anywhere) both job-locally, for this
    job's remaining files, and to the persistent Drive cache, so the next
    job/cold container skips re-embedding the noise wav too. Best-effort,
    for the same reason as restore_noise_embeddings(); service/
    embeddings_cache_folder_id optional the same way.
    """
    produced = os.path.join(output_path, 'noise_embeddings.pkl')
    if not os.path.exists(produced):
        return

    if not os.path.exists(NOISE_EMBEDDINGS_CACHE_PATH):
        try:
            os.makedirs(os.path.dirname(NOISE_EMBEDDINGS_CACHE_PATH), exist_ok=True)
            shutil.copy2(produced, NOISE_EMBEDDINGS_CACHE_PATH)
        except OSError as e:
            print(f"  (could not stash noise embeddings locally, next file will re-embed: {e})")

    if service is None or embeddings_cache_folder_id is None:
        return
    try:
        if not find_file(service, embeddings_cache_folder_id, NOISE_EMBEDDINGS_CACHE_NAME):
            print(f"  Caching noise embeddings to Drive: {NOISE_EMBEDDINGS_CACHE_NAME}")
            upload_file(service, embeddings_cache_folder_id, produced, NOISE_EMBEDDINGS_CACHE_NAME)
    except Exception as e:
        print(f"  (caching noise embeddings to Drive failed, will re-embed next job: {e})")


def process_one_file(service, file_info, run_output_folder_id, noise_components, noise_var_scale,
                      hit_gap_seconds, embeddings_cache_folder_id=None, require_full_match=False,
                      pvl_path=None):
    """
    Download -> convert to mono -> find -> annotate -> upload -> delete
    local files, for a single Drive file. Returns a result dict; never
    raises (errors are captured in the "status"/"error" fields) so one bad
    file doesn't abort the rest of the folder.

    When embeddings_cache_folder_id is given, <stem>.pkl there is pulled in
    as this run's embeddings.pkl before find (skipping the Whisper decode
    entirely) and written back afterwards if it wasn't cached yet. The
    embedding depends only on the audio, not on any --noise-* setting, so a
    cache entry stays valid across runs that sweep those parameters.

    When pvl_path is given (the job's PVL spreadsheet, already downloaded
    once for the whole job -- see handler()), run_annotate() merges this
    file's motif_hits.csv into it, keyed by an encounter number read from
    this recording's own filename (see encounter_from_filename()) --
    best-effort, see run_annotate()'s docstring.
    """
    name = file_info['name']
    stem = Path(name).stem
    cache_name = f"{stem}.{EMBEDDINGS_CACHE_VERSION}.pkl"
    work_dir = tempfile.mkdtemp(prefix='motif_find_')
    try:
        output_path = os.path.join(work_dir, 'output')
        os.makedirs(output_path, exist_ok=True)

        embeddings_path = os.path.join(output_path, 'embeddings.pkl')
        embeddings_cached = False
        if embeddings_cache_folder_id:
            # Best-effort: a cache miss, a failed lookup, or a download that
            # dies partway must fall back to a fresh decode, never fail the
            # file -- and a half-written pickle has to be cleared out, or
            # find would try to unpickle a truncated file.
            try:
                cached_id = find_file(service, embeddings_cache_folder_id, cache_name)
                if cached_id:
                    print(f"  Using cached embeddings from Drive: {cache_name}")
                    download_file(service, cached_id, embeddings_path)
                    embeddings_cached = True
            except Exception as e:
                print(f"  (embeddings cache fetch failed, re-embedding: {e})")
                if os.path.exists(embeddings_path):
                    os.remove(embeddings_path)
                embeddings_cached = False

        # A cached embedding makes the source audio unnecessary -- find only
        # needs the wav to cut submodel_<n>_hits.wav clips from, which it
        # still does, so the download/convert can't be skipped outright.
        raw_path = os.path.join(work_dir, name)
        print(f"  Downloading: {name}")
        download_file(service, file_info['id'], raw_path)

        mono_path = os.path.join(work_dir, 'mono.wav')
        print(f"  Converting to mono (first channel): {name}")
        convert_to_mono_first_channel(raw_path, mono_path)

        restore_noise_embeddings(output_path, service, embeddings_cache_folder_id)

        print(f"  Running find: {name}")
        run_find(mono_path, output_path, noise_components, noise_var_scale, hit_gap_seconds,
                 require_full_match=require_full_match)

        save_noise_embeddings(output_path, service, embeddings_cache_folder_id)

        hits = count_hits(os.path.join(output_path, 'motif_hits.csv'))

        if pvl_path:
            try:
                encounter = encounter_from_filename(stem)
            except (ValueError, IndexError) as e:
                print(f"  (annotation skipped: couldn't read an encounter number from '{name}' "
                      f"-- expected \"E<number>_...\": {e})")
            else:
                run_annotate(output_path, pvl_path, encounter)

        embeddings_uploaded = False
        if embeddings_cache_folder_id and not embeddings_cached and os.path.exists(embeddings_path):
            # Also best-effort: the decode already succeeded and the results
            # are about to upload, so failing to populate the cache costs a
            # re-decode next time but must not fail this file.
            try:
                print(f"  Caching embeddings to Drive: {cache_name}")
                upload_file(service, embeddings_cache_folder_id, embeddings_path, cache_name)
                embeddings_uploaded = True
            except Exception as e:
                print(f"  (caching embeddings failed, will re-embed next run: {e})")

        file_output_folder_id = create_output_folder(service, run_output_folder_id, stem)
        print(f"  Uploading results: {name}")
        uploaded = upload_folder_contents(service, file_output_folder_id, output_path, skip_names=UPLOAD_SKIP_NAMES)

        return {
            "name": name,
            "status": "success",
            "hits": hits,
            "uploaded": uploaded,
            "output_folder_id": file_output_folder_id,
            "embeddings_from_cache": embeddings_cached,
            "embeddings_cached": embeddings_uploaded,
        }
    except Exception as e:
        import traceback
        print(f"  ERROR processing {name}: {e}\n{traceback.format_exc()}")
        return {"name": name, "status": "error", "error": str(e)}
    finally:
        # "after each processed file delete all files for this model
        # locally" -- downloaded raw audio, the mono conversion, and every
        # find output (embeddings.pkl, motif_hits.csv, submodel_*_hits.wav)
        # for this one recording, so disk use stays bounded across however
        # many recordings are in the folder.
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"  Cleaned up local files for: {name}")


def handle_extend(job_input):
    """
    Runs every remaining stage of the L2 extend pipeline in one job
    submission -- mines new candidate material, (re)trains a model from the
    combined pool, then runs `find` with it and annotates against
    pvl_file_id -- so a single call takes the packaged model all the way to
    fresh find results, with no need to resubmit between stages. Each stage
    is skipped if the extend state folder shows it's already done, so a job
    that times out mid-pipeline resumes from wherever it left off on the
    next submission instead of redoing finished work:

      A. No candidates.pkl in the extend state folder yet: embed every
         recording in gdrive_folder_id (Drive-cached exactly like `find`'s
         embeddings cache, so a resumed call never re-decodes a file),
         extract non-NOISE regions from those embeddings
         (motif_discovery.py's extract_non_noise_candidates(), via the
         extend-candidates subcommand), combine them with the packaged
         /app/motif/l2/candidates.pkl, and upload the result as this job
         family's candidates.pkl. Also uploads this cycle's new regions as
         l2_<timestamp>.csv/l2_<timestamp>.wav -- one starts,stops-into-
         one-wav pair in the same format train's own csv_path/wav_path
         input uses, so what got mined this cycle stays auditable/reusable
         on its own, not just buried as embeddings inside candidates.pkl.
         Accumulates across cycles (never overwritten, unlike
         candidates.pkl/phmm_extended.pkl).
      B. Candidates exist but no phmm_extended.pkl yet: fit a new profile
         HMM from that combined pool (train-candidates -- same DTW-filter +
         BIC-sweep discovery path `train` uses) and upload it.
      C. Run `find` with the current model (whatever stage B just
         produced, or the one already on Drive if B was skipped this call)
         against every recording, annotate against pvl_file_id if given,
         upload results.

    Every recording is embedded once up front, before any of A/B/C run --
    stage A needs that audio to mine candidates from, and stage C (which
    this call always reaches) needs it to run find, so there's no case
    left where skipping the embed loop saves anything the way it used to
    when a call could stop at stage B alone.

    job_input keys (all beyond gdrive_folder_id are optional):
        gdrive_folder_id      -- one folder id or a list of them; every file
                                 across all of them feeds the same shared
                                 candidates pool/model
        output_gdrive_folder_id  -- same meaning as find's
        min_region_frames    -- default 10, see extend-candidates' own flag
        all_medoids           -- default True, see run_train_candidates()
        dtw_restarts, dtw_density_tolerance, min_match_states -- optional,
                                 forwarded to train-candidates'
                                 --dtw-restarts/--dtw-density-tolerance/
                                 --min-match-states (omitted, using
                                 motif_discovery.py's own CLI defaults, when
                                 not given); see run_train_candidates()
        noise_components, noise_var_scale, hit_gap_seconds, require_full_match
                              -- same meaning as find's, used only at the find stage
        pvl_file_id           -- one id shared by every folder, or a list matched
                                 positionally to gdrive_folder_id (null for a
                                 folder with no PVL); see _as_pvl_id_list(). Also
                                 only matters at the find stage.

    Returns a dict with "stage": "find" (this always finishes there now)
    and a "stages_completed" list saying which of candidates_built /
    model_trained / find actually ran in this call -- a stage already done
    on a prior submission is skipped, not repeated, so this list shrinks
    the closer a job family already was to done before this call started.
    """
    gdrive_folder_ids = _as_folder_id_list(job_input.get('gdrive_folder_id'))
    output_gdrive_folder_id = job_input.get('output_gdrive_folder_id')
    if not gdrive_folder_ids:
        return {"error": "gdrive_folder_id is required"}

    min_region_frames = int(job_input.get('min_region_frames', 10))
    noise_components = int(job_input.get('noise_components', 6))
    noise_var_scale = float(job_input.get('noise_var_scale', 1.0))
    hit_gap_seconds = float(job_input.get('hit_gap_seconds', 0.5))
    require_full_match = bool(job_input.get('require_full_match', False))
    # See run_train_candidates()'s docstring for why this defaults to True
    # here specifically (unlike train-candidates' own CLI default).
    all_medoids = bool(job_input.get('all_medoids', True))
    dtw_restarts = job_input.get('dtw_restarts')
    dtw_restarts = int(dtw_restarts) if dtw_restarts is not None else None
    dtw_density_tolerance = job_input.get('dtw_density_tolerance')
    dtw_density_tolerance = float(dtw_density_tolerance) if dtw_density_tolerance is not None else None
    min_match_states = job_input.get('min_match_states')
    min_match_states = int(min_match_states) if min_match_states is not None else None
    dtw_no_path_norm = bool(job_input.get('dtw_no_path_norm', False))
    try:
        pvl_ids_by_folder = _as_pvl_id_list(job_input.get('pvl_file_id'), len(gdrive_folder_ids))
    except ValueError as e:
        return {"error": str(e)}
    folder_to_pvl_id = dict(zip(gdrive_folder_ids, pvl_ids_by_folder))

    service = get_drive_service()
    # output_gdrive_folder_id, or the first input folder if not given -- same
    # fallback find's handler() uses, just against a list here since multiple
    # source folders all feed one shared candidates pool/model.
    parent_folder_for_output = output_gdrive_folder_id or gdrive_folder_ids[0]

    print(f"\nListing WAV/M4A files in {len(gdrive_folder_ids)} Google Drive folder(s)...")
    files = []
    for folder_id in gdrive_folder_ids:
        folder_files = list_audio_files(service, folder_id)
        print(f"  {folder_id}: {len(folder_files)} audio files "
              f"(pvl_file_id={folder_to_pvl_id[folder_id] or 'none'})")
        for f in folder_files:
            f['_source_folder_id'] = folder_id  # carried through to embedded[]'s pvl_file_id below
        files.extend(folder_files)
    if not files:
        return {"error": "No WAV/M4A files found in the specified Google Drive folder(s)"}
    print(f"Found {len(files)} audio files total across {len(gdrive_folder_ids)} folder(s)")

    state_folder_id = find_or_create_folder(service, parent_folder_for_output, EXTEND_STATE_FOLDER_NAME)
    embeddings_cache_folder_id = find_or_create_folder(service, parent_folder_for_output, EMBEDDINGS_CACHE_FOLDER_NAME)

    training_sample_rate = read_training_sample_rate(TRAINING_METADATA_PATH)
    print(f"Baked model's training rate: {training_sample_rate} Hz (embedding new audio at this rate to match)")

    candidates_id = find_file(service, state_folder_id, EXTEND_CANDIDATES_NAME)
    model_id = find_file(service, state_folder_id, EXTEND_MODEL_NAME) if candidates_id else None
    stages_completed = []

    # All three stages funnel through here now -- stage A needs every
    # recording's audio to mine candidates from, and stage C (which this
    # call always reaches) needs it to run find, so embed everything up
    # front regardless of which of A/B/C actually run below. Drive-cached
    # exactly like `find`'s own embeddings cache, so a call that's only
    # here for stage B/C because stage A already ran on a prior submission
    # doesn't pay to re-decode anything.
    work_dirs = []
    embedded = []
    try:
        for i, file_info in enumerate(files):
            name = file_info['name']
            stem = Path(name).stem
            print(f"\n[{i + 1}/{len(files)}] embedding {name}")
            cache_name = f"{stem}.{EMBEDDINGS_CACHE_VERSION}.pkl"
            work_dir = tempfile.mkdtemp(prefix='motif_extend_')
            work_dirs.append(work_dir)
            output_path = os.path.join(work_dir, 'output')
            os.makedirs(output_path, exist_ok=True)
            embeddings_path = os.path.join(output_path, 'embeddings.pkl')

            embeddings_cached = False
            try:
                cached_id = find_file(service, embeddings_cache_folder_id, cache_name)
                if cached_id:
                    print(f"  Using cached embeddings from Drive: {cache_name}")
                    download_file(service, cached_id, embeddings_path)
                    embeddings_cached = True
            except Exception as e:
                print(f"  (embeddings cache fetch failed, re-embedding: {e})")
                if os.path.exists(embeddings_path):
                    os.remove(embeddings_path)
                embeddings_cached = False

            raw_path = os.path.join(work_dir, name)
            mono_path = os.path.join(work_dir, 'mono.wav')
            print(f"  Downloading: {name}")
            download_file(service, file_info['id'], raw_path)
            print(f"  Converting to mono (first channel): {name}")
            convert_to_mono_first_channel(raw_path, mono_path)

            if not embeddings_cached:
                print(f"  Embedding: {name}")
                run_embed(mono_path, output_path, training_sample_rate)
                if os.path.exists(embeddings_path):
                    try:
                        print(f"  Caching embeddings to Drive: {cache_name}")
                        upload_file(service, embeddings_cache_folder_id, embeddings_path, cache_name)
                    except Exception as e:
                        print(f"  (caching embeddings failed, will re-embed next run: {e})")

            embedded.append({
                "file_info": file_info, "stem": stem, "work_dir": work_dir,
                "output_path": output_path, "embeddings_path": embeddings_path, "mono_path": mono_path,
                "pvl_file_id": folder_to_pvl_id.get(file_info.get('_source_folder_id')),
            })

        # ---- Stage: no combined candidates.pkl yet -> build one ----
        if not candidates_id:
            print(f"\nNo {EXTEND_CANDIDATES_NAME} in extend state folder yet -- building one")
            embroot = tempfile.mkdtemp(prefix='motif_extend_embroot_')
            try:
                for e in embedded:
                    dest = os.path.join(embroot, e["stem"])
                    os.makedirs(dest, exist_ok=True)
                    shutil.copy2(e["embeddings_path"], os.path.join(dest, "embeddings.pkl"))
                    # extend-candidates slices this file's new regions'
                    # actual audio out of here (when present) to also write
                    # l2_<timestamp>.csv/.wav -- see its docstring.
                    shutil.copy2(e["mono_path"], os.path.join(dest, "audio.wav"))

                baked_local = os.path.join(embroot, "baked_candidates.pkl")
                shutil.copy2(BAKED_CANDIDATES_PATH, baked_local)
                combined_local = os.path.join(embroot, EXTEND_CANDIDATES_NAME)
                run_extend_candidates(baked_local, embroot, combined_local, min_region_frames)

                print(f"Uploading combined candidates to extend state folder: {EXTEND_CANDIDATES_NAME}")
                upload_file(service, state_folder_id, combined_local, EXTEND_CANDIDATES_NAME)

                # This cycle's freshly-mined regions, in the same
                # starts,stops-into-one-wav format train's own csv_path/
                # wav_path input uses -- uploaded alongside (not
                # overwriting) whatever earlier cycles already wrote, so
                # they accumulate as a dated history rather than replacing
                # each other like candidates.pkl/phmm_extended.pkl do.
                new_candidate_csvs = sorted(Path(embroot).glob("l2_*.csv"))
                new_candidate_wavs = sorted(Path(embroot).glob("l2_*.wav"))
                for csv_file in new_candidate_csvs:
                    print(f"Uploading new-candidates CSV to extend state folder: {csv_file.name}")
                    upload_file(service, state_folder_id, str(csv_file), csv_file.name, mime_type='text/csv')
                for wav_file in new_candidate_wavs:
                    print(f"Uploading new-candidates wav to extend state folder: {wav_file.name}")
                    upload_file(service, state_folder_id, str(wav_file), wav_file.name, mime_type='audio/wav')
            finally:
                shutil.rmtree(embroot, ignore_errors=True)

            candidates_id = find_file(service, state_folder_id, EXTEND_CANDIDATES_NAME)
            stages_completed.append("candidates_built")

        # ---- Stage: candidates exist but no trained model yet -> train ----
        if not model_id:
            print(f"\n{EXTEND_CANDIDATES_NAME} exists but no {EXTEND_MODEL_NAME} yet -- training a new model")
            train_work_dir = tempfile.mkdtemp(prefix='motif_extend_train_')
            try:
                candidates_local = os.path.join(train_work_dir, EXTEND_CANDIDATES_NAME)
                download_file(service, candidates_id, candidates_local)
                train_output = os.path.join(train_work_dir, 'output')
                os.makedirs(train_output, exist_ok=True)
                run_train_candidates(candidates_local, train_output, training_sample_rate, all_medoids=all_medoids,
                                      dtw_restarts=dtw_restarts, dtw_density_tolerance=dtw_density_tolerance,
                                      min_match_states=min_match_states, dtw_no_path_norm=dtw_no_path_norm)

                model_local = os.path.join(train_output, EXTEND_MODEL_NAME)
                metadata_local = os.path.join(train_output, EXTEND_TRAINING_METADATA_NAME)
                print(f"Uploading trained model to extend state folder: {EXTEND_MODEL_NAME}")
                upload_file(service, state_folder_id, model_local, EXTEND_MODEL_NAME)
                if os.path.exists(metadata_local):
                    upload_file(service, state_folder_id, metadata_local, EXTEND_TRAINING_METADATA_NAME)
                metrics_local = os.path.join(train_output, 'metrics.csv')
                if os.path.exists(metrics_local):
                    upload_file(service, state_folder_id, metrics_local, 'metrics.csv')
            finally:
                shutil.rmtree(train_work_dir, ignore_errors=True)

            model_id = find_file(service, state_folder_id, EXTEND_MODEL_NAME)
            stages_completed.append("model_trained")

        # ---- Stage: run find against every recording + annotate ----
        print(f"\nRunning find against all recordings with the current {EXTEND_MODEL_NAME}")
        model_work_dir = tempfile.mkdtemp(prefix='motif_extend_model_')
        pvl_work_dir = None
        try:
            model_local = os.path.join(model_work_dir, EXTEND_MODEL_NAME)
            download_file(service, model_id, model_local)
            metadata_id = find_file(service, state_folder_id, EXTEND_TRAINING_METADATA_NAME)
            if metadata_id:
                download_file(service, metadata_id, os.path.join(model_work_dir, EXTEND_TRAINING_METADATA_NAME))

            # Distinct PVL ids only, downloaded once each -- multiple
            # folders may share the same PVL (or none at all). Keyed by
            # pvl_file_id so each file below can look up the one that
            # matches its own source folder (see folder_to_pvl_id above).
            distinct_pvl_ids = sorted({pid for pid in pvl_ids_by_folder if pid})
            pvl_paths_by_id = {}
            if distinct_pvl_ids:
                pvl_work_dir = tempfile.mkdtemp(prefix='motif_extend_pvl_')
                for idx, pid in enumerate(distinct_pvl_ids):
                    local_path = os.path.join(pvl_work_dir, f'pvl_{idx}.xlsx')
                    print(f"Downloading PVL file: {pid}")
                    try:
                        download_file(service, pid, local_path)
                        pvl_paths_by_id[pid] = local_path
                    except Exception as e:
                        print(f"WARNING: failed to download PVL file {pid} ({e}) -- "
                              f"skipping annotation for its folder(s)")

            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            output_folder_name = f"motif_extend_find_{timestamp}"
            run_output_folder_id = create_output_folder(service, parent_folder_for_output, output_folder_name)

            results = []
            for i, e in enumerate(embedded):
                name = e["file_info"]["name"]
                stem = e["stem"]
                print(f"\n[{i + 1}/{len(embedded)}] {name}")
                try:
                    restore_noise_embeddings(e["output_path"], service, embeddings_cache_folder_id)

                    print(f"  Running find: {name}")
                    run_find(e["mono_path"], e["output_path"], noise_components, noise_var_scale,
                             hit_gap_seconds, require_full_match=require_full_match, model_path=model_local)

                    save_noise_embeddings(e["output_path"], service, embeddings_cache_folder_id)

                    hits = count_hits(os.path.join(e["output_path"], "motif_hits.csv"))
                    pvl_path = pvl_paths_by_id.get(e["pvl_file_id"])
                    if pvl_path:
                        try:
                            encounter = encounter_from_filename(stem)
                        except (ValueError, IndexError) as err:
                            print(f"  (annotation skipped: couldn't read an encounter number from '{name}' "
                                  f"-- expected \"E<number>_...\": {err})")
                        else:
                            run_annotate(e["output_path"], pvl_path, encounter)
                    file_output_folder_id = create_output_folder(service, run_output_folder_id, stem)
                    uploaded = upload_folder_contents(service, file_output_folder_id, e["output_path"],
                                                       skip_names=UPLOAD_SKIP_NAMES)
                    results.append({"name": name, "status": "success", "hits": hits, "uploaded": uploaded,
                                     "output_folder_id": file_output_folder_id})
                except Exception as err:
                    import traceback
                    print(f"  ERROR processing {name}: {err}\n{traceback.format_exc()}")
                    results.append({"name": name, "status": "error", "error": str(err)})

            stages_completed.append("find")
            build_and_upload_pvl_db(service, results, run_output_folder_id)
            return {
                "status": "success",
                "stage": "find",
                "stages_completed": stages_completed,
                "output_folder_id": run_output_folder_id,
                "output_folder_name": output_folder_name,
                "files_found": len(files),
                "files_processed": sum(1 for r in results if r["status"] == "success"),
                "files_failed": sum(1 for r in results if r["status"] == "error"),
                "results": results,
            }
        finally:
            shutil.rmtree(model_work_dir, ignore_errors=True)
            if pvl_work_dir:
                shutil.rmtree(pvl_work_dir, ignore_errors=True)
    finally:
        for wd in work_dirs:
            shutil.rmtree(wd, ignore_errors=True)


def handler(job):
    """RunPod serverless handler function."""
    job_input = job.get('input', {})

    # "extend" is a separate, staged pipeline (see handle_extend()'s
    # docstring) -- everything below this dispatch is the original `find`
    # job, unchanged, and stays the default when mode is omitted.
    mode = job_input.get('mode', 'find')
    if mode == 'extend':
        try:
            return handle_extend(job_input)
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(f"ERROR: {error_msg}")
            return {"error": error_msg}

    gdrive_folder_id = job_input.get('gdrive_folder_id')
    output_gdrive_folder_id = job_input.get('output_gdrive_folder_id')
    noise_components = job_input.get('noise_components', 6)
    # 1.0, NOT >1: every emission Gaussian is 128-dim, so scaling the noise
    # model's variances by s shifts its log-likelihood by -D/2*log2(s) = -64
    # bits at s=2 -- a flat handicap that makes the match states win every
    # frame regardless of the audio, and collapses mean_llr_score into a
    # meaningless constant band. At 1.0 both sides sit at the same var_floor
    # and the ratio actually discriminates, so filter hits on
    # mean_llr_score instead of suppressing them with this.
    noise_var_scale = job_input.get('noise_var_scale', 1.0)
    hit_gap_seconds = job_input.get('hit_gap_seconds', 0.5)
    use_embeddings_cache = job_input.get('use_embeddings_cache', True)
    embeddings_cache_folder_id = job_input.get('embeddings_cache_folder_id')
    # See motif_discovery.py's --require-full-match / phmm.viterbi()'s
    # docstring: disallows partial matches (entering/exiting a submodel's
    # match-state chain at an interior state). Off by default, matching
    # find's own default.
    require_full_match = bool(job_input.get('require_full_match', False))
    # Optional: a Drive file id for a PVL .xlsx shotlog. If given, every
    # file's motif_hits.csv gets merged into it via annotate.py (see
    # process_one_file()/run_annotate()); if not, annotation is simply
    # skipped -- find itself is unaffected either way.
    pvl_file_id = job_input.get('pvl_file_id')

    if not gdrive_folder_id:
        return {"error": "gdrive_folder_id is required"}

    try:
        noise_components = int(noise_components)
        noise_var_scale = float(noise_var_scale)
        hit_gap_seconds = float(hit_gap_seconds)
    except (TypeError, ValueError):
        return {"error": "noise_components must be an int, noise_var_scale/hit_gap_seconds must be numeric"}

    print(f"Input Google Drive folder: {gdrive_folder_id}")
    print(f"Output Google Drive folder: {output_gdrive_folder_id or '(same as input)'}")
    print(f"noise_components={noise_components} noise_var_scale={noise_var_scale} "
          f"hit_gap_seconds={hit_gap_seconds} require_full_match={require_full_match}")

    try:
        print("\nInitializing Google Drive connection...")
        service = get_drive_service()

        print("\nListing WAV/M4A files in Google Drive folder...")
        files = list_audio_files(service, gdrive_folder_id)
        if not files:
            return {"error": "No WAV/M4A files found in the specified Google Drive folder"}
        print(f"Found {len(files)} audio files")

        parent_folder_for_output = output_gdrive_folder_id or gdrive_folder_id
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_folder_name = f"motif_find_{timestamp}"
        print(f"\nCreating output folder in Google Drive: {output_folder_name}")
        run_output_folder_id = create_output_folder(service, parent_folder_for_output, output_folder_name)

        # Deliberately NOT under run_output_folder_id: the cache has to
        # outlive this run's timestamped folder for a later run to find it.
        cache_folder_id = None
        if use_embeddings_cache:
            cache_folder_id = embeddings_cache_folder_id or find_or_create_folder(
                service, parent_folder_for_output, EMBEDDINGS_CACHE_FOLDER_NAME
            )
            print(f"Embeddings cache folder: {cache_folder_id}")

        # Downloaded once per job (not per file, unlike the embeddings
        # cache): one PVL covers many encounters/files, and re-downloading
        # it per file would be pure waste. Lives in its own tempdir (not a
        # fixed path like NOISE_EMBEDDINGS_CACHE_PATH) so a stale PVL from
        # a *previous* job can never leak into this one -- every handler()
        # call gets a fresh mkdtemp(), and it's cleaned up below regardless
        # of how the job ends.
        pvl_path = None
        pvl_work_dir = None
        if pvl_file_id:
            pvl_work_dir = tempfile.mkdtemp(prefix='motif_pvl_')
            pvl_path = os.path.join(pvl_work_dir, 'pvl.xlsx')
            print(f"\nDownloading PVL file: {pvl_file_id}")
            try:
                download_file(service, pvl_file_id, pvl_path)
            except Exception as e:
                print(f"WARNING: failed to download PVL file ({e}) -- skipping annotation for this job")
                pvl_path = None

        try:
            results = []
            for i, file_info in enumerate(files):
                print(f"\n[{i + 1}/{len(files)}] {file_info['name']}")
                results.append(process_one_file(
                    service, file_info, run_output_folder_id,
                    noise_components, noise_var_scale, hit_gap_seconds,
                    embeddings_cache_folder_id=cache_folder_id,
                    require_full_match=require_full_match,
                    pvl_path=pvl_path,
                ))
        finally:
            if pvl_work_dir:
                shutil.rmtree(pvl_work_dir, ignore_errors=True)

        files_processed = sum(1 for r in results if r["status"] == "success")
        files_failed = sum(1 for r in results if r["status"] == "error")

        build_and_upload_pvl_db(service, results, run_output_folder_id)

        return {
            "status": "success",
            "output_folder_id": run_output_folder_id,
            "output_folder_name": output_folder_name,
            "embeddings_cache_folder_id": cache_folder_id,
            "files_found": len(files),
            "files_processed": files_processed,
            "files_failed": files_failed,
            "results": results,
        }

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"ERROR: {error_msg}")
        return {"error": error_msg}


# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
