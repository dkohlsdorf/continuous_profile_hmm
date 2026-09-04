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
        }
    }

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
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import runpod

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

MOTIF_DISCOVERY_SCRIPT = "/app/motif_discovery.py"
APP_DIR = "/app"
MODEL_PATH = "/app/motif/l2/phmm_l2.pkl"
NOISE_WAV_PATH = "/app/assets/naive_noise.wav"

# motif_hits.csv + submodel_<n>_hits.wav are this run's deliverables.
# embeddings.pkl goes to the persistent Drive cache folder instead (see
# EMBEDDINGS_CACHE_FOLDER_NAME), not into the timestamped results folder --
# a results folder is per-run, so a copy there could never be found again
# by a later run looking to skip decoding. noise_embeddings.pkl is
# find_in_file()'s own same-run cache for the packaged --noise-wav (see
# motif_discovery.py's build_noise_model() docstring) -- meaningless
# outside this container, since assets/naive_noise.wav isn't on Drive.
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
             require_full_match=False):
    """
    Run motif_discovery.py find for one wav file against the packaged L2
    model + noise recording, streaming its output live. cwd=APP_DIR because
    CONFIG['checkpoint_path'] (lib_phmm/config.py) is a path relative to cwd,
    not to motif_discovery.py's location.

    require_full_match forwards to find's --require-full-match (see there
    and phmm.viterbi()'s docstring) -- disallows partial matches (entering
    or exiting a submodel's match-state chain at an interior state). Off
    by default, matching find's own default.
    """
    cmd = [
        'python', '-u', MOTIF_DISCOVERY_SCRIPT, 'find',
        MODEL_PATH, wav_path, output_path,
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


def count_hits(motif_hits_csv):
    """Row count of motif_hits.csv minus its header, or 0 if it's missing/empty."""
    if not os.path.exists(motif_hits_csv):
        return 0
    with open(motif_hits_csv) as f:
        return max(0, sum(1 for _ in f) - 1)


def restore_noise_embeddings(output_path):
    """
    Copy the job-local noise_embeddings.pkl (if a previous file in this job
    already produced one) into output_path, so build_noise_model() loads it
    instead of re-embedding the packaged noise wav from scratch.

    Best-effort: this is purely a speedup, so a broken/unwritable cache must
    never take down the actual run -- worst case find re-embeds the noise.
    """
    try:
        if os.path.exists(NOISE_EMBEDDINGS_CACHE_PATH):
            shutil.copy2(NOISE_EMBEDDINGS_CACHE_PATH, os.path.join(output_path, 'noise_embeddings.pkl'))
            print("  Reusing noise embeddings from this job's earlier file")
    except OSError as e:
        print(f"  (noise embedding cache unreadable, re-embedding: {e})")


def save_noise_embeddings(output_path):
    """
    Stash this run's noise_embeddings.pkl for the job's remaining files.
    Best-effort, for the same reason as restore_noise_embeddings().
    """
    produced = os.path.join(output_path, 'noise_embeddings.pkl')
    try:
        if os.path.exists(produced) and not os.path.exists(NOISE_EMBEDDINGS_CACHE_PATH):
            os.makedirs(os.path.dirname(NOISE_EMBEDDINGS_CACHE_PATH), exist_ok=True)
            shutil.copy2(produced, NOISE_EMBEDDINGS_CACHE_PATH)
    except OSError as e:
        print(f"  (could not stash noise embeddings, next file will re-embed: {e})")


def process_one_file(service, file_info, run_output_folder_id, noise_components, noise_var_scale,
                      hit_gap_seconds, embeddings_cache_folder_id=None, require_full_match=False):
    """
    Download -> convert to mono -> find -> upload -> delete local files, for
    a single Drive file. Returns a result dict; never raises (errors are
    captured in the "status"/"error" fields) so one bad file doesn't abort
    the rest of the folder.

    When embeddings_cache_folder_id is given, <stem>.pkl there is pulled in
    as this run's embeddings.pkl before find (skipping the Whisper decode
    entirely) and written back afterwards if it wasn't cached yet. The
    embedding depends only on the audio, not on any --noise-* setting, so a
    cache entry stays valid across runs that sweep those parameters.
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

        restore_noise_embeddings(output_path)

        print(f"  Running find: {name}")
        run_find(mono_path, output_path, noise_components, noise_var_scale, hit_gap_seconds,
                 require_full_match=require_full_match)

        save_noise_embeddings(output_path)

        hits = count_hits(os.path.join(output_path, 'motif_hits.csv'))

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


def handler(job):
    """RunPod serverless handler function."""
    job_input = job.get('input', {})

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

        results = []
        for i, file_info in enumerate(files):
            print(f"\n[{i + 1}/{len(files)}] {file_info['name']}")
            results.append(process_one_file(
                service, file_info, run_output_folder_id,
                noise_components, noise_var_scale, hit_gap_seconds,
                embeddings_cache_folder_id=cache_folder_id,
                require_full_match=require_full_match,
            ))

        files_processed = sum(1 for r in results if r["status"] == "success")
        files_failed = sum(1 for r in results if r["status"] == "error")

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
