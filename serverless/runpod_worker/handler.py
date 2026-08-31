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
            "noise_var_scale": 2.0,     # optional, default 2.0 (noise-model variance inflation)
            "hit_gap_seconds": 0.5      # optional, default 0.5 (silence gap between hits in submodel_<n>_hits.wav)
        }
    }

Output:
    {
        "status": "success",
        "output_folder_id": "created-output-folder-id",
        "output_folder_name": "motif_find_2026-08-31_12-30-45",
        "files_found": 14,
        "files_processed": 14,
        "files_failed": 0,
        "results": [
            {"name": "...", "status": "success", "hits": 235, "uploaded": 22, "output_folder_id": "..."},
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

# motif_hits.csv + submodel_<n>_hits.wav are the deliverables; embeddings.pkl
# is only a same-run cache (see motif_discovery.py's build_noise_model()/
# find_in_file() docstrings) and isn't worth uploading.
UPLOAD_SKIP_NAMES = {"embeddings.pkl"}


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


def list_wav_files(service, folder_id):
    """
    List (not download) every WAV file in a Google Drive folder.

    Split out from downloading so the caller can process (download, find,
    upload, delete) one file at a time instead of pulling the whole folder
    to disk up front.
    """
    query = f"'{folder_id}' in parents and (mimeType='audio/wav' or mimeType='audio/x-wav' or name contains '.wav') and trashed=false"
    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType)",
        pageSize=1000,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()

    # The Drive query's "name contains '.wav'" clause can match a Google-native
    # file (Doc/Sheet/Slides/etc.) whose name merely contains ".wav" as a
    # substring; those have no binary content and 403 on download (only
    # export), so exclude any Google Workspace mimeType and require the name
    # to actually end in .wav.
    return [
        f for f in results.get('files', [])
        if not f.get('mimeType', '').startswith('application/vnd.google-apps')
        and f['name'].lower().endswith('.wav')
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


def run_find(wav_path, output_path, noise_components, noise_var_scale, hit_gap_seconds):
    """
    Run motif_discovery.py find for one wav file against the packaged L2
    model + noise recording, streaming its output live. cwd=APP_DIR because
    CONFIG['checkpoint_path'] (lib_phmm/config.py) is a path relative to cwd,
    not to motif_discovery.py's location.
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


def process_one_file(service, file_info, run_output_folder_id, noise_components, noise_var_scale, hit_gap_seconds):
    """
    Download -> convert to mono -> find -> upload -> delete local files, for
    a single Drive file. Returns a result dict; never raises (errors are
    captured in the "status"/"error" fields) so one bad file doesn't abort
    the rest of the folder.
    """
    name = file_info['name']
    stem = Path(name).stem
    work_dir = tempfile.mkdtemp(prefix='motif_find_')
    try:
        raw_path = os.path.join(work_dir, name)
        print(f"  Downloading: {name}")
        download_file(service, file_info['id'], raw_path)

        mono_path = os.path.join(work_dir, 'mono.wav')
        print(f"  Converting to mono (first channel): {name}")
        convert_to_mono_first_channel(raw_path, mono_path)

        output_path = os.path.join(work_dir, 'output')
        os.makedirs(output_path, exist_ok=True)
        print(f"  Running find: {name}")
        run_find(mono_path, output_path, noise_components, noise_var_scale, hit_gap_seconds)

        hits = count_hits(os.path.join(output_path, 'motif_hits.csv'))

        file_output_folder_id = create_output_folder(service, run_output_folder_id, stem)
        print(f"  Uploading results: {name}")
        uploaded = upload_folder_contents(service, file_output_folder_id, output_path, skip_names=UPLOAD_SKIP_NAMES)

        return {
            "name": name,
            "status": "success",
            "hits": hits,
            "uploaded": uploaded,
            "output_folder_id": file_output_folder_id,
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
    noise_var_scale = job_input.get('noise_var_scale', 2.0)
    hit_gap_seconds = job_input.get('hit_gap_seconds', 0.5)

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
    print(f"noise_components={noise_components} noise_var_scale={noise_var_scale} hit_gap_seconds={hit_gap_seconds}")

    try:
        print("\nInitializing Google Drive connection...")
        service = get_drive_service()

        print("\nListing WAV files in Google Drive folder...")
        files = list_wav_files(service, gdrive_folder_id)
        if not files:
            return {"error": "No WAV files found in the specified Google Drive folder"}
        print(f"Found {len(files)} WAV files")

        parent_folder_for_output = output_gdrive_folder_id or gdrive_folder_id
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_folder_name = f"motif_find_{timestamp}"
        print(f"\nCreating output folder in Google Drive: {output_folder_name}")
        run_output_folder_id = create_output_folder(service, parent_folder_for_output, output_folder_name)

        results = []
        for i, file_info in enumerate(files):
            print(f"\n[{i + 1}/{len(files)}] {file_info['name']}")
            results.append(process_one_file(
                service, file_info, run_output_folder_id,
                noise_components, noise_var_scale, hit_gap_seconds,
            ))

        files_processed = sum(1 for r in results if r["status"] == "success")
        files_failed = sum(1 for r in results if r["status"] == "error")

        return {
            "status": "success",
            "output_folder_id": run_output_folder_id,
            "output_folder_name": output_folder_name,
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
