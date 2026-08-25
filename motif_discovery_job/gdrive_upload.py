"""
gdrive_upload.py

Uploads every file directly inside a local folder to a Google Drive
folder. Auth is the OAuth path from
dolphin_msa/serverless/runpod_worker/handler.py's get_drive_service()
(copied, trimmed of the download/email/RunPod pieces this job doesn't
need, and of the service-account path -- this deployment uses OAuth
only), via env vars.

Env vars (get a refresh token by running
dolphin_msa/serverless/get_refresh_token.py locally once):
    GOOGLE_REFRESH_TOKEN
    GOOGLE_CLIENT_ID
    GOOGLE_CLIENT_SECRET

Usage:
    python gdrive_upload.py <local_folder> <gdrive_folder_id>
"""
import os
import sys

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


def get_drive_service():
    refresh_token = os.environ.get('GOOGLE_REFRESH_TOKEN')
    client_id = os.environ.get('GOOGLE_CLIENT_ID')
    client_secret = os.environ.get('GOOGLE_CLIENT_SECRET')

    if not (refresh_token and client_id and client_secret):
        raise ValueError(
            "No Google credentials configured. Set all three of:\n"
            "  GOOGLE_REFRESH_TOKEN, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET"
        )

    credentials = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri='https://oauth2.googleapis.com/token',
        client_id=client_id,
        client_secret=client_secret,
        scopes=['https://www.googleapis.com/auth/drive']
    )
    credentials.refresh(Request())
    return build('drive', 'v3', credentials=credentials)


def upload_files(service, folder_id, local_folder):
    mime_types = {
        '.wav': 'audio/wav',
        '.pkl': 'application/octet-stream',
        '.png': 'image/png',
        '.csv': 'text/csv',
        '.pdf': 'application/pdf',
        '.json': 'application/json',
    }

    uploaded_count = 0
    for filename in sorted(os.listdir(local_folder)):
        local_path = os.path.join(local_folder, filename)
        if not os.path.isfile(local_path):
            continue

        print(f"  Uploading: {filename}")
        _, ext = os.path.splitext(filename)
        mime_type = mime_types.get(ext, 'application/octet-stream')

        file_metadata = {'name': filename, 'parents': [folder_id]}
        media = MediaFileUpload(local_path, mimetype=mime_type, resumable=True)
        service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id',
            supportsAllDrives=True,
        ).execute()
        uploaded_count += 1

    return uploaded_count


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gdrive_upload.py <local_folder> <gdrive_folder_id>", file=sys.stderr)
        sys.exit(1)

    local_folder, folder_id = sys.argv[1], sys.argv[2]
    if not os.path.isdir(local_folder):
        print(f"ERROR: {local_folder} is not a directory", file=sys.stderr)
        sys.exit(1)

    service = get_drive_service()
    n = upload_files(service, folder_id, local_folder)
    print(f"Uploaded {n} files from {local_folder} to Google Drive folder {folder_id}")
