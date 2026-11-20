"""
Google Drive helpers shared by the web UI scripts (preprocessing.py,
display_set.py), plus the layout of the webui_preprocessing output folder.

Auth: same env vars as serverless/runpod_worker/handler.py
(GOOGLE_REFRESH_TOKEN + GOOGLE_CLIENT_ID + GOOGLE_CLIENT_SECRET, or
GOOGLE_SERVICE_ACCOUNT_JSON).
"""

import base64
import io
import json
import os
import threading

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload, MediaIoBaseUpload

OUTPUT_FOLDER_NAME = "webui_preprocessing"
SUBFOLDERS = ("spectrograms", "region_spectrograms", "embeddings")
FOLDER_MIME = "application/vnd.google-apps.folder"


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


def thread_service():
    """googleapiclient services aren't thread-safe: one per worker thread."""
    if not hasattr(_thread_local, "service"):
        _thread_local.service = get_drive_service()
    return _thread_local.service


def upload(folder_id, path, mime):
    thread_service().files().create(
        body={"name": path.name, "parents": [folder_id]},
        media_body=MediaFileUpload(str(path), mimetype=mime),
        fields="id", supportsAllDrives=True).execute(num_retries=5)
