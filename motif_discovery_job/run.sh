#!/bin/sh
# Runs motif_discovery.py with whatever args the container was given,
# then optionally uploads OUTPUT_PATH's contents to Google Drive if
# GDRIVE_OUTPUT_FOLDER_ID is set (see gdrive_upload.py for the required
# credential env vars). Uploading is entirely opt-in -- with
# GDRIVE_OUTPUT_FOLDER_ID unset this is just `python motif_discovery.py "$@"`.
set -e

python motif_discovery.py "$@"

if [ -n "$GDRIVE_OUTPUT_FOLDER_ID" ]; then
    if [ -z "$OUTPUT_PATH" ]; then
        echo "GDRIVE_OUTPUT_FOLDER_ID is set but OUTPUT_PATH is not -- skipping upload" >&2
        echo "(OUTPUT_PATH should match the output_path positional arg passed above)" >&2
        exit 1
    fi
    echo "=========================================="
    echo "Uploading $OUTPUT_PATH to Google Drive folder $GDRIVE_OUTPUT_FOLDER_ID"
    echo "=========================================="
    python gdrive_upload.py "$OUTPUT_PATH" "$GDRIVE_OUTPUT_FOLDER_ID"
fi
