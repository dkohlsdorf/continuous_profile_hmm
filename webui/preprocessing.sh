#!/bin/sh
# Full web UI data pipeline for one extend-find run:
#   1. preprocessing.py          hits -> spectrograms / region spectrograms / embeddings + results.csv on Drive
#   2. display_set.py            download results.csv, pick the display set, download its files
#   3. greedy_seriation_prep.py  pairwise DTW between display-set hits + submodel linkages (~35 min)
#   4. prepdb.py                 display set + linkages -> web.db
#
#   webui/preprocessing.sh RUN_FOLDER_ID CACHE_FOLDER_ID WAV_FOLDER_ID [WAV_FOLDER_ID ...]
#
# Env: DATA_DIR (default ~/Documents/dolphin_web_data), EMBEDDING_MODEL (default whisper),
# plus the Google Drive auth vars (see gdrive_helper.py). Every step is restartable:
# preprocessing.py skips finished recordings, display_set.py skips files on disk.
# Load the result into D1 afterwards with `npm run db:local` / `npm run db:remote` in webui/app.
set -eu

if [ $# -lt 3 ]; then
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi
RUN_FOLDER=$1
CACHE_FOLDER=$2
shift 2

HERE=$(cd "$(dirname "$0")" && pwd)
DATA_DIR=${DATA_DIR:-$HOME/Documents/dolphin_web_data}
mkdir -p "$DATA_DIR"

echo "== 1/4 preprocessing"
python "$HERE/preprocessing.py" "$RUN_FOLDER" "$CACHE_FOLDER" --wav-folders "$@" \
  --embedding-model "${EMBEDDING_MODEL:-whisper}"

echo "== 2/4 display set"
python "$HERE/display_set.py" --run-folder "$RUN_FOLDER" --out-dir "$DATA_DIR" \
  --results "$DATA_DIR/results.csv" --display "$DATA_DIR/results_display_set.csv"

echo "== 3/4 pairwise DTW + submodel linkages"
python "$HERE/greedy_seriation_prep.py" --data-dir "$DATA_DIR"

echo "== 4/4 database"
python "$HERE/prepdb.py" --data-dir "$DATA_DIR"
