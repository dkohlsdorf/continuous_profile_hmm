# Build context is the repo root (see docker-compose.yml's build.context:
# "..") since this needs lib_phmm/, motif_discovery.py, and model/ from
# there -- so every COPY source path below is repo-root-relative, not
# relative to this Dockerfile's own folder.
FROM python:3.11-slim

# build-essential: g++ for the pybind11 C++ extensions (profile_hmm,
# hierarchical_kmedian_dtw). libsndfile1: torchaudio's sox_io/soundfile
# backend needs it to load/save wav files.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Container stdout isn't a TTY, so Python fully-buffers it by default --
# print()s (progress banners, --verbose output, sweep status) can sit
# unflushed for a long time otherwise, which looks like "no logs" in
# Cloud Logging even though the run is progressing normally. Covers
# every python invocation in the container (motif_discovery.py via
# run.sh, gdrive_upload.py) without needing flush=True everywhere.
ENV PYTHONUNBUFFERED=1

# Python deps in their own layer so editing motif_discovery.py doesn't
# force a full torch/transformers reinstall on every rebuild.
COPY motif_discovery_job/docker-requirements.txt .
RUN pip install --no-cache-dir -r docker-requirements.txt

# Source + the fine-tuned checkpoint (static app data) + the run/upload
# wrapper scripts.
COPY lib_phmm/ lib_phmm/
COPY motif_discovery.py processing.py compare_metrics.py ./
COPY model/ model/
COPY motif_discovery_job/run.sh motif_discovery_job/gdrive_upload.py ./

# Build the C++ extensions against this container's Python/platform --
# any .so copied in from the host (e.g. built on macOS/arm64) would be
# wrong here regardless, so this always rebuilds from source.
RUN cd lib_phmm && python setup.py build_ext --inplace

# Baked-in L2 run data (motif_discovery_job/data/, staged from
# ../../audio/train and ../motif/l2/candidates.pkl) so the image is
# self-contained on platforms with no way to bind-mount local host
# paths (e.g. GCP Batch). Last/own layer since it's the thing most
# likely to change without the code changing. docker-compose.yml's
# local bind mounts still shadow these paths for local runs -- this is
# purely the no-mount fallback.
COPY motif_discovery_job/data/train/ /audio/train/
COPY motif_discovery_job/data/motif/ motif/

# run.sh runs motif_discovery.py with whatever args/CMD the container is
# given, then optionally uploads the output folder to Google Drive if
# GDRIVE_OUTPUT_FOLDER_ID is set -- see run.sh/gdrive_upload.py.
ENTRYPOINT ["sh", "run.sh"]
