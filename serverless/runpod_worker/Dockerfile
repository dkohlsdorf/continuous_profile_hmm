# Dockerfile for the motif `find` RunPod serverless worker.
# Packages motif_discovery.py + lib_phmm, the fine-tuned Whisper embedding
# checkpoint, the trained L2 motif model, and the noise recording used for
# --noise-wav, so a job only has to hand over Google Drive file IDs.
#
# Build from the repo root (continuous_profile_hmm/), not this directory:
#   docker build -f serverless/runpod_worker/Dockerfile -t motif-find-worker .

FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# process() in lib_phmm/signals.py calls the CPU-side Whisper feature
# extractor once per 5120-sample window -- tens of thousands of times for
# a long recording (a 75,259-window file was observed OOM-killed, return
# code -9, on a MIG-sliced pod). Left unset, OpenBLAS/OMP/MKL each size
# their thread pool to every visible core on the host CPU (an EPYC 9554,
# 64c/128t), so a workload of many tiny per-window calls spends its time
# spinning up/tearing down huge thread pools instead of the ~0.1s of real
# work each one does -- glibc's per-thread malloc arenas are a well-known
# way for that churn to bloat RSS over exactly this many iterations,
# unrelated to actual data volume (the embeddings list itself is tiny:
# 75,259 x (1,128) float32 is ~38MB). Capped here since it's a container-
# wide setting these libraries only read once, at first use.
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4

WORKDIR /app

# build-essential/g++/gcc: compiling lib_phmm's pybind11 C++ extensions.
# libsndfile1: torchaudio's sound backend. ffmpeg: the mono-conversion step
# in handler.py, and torchaudio/transformers' own audio decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    gcc \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY serverless/runpod_worker/requirements.txt .

# torch/torchaudio/torchvision are pulled from the cu128 index specifically
# (rather than the base image's own build, or the default PyPI cu126 build)
# because newer GPU architectures (e.g. Blackwell) have no compiled kernels
# in the older builds, causing "no kernel image is available for execution
# on the device" on a RunPod pod with a newer card.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir runpod google-api-python-client google-auth

# Project source. lib_phmm's checked-in profile_hmm.cpython-*.so /
# hierarchical_kmedian_dtw.cpython-*.so are macOS arm64 builds (see
# .dockerignore) and won't load here -- rebuilt below for this container's
# Linux/Python instead.
COPY motif_discovery.py .
COPY lib_phmm/ ./lib_phmm/

RUN cd lib_phmm && pip install --no-cache-dir -e . --no-build-isolation && cd ..

RUN python -c "import lib_phmm.profile_hmm as phmm; print('profile_hmm loaded')" && \
    python -c "import lib_phmm.hierarchical_kmedian_dtw as hkd; print('hierarchical_kmedian_dtw loaded')"

# Pre-download the Whisper-small base weights lib_phmm/model.py's
# lora_encoder() wraps, so a cold job doesn't stall on the HF Hub.
RUN python -c "from transformers import WhisperModel, WhisperProcessor; \
    WhisperProcessor.from_pretrained('openai/whisper-small'); \
    WhisperModel.from_pretrained('openai/whisper-small'); \
    print('Whisper small cached')"

# Fine-tuned embedding/classification checkpoint (CONFIG['checkpoint_path']
# in lib_phmm/config.py is this same relative path, resolved against
# whatever the process's cwd is -- handler.py always runs motif_discovery.py
# with cwd=/app).
COPY model/wdp_whisper_emb_model_L2.pt ./model/wdp_whisper_emb_model_L2.pt

# Trained L2 motif model + the --noise-wav recording, staged into
# serverless/runpod_worker/assets/ (see .gitignore) since /motif/ is
# excluded from this build context and the noise wav lives outside the
# repo entirely -- re-copy both there to refresh.
COPY serverless/runpod_worker/assets/phmm_l2.pkl ./motif/l2/phmm_l2.pkl
COPY serverless/runpod_worker/assets/naive_noise.wav ./assets/naive_noise.wav

# Records the rate the model was trained at, read back by find to resample
# both search and noise audio to match (see load_training_sample_rate() in
# motif_discovery.py). `train` writes this next to the model it produces;
# it's staged here because the packaged phmm_l2.pkl predates that.
COPY serverless/runpod_worker/assets/training_metadata.json ./motif/l2/training_metadata.json

COPY serverless/runpod_worker/handler.py .

RUN mkdir -p /app/work

CMD ["python", "-u", "handler.py"]
