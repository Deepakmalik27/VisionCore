# Reception analytics — pinned runtime.
#
# WHY PINNED SO HARD
#   On Kaggle, Cell 1 force-reinstalls scipy to restore its Cython
#   C-extensions and hand-attaches 38 string ufuncs into numpy._core.umath so
#   InsightFace stays importable. None of that is the pipeline — it is scar
#   tissue from fighting a pre-installed environment on every single run. Pin
#   once here and that whole failure class disappears.
#
#   Every version below is what run 68b97311f9 actually loaded and proved
#   healthy. Do not relax them to ranges; ">=" is how the fleet drifts.
#
# VERIFY BEFORE TRUSTING
#   That run reported torch 2.10.0+cu128. Confirm this base tag exists and
#   that `torch.cuda.is_available()` is True on your instance before running a
#   real night; adjust CUDA_TAG / TORCH_INDEX together if not.

ARG CUDA_TAG=12.8.1-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev \
      ffmpeg \
      libgl1 libglib2.0-0 \
      git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# torch first, from the CUDA-matched index
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN python3 -m pip install torch torchvision --index-url ${TORCH_INDEX}

# exact versions from the last healthy run
RUN python3 -m pip install \
      numpy==2.0.2 \
      scipy==1.16.3 \
      supervision==0.26.1 \
      boxmot==19.0.0 \
      onnxruntime-gpu==1.19.0 \
      "ultralytics>=8.3.0" \
      insightface \
      opencv-python-headless \
      pandas openpyxl jinja2 matplotlib tqdm scikit-learn \
      gdown \
      papermill jupyter nbformat ipykernel \
      pyyaml \
      "psycopg[binary]==3.3.4"

WORKDIR /app

# Source of truth. kevacv is copied so tests run in-container; the notebook
# still carries its own embedded copy for Kaggle — tools/embed_kevacv.py keeps
# the two identical and `--check` fails the build if they drift.
COPY kevacv/   /app/kevacv/
COPY tests/    /app/tests/
COPY tools/    /app/tools/
COPY notebooks/ /app/notebooks/
COPY zones/    /app/zones/
COPY config/   /app/config/
COPY run.sh    /app/run.sh
RUN chmod +x /app/run.sh

# Fail the build rather than discover the drift an hour into a run.
RUN python3 tools/embed_kevacv.py --check

# data/ output/ models/ are volumes: the 3.2 GB chunk is pulled ONCE onto a
# persistent disk, not re-downloaded every run the way Kaggle forces.
VOLUME ["/app/data", "/app/output", "/app/models", "/app/logs"]

HEALTHCHECK --interval=60s --timeout=15s --start-period=10s --retries=3 \
  CMD python3 -c "import kevacv, torch; \
      assert torch.cuda.is_available(), 'no GPU'; \
      print('ok', kevacv.__version__)" || exit 1

# Default is the CODEBASE path (kevacv.pipeline), not the notebook.
ENTRYPOINT ["/app/run.sh"]
CMD ["--check"]
