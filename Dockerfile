# The viewer in a container, for hosts whose own system is too old for the
# pinned wheels (torch and vtk need glibc >= 2.28, e.g. not Ubuntu 18.04).
# CPU only; renders in software (Mesa) through VTK's offscreen fallback.
#
#   docker compose build
#   docker compose run --rm viewer python -m tools.fetch_scans --demo --splits
#   docker compose up            # http://127.0.0.1:8000
#
# See "Docker" in the README for the LLM and VR.
FROM python:3.12-slim-bookworm

# OpenGL/EGL through Mesa for VTK's offscreen rendering; libgomp for torch.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libegl1 libegl-mesa0 libgl1-mesa-dri libglx-mesa0 libosmesa6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so editing the code does not reinstall them. The CPU
# build of torch (~200 MB instead of ~3 GB of CUDA libraries).
COPY requirements.txt .
RUN pip install --no-cache-dir "torch==$(sed -n 's/^torch==//p' requirements.txt)" \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# compose.yaml runs the container as the host user (files in out/ stay
# theirs); the Whisper cache must be writable for any uid.
RUN mkdir -p /cache/huggingface && chmod -R 777 /cache

ENV PYTHONUNBUFFERED=1 \
    VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow \
    HF_HOME=/cache/huggingface

EXPOSE 8000
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000"]
