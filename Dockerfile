FROM python:3.11-slim

# libgl1/libglx-mesa0/libosmesa6: offscreen OpenGL for VTK's volume mapper
# (falls back to software rendering without a passed-through GPU; see /dev/dri
# note in docker-compose.yml for hardware acceleration on the app service)
# libportaudio2: sounddevice's native dependency (import-time only in this
# app -- server.py never opens a mic, audio arrives as browser uploads)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglx-mesa0 libosmesa6 libxrender1 libxext6 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
