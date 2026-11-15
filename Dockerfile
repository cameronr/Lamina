# Lamina in a container (Linux, NVIDIA GPUs of the RTX 30 series or newer, CUDA 13). The first stage builds the engine
# and the image encoder with the portable preset (build.sh) for the CUDA architectures below (docker build has no GPU
# to read them from); the runtime stage starts with setup.py --no-build, which downloads and prepares the model in
# /data once and checks it on later starts, then serves it on port 8080 with the API key the container is given.
#
#   docker build -t lamina .                                                       (RTX 30, 40 and 50)
#   docker build -t lamina --build-arg CUDA_ARCHITECTURES="86-real;86-virtual" .   (RTX 30 only: a faster build)
#   docker build -t lamina --build-arg JOBS=2 .                                    (fewer build jobs at once)
#   docker run --rm --gpus all --ulimit memlock=-1 -p 127.0.0.1:8081:8080 -v lamina-data:/data \
#              -e MODEL=iq3_s -e API_KEY=<a long random key> lamina
#
# The settings (environment variables): MODEL (iq3_s, ud-q4_k_xl or qwen36-ud-q4_k_xl), API_KEY, and optionally
# VISION (on, off or gpu), CONTEXT (tokens), MAIN_GPU (a name, UUID or nvidia-smi index). The image runs on any x86-64 CPU
# with AVX2 (--portable). Pinned memory needs --ulimit memlock=-1.

ARG CUDA_VERSION=13.0.0

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates cmake libssl-dev \
    ninja-build && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/lamina
COPY . .
ARG CUDA_ARCHITECTURES="86-real;89-real;120-real;120-virtual"
# build jobs at once (default: half the CPU's threads, fewer when the build's free RAM holds fewer at 1.5 GB a job)
ARG JOBS=
RUN sh build.sh portable -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" ${JOBS:+-DLAMINA_JOBS=$JOBS} && \
    cmake --install build-portable --component lamina --prefix /opt/lamina/dist

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 libssl3 python3 python3-venv tini \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/lamina
COPY . .
RUN python3 -m venv .venv
# (the image encoder beside the server, which takes it from its own folder)
COPY --from=build /opt/lamina/dist/lamina /opt/lamina/dist/lamina-server /opt/lamina/dist/lamina-vision build/
VOLUME ["/data"]
EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/lamina/docker/entrypoint.sh"]
