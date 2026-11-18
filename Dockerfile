# The fork's prod image (lamina). Upstream removed its image (db378e8ba); this one stays
# fork-local for run-lamina.sh (dotfiles/lamina), which overrides the entrypoint with its own
# lamina-server --config call. The build stage is upstream's portable preset (build.sh); the
# runtime stage lays the programs flat in /opt/lamina/build so lamina-server finds the
# lamina-vision beside it, as the server expects since the vision "exe" config key was removed.
#
#   docker build -t lamina:main .                                                      (RTX 30, 40 and 50)
#   docker build -t lamina:main --build-arg CUDA_ARCHITECTURES="86-real;89-real" .    (this box's pair)
#
# Pinned memory needs --ulimit memlock=-1; AVX2 or better CPUs (the server refuses below it).

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
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 libssl3 tini \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/lamina
# (the image encoder beside the server, which takes it from its own folder)
COPY --from=build /opt/lamina/dist/lamina /opt/lamina/dist/lamina-server /opt/lamina/dist/lamina-vision build/
VOLUME ["/data"]
EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/opt/lamina/build/lamina-server", "--help"]
