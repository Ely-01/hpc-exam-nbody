FROM ubuntu:22.04

LABEL org.opencontainers.image.title="hpc-exam-nbody"
LABEL org.opencontainers.image.description="Direct N-body MPI+OpenMP build environment"

ENV DEBIAN_FRONTEND=noninteractive
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none
ENV OMP_PROC_BIND=close
ENV OMP_PLACES=cores

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        make \
        openmpi-bin \
        libopenmpi-dev \
        python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

CMD ["/bin/bash"]
