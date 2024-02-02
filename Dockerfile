FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=US/Eastern

RUN apt update && \
    apt install -y \
        tzdata sudo git \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        libeigen3-dev libgtest-dev libabsl-dev \
        libopencv-dev libyaml-cpp-dev libgmock-dev

COPY scripts/install-ceres.sh /tmp/install-ceres.sh
RUN /tmp/install-ceres.sh

WORKDIR /workdir
RUN chmod 777 /workdir