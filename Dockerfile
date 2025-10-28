FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=US/Eastern

ARG PYTHON_VERSION_MAJOR=3
ARG PYTHON_VERSION_MINOR=11

RUN apt update && \
    apt install -y \
        tzdata sudo git \
        python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} \
        python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}-venv \
        python${PYTHON_VERSION_MAJOR}-dev python${PYTHON_VERSION_MAJOR}-pip \
        libeigen3-dev libgtest-dev libabsl-dev \
        libopencv-dev libyaml-cpp-dev libgmock-dev patchelf


COPY scripts/install-ceres.sh /tmp/install-ceres.sh
RUN /tmp/install-ceres.sh

RUN pip install setuptools build auditwheel pybind11[global]

WORKDIR /calico
RUN chmod 777 /calico