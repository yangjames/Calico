#!/bin/bash

DOCKER_BASE=$(dirname "$(realpath "${BASH_SOURCE[0]}")")/..

if [[ $USER == "root" ]]; then
    echo "Cannot be root!"
    exit 1
fi

# build local dockerfile to run bg apps in containers
export DOCKER_BUILDKIT=1
export UBUNTU_VERSION=22.04

(
    cd "${DOCKER_BASE}" || exit 1

    docker build --progress plain --pull -f Dockerfile \
        -t ghcr.io/yangjames/calico:latest \
        --build-arg UBUNTU_VERSION .
)
