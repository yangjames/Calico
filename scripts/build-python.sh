#!/bin/bash

DOCKER_BASE=$(dirname "$(realpath "${BASH_SOURCE[0]}")")/..

./scripts/build-docker.sh || exit 1

# If we don't do this, we'll end up with files owned by root
# all over the place.
export USER_ID=$(id -u)
export GID=$(id -g)

docker run -it --rm \
    -v $(pwd):/calico:rw \
    ghcr.io/yangjames/calico:latest \
    /bin/bash -c ". /venv/bin/activate && python3 -m build --wheel && auditwheel repair --plat manylinux_2_35_x86_64 dist/calico-*-linux_x86_64.whl"