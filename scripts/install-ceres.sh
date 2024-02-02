#!/bin/bash

# CMake
# google-glog + gflags
# Use ATLAS for BLAS & LAPACK
# Eigen3
# SuiteSparse (optional)
sudo apt-get install -y cmake libgoogle-glog-dev \
    libgflags-dev libatlas-base-dev libeigen3-dev \
    libsuitesparse-dev

cd /tmp || exit
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver || exit

cmake .
make -j3
#make test
# Optionally install Ceres, it can also be exported using CMake which
# allows Ceres to be used without requiring installation, see the documentation
# for the EXPORT_BUILD_DIR option for more information.
sudo make install