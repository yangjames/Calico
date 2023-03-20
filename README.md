# Calico
Yet another sensor calibration library...

# WIP
## Build dependencies

CMake (3.5 or higher): `sudo apt install cmake`

TODO: bazel support

## Library dependencies

BLAS & LAPACK: `sudo apt install libatlas-base-dev` 

glog (for Ceres solver): `sudo apt install libgoogle-glog-dev libgflags-dev`

Eigen3: `sudo apt install libeigen3-dev`

GTest: `sudo apt install libgtest-dev libgmock-dev`

Abseil: `sudo apt install libabsl-dev`

Ceres Solver: Our least-squares optimization backend. See [Ceres solver installation page](http://ceres-solver.org/installation.html) for more details.
```
cd ~
wget http://ceres-solver.org/ceres-solver-2.1.0.tar.gz
tar zxf ceres-solver-2.1.0.tar.gz
mkdir ceres-bin
cd ceres-bin
cmake ../ceres-solver-2.1.0
make -j3
sudo make install
```