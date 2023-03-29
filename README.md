# Calico
<img src="doc/calico.png" width="200">

# WIP
## Build dependencies

CMake (3.5 or higher): `sudo apt install cmake`

TODO: bazel support


Simple install:
## Ubuntu 22.04
```
sudo apt install -y build-essential libatlas-base-dev libgoogle-glog-dev libgflags-dev libeigen3-dev libgtest-dev libgmock-dev libabsl-dev python3-dev python3-pytest pybind11-dev libceres-dev libyaml-cpp-dev libopencv-dev python3-opencv
```
## Ubuntu 20.04
```
sudo apt install -y build-essential libatlas-base-dev libgoogle-glog-dev libgflags-dev libeigen3-dev libgtest-dev libgmock-dev python3-dev python3-pytest pybind11-dev libyaml-cpp-dev libopencv-dev python3-opencv
```
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
```
cd ~
git clone https://github.com/abseil/abseil-cpp.git
cd ~/abseil-cpp
cmake -S . -B build -DCMAKE_CXX_STANDARD=17 -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build build
sudo cmake --install build
```
## Library dependencies

BLAS & LAPACK: `sudo apt install libatlas-base-dev` 

glog (for Ceres solver): `sudo apt install libgoogle-glog-dev libgflags-dev`

Eigen3: `sudo apt install libeigen3-dev`

GTest: `sudo apt install libgtest-dev libgmock-dev`

Abseil: `sudo apt install libabsl-dev`

Pybind11: `sudo apt install pybind11-dev`

Ceres Solver: Our least-squares optimization backend. See [Ceres solver installation page](http://ceres-solver.org/installation.html) for more details.
