# Calico
Yet another sensor calibration library...

# WIP
Current dependencies:

Eigen3: `sudo apt install libeigen3-dev`

GTest: `sudo apt install libgtest-dev`

Abseil: Built with `std=c++17`. See [Abseil install page](https://abseil.io/docs/cpp/quickstart-cmake#getting-the-abseil-code) for more details.
```
cd ~
git clone https://github.com/abseil/abseil-cpp.git
mkdir ~/abseil-cpp/build
cd ~/abseil-cpp/build
cmake -DABSL_BUILD_TESTING=ON -DABSL_USE_GOOGLETEST_HEAD=ON -DCMAKE_CXX_STANDARD=17 ..
cmake --build . --target all
sudo make install
```