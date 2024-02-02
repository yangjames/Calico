cmake_minimum_required(VERSION 3.4...3.18)
project(calico)

include(${CMAKE_CURRENT_LIST_DIR}/../CMakeLists.txt)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYBIND_LIBRARY_OUTPUT_DIRECTORY})

# # Load dependencies.
find_package(pybind11 REQUIRED)

# Add top level include directory.
include_directories(
  ${CMAKE_CURRENT_LIST_DIR}
)

pybind11_add_module(calico src/calico.cpp)
target_link_libraries(
  calico PUBLIC
  Eigen3::Eigen
  accelerometer
  aprilgrid_detector
  batch_optimizer
  camera
  gyroscope
  trajectory
  world_model
)


# CALICO_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(calico
                           PRIVATE VERSION_INFO=${CALICO_VERSION_INFO})