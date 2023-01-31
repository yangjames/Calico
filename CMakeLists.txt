cmake_minimum_required(VERSION 3.5)
project (Calico)

# std=c++17 or higher required for Abseil.
set(CMAKE_CXX_STANDARD 20)

# Load dependencies.
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(GTest REQUIRED)
find_package(absl REQUIRED)
find_package(Ceres REQUIRED)

# Add top level include directory.
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Typedefs test.
add_executable(
  typedefs_test
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/typedefs_test.cpp
)
target_link_libraries(
  typedefs_test
  Eigen3::Eigen
  GTest::Main
)
add_test(TypeDefsTest typedefs_test)

# Build camera library and tests.
add_library(
  camera_lib
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera_models.cpp
)
target_link_libraries(
  camera_lib
  Eigen3::Eigen
  absl::status
  absl::statusor
  absl::flat_hash_map
  Ceres::ceres
)
add_executable(
  camera_lib_test
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera_lib_test.cpp
)
target_link_libraries(
  camera_lib_test
  camera_lib
  GTest::Main
  gmock
)
add_test(CameraLibTest camera_lib_test)

