cmake_minimum_required(VERSION 3.0)
project (Calico)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(GTest REQUIRED)
find_package(absl REQUIRED)

# Camera library.
message(STATUS "Current source dir: ${CMAKE_CURRENT_SOURCE_DIR}")
add_library(
  camera_lib
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera/camera_models.h
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera/camera_models.cpp
)

target_link_libraries(
  camera_lib
  Eigen3::Eigen
  absl::status
  absl::statusor
)
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Camera test
add_executable(
  camera_lib_test
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera/camera_lib_test.cpp
)
target_link_libraries(
  camera_lib_test
  camera_lib
  GTest::Main
)
add_test(CameraLibTest camera_lib_test)
