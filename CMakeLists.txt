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

# Build world model library and tests.
add_library(
  world_model
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/world_model.cpp
)
target_link_libraries(
  world_model
  Eigen3::Eigen
  absl::status
  absl::statusor
  absl::flat_hash_map
  Ceres::ceres
)
add_executable(
  world_model_test
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/world_model_test.cpp
)
target_link_libraries(
  world_model_test
  world_model
  GTest::Main
  gmock
)
add_test(WorldModelTest world_model_test)

# Build camera library and tests.
add_library(
  camera
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera_models.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera_cost_functor.cpp
)
target_link_libraries(
  camera
  Eigen3::Eigen
  absl::status
  absl::statusor
  absl::flat_hash_map
  Ceres::ceres
  world_model
)

add_executable(
  camera_models_test
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera_models_test.cpp
)
target_link_libraries(
  camera_models_test
  camera
  GTest::Main
  gmock
)
add_test(CameraModelsTest camera_models_test)

add_executable(
  camera_cost_functor_test
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera_cost_functor_test.cpp
)
target_link_libraries(
  camera_cost_functor_test
  camera
  GTest::Main
  gmock
)
add_test(CameraCostFunctorTest camera_cost_functor_test)

add_executable(
  camera_test
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera_test.cpp
)
target_link_libraries(
  camera_test
  camera
  GTest::Main
  gmock
)
add_test(CameraLibTest camera_test)

# Build batch optimizer and tests.
add_library(
  batch_optimizer
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/batch_optimizer.cpp
)
target_link_libraries(
  batch_optimizer
  camera
  world_model
)

add_executable(
  batch_optimizer_test
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/batch_optimizer_test.cpp
)
target_link_libraries(
  batch_optimizer_test
  camera
  world_model
  batch_optimizer
  GTest::Main
  gmock
)
add_test(BatchOptimizerTest batch_optimizer_test)
