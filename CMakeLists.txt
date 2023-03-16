cmake_minimum_required(VERSION 3.5)
project (Calico)

# std=c++17 or higher required for Abseil.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Load dependencies.
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(GTest REQUIRED)
find_package(absl REQUIRED)
find_package(Ceres REQUIRED)
enable_testing()

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
  gmock
)
add_test(TypeDefsTest typedefs_test)

# Build spline test.
add_executable(
  bspline_test
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/bspline_test.cpp
)
target_link_libraries(
  bspline_test
  Eigen3::Eigen
  absl::status
  absl::statusor
  GTest::Main
  gmock
)
add_test(BSplineTest bspline_test)

# Build trajectory library and tests.
add_library(
  trajectory
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/trajectory.cpp
)
target_link_libraries(
  trajectory
  Eigen3::Eigen
  absl::status
  absl::statusor
  absl::flat_hash_map
  Ceres::ceres
)
add_executable(
  trajectory_test
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/trajectory_test.cpp
  )
target_link_libraries(
  trajectory_test
  trajectory
  GTest::Main
  gmock
)

# Geometry test.
add_executable(
  geometry_test
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/geometry_test.cpp
  )
target_link_libraries(
  geometry_test
  trajectory
  Eigen3::Eigen
  GTest::Main
  gmock
)

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

# Build gyroscope library and tests.
add_library(
  gyroscope_models
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/gyroscope_models.cpp
)
target_link_libraries(
  gyroscope_models
  absl::statusor
  Eigen3::Eigen
)
add_executable(
  gyroscope_models_test
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/gyroscope_models_test.cpp
  )
target_link_libraries(
  gyroscope_models_test
  gyroscope_models
  GTest::Main
  gmock
)
add_test(GyroscopeModelsTest gyroscope_models_test)

add_library(
  gyroscope_cost_functor
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/gyroscope_cost_functor.cpp
)
target_link_libraries(
  gyroscope_cost_functor
  gyroscope_models
  trajectory
)
add_executable(
  gyroscope_cost_functor_test
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/gyroscope_cost_functor_test.cpp
)
target_link_libraries(
  gyroscope_cost_functor_test
  gyroscope_cost_functor
  GTest::Main
  gmock
)
add_test(GyroscopeCostFunctorTest gyroscope_cost_functor_test)

add_library(
  gyroscope
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/gyroscope.cpp
)
target_link_libraries(
  gyroscope
  gyroscope_cost_functor
  gyroscope_models
  trajectory
  world_model
)

add_executable(
  gyroscope_test
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/gyroscope_test.cpp
)
target_link_libraries(
  gyroscope_test
  gyroscope
  GTest::Main
  gmock
)
add_test(GyroscopeLibTest gyroscope_test)
  

# Build camera library and tests.
add_library(
  camera_models
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera_models.cpp
)
target_link_libraries(
  camera_models
  absl::statusor
  Eigen3::Eigen
)
add_executable(
  camera_models_test
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera_models_test.cpp
)
target_link_libraries(
  camera_models_test
  camera_models
  GTest::Main
  gmock
)
add_test(CameraModelsTest camera_models_test)

add_library(
  camera_cost_functor
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera_cost_functor.cpp
)
target_link_libraries(
  camera_cost_functor
  camera_models
  trajectory
)
add_executable(
  camera_cost_functor_test
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera_cost_functor_test.cpp
)
target_link_libraries(
  camera_cost_functor_test
  camera_cost_functor
  GTest::Main
  gmock
)
add_test(CameraCostFunctorTest camera_cost_functor_test)

add_library(
  camera
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera.cpp
)
target_link_libraries(
  camera
  camera_models
  camera_cost_functor
  # absl::status
  # absl::statusor
  # absl::flat_hash_map
  # Ceres::ceres
  # Eigen3::Eigen
  trajectory
  world_model
)

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
