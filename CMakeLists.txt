cmake_minimum_required(VERSION 3.5)
project (Calico)

# Turn off testing by default.
option(BUILD_TESTING "Build unit tests." OFF)
include(CTest)

# std=c++17 or higher required for Abseil.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")

# Load dependencies.
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(GTest REQUIRED)
find_package(absl REQUIRED)
find_package(Ceres REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# Add top level include directory.
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

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

# World model library.
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
  absl::strings
  Ceres::ceres
)
# Gyroscope library.
add_library(
  gyroscope_models
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/gyroscope_models.cpp
)
target_link_libraries(
  gyroscope_models
  absl::statusor
  Eigen3::Eigen
)

add_library(
  gyroscope_cost_functor
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/gyroscope_cost_functor.cpp
)
target_link_libraries(
  gyroscope_cost_functor
  gyroscope_models
  trajectory
)

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

# Accelerometer library and tests.
add_library(
  accelerometer_models
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/accelerometer_models.cpp
)
target_link_libraries(
  accelerometer_models
  absl::statusor
  Eigen3::Eigen
)

add_library(
  accelerometer_cost_functor
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/accelerometer_cost_functor.cpp
)
target_link_libraries(
  accelerometer_cost_functor
  accelerometer_models
  trajectory
)

add_library(
  accelerometer
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/accelerometer.cpp
)
target_link_libraries(
  accelerometer
  accelerometer_cost_functor
  accelerometer_models
  trajectory
  world_model
)

# Camera library.
add_library(
  camera_models
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera_models.cpp
)
target_link_libraries(
  camera_models
  absl::statusor
  Eigen3::Eigen
)

add_library(
  camera_cost_functor
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera_cost_functor.cpp
)
target_link_libraries(
  camera_cost_functor
  camera_models
  trajectory
)

add_library(
  camera
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera.cpp
)
target_link_libraries(
  camera
  camera_models
  camera_cost_functor
  trajectory
  world_model
)

# Batch optimizer.
add_library(
  batch_optimizer
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/batch_optimizer.cpp
)
target_link_libraries(
  batch_optimizer
  accelerometer
  camera
  gyroscope
  world_model
)

# AprilTags c++ library.
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/third_party/apriltags/
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/third_party/apriltags/apriltags/
)
add_library(
  apriltags
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/third_party/apriltags/src/Edge.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/third_party/apriltags/src/FloatImage.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/third_party/apriltags/src/Gaussian.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/third_party/apriltags/src/GLine2D.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/third_party/apriltags/src/GLineSegment2D.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/third_party/apriltags/src/GrayModel.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/third_party/apriltags/src/Homography33.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/third_party/apriltags/src/MathUtil.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/third_party/apriltags/src/Quad.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/third_party/apriltags/src/Segment.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/third_party/apriltags/src/TagFamily.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/third_party/apriltags/src/TagDetection.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/third_party/apriltags/src/TagDetector.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/third_party/apriltags/src/UnionFindSimple.cc
)
target_link_libraries(
  apriltags
  Eigen3::Eigen
)

# AprilGrid Detector
add_library(
  aprilgrid_detector
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/chart_detectors/aprilgrid_detector.cpp
)
target_link_libraries(
  aprilgrid_detector
  world_model
  apriltags
  ${OpenCV_LIBS}
  ${YAML_CPP_LIBRARIES}
)

# Python bindings.
pybind11_add_module(
  calico
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/python/calico.cpp
)
target_link_libraries(
  calico PUBLIC
  accelerometer
  aprilgrid_detector
  batch_optimizer
  camera
  gyroscope
  trajectory
  world_model
  Eigen3::Eigen
)
set_target_properties(
  calico PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/calico
)
file(
  COPY
  #${CMAKE_CURRENT_SOURCE_DIR}/calico/python/__init__.py
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/python/utils.py
  DESTINATION ${PROJECT_BINARY_DIR}/calico
)

# Install cpp library.
install(
  TARGETS
  accelerometer
  accelerometer_cost_functor
  accelerometer_models
  apriltags
  batch_optimizer
  camera
  camera_cost_functor
  camera_models
  gyroscope
  gyroscope_cost_functor
  gyroscope_models
  trajectory
  world_model
  aprilgrid_detector
  EXPORT
  CalicoConfig
  DESTINATION lib/calico
)
install(
  EXPORT
  CalicoConfig
  DESTINATION lib/cmake/calico
)
set(top_level_headers
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/profiler.h
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/trajectory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/bspline.h
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/test_utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/optimization_utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/batch_optimizer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/typedefs.h
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/matchers.h
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/status_builder.h
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/geometry.h
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/world_model.h
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/statusor_macros.h
)
install(
  FILES ${top_level_headers}
  DESTINATION include/calico
)
set(sensor_headers
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/gyroscope_models.h
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera.h
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/accelerometer_cost_functor.h
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/sensor_base.h
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/gyroscope_cost_functor.h
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera_cost_functor.h
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/camera_models.h
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/accelerometer_models.h
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/gyroscope.h
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/sensors/accelerometer.h
)
install(
  FILES ${sensor_headers}
  DESTINATION include/calico/sensors/
)
# Install python library.
install(
  TARGETS calico
  DESTINATION ${Python_SITELIB}/calico
)
install(
  FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/python/utils.py
  ${CMAKE_CURRENT_SOURCE_DIR}/calico/python/__init__.py
  DESTINATION ${Python_SITELIB}/calico
)

if (BUILD_TESTING)
  # Typedefs test.
  add_executable(
    typedefs_test
    ${CMAKE_CURRENT_SOURCE_DIR}/calico/test/typedefs_test.cpp
    )
  target_link_libraries(
    typedefs_test
    Eigen3::Eigen
    GTest::Main
    gmock
    )
  add_test(TypeDefsTest typedefs_test)
  # Geometry test.
  add_executable(
    geometry_test
    ${CMAKE_CURRENT_SOURCE_DIR}/calico/test/geometry_test.cpp
    )
  target_link_libraries(
    geometry_test
    trajectory
    Eigen3::Eigen
    GTest::Main
    gmock
    )
  add_test(GeometryTest geometry_test)
  # Spline test.
  add_executable(
    bspline_test
    ${CMAKE_CURRENT_SOURCE_DIR}/calico/test/bspline_test.cpp
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
  # Trajectory test.
  add_executable(
    trajectory_test
    ${CMAKE_CURRENT_SOURCE_DIR}/calico/test/trajectory_test.cpp
    )
  target_link_libraries(
    trajectory_test
    trajectory
    GTest::Main
    gmock
    )
  add_test(TrajectoryTest trajectory_test)
  # World model test.
  add_executable(
    world_model_test
    ${CMAKE_CURRENT_SOURCE_DIR}/calico/test/world_model_test.cpp
    )
  target_link_libraries(
    world_model_test
    world_model
    GTest::Main
    gmock
    )
  add_test(WorldModelTest world_model_test)
  # Gyroscope models test.
  add_executable(
    gyroscope_models_test
    ${CMAKE_CURRENT_SOURCE_DIR}/calico/test/gyroscope_models_test.cpp
    )
  target_link_libraries(
    gyroscope_models_test
    gyroscope_models
    GTest::Main
    gmock
    )
  add_test(GyroscopeModelsTest gyroscope_models_test)
  # Gyroscope cost functor test.
  add_executable(
    gyroscope_cost_functor_test
    ${CMAKE_CURRENT_SOURCE_DIR}/calico/test/gyroscope_cost_functor_test.cpp
    )
  target_link_libraries(
    gyroscope_cost_functor_test
    gyroscope_cost_functor
    GTest::Main
    gmock
    )
  add_test(GyroscopeCostFunctorTest gyroscope_cost_functor_test)
  # Gyroscope test.
  add_executable(
    gyroscope_test
    ${CMAKE_CURRENT_SOURCE_DIR}/calico/test/gyroscope_test.cpp
    )
  target_link_libraries(
    gyroscope_test
    gyroscope
    GTest::Main
    gmock
    )
  add_test(GyroscopeLibTest gyroscope_test)
  # Accelerometer models test.
  add_executable(
    accelerometer_models_test
    ${CMAKE_CURRENT_SOURCE_DIR}/calico/test/accelerometer_models_test.cpp
    )
  target_link_libraries(
    accelerometer_models_test
    accelerometer_models
    GTest::Main
    gmock
    )
  add_test(AccelerometerModelsTest accelerometer_models_test)
  # Accelerometer cost functor test.
  add_executable(
    accelerometer_cost_functor_test
    ${CMAKE_CURRENT_SOURCE_DIR}/calico/test/accelerometer_cost_functor_test.cpp
    )
  target_link_libraries(
    accelerometer_cost_functor_test
    accelerometer_cost_functor
    GTest::Main
    gmock
    )
  add_test(AccelerometerCostFunctorTest accelerometer_cost_functor_test)
  # Accelerometer test.
  add_executable(
    accelerometer_test
    ${CMAKE_CURRENT_SOURCE_DIR}/calico/test/accelerometer_test.cpp
    )
  target_link_libraries(
    accelerometer_test
    accelerometer
    GTest::Main
    gmock
    )
  add_test(AccelerometerLibTest accelerometer_test)
  # Camera models test.
  add_executable(
    camera_models_test
    ${CMAKE_CURRENT_SOURCE_DIR}/calico/test/camera_models_test.cpp
    )
  target_link_libraries(
    camera_models_test
    camera_models
    GTest::Main
    gmock
    )
  add_test(CameraModelsTest camera_models_test)
  # Camera cost functor test.
  add_executable(
    camera_cost_functor_test
    ${CMAKE_CURRENT_SOURCE_DIR}/calico/test/camera_cost_functor_test.cpp
    )
  target_link_libraries(
    camera_cost_functor_test
    camera_cost_functor
    GTest::Main
    gmock
    )
  add_test(CameraCostFunctorTest camera_cost_functor_test)
  # Camera test.
  add_executable(
    camera_test
    ${CMAKE_CURRENT_SOURCE_DIR}/calico/test/camera_test.cpp
    )
  target_link_libraries(
    camera_test
    camera
    GTest::Main
    gmock
    )
  add_test(CameraLibTest camera_test)
  # Batch optimizer test.
  add_executable(
    batch_optimizer_test
    ${CMAKE_CURRENT_SOURCE_DIR}/calico/test/batch_optimizer_test.cpp
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
  # Python bindings test.
  add_test(
    NAME python_bindings_test
    COMMAND python3 -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/calico/test/python_bindings_test.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/calico/test
    )
  # Python utilities test.
  add_test(
    NAME python_utils_test
    COMMAND python3 -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/calico/test/python_utils_test.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/calico/test
    )
  set_tests_properties(
    python_bindings_test python_utils_test
    PROPERTIES ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR}:$ENV{PYTHONPATH}"
    )
endif()
