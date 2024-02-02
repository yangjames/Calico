cmake_minimum_required(VERSION 3.4...3.18)
project(calico)

# Load dependencies.
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(GTest REQUIRED)
find_package(absl REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(pybind11 REQUIRED)

# Add top level include directory.
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../
  ${CMAKE_CURRENT_SOURCE_DIR}/../calico/third_party/apriltags
)

pybind11_add_module(calico src/calico.cpp)
target_link_libraries(
  calico PRIVATE
  Eigen3::Eigen
  absl::status
  absl::statusor
  absl::flat_hash_map
  Ceres::ceres
  ${OpenCV_LIBS}
  ${YAML_CPP_LIBRARIES}
)
target_link_libraries(
  calico PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../libaccelerometer.a
  ${CMAKE_CURRENT_SOURCE_DIR}/../libaccelerometer_models.a
  ${CMAKE_CURRENT_SOURCE_DIR}/../libaccelerometer_cost_functor.a
  ${CMAKE_CURRENT_SOURCE_DIR}/../libapriltags.a
  ${CMAKE_CURRENT_SOURCE_DIR}/../libaprilgrid_detector.a
  ${CMAKE_CURRENT_SOURCE_DIR}/../libbatch_optimizer.a
  ${CMAKE_CURRENT_SOURCE_DIR}/../libcamera.a
  ${CMAKE_CURRENT_SOURCE_DIR}/../libcamera_cost_functor.a
  ${CMAKE_CURRENT_SOURCE_DIR}/../libcamera_models.a
  ${CMAKE_CURRENT_SOURCE_DIR}/../libgyroscope.a
  ${CMAKE_CURRENT_SOURCE_DIR}/../libgyroscope_models.a
  ${CMAKE_CURRENT_SOURCE_DIR}/../libgyroscope_cost_functor.a
  ${CMAKE_CURRENT_SOURCE_DIR}/../libtrajectory.a
  ${CMAKE_CURRENT_SOURCE_DIR}/../libworld_model.a
  ${CMAKE_CURRENT_SOURCE_DIR}/../libcamera_models.a
  ${CMAKE_CURRENT_SOURCE_DIR}/../libcamera.a
)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(calico
                           PRIVATE VERSION_INFO=${CALICO_VERSION_INFO})