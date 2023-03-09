#include "calico/sensors/camera_cost_functor.h"

#include "Eigen/Dense"
#include "gmock/gmock.h"
#include "gtest/gtest.h"


namespace calico::sensors {
namespace {

// Test the creation of camera cost functions and convenience functions.
class CameraCostFunctionCreationFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    DefaultSyntheticTest synthetic_test;
    synthetic_test
  }
};

struct CameraCostFunctionCreationTestCase {
  std::string test_name;
  CameraIntrinsicsModel camera_model;
  Eigen::Vector2d pixel;
  Eigen::VectorXd intrinsics;
  Pose3d extrinsics;
  Eigen::Vector3d t_model_point;
  Pose3d T_world_model;
  Pose3d T_world_sensorrig;
};
using CameraCostFunctionCreationTest =
  ::testing::TestWithParam<CameraCostFunctionCreationTestCase>;

TEST_P(CameraCostFunctionCreationTest, Instantiation) {
  CameraCostFunctionCreationTestCase test_case = GetParam();
  std::vector<double*> parameters;
  auto* cost_function =
    CameraCostFunctor::CreateCostFunction(
        test_case.pixel, test_case.camera_model, test_case.intrinsics,
        test_case.extrinsics, test_case.t_model_point, test_case.T_world_model,
        test_case.T_world_sensorrig, parameters);
  ASSERT_NE(cost_function, nullptr);
  delete cost_function;
}

INSTANTIATE_TEST_SUITE_P(
    CameraCostFunctionCreationTests, CameraCostFunctionCreationTest,
    testing::ValuesIn<CameraCostFunctionCreationTestCase>({
        {
          "OpenCv5",
          CameraIntrinsicsModel::kOpenCv5,
          Eigen::Vector2d::Random(),
          Eigen::VectorXd::Random(OpenCv5Model::kNumberOfParameters),
          Pose3d(),
          Eigen::Vector3d::Random(),
          Pose3d(),
          Pose3d(),
        },
      }),
    [](const testing::TestParamInfo
        <CameraCostFunctionCreationTest::ParamType>& info) {
      return info.param.test_name;
    });

} // namespace
} // namespace calico::sensors
