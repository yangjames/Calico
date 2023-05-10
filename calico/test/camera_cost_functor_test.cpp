#include "calico/sensors/camera_cost_functor.h"

#include "calico/test_utils.h"
#include "Eigen/Dense"
#include "gmock/gmock.h"
#include "gtest/gtest.h"


namespace calico::sensors {
namespace {

struct CameraCostFunctionCreationTestCase {
  std::string test_name;
  CameraIntrinsicsModel camera_model;
  Eigen::Vector2d pixel;
  double sigma;
  Eigen::VectorXd intrinsics;
  Pose3d extrinsics;
  double latency;
  Eigen::Vector3d t_model_point;
  Pose3d T_world_model;
};

// Test the creation of camera cost functions and convenience functions.
class CameraCostFunctionCreationTest :
    public ::testing::TestWithParam<CameraCostFunctionCreationTestCase> {
 protected:
  void SetUp() override {
    DefaultSyntheticTest synthetic_test;
    absl::flat_hash_map<double, Pose3d> poses_world_sensorrig =
      synthetic_test.TrajectoryAsMap();
    timestamps = synthetic_test.TrajectoryMapKeys();
    ASSERT_OK(trajectory_world_sensorrig.FitSpline(poses_world_sensorrig));
  }
  std::vector<double> timestamps;
  Trajectory trajectory_world_sensorrig;
};

TEST_P(CameraCostFunctionCreationTest, Instantiation) {
  CameraCostFunctionCreationTestCase test_case = GetParam();
  std::vector<double*> parameters;
  for (const auto& stamp : timestamps) {
    TrajectoryEvaluationParams segment =
        trajectory_world_sensorrig.GetEvaluationParams(stamp);
    auto* cost_function =
        CameraCostFunctor::CreateCostFunction(
            test_case.pixel, test_case.sigma, test_case.camera_model,
            test_case.intrinsics, test_case.extrinsics, test_case.latency,
            test_case.t_model_point, test_case.T_world_model,
            trajectory_world_sensorrig, stamp, parameters);
    ASSERT_NE(cost_function, nullptr);
    delete cost_function;
  }
}

INSTANTIATE_TEST_SUITE_P(
    CameraCostFunctionCreationTests, CameraCostFunctionCreationTest,
    testing::ValuesIn<CameraCostFunctionCreationTestCase>({
        {
          "OpenCv5",
          CameraIntrinsicsModel::kOpenCv5,
          Eigen::Vector2d::Random(), 1,
          Eigen::VectorXd::Random(OpenCv5Model::kNumberOfParameters),
          Pose3d(),
          5,
          Eigen::Vector3d::Random(),
          Pose3d(),
        },
        {
          "KannalaBrandt",
          CameraIntrinsicsModel::kKannalaBrandt,
          Eigen::Vector2d::Random(), 1,
          Eigen::VectorXd::Random(KannalaBrandtModel::kNumberOfParameters),
          Pose3d(),
          5,
          Eigen::Vector3d::Random(),
          Pose3d(),
        },
      }),
    [](const testing::TestParamInfo
        <CameraCostFunctionCreationTest::ParamType>& info) {
      return info.param.test_name;
    });

} // namespace
} // namespace calico::sensors
