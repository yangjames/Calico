#include "calico/sensors/imager_cost_functor.h"

#include <string>
#include <vector>

#include "Eigen/Dense"
#include "calico/test_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace calico::sensors {
namespace {

struct ImagerCostFunctionCreationTestCase {
  std::string test_name;
  CameraIntrinsicsModel camera_model;
  Eigen::Vector2d pixel;
  double sigma;
  Eigen::VectorXd intrinsics;
  Pose3d imager_extrinsics;
  Pose3d sensor_extrinsics;
  double latency;
  Eigen::Vector3d t_model_point;
  Pose3d T_world_model;
};

class ImagerCostFunctionCreationTest
    : public ::testing::TestWithParam<ImagerCostFunctionCreationTestCase> {
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

TEST_P(ImagerCostFunctionCreationTest, Instantiation) {
  ImagerCostFunctionCreationTestCase test_case = GetParam();
  std::vector<double*> parameters;
  for (const auto& stamp : timestamps) {
    auto* cost_function = ImagerCostFunctor::CreateCostFunction(
        test_case.pixel, test_case.sigma, test_case.camera_model,
        test_case.intrinsics, test_case.imager_extrinsics,
        test_case.sensor_extrinsics, test_case.latency, test_case.t_model_point,
        test_case.T_world_model, trajectory_world_sensorrig, stamp, parameters);
    ASSERT_NE(cost_function, nullptr);
    delete cost_function;
  }
}

INSTANTIATE_TEST_SUITE_P(
    ImagerCostFunctionCreationTests, ImagerCostFunctionCreationTest,
    testing::ValuesIn<ImagerCostFunctionCreationTestCase>({
        {
            "OpenCv5",
            CameraIntrinsicsModel::kOpenCv5,
            Eigen::Vector2d::Random(),
            /*sigma=*/1.0,
            Eigen::VectorXd::Random(OpenCv5Model::kNumberOfParameters),
            /*imager_extrinsics=*/Pose3d(),
            /*sensor_extrinsics=*/Pose3d(),
            /*latency=*/0.0,
            /*t_model_point=*/Eigen::Vector3d::Random(),
            /*T_world_model=*/Pose3d(),
        },
        {
            "KannalaBrandt",
            CameraIntrinsicsModel::kKannalaBrandt,
            Eigen::Vector2d::Random(),
            /*sigma=*/1.0,
            Eigen::VectorXd::Random(KannalaBrandtModel::kNumberOfParameters),
            /*imager_extrinsics=*/Pose3d(),
            /*sensor_extrinsics=*/Pose3d(),
            /*latency=*/0.0,
            /*t_model_point=*/Eigen::Vector3d::Random(),
            /*T_world_model=*/Pose3d(),
        },
    }),
    [](const testing::TestParamInfo<ImagerCostFunctionCreationTest::ParamType>&
           info) { return info.param.test_name; });  // NOLINT

}  // namespace
}  // namespace calico::sensors
