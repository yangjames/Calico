#include "calico/sensors/gyroscope_cost_functor.h"

#include "calico/test_utils.h"
#include "Eigen/Dense"
#include "gmock/gmock.h"
#include "gtest/gtest.h"


namespace calico::sensors {
namespace {

struct GyroscopeCostFunctionCreationTestCase {
  std::string test_name;
  GyroscopeIntrinsicsModel gyroscope_model;
  Eigen::Vector3d measurement;
  double sigma;
  Eigen::VectorXd intrinsics;
  Pose3d extrinsics;
  double latency;
};

// Test the creation of gyroscope cost functions and convenience functions.
class GyroscopeCostFunctionCreationTest :
    public ::testing::TestWithParam<GyroscopeCostFunctionCreationTestCase> {
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

TEST_P(GyroscopeCostFunctionCreationTest, Instantiation) {
  GyroscopeCostFunctionCreationTestCase test_case = GetParam();
  std::vector<double*> parameters;
  for (const auto& stamp : timestamps) {
    TrajectoryEvaluationParams segment =
        trajectory_world_sensorrig.GetEvaluationParams(stamp);
    auto* cost_function =
        GyroscopeCostFunctor::CreateCostFunction(
            test_case.measurement, test_case.sigma, test_case.gyroscope_model,
            test_case.intrinsics, test_case.extrinsics, test_case.latency,
            trajectory_world_sensorrig, stamp, parameters);
    ASSERT_NE(cost_function, nullptr);
    delete cost_function;
  }
}

INSTANTIATE_TEST_SUITE_P(
    GyroscopeCostFunctionCreationTests, GyroscopeCostFunctionCreationTest,
    testing::ValuesIn<GyroscopeCostFunctionCreationTestCase>({
        {
          "ScaleOnly",
          GyroscopeIntrinsicsModel::kGyroscopeScaleOnly,
          Eigen::Vector3d::Random(), 1,
          Eigen::VectorXd::Random(GyroscopeScaleOnlyModel::kNumberOfParameters),
          Pose3d(),
          5,
        },
      }),
    [](const testing::TestParamInfo
        <GyroscopeCostFunctionCreationTest::ParamType>& info) {
      return info.param.test_name;
    });

} // namespace
} // namespace calico::sensors
