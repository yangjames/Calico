#include "calico/sensors/accelerometer_cost_functor.h"

#include "calico/test_utils.h"
#include "Eigen/Dense"
#include "gmock/gmock.h"
#include "gtest/gtest.h"


namespace calico::sensors {
namespace {

struct AccelerometerCostFunctionCreationTestCase {
  std::string test_name;
  AccelerometerIntrinsicsModel accelerometer_model;
  Eigen::Vector3d measurement;
  Eigen::VectorXd intrinsics;
  Pose3d extrinsics;
  double latency;
  Eigen::Vector3d gravity;
};

// Test the creation of accelerometer cost functions and convenience functions.
class AccelerometerCostFunctionCreationTest :
    public ::testing::TestWithParam<AccelerometerCostFunctionCreationTestCase> {
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

TEST_P(AccelerometerCostFunctionCreationTest, Instantiation) {
  AccelerometerCostFunctionCreationTestCase test_case = GetParam();
  std::vector<double*> parameters;
  for (const auto& stamp : timestamps) {
    TrajectoryEvaluationParams segment =
        trajectory_world_sensorrig.GetEvaluationParams(stamp);
    auto* cost_function =
        AccelerometerCostFunctor::CreateCostFunction(
            test_case.measurement, test_case.accelerometer_model,
            test_case.intrinsics, test_case.extrinsics, test_case.latency,
            test_case.gravity, trajectory_world_sensorrig, stamp, parameters);
    ASSERT_NE(cost_function, nullptr);
    delete cost_function;
  }
}

INSTANTIATE_TEST_SUITE_P(
    AccelerometerCostFunctionCreationTests, AccelerometerCostFunctionCreationTest,
    testing::ValuesIn<AccelerometerCostFunctionCreationTestCase>({
        {
          "AccelerometerScaleOnly",
          AccelerometerIntrinsicsModel::kAccelerometerScaleOnly,
          Eigen::Vector3d::Random(),
          Eigen::VectorXd::Random(
              AccelerometerScaleOnlyModel::kNumberOfParameters),
          Pose3d(),
          5,
          Eigen::Vector3d::Random(),
        },
      }),
    [](const testing::TestParamInfo
        <AccelerometerCostFunctionCreationTest::ParamType>& info) {
      return info.param.test_name;
    });

} // namespace
} // namespace calico::sensors
