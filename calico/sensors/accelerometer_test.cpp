#include "calico/sensors/accelerometer.h"

#include "calico/matchers.h"
#include "calico/test_utils.h"
#include "calico/typedefs.h"
#include "Eigen/Dense"
#include "gmock/gmock.h"
#include "gtest/gtest.h"


namespace calico::sensors {
namespace {

struct AccelerometerTestCase {
  std::string test_name;
  AccelerometerIntrinsicsModel accelerometer_model;
  int parameter_size;
};

class AccelerometerTest
  : public ::testing::TestWithParam<AccelerometerTestCase> {
 protected:
  void SetUp() override {
    DefaultSyntheticTest synthetic_test;
    const auto trajectory = synthetic_test.TrajectoryAsMap();
    EXPECT_OK(trajectory_world_sensorrig.AddPoses(trajectory));
    stamps = synthetic_test.TrajectoryMapKeys();
  }
  std::vector<double> stamps;
  Trajectory trajectory_world_sensorrig;
};

TEST_P(AccelerometerTest, SettersAndGetters) {
  const auto params = GetParam();
  Accelerometer accelerometer;
  const Pose3d extrinsics(Eigen::Quaterniond::UnitRandom(),
                          Eigen::Vector3d::Random());
  const double latency = 3;
  Eigen::VectorXd intrinsics(params.parameter_size);
  intrinsics.setRandom();
  accelerometer.SetName(params.test_name);
  EXPECT_OK(accelerometer.SetModel(params.accelerometer_model));
  accelerometer.SetExtrinsics(extrinsics);
  EXPECT_OK(accelerometer.SetLatency(latency));
  EXPECT_OK(accelerometer.SetIntrinsics(intrinsics));
  EXPECT_EQ(accelerometer.GetName(), params.test_name);
  EXPECT_EQ(accelerometer.GetModel(), params.accelerometer_model);
  EXPECT_THAT(accelerometer.GetExtrinsics(), PoseEq(extrinsics));
  EXPECT_THAT(accelerometer.GetIntrinsics(), EigenEq(intrinsics));
  EXPECT_EQ(accelerometer.GetLatency(), latency);
}

TEST_P(AccelerometerTest, AddSingleMeasurementOnlyUniqueAllowed) {
  const AccelerometerMeasurement measurement{
    .measurement = Eigen::Vector3d::Random(),
    .id = {.stamp = 0, .sequence = 0},
  };
  Accelerometer accelerometer;
  EXPECT_EQ(accelerometer.NumberOfMeasurements(), 0);
  EXPECT_OK(accelerometer.AddMeasurement(measurement));
  EXPECT_EQ(accelerometer.NumberOfMeasurements(), 1);
  // Add the same measurement and expect an error.
  EXPECT_EQ(accelerometer.AddMeasurement(measurement).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(accelerometer.NumberOfMeasurements(), 1);
}

TEST_P(AccelerometerTest, AddMultipleMeasurementsOnlyUniqueAllowed) {
  std::vector<AccelerometerMeasurement> measurements {
    {
      .measurement = Eigen::Vector3d::Random(),
      .id = {.stamp = 0, .sequence = 0},
    },
    {
      .measurement = Eigen::Vector3d::Random(),
      .id = {.stamp = 1, .sequence = 1},
    }
  };
  Accelerometer accelerometer;
  EXPECT_OK(accelerometer.AddMeasurements(measurements));
  EXPECT_EQ(accelerometer.NumberOfMeasurements(), measurements.size());
  accelerometer.ClearMeasurements();
  measurements.push_back(measurements.front());
  EXPECT_EQ(accelerometer.AddMeasurements(measurements).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(accelerometer.NumberOfMeasurements(), measurements.size() - 1);
}

TEST_P(AccelerometerTest, RemoveMeasurement) {
  const AccelerometerMeasurement measurement{};
  Accelerometer accelerometer;
  EXPECT_OK(accelerometer.AddMeasurement(measurement));
  EXPECT_EQ(accelerometer.NumberOfMeasurements(), 1);
  EXPECT_OK(accelerometer.RemoveMeasurementById(measurement.id));
  EXPECT_EQ(accelerometer.NumberOfMeasurements(), 0);
  EXPECT_EQ(accelerometer.RemoveMeasurementById(measurement.id).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_P(AccelerometerTest, RemoveMultipleMeasurements) {
  std::vector<AccelerometerMeasurement> measurements {
    {
      .measurement = Eigen::Vector3d::Random(),
      .id = {.stamp = 0, .sequence = 0},
    },
    {
      .measurement = Eigen::Vector3d::Random(),
      .id = {.stamp = 1, .sequence = 1},
    }
  };
  Accelerometer accelerometer;
  EXPECT_OK(accelerometer.AddMeasurements(measurements));
  EXPECT_EQ(accelerometer.NumberOfMeasurements(), measurements.size());

  std::vector<AccelerometerObservationId> ids;
  for (const auto& measurement : measurements) {
    ids.push_back(measurement.id);
  }
  EXPECT_OK(accelerometer.RemoveMeasurementsById(ids));
  EXPECT_EQ(accelerometer.NumberOfMeasurements(), 0);
  EXPECT_EQ(accelerometer.RemoveMeasurementsById(ids).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_P(AccelerometerTest, AddCalibrationParametersToProblem) {
  const auto params = GetParam();
  Eigen::VectorXd intrinsics(params.parameter_size);
  intrinsics.setRandom();
  Pose3d extrinsics(Eigen::Quaterniond::UnitRandom(),
                           Eigen::Vector3d::Random());
  Accelerometer accelerometer;
  EXPECT_OK(accelerometer.SetModel(params.accelerometer_model));
  EXPECT_OK(accelerometer.SetIntrinsics(intrinsics));
  accelerometer.SetExtrinsics(extrinsics);
  ceres::Problem problem;
  ASSERT_OK_AND_ASSIGN(const int num_parameters,
                       accelerometer.AddParametersToProblem(problem));
  EXPECT_EQ(problem.NumParameters(), num_parameters);
}

TEST_P(AccelerometerTest, PerfectDataPerfectResiduals) {
  const auto params = GetParam();
  Eigen::VectorXd intrinsics(params.parameter_size);
  intrinsics.setRandom();
  Pose3d extrinsics(Eigen::Quaterniond::UnitRandom(),
                           Eigen::Vector3d::Random());
  WorldModel world_model;
  Accelerometer accelerometer;
  EXPECT_OK(accelerometer.SetModel(params.accelerometer_model));
  EXPECT_OK(accelerometer.SetIntrinsics(intrinsics));
  accelerometer.SetExtrinsics(extrinsics);
  std::vector<AccelerometerMeasurement> measurements;
  ASSERT_OK_AND_ASSIGN(measurements,
      accelerometer.Project(stamps, trajectory_world_sensorrig, world_model));
  ceres::Problem problem;
  ASSERT_OK_AND_ASSIGN(const int num_residuals,
      accelerometer.AddResidualsToProblem(problem, trajectory_world_sensorrig,
                                      world_model));
  EXPECT_EQ(problem.NumResiduals(), 3 * num_residuals);
  double cost = std::numeric_limits<double>::infinity();
  problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr,
                   nullptr);
  EXPECT_EQ(cost, 0.0);
}

INSTANTIATE_TEST_SUITE_P(
    AccelerometerTests, AccelerometerTest,
    testing::ValuesIn<AccelerometerTestCase>({
        {
          "ScaleOnly",
          AccelerometerIntrinsicsModel::kAccelerometerScaleOnly,
          AccelerometerScaleOnlyModel::kNumberOfParameters,
        },
        {
          "ScaleAndBias",
          AccelerometerIntrinsicsModel::kAccelerometerScaleAndBias,
          AccelerometerScaleAndBiasModel::kNumberOfParameters,
        },
      }),
    [](const testing::TestParamInfo<AccelerometerTest::ParamType>& info
         ) {
      return info.param.test_name;
    });

} // namespace
} // namespace calico::sensors
