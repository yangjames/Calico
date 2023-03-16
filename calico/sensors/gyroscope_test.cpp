#include "calico/sensors/gyroscope.h"

#include "calico/matchers.h"
#include "calico/test_utils.h"
#include "calico/typedefs.h"
#include "Eigen/Dense"
#include "gmock/gmock.h"
#include "gtest/gtest.h"


namespace calico::sensors {
namespace {

struct GyroscopeTestCase {
  std::string test_name;
  GyroscopeIntrinsicsModel gyroscope_model;
  int parameter_size;
};

class GyroscopeTest
  : public ::testing::TestWithParam<GyroscopeTestCase> {
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

TEST_P(GyroscopeTest, SettersAndGetters) {
  const auto params = GetParam();
  Gyroscope gyroscope;
  const Pose3d extrinsics(Eigen::Quaterniond::UnitRandom(),
                          Eigen::Vector3d::Random());
  const double latency = 3;
  Eigen::VectorXd intrinsics(params.parameter_size);
  intrinsics.setRandom();
  gyroscope.SetName(params.test_name);
  EXPECT_OK(gyroscope.SetModel(params.gyroscope_model));
  gyroscope.SetExtrinsics(extrinsics);
  EXPECT_OK(gyroscope.SetLatency(latency));
  EXPECT_OK(gyroscope.SetIntrinsics(intrinsics));
  EXPECT_EQ(gyroscope.GetName(), params.test_name);
  EXPECT_EQ(gyroscope.GetModel(), params.gyroscope_model);
  EXPECT_THAT(gyroscope.GetExtrinsics(), PoseEq(extrinsics));
  EXPECT_THAT(gyroscope.GetIntrinsics(), EigenEq(intrinsics));
  EXPECT_EQ(gyroscope.GetLatency(), latency);
}

TEST_P(GyroscopeTest, AddSingleMeasurementOnlyUniqueAllowed) {
  const GyroscopeMeasurement measurement{
    .measurement = Eigen::Vector3d::Random(),
    .id = {.stamp = 0, .sequence = 0},
  };
  Gyroscope gyroscope;
  EXPECT_EQ(gyroscope.NumberOfMeasurements(), 0);
  EXPECT_OK(gyroscope.AddMeasurement(measurement));
  EXPECT_EQ(gyroscope.NumberOfMeasurements(), 1);
  // Add the same measurement and expect an error.
  EXPECT_EQ(gyroscope.AddMeasurement(measurement).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(gyroscope.NumberOfMeasurements(), 1);
}

TEST_P(GyroscopeTest, AddMultipleMeasurementsOnlyUniqueAllowed) {
  std::vector<GyroscopeMeasurement> measurements {
    {
      .measurement = Eigen::Vector3d::Random(),
      .id = {.stamp = 0, .sequence = 0},
    },
    {
      .measurement = Eigen::Vector3d::Random(),
      .id = {.stamp = 1, .sequence = 1},
    }
  };
  Gyroscope gyroscope;
  EXPECT_OK(gyroscope.AddMeasurements(measurements));
  EXPECT_EQ(gyroscope.NumberOfMeasurements(), measurements.size());
  gyroscope.ClearMeasurements();
  measurements.push_back(measurements.front());
  EXPECT_EQ(gyroscope.AddMeasurements(measurements).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(gyroscope.NumberOfMeasurements(), measurements.size() - 1);
}

TEST_P(GyroscopeTest, RemoveMeasurement) {
  const GyroscopeMeasurement measurement{};
  Gyroscope gyroscope;
  EXPECT_OK(gyroscope.AddMeasurement(measurement));
  EXPECT_EQ(gyroscope.NumberOfMeasurements(), 1);
  EXPECT_OK(gyroscope.RemoveMeasurementById(measurement.id));
  EXPECT_EQ(gyroscope.NumberOfMeasurements(), 0);
  EXPECT_EQ(gyroscope.RemoveMeasurementById(measurement.id).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_P(GyroscopeTest, RemoveMultipleMeasurements) {
  std::vector<GyroscopeMeasurement> measurements {
    {
      .measurement = Eigen::Vector3d::Random(),
      .id = {.stamp = 0, .sequence = 0},
    },
    {
      .measurement = Eigen::Vector3d::Random(),
      .id = {.stamp = 1, .sequence = 1},
    }
  };
  Gyroscope gyroscope;
  EXPECT_OK(gyroscope.AddMeasurements(measurements));
  EXPECT_EQ(gyroscope.NumberOfMeasurements(), measurements.size());

  std::vector<GyroscopeObservationId> ids;
  for (const auto& measurement : measurements) {
    ids.push_back(measurement.id);
  }
  EXPECT_OK(gyroscope.RemoveMeasurementsById(ids));
  EXPECT_EQ(gyroscope.NumberOfMeasurements(), 0);
  EXPECT_EQ(gyroscope.RemoveMeasurementsById(ids).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_P(GyroscopeTest, AddCalibrationParametersToProblem) {
  const auto params = GetParam();
  Eigen::VectorXd intrinsics(params.parameter_size);
  intrinsics.setRandom();
  Pose3d extrinsics(Eigen::Quaterniond::UnitRandom(),
                           Eigen::Vector3d::Random());
  Gyroscope gyroscope;
  EXPECT_OK(gyroscope.SetModel(params.gyroscope_model));
  EXPECT_OK(gyroscope.SetIntrinsics(intrinsics));
  gyroscope.SetExtrinsics(extrinsics);
  ceres::Problem problem;
  ASSERT_OK_AND_ASSIGN(const int num_parameters,
                       gyroscope.AddParametersToProblem(problem));
  EXPECT_EQ(problem.NumParameters(), num_parameters);
}

TEST_P(GyroscopeTest, PerfectDataPerfectResiduals) {
  const auto params = GetParam();
  Eigen::VectorXd intrinsics(params.parameter_size);
  intrinsics.setRandom();
  Pose3d extrinsics(Eigen::Quaterniond::UnitRandom(),
                           Eigen::Vector3d::Random());
  Gyroscope gyroscope;
  EXPECT_OK(gyroscope.SetModel(params.gyroscope_model));
  EXPECT_OK(gyroscope.SetIntrinsics(intrinsics));
  gyroscope.SetExtrinsics(extrinsics);
  std::vector<GyroscopeMeasurement> measurements;
  ASSERT_OK_AND_ASSIGN(measurements,
                       gyroscope.Project(stamps, trajectory_world_sensorrig));
  ceres::Problem problem;
  WorldModel world_model;
  ASSERT_OK_AND_ASSIGN(const int num_residuals,
      gyroscope.AddResidualsToProblem(problem, trajectory_world_sensorrig,
                                      world_model));
  EXPECT_EQ(problem.NumResiduals(), 3 * num_residuals);
  double cost = std::numeric_limits<double>::infinity();
  problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr,
                   nullptr);
  EXPECT_EQ(cost, 0.0);
}

INSTANTIATE_TEST_SUITE_P(
    GyroscopeTests, GyroscopeTest,
    testing::ValuesIn<GyroscopeTestCase>({
        {
          "ScaleOnly",
          GyroscopeIntrinsicsModel::kScaleOnly,
          ScaleOnlyModel::kNumberOfParameters,
        },
        {
          "ScaleAndBias",
          GyroscopeIntrinsicsModel::kScaleAndBias,
          ScaleAndBiasModel::kNumberOfParameters,
        },
      }),
    [](const testing::TestParamInfo<GyroscopeTest::ParamType>& info
         ) {
      return info.param.test_name;
    });

} // namespace
} // namespace calico::sensors
