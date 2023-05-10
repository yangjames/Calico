#include "calico/sensors/gyroscope.h"

#include "calico/geometry.h"
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
    EXPECT_OK(trajectory_world_sensorrig.FitSpline(trajectory));
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

TEST_P(GyroscopeTest, AnalyticallyVsNumericallyDiffedKinematicsMatch) {
  constexpr double kSmallError = 1e-5;
  const Pose3d T_sensorrig_gyro(Eigen::Quaterniond::UnitRandom(),
                                Eigen::Vector3d::Zero());
  const Eigen::Matrix3d R_sensorrig_gyro =
      T_sensorrig_gyro.rotation().toRotationMatrix();
  const Eigen::Vector3d t_sensorrig_gyro = T_sensorrig_gyro.translation();

  WorldModel world_model;
  Gyroscope gyroscope;
  ASSERT_OK(gyroscope.SetModel(GyroscopeIntrinsicsModel::kGyroscopeScaleOnly));
  Eigen::VectorXd intrinsics(GyroscopeScaleOnlyModel::kNumberOfParameters);
  intrinsics.setOnes();
  ASSERT_OK(gyroscope.SetIntrinsics(intrinsics));
  gyroscope.SetExtrinsics(T_sensorrig_gyro);
  std::vector<GyroscopeMeasurement> measurements;
  ASSERT_OK_AND_ASSIGN(measurements,
      gyroscope.Project(stamps, trajectory_world_sensorrig, world_model));

  const double dt = 1e-5;
  const double dt_inv = 1.0 / dt;

  for (int i = 0; i < stamps.size() - 2; ++i) {
    const double stamp_i = stamps.at(i);
    const double stamp_ii = stamp_i + dt;
    const Eigen::Vector<double, 6> pose_vector_i =
        trajectory_world_sensorrig.spline().Interpolate(
            {stamp_i}, /*derivative=*/0).value()[0];
    const Eigen::Vector<double, 6> pose_vector_ii =
        trajectory_world_sensorrig.spline().Interpolate(
            {stamp_ii}, /*derivative=*/0).value()[0];
    const Eigen::Vector<double, 6> pose_dot_vector_i =
        trajectory_world_sensorrig.spline().Interpolate(
            {stamp_i}, /*derivative=*/1).value()[0];
    const Eigen::Vector3d phi_sensorrig_world_i = -pose_vector_i.head(3);
    const Eigen::Vector3d phi_sensorrig_world_ii = -pose_vector_ii.head(3);

    const Eigen::Matrix3d R_sensorrig_world_i = ExpSO3(phi_sensorrig_world_i);
    const Eigen::Matrix3d R_sensorrig_world_ii = ExpSO3(phi_sensorrig_world_ii);

    const Eigen::Matrix3d dR =
        R_sensorrig_world_ii * R_sensorrig_world_i.transpose();
    const Eigen::Vector3d omega_sensorrig_world = dt_inv * LnSO3(dR);
    const Eigen::Vector3d omega_sensorrig_world_sensorrig = -omega_sensorrig_world;
    const Eigen::Vector3d omega_gyroscope_world =
        R_sensorrig_gyro.transpose() * omega_sensorrig_world_sensorrig;

    const auto measurement = measurements.at(i);
    const Eigen::Vector3d error = omega_gyroscope_world - measurement.measurement;
    ASSERT_LT(error.squaredNorm(), kSmallError);
  }
}

TEST_P(GyroscopeTest, PerfectDataPerfectResiduals) {
  const auto params = GetParam();
  Eigen::VectorXd intrinsics(params.parameter_size);
  intrinsics.setRandom();
  Pose3d extrinsics(Eigen::Quaterniond::UnitRandom(),
                           Eigen::Vector3d::Random());
  WorldModel world_model;
  Gyroscope gyroscope;
  EXPECT_OK(gyroscope.SetModel(params.gyroscope_model));
  EXPECT_OK(gyroscope.SetIntrinsics(intrinsics));
  gyroscope.SetExtrinsics(extrinsics);
  std::vector<GyroscopeMeasurement> measurements;
  ASSERT_OK_AND_ASSIGN(measurements,
      gyroscope.Project(stamps, trajectory_world_sensorrig, world_model));
  ASSERT_OK(gyroscope.AddMeasurements(measurements));
  ceres::Problem problem;
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
          "GyroscopeScaleOnly",
          GyroscopeIntrinsicsModel::kGyroscopeScaleOnly,
          GyroscopeScaleOnlyModel::kNumberOfParameters,
        },
        {
          "GyroscopeScaleAndBias",
          GyroscopeIntrinsicsModel::kGyroscopeScaleAndBias,
          GyroscopeScaleAndBiasModel::kNumberOfParameters,
        },
      }),
    [](const testing::TestParamInfo<GyroscopeTest::ParamType>& info
         ) {
      return info.param.test_name;
    });

} // namespace
} // namespace calico::sensors
