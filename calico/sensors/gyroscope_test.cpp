#include "calico/sensors/gyroscope.h"

#include "calico/matchers.h"
#include "calico/typedefs.h"
#include "calico/sensors/gyroscope_cost_functor.h"
#include "calico/sensors/gyroscope_models.h"
#include "Eigen/Dense"
#include "gmock/gmock.h"
#include "gtest/gtest.h"


namespace calico::sensors {
namespace {


class GyroscopeContainerTest : public ::testing::Test {
 protected:
  static constexpr absl::string_view kGyroscopeName = "gyroscope";
  static constexpr GyroscopeIntrinsicsModel kGyroscopeModel =
      GyroscopeIntrinsicsModel::kScaleOnly;
  static constexpr int kNumSamples = 10;
  static constexpr double kLatency = 0.5;
  const Pose3d kExtrinsics = Pose3d(
      Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
  Eigen::VectorXd kIntrinsics;
  void SetUp() override {
    kIntrinsics.resize(ScaleOnlyModel::kNumberOfParameters);
    kIntrinsics.setRandom();
    for (int sample_id = 0; sample_id < kNumSamples; ++sample_id) {
      measurements_.push_back(
          GyroscopeMeasurement{
            .measurement = Eigen::Vector3d::Random(),
            .id = {
              .stamp = static_cast<double>(sample_id),
              .sequence = sample_id
            }
          });
    }
  }
  Gyroscope gyroscope_;
  std::vector<GyroscopeMeasurement> measurements_;
};

TEST_F(GyroscopeContainerTest, SettersAndGetters) {
  // Pre-assignment.
  EXPECT_THAT(gyroscope_.GetName(), ::testing::IsEmpty());
  EXPECT_EQ(gyroscope_.GetModel(), GyroscopeIntrinsicsModel::kNone);
  EXPECT_THAT(gyroscope_.GetExtrinsics(), PoseEq(Pose3d()));
  EXPECT_THAT(gyroscope_.GetIntrinsics(), EigenEq(Eigen::VectorXd()));
  EXPECT_EQ(gyroscope_.GetLatency(), 0);
  // Post-assignment.
  gyroscope_.SetName(kGyroscopeName);
  EXPECT_OK(gyroscope_.SetModel(kGyroscopeModel));
  gyroscope_.SetExtrinsics(kExtrinsics);
  EXPECT_OK(gyroscope_.SetLatency(kLatency));
  EXPECT_OK(gyroscope_.SetIntrinsics(kIntrinsics));
  EXPECT_EQ(gyroscope_.GetName(), kGyroscopeName);
  EXPECT_EQ(gyroscope_.GetModel(), kGyroscopeModel);
  EXPECT_THAT(gyroscope_.GetExtrinsics(), PoseEq(kExtrinsics));
  EXPECT_THAT(gyroscope_.GetIntrinsics(), EigenEq(kIntrinsics));
  EXPECT_EQ(gyroscope_.GetLatency(), kLatency);
}

TEST_F(GyroscopeContainerTest, AddSingleMeasurementOnlyUniqueAllowed) {
  const GyroscopeMeasurement measurement{
    .measurement = Eigen::Vector3d::Random(),
    .id = {.stamp = 0, .sequence = 0},
  };
  gyroscope_.ClearMeasurements();
  EXPECT_EQ(gyroscope_.NumberOfMeasurements(), 0);
  EXPECT_OK(gyroscope_.AddMeasurement(measurement));
  EXPECT_EQ(gyroscope_.NumberOfMeasurements(), 1);
  // Add the same measurement and expect an error.
  EXPECT_EQ(gyroscope_.AddMeasurement(measurement).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(gyroscope_.NumberOfMeasurements(), 1);
}

TEST_F(GyroscopeContainerTest, AddMultipleMeasurementsOnlyUniqueAllowed) {
  std::vector<GyroscopeMeasurement> measurements = measurements_;
  gyroscope_.ClearMeasurements();
  EXPECT_EQ(gyroscope_.NumberOfMeasurements(), 0);
  EXPECT_EQ(gyroscope_.AddMeasurements(measurements).code(),
            absl::StatusCode::kOk);
  EXPECT_EQ(gyroscope_.NumberOfMeasurements(), measurements.size());
  const GyroscopeMeasurement redundant_measurement {
    .id = {
      .stamp = 0,
      .sequence = 0,
    }
  };
  measurements.push_back(redundant_measurement);
  gyroscope_.ClearMeasurements();
  EXPECT_EQ(gyroscope_.AddMeasurements(measurements).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(gyroscope_.NumberOfMeasurements(), measurements.size() - 1);
}

TEST_F(GyroscopeContainerTest, RemoveMeasurement) {
  const GyroscopeMeasurement measurement{};
  gyroscope_.ClearMeasurements();
  EXPECT_OK(gyroscope_.AddMeasurement(measurement));
  EXPECT_EQ(gyroscope_.NumberOfMeasurements(), 1);
  EXPECT_OK(gyroscope_.RemoveMeasurementById(measurement.id));
  EXPECT_EQ(gyroscope_.NumberOfMeasurements(), 0);
  EXPECT_EQ(gyroscope_.RemoveMeasurementById(measurement.id).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(GyroscopeContainerTest, RemoveMultipleMeasurements) {
  std::vector<GyroscopeMeasurement> measurements = measurements_;
  gyroscope_.ClearMeasurements();
  EXPECT_EQ(gyroscope_.NumberOfMeasurements(), 0);
  EXPECT_OK(gyroscope_.AddMeasurements(measurements));
  EXPECT_EQ(gyroscope_.NumberOfMeasurements(), measurements.size());

  std::vector<GyroscopeObservationId> ids;
  for (const auto& measurement : measurements) {
    ids.push_back(measurement.id);
  }
  EXPECT_OK(gyroscope_.RemoveMeasurementsById(ids));
  EXPECT_EQ(gyroscope_.NumberOfMeasurements(), 0);
  EXPECT_EQ(gyroscope_.RemoveMeasurementsById(ids).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(GyroscopeContainerTest, AddCalibrationParametersToProblem) {
  EXPECT_OK(gyroscope_.SetModel(kGyroscopeModel));
  EXPECT_OK(gyroscope_.SetIntrinsics(kIntrinsics));
  gyroscope_.SetExtrinsics(kExtrinsics);
  ceres::Problem problem;
  ASSERT_OK_AND_ASSIGN(const int num_parameters,
                       gyroscope_.AddParametersToProblem(problem));
  EXPECT_EQ(problem.NumParameters(), num_parameters);
}

} // namespace
} // namespace calico::sensors
