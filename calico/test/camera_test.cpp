#include "calico/sensors/camera.h"

#include "calico/matchers.h"
#include "calico/typedefs.h"
#include "calico/sensors/camera_cost_functor.h"
#include "calico/sensors/camera_models.h"
#include "Eigen/Dense"
#include "gmock/gmock.h"
#include "gtest/gtest.h"


namespace calico::sensors {
namespace {

class CameraContainerTest : public ::testing::Test {
 protected:
  const std::string kCameraName = "camera";
  static constexpr CameraIntrinsicsModel kCameraModel =
      CameraIntrinsicsModel::kOpenCv5;
  const Pose3d kExtrinsics = Pose3d(
      Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
  const Eigen::VectorXd kIntrinsics =
      Eigen::VectorXd::Random(OpenCv5Model::kNumberOfParameters);
  static constexpr int kNumFeatures = 10;
  static constexpr int kNumModels = 10;
  static constexpr int kNumImages = 10;

  void SetUp() override {
    for (int image_id = 0; image_id < kNumImages; ++image_id) {
      for (int model_id= 0; model_id < kNumModels; ++model_id) {
        for (int feature_id = 0; feature_id < kNumFeatures; ++feature_id) {
          measurements_.push_back(CameraMeasurement{
              .id = {
                .stamp = static_cast<double>(image_id),
                .image_id = image_id,
                .model_id = model_id,
                .feature_id = feature_id
              }
            });
        }
      }
    }
  }

  Camera camera_;
  std::vector<CameraMeasurement> measurements_;
};

TEST_F(CameraContainerTest, SettersAndGetters) {
  // Pre-assignment.
  EXPECT_THAT(camera_.GetName(), ::testing::IsEmpty());
  EXPECT_EQ(camera_.GetModel(), CameraIntrinsicsModel::kNone);
  EXPECT_THAT(camera_.GetExtrinsics(), PoseEq(Pose3d()));
  EXPECT_THAT(camera_.GetIntrinsics(), EigenEq(Eigen::VectorXd()));
  // Post-assignment.
  camera_.SetName(kCameraName);
  EXPECT_OK(camera_.SetModel(kCameraModel));
  camera_.SetExtrinsics(kExtrinsics);
  EXPECT_OK(camera_.SetIntrinsics(kIntrinsics));
  EXPECT_EQ(camera_.GetName(), kCameraName);
  EXPECT_EQ(camera_.GetModel(), kCameraModel);
  EXPECT_THAT(camera_.GetExtrinsics(), PoseEq(kExtrinsics));
  EXPECT_THAT(camera_.GetIntrinsics(), EigenEq(kIntrinsics));
}

TEST_F(CameraContainerTest, AddSingleMeasurementOnlyUniqueAllowed) {
  const CameraMeasurement measurement{
    .pixel = Eigen::Vector2d::Random(),
    .id = {.image_id = 0, .model_id = 1, .feature_id = 2},
  };
  camera_.ClearMeasurements();
  EXPECT_EQ(camera_.NumberOfMeasurements(), 0);
  EXPECT_OK(camera_.AddMeasurement(measurement));
  EXPECT_EQ(camera_.NumberOfMeasurements(), 1);
  // Add the same measurement and expect an error.
  EXPECT_EQ(camera_.AddMeasurement(measurement).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(camera_.NumberOfMeasurements(), 1);
}

TEST_F(CameraContainerTest, AddMultipleMeasurementsOnlyUniqueAllowed) {
  std::vector<CameraMeasurement> measurements = measurements_;
  camera_.ClearMeasurements();
  EXPECT_EQ(camera_.NumberOfMeasurements(), 0);
  EXPECT_EQ(camera_.AddMeasurements(measurements).code(),
            absl::StatusCode::kOk);
  EXPECT_EQ(camera_.NumberOfMeasurements(), measurements.size());
  const CameraMeasurement redundant_measurement {
    .id = {
      .stamp = 0,
      .image_id = 0,
      .model_id = 0,
      .feature_id = 0
    }
  };
  measurements.push_back(redundant_measurement);
  camera_.ClearMeasurements();
  EXPECT_EQ(camera_.AddMeasurements(measurements).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(camera_.NumberOfMeasurements(), measurements.size() - 1);
}

TEST_F(CameraContainerTest, AddCalibrationParametersToProblem) {
  EXPECT_OK(camera_.SetModel(kCameraModel));
  EXPECT_OK(camera_.SetIntrinsics(kIntrinsics));
  camera_.SetExtrinsics(kExtrinsics);
  ceres::Problem problem;
  ASSERT_OK_AND_ASSIGN(const int num_parameters,
                       camera_.AddParametersToProblem(problem));
  EXPECT_EQ(problem.NumParameters(), num_parameters);
}

} // namespace
} // namespace calico::sensors
