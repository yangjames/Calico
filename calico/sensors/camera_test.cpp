#include "calico/typedefs.h"
#include "calico/sensors/camera.h"
#include "calico/sensors/camera_cost_functor.h"
#include "calico/sensors/camera_models.h"

#include "Eigen/Dense"
#include "gmock/gmock.h"
#include "gtest/gtest.h"


namespace calico::sensors {
namespace {

MATCHER_P(PoseEq, expected_pose, "") {
  return (arg.rotation().isApprox(expected_pose.rotation()) &&
          arg.translation().isApprox(expected_pose.translation()));
}

MATCHER_P(EigenEq, expected_vector, "") {
  return (arg.isApprox(expected_vector));
}

MATCHER_P(ImageSizeEq, expected_image_size, "") {
  return (arg.width == expected_image_size.width &&
          arg.height == expected_image_size.height);
}

class CameraContainerTest : public ::testing::Test {
 protected:
  static constexpr absl::string_view kCameraName = "camera";
  const ImageSize kImageSize { .width = 1280, .height = 800, };
  static constexpr CameraIntrinsicsModel kCameraModel =
      CameraIntrinsicsModel::kOpenCv5;
  const Pose3 kExtrinsics = Pose3(
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
  EXPECT_THAT(camera_.GetExtrinsics(), PoseEq(Pose3()));
  EXPECT_THAT(camera_.GetIntrinsics(), EigenEq(Eigen::VectorXd()));
  EXPECT_THAT(camera_.GetImageSize(), ImageSizeEq(ImageSize()));
  // Post-assignment.
  camera_.SetName(kCameraName);
  EXPECT_EQ(camera_.SetModel(kCameraModel).code(), absl::StatusCode::kOk);
  camera_.SetExtrinsics(kExtrinsics);
  const absl::Status set_intrinsics_status = camera_.SetIntrinsics(kIntrinsics);
  EXPECT_EQ(set_intrinsics_status.code(), absl::StatusCode::kOk)
    << set_intrinsics_status;
  EXPECT_EQ(camera_.SetImageSize(kImageSize).code(), absl::StatusCode::kOk);
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
  EXPECT_EQ(camera_.AddMeasurement(measurement).code(), absl::StatusCode::kOk);
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

TEST_F(CameraContainerTest, RemoveMeasurement) {
  const CameraMeasurement measurement{};
  camera_.ClearMeasurements();
  EXPECT_EQ(camera_.AddMeasurement(measurement).code(), absl::StatusCode::kOk);
  EXPECT_EQ(camera_.NumberOfMeasurements(), 1);
  EXPECT_EQ(camera_.RemoveMeasurementById(measurement.id).code(),
            absl::StatusCode::kOk);
  EXPECT_EQ(camera_.NumberOfMeasurements(), 0);
  EXPECT_EQ(camera_.RemoveMeasurementById(measurement.id).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(CameraContainerTest, RemoveMultipleMeasurements) {
  std::vector<CameraMeasurement> measurements = measurements_;
  camera_.ClearMeasurements();
  EXPECT_EQ(camera_.NumberOfMeasurements(), 0);
  EXPECT_EQ(camera_.AddMeasurements(measurements).code(),
            absl::StatusCode::kOk);
  EXPECT_EQ(camera_.NumberOfMeasurements(), measurements.size());

  std::vector<ObservationId> ids;
  for (const auto& measurement : measurements) {
    ids.push_back(measurement.id);
  }
  EXPECT_EQ(camera_.RemoveMeasurementsById(ids).code(), absl::StatusCode::kOk);
  EXPECT_EQ(camera_.NumberOfMeasurements(), 0);
  EXPECT_EQ(camera_.RemoveMeasurementsById(ids).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(CameraContainerTest, AddCalibrationParametersToProblem) {
  EXPECT_EQ(camera_.SetModel(kCameraModel).code(), absl::StatusCode::kOk);
  EXPECT_EQ(camera_.SetIntrinsics(kIntrinsics).code(), absl::StatusCode::kOk);
  camera_.SetExtrinsics(kExtrinsics);
  ceres::Problem problem;
  const absl::StatusOr<int> num_parameters =
    camera_.AddParametersToProblem(problem);
  ASSERT_EQ(num_parameters.status().code(), absl::StatusCode::kOk);
  EXPECT_EQ(problem.NumParameters(), *num_parameters);
}

} // namespace
} // namespace calico::sensors
