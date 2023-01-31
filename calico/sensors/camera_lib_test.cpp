#include "calico/typedefs.h"
#include "calico/sensors/camera.h"
#include "calico/sensors/camera_models.h"

#include <ranges>

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

// Creation of camera models.
struct CameraModelCreationTestCase {
  std::string test_name;
  CameraIntrinsicsModel camera_model;
  int parameter_size;
};

using CameraModelCreationTest =
    ::testing::TestWithParam<CameraModelCreationTestCase>;

TEST_P(CameraModelCreationTest, TestModelGetters) {
  const CameraModelCreationTestCase& test_case = GetParam();
  const auto camera_model =
      CameraModel::Create(test_case.camera_model);
  EXPECT_NE(camera_model, nullptr);
  EXPECT_EQ(camera_model->GetType(), test_case.camera_model);
  EXPECT_EQ(camera_model->NumberOfParameters(),
            test_case.parameter_size);
}

INSTANTIATE_TEST_SUITE_P(
    CameraModelCreationTests, CameraModelCreationTest,
    testing::ValuesIn<CameraModelCreationTestCase>({
        {
          "OpenCv5",
          CameraIntrinsicsModel::kOpenCv5,
          OpenCv5Model::kNumberOfParameters
        },
      }),
    [](const testing::TestParamInfo<CameraModelCreationTest::ParamType>& info) {
      return info.param.test_name;
    });


class CameraTest : public ::testing::Test {
 protected:
  static constexpr absl::string_view kCameraName = "camera";
  static constexpr int kImageWidth = 1280;
  static constexpr int kImageHeight = 800;
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

TEST_F(CameraTest, SettersAndGetters) {
  // Pre-assignment.
  EXPECT_THAT(camera_.GetName(), ::testing::IsEmpty());
  EXPECT_EQ(camera_.GetCameraModel(), CameraIntrinsicsModel::kNone);
  EXPECT_THAT(camera_.GetExtrinsics(), PoseEq(Pose3()));
  EXPECT_THAT(camera_.GetIntrinsics(), EigenEq(Eigen::VectorXd()));
  // Post-assignment.
  camera_.SetName(kCameraName);
  EXPECT_EQ(camera_.SetCameraModel(kCameraModel).code(), absl::StatusCode::kOk);
  camera_.SetExtrinsics(kExtrinsics);
  const absl::Status set_intrinsics_status = camera_.SetIntrinsics(kIntrinsics);
  EXPECT_EQ(set_intrinsics_status.code(), absl::StatusCode::kOk)
    << set_intrinsics_status;
  EXPECT_EQ(camera_.GetName(), kCameraName);
  EXPECT_EQ(camera_.GetCameraModel(), kCameraModel);
  EXPECT_THAT(camera_.GetExtrinsics(), PoseEq(kExtrinsics));
  EXPECT_THAT(camera_.GetIntrinsics(), EigenEq(kIntrinsics));
}

TEST_F(CameraTest, AddSingleMeasurementOnlyUniqueAllowed) {
  const CameraMeasurement measurement{
    .pixel = Eigen::Vector2d::Random(),
    .id = {.image_id = 0, .model_id = 1, .feature_id = 2},
    .stamp = 123.456
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

TEST_F(CameraTest, AddMultipleMeasurementsOnlyUniqueAllowed) {
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

TEST_F(CameraTest, RemoveMeasurement) {
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

TEST_F(CameraTest, RemoveMultipleMeasurements) {
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

} // namespace
} // namespace calico::sensors
