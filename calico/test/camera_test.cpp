#include "calico/sensors/camera.h"

#include "calico/matchers.h"
#include "calico/typedefs.h"
#include "calico/sensors/camera_cost_functor.h"
#include "calico/sensors/camera_models.h"
#include "calico/world_model.h"
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
  EXPECT_THAT(camera_.AddMeasurement(measurement),
              StatusCodeIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(camera_.NumberOfMeasurements(), 1);
}

TEST_F(CameraContainerTest, AddMultipleMeasurementsOnlyUniqueAllowed) {
  std::vector<CameraMeasurement> measurements = measurements_;
  camera_.ClearMeasurements();
  EXPECT_EQ(camera_.NumberOfMeasurements(), 0);
  EXPECT_OK(camera_.AddMeasurements(measurements));
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
  EXPECT_THAT(camera_.AddMeasurements(measurements),
              StatusCodeIs(absl::StatusCode::kInvalidArgument));
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

TEST(CameraProjectionTest, LandmarkInView) {
  // Construct a scene where a camera is sitting still, hovering 1m above the origin for 1 second.
  Eigen::Quaterniond q_world_camera(/*w=*/0.0, /*x=*/1.0, /*y=*/0.0, /*z=*/0.0);
  Eigen::Vector3d t_world_camera(0.0, 0.0, 1.0);
  Trajectory trajectory;
  ASSERT_OK(trajectory.FitSpline({
    {0.0, Pose3d(q_world_camera, t_world_camera)},
    {1.0, Pose3d(q_world_camera, t_world_camera)},
  }));
  // Construct a landmark placed at the origin.
  Landmark landmark;
  WorldModel world_model;
  ASSERT_OK(world_model.AddLandmark(&landmark, /*take_ownership=*/false));
  // Construct the camera.
  Camera camera;
  ASSERT_OK(camera.SetModel(CameraIntrinsicsModel::kOpenCv5));
  Eigen::VectorXd intrinsics(OpenCv5Model::kNumberOfParameters);
  intrinsics << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  ASSERT_OK(camera.SetIntrinsics(intrinsics));

  // Project the landmark into the camera.
  ASSERT_OK_AND_ASSIGN(
    const std::vector<CameraMeasurement> measurements,
    camera.Project(std::vector<double>{0.0}, trajectory, world_model)
  );
  ASSERT_EQ(measurements.size(), 1);
}

TEST(CameraProjectionTest, LandmarkOutOfView) {
  // Construct a scene where a camera is sitting still, hovering 1m above the origin for 1 second.
  const Eigen::Quaterniond q_world_camera(/*w=*/0.0, /*x=*/1.0, /*y=*/0.0, /*z=*/0.0);
  const Eigen::Vector3d t_world_camera(0.0, 0.0, 1.0);
  Trajectory trajectory;
  ASSERT_OK(trajectory.FitSpline({
    {0.0, Pose3d(q_world_camera, t_world_camera)},
    {1.0, Pose3d(q_world_camera, t_world_camera)},
  }));
  // Construct a landmark placed behind the camera.
  Landmark landmark{.point = Eigen::Vector3d(0.0, 0.0, 2.0)};
  WorldModel world_model;
  ASSERT_OK(world_model.AddLandmark(&landmark, /*take_ownership=*/false));
  // Construct the camera.
  Camera camera;
  ASSERT_OK(camera.SetModel(CameraIntrinsicsModel::kOpenCv5));
  Eigen::VectorXd intrinsics(OpenCv5Model::kNumberOfParameters);
  intrinsics << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  ASSERT_OK(camera.SetIntrinsics(intrinsics));
  // Project the landmark into the camera.
  ASSERT_OK_AND_ASSIGN(
    const std::vector<CameraMeasurement> measurements,
    camera.Project(std::vector<double>{0.0}, trajectory, world_model)
  );
  ASSERT_EQ(measurements.size(), 0);
}

TEST(CameraProjectionTest, RigidBodyInView) {
  // Construct a scene where a camera is sitting still, hovering 1m above the origin for 1 second.
  const Eigen::Quaterniond q_world_camera(/*w=*/0.0, /*x=*/1.0, /*y=*/0.0, /*z=*/0.0);
  const Eigen::Vector3d t_world_camera(0.0, 0.0, 1.0);
  Trajectory trajectory;
  ASSERT_OK(trajectory.FitSpline({
    {0.0, Pose3d(q_world_camera, t_world_camera)},
    {1.0, Pose3d(q_world_camera, t_world_camera)},
  }));
  // Construct a rigidbody placed at the origin.
  RigidBody rigidbody {
    .model_definition = {
      {0, Eigen::Vector3d(-0.5, -0.5, 0.0)},
      {1, Eigen::Vector3d(-0.5, 0.5, 0.0)},
      {2, Eigen::Vector3d(0.5, 0.5, 0.0)},
      {3, Eigen::Vector3d(0.5, -0.5, 0.0)},
    },
    .id = 0
  };
  WorldModel world_model;
  ASSERT_OK(world_model.AddRigidBody(&rigidbody, /*take_ownership=*/false));
  // Construct the camera.
  Camera camera;
  ASSERT_OK(camera.SetModel(CameraIntrinsicsModel::kOpenCv5));
  Eigen::VectorXd intrinsics(OpenCv5Model::kNumberOfParameters);
  intrinsics << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  ASSERT_OK(camera.SetIntrinsics(intrinsics));
  // Project the landmark into the camera.
  ASSERT_OK_AND_ASSIGN(
    const std::vector<CameraMeasurement> measurements,
    camera.Project(std::vector<double>{0.0}, trajectory, world_model)
  );
  ASSERT_EQ(measurements.size(), 4);
}

TEST(CameraProjectionTest, RigidBodyOutOfView) {
  // Construct a scene where a camera is sitting still, hovering 1m above the origin for 1 second.
  const Eigen::Quaterniond q_world_camera(/*w=*/0.0, /*x=*/1.0, /*y=*/0.0, /*z=*/0.0);
  const Eigen::Vector3d t_world_camera(0.0, 0.0, 1.0);
  Trajectory trajectory;
  ASSERT_OK(trajectory.FitSpline({
    {0.0, Pose3d(q_world_camera, t_world_camera)},
    {1.0, Pose3d(q_world_camera, t_world_camera)},
  }));
  // Construct a rigidbody placed at the origin.
  RigidBody rigidbody {
    .model_definition = {
      {0, Eigen::Vector3d(-0.5, -0.5, 0.0)},
      {1, Eigen::Vector3d(-0.5, 0.5, 0.0)},
      {2, Eigen::Vector3d(0.5, 0.5, 0.0)},
      {3, Eigen::Vector3d(0.5, -0.5, 0.0)},
    },
    .T_world_rigidbody = Pose3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.0, 0.0, 2.0)),
    .id = 0
  };
  WorldModel world_model;
  ASSERT_OK(world_model.AddRigidBody(&rigidbody, /*take_ownership=*/false));
  // Construct the camera.
  Camera camera;
  ASSERT_OK(camera.SetModel(CameraIntrinsicsModel::kOpenCv5));
  Eigen::VectorXd intrinsics(OpenCv5Model::kNumberOfParameters);
  intrinsics << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  ASSERT_OK(camera.SetIntrinsics(intrinsics));
  // Project the landmark into the camera.
  ASSERT_OK_AND_ASSIGN(
    const std::vector<CameraMeasurement> measurements,
    camera.Project(std::vector<double>{0.0}, trajectory, world_model)
  );
  ASSERT_EQ(measurements.size(), 0);
}
} // namespace
} // namespace calico::sensors
