#include "calico/batch_optimizer.h"

#include "calico/sensors/camera_models.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace calico {
namespace {

class BatchOptimizerTest : public ::testing::Test {
 protected:
  static constexpr double kSamplePlaneWidth = 1.5;
  static constexpr double kSamplePlaneHeight = 1.5;
  static constexpr double kDelta = 0.025;
  static constexpr int kNumXPoints = static_cast<int>(kSamplePlaneWidth / kDelta) + 1;
  static constexpr int kNumYPoints = static_cast<int>(kSamplePlaneHeight / kDelta) + 1;
  Eigen::Matrix3d R_world_sensorrig;
  Eigen::Vector3d t_world_sensorrig;
  std::vector<Eigen::Vector3d> t_world_points;
  // Set up a synthetic world with a plane of points measuring 1.5x1.5m on the
  // ground with a camera facing downward at 1m above the center of the plane.
  void SetUp() override {
    R_world_sensorrig <<
      1, 0, 0,
      0, -1, 0,
      0, 0, -1;
    t_world_sensorrig = Eigen::Vector3d(kSamplePlaneWidth / 2.0,
                                        kSamplePlaneHeight / 2.0, 1.0);
    for (int i = 0; i < kNumXPoints; ++i) {
      for (int j = 0; j < kNumYPoints; ++j) {
        const double x = i * kDelta;
        const double y = j * kDelta;
        t_world_points.push_back(Eigen::Vector3d(x, y, 0.0));
      }
    }
  }
};

TEST_F(BatchOptimizerTest, OpenCv5ToyStereoCalibration) {
  // Intrinsics and extrinsics for our test camera.
  const sensors::CameraIntrinsicsModel kCameraModel =
      sensors::CameraIntrinsicsModel::kOpenCv5;
  constexpr double kStereoRotationAngle = 5.0 * M_PI / 180.0;
  constexpr double kStereoBaseline = 0.08;
  Eigen::VectorXd true_intrinsics(sensors::OpenCv5Model::kNumberOfParameters);
  true_intrinsics <<
    785, 640, 400, -3.149e-1, 1.069e-1, 1.616e-4, 1.141e-4, -1.853e-2;
  Pose3 true_extrinsics_left;
  Pose3 true_extrinsics_right;
  true_extrinsics_right.rotation() =
      Eigen::Quaterniond(
          Eigen::AngleAxisd(
              kStereoRotationAngle, Eigen::Vector3d::Random().normalized()));
  true_extrinsics_right.translation() =
      kStereoBaseline * Eigen::Vector3d::Random();

  // Ground truth cameras to be used for synthetic measurement generation.
  sensors::Camera true_camera_left;
  const auto set_true_left_model_status =
      true_camera_left.SetModel(kCameraModel);
  EXPECT_EQ(set_true_left_model_status.code(), absl::StatusCode::kOk);
  const auto set_true_left_intrinsics_status =
      true_camera_left.SetIntrinsics(true_intrinsics);
  EXPECT_EQ(set_true_left_intrinsics_status.code(), absl::StatusCode::kOk);
  true_camera_left.SetExtrinsics(true_extrinsics_left);
  sensors::Camera true_camera_right;
  const auto set_true_right_model_status =
      true_camera_right.SetModel(kCameraModel);
  EXPECT_EQ(set_true_right_model_status.code(), absl::StatusCode::kOk);
  const auto set_true_right_intrinsics_status =
      true_camera_right.SetIntrinsics(true_intrinsics);
  EXPECT_EQ(set_true_right_intrinsics_status.code(), absl::StatusCode::kOk);
  true_camera_right.SetExtrinsics(true_extrinsics_right);

  // "Trajectory" of the rigid body, just one stationary pose.
  absl::flat_hash_map<int, Pose3> body_trajectory {
      {1, Pose3(Eigen::Quaterniond(R_world_sensorrig), t_world_sensorrig)},
  };

  // World model consisting of a single planar object.
  RigidBody planar_target{
    .world_pose_is_constant = true,
    .model_definition_is_constant = true,
  };
  for (int i = 0; i < t_world_points.size(); ++i) {
    planar_target.model_definition[i] = t_world_points[i];
  }
  WorldModel world_model;
  const auto add_rigidbody_status = world_model.AddRigidBody(planar_target);
  EXPECT_EQ(add_rigidbody_status.code(), absl::StatusCode::kOk);

  // Generate measurements.
  const auto measurements_left =
      true_camera_left.Project(body_trajectory, world_model);
  const auto measurements_right =
      true_camera_right.Project(body_trajectory, world_model);

  // Create optimization cameras.
  sensors::Camera* camera_left = new sensors::Camera();
  const auto set_left_model_status = camera_left->SetModel(kCameraModel);
  EXPECT_EQ(set_left_model_status.code(), absl::StatusCode::kOk);
  camera_left->EnableExtrinsicsParameters(false);
  camera_left->EnableIntrinsicsParameters(true);
  const auto set_left_measurements_status =
      camera_left->AddMeasurements(measurements_left);
  EXPECT_EQ(set_left_measurements_status.code(), absl::StatusCode::kOk);
  sensors::Camera* camera_right = new sensors::Camera();
  const auto set_right_model_status = camera_right->SetModel(kCameraModel);
  EXPECT_EQ(set_right_model_status.code(), absl::StatusCode::kOk);
  camera_right->EnableExtrinsicsParameters(true);
  camera_right->EnableIntrinsicsParameters(true);
  const auto set_right_measurements_status =
      camera_right->AddMeasurements(measurements_right);
  EXPECT_EQ(set_right_measurements_status.code(), absl::StatusCode::kOk);

  // Construct optimization problem and optimize.
  BatchOptimizer optimizer;
  optimizer.AddSensor(camera_left);
  optimizer.AddSensor(camera_right);
  optimizer.AddWorldModel(world_model);
  optimizer.AddTrajectory(body_trajectory);
  const auto summary = optimizer.Optimize();
  EXPECT_EQ(summary.status().code(), absl::StatusCode::kOk);
  std::cout << (*summary).FullReport() << std::endl;
}

} // namespace
} // namespace calico
