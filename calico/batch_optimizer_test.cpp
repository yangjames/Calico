#include "calico/batch_optimizer.h"

#include "calico/matchers.h"
#include "calico/sensors/camera_models.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace calico {
namespace {

class BatchOptimizerTest : public ::testing::Test {
 protected:
  static constexpr double kDeg2Rad = M_PI / 180.0;
  // Planar points specs.
  static constexpr double kSamplePlaneWidth = 1.5;
  static constexpr double kSamplePlaneHeight = 1.5;
  static constexpr double kDelta = 0.3;
  static constexpr int kNumXPoints = static_cast<int>(kSamplePlaneWidth / kDelta) + 1;
  static constexpr int kNumYPoints = static_cast<int>(kSamplePlaneHeight / kDelta) + 1;
  std::vector<Eigen::Vector3d> t_world_points;
  // Sensor rig trajectory specs.
  static constexpr int kNumIncrements = 10;
  static constexpr double kMinPos = 0.75;
  static constexpr double kMaxPos = 1.25;
  static constexpr double kMinAngle = -30 * kDeg2Rad;
  static constexpr double kMaxAngle = 30 * kDeg2Rad;
  absl::flat_hash_map<int, Pose3> trajectory_world_sensorrig;

  // Set up a synthetic world with a plane of points measuring 1.5x1.5m on the
  // ground with a camera facing downward at 1m above the center of the plane
  // and making small movements.
  void SetUp() override {
    // Construct sensor rig trajectory.
    int pose_idx = 0;
    Eigen::Matrix3d R_world_sensorrig0;
    R_world_sensorrig0 <<
      1, 0, 0,
      0, -1, 0,
      0, 0, -1;
    const Eigen::Quaterniond q_world_sensorrig0(R_world_sensorrig0);
    const Eigen::Vector3d t_world_sensorrig0 =
      Eigen::Vector3d(kSamplePlaneWidth / 2.0, kSamplePlaneHeight / 2.0, 1.0);
    // Construct angular and linear motion sequences.
    std::vector<double> angle_displacements;
    std::vector<double> position_displacements;
    for (int i = 0; i < kNumIncrements; ++i) {
      const double delta = double(i) / kNumIncrements;
      const double angle = kMinAngle + (kMaxAngle - kMinAngle) * delta;
      const double position = kMinPos + (kMaxPos - kMinPos) * delta;
      angle_displacements.push_back(angle);
      position_displacements.push_back(position);
    }
    for (const Eigen::Vector3d& axis : std::vector{
             Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(),
                 Eigen::Vector3d::UnitZ()}) {
      for (const auto& angle : angle_displacements) {
        const Eigen::Quaterniond q_sensorrig0_sensorrig(
            Eigen::AngleAxisd(angle, axis));
        const Eigen::Quaterniond q_world_sensorrig =
            q_world_sensorrig0 * q_sensorrig0_sensorrig;
        trajectory_world_sensorrig[pose_idx++] =
            Pose3(q_world_sensorrig, t_world_sensorrig0);
      }
      for (const auto& position : position_displacements) {
        const Eigen::Vector3d t_world_sensorrig =
          t_world_sensorrig0 + axis * position;
        trajectory_world_sensorrig[pose_idx++] =
            Pose3(q_world_sensorrig0, t_world_sensorrig);
      }
    }
    
    // Construct planar points.
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
      true_camera_left.Project(trajectory_world_sensorrig, world_model);
  const auto measurements_right =
      true_camera_right.Project(trajectory_world_sensorrig, world_model);

  // Create optimization cameras.
  Eigen::VectorXd initial_intrinsics = 1.01 * true_intrinsics;
  initial_intrinsics.tail(5).setZero();
  const Pose3 initial_extrinsics = true_extrinsics_right;
  
  sensors::Camera* camera_left = new sensors::Camera();
  camera_left->SetName("Left");
  const auto set_left_model_status = camera_left->SetModel(kCameraModel);
  EXPECT_EQ(set_left_model_status.code(), absl::StatusCode::kOk);
  const auto set_left_intrinsics_status =
    camera_left->SetIntrinsics(initial_intrinsics);
  EXPECT_EQ(set_left_intrinsics_status.code(), absl::StatusCode::kOk);
  camera_left->EnableExtrinsicsParameters(false);
  camera_left->EnableIntrinsicsParameters(true);
  const auto set_left_measurements_status =
      camera_left->AddMeasurements(measurements_left);
  EXPECT_EQ(set_left_measurements_status.code(), absl::StatusCode::kOk);
  sensors::Camera* camera_right = new sensors::Camera();
  camera_right->SetName("Right");
  const auto set_right_model_status = camera_right->SetModel(kCameraModel);
  EXPECT_EQ(set_right_model_status.code(), absl::StatusCode::kOk);
  const auto set_right_intrinsics_status =
      camera_right->SetIntrinsics(initial_intrinsics);
  EXPECT_EQ(set_right_intrinsics_status.code(), absl::StatusCode::kOk);
  camera_right->SetExtrinsics(initial_extrinsics);
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
  optimizer.AddTrajectory(trajectory_world_sensorrig);
  const auto summary = optimizer.Optimize();
  EXPECT_EQ(summary.status().code(), absl::StatusCode::kOk);

  // Expect near perfect calibration results due to perfect data.
  constexpr double kSmallNumber = 1e-9;
  EXPECT_EQ((*summary).termination_type, ceres::CONVERGENCE);
  EXPECT_LT((*summary).final_cost, kSmallNumber);
  EXPECT_TRUE(true_intrinsics.isApprox(camera_left->GetIntrinsics(),
                                       kSmallNumber));
  EXPECT_TRUE(true_intrinsics.isApprox(camera_right->GetIntrinsics(),
                                       kSmallNumber));
  EXPECT_THAT(true_extrinsics_right, PoseIsApprox(camera_right->GetExtrinsics(),
                                                  kSmallNumber));
}

} // namespace
} // namespace calico
