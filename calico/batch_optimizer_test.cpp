#include "calico/batch_optimizer.h"

#include "calico/matchers.h"
#include "calico/test_utils.h"
#include "calico/sensors/camera_models.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace calico {
namespace {

class BatchOptimizerTest : public ::testing::Test {
 protected:
  absl::flat_hash_map<double, Pose3d> poses_world_sensorrig;
  std::vector<Eigen::Vector3d> t_world_points;
  std::vector<double> stamps;

  void SetUp() override {
    DefaultSyntheticTest testing_fixture;
    poses_world_sensorrig = testing_fixture.TrajectoryAsMap();
    t_world_points = testing_fixture.WorldPoints();
    stamps = testing_fixture.TrajectoryMapKeys();
  }
};

TEST_F(BatchOptimizerTest, OpenCv5ToyStereoCalibration) {
  // Intrinsics and extrinsics for our test camera.
  const sensors::CameraIntrinsicsModel kCameraModel =
      sensors::CameraIntrinsicsModel::kOpenCv5;
  constexpr double kStereoRotationAngle = 5.0 * M_PI / 180.0;
  constexpr double kStereoBaseline = 0.08;
  constexpr double kLatency = 0.01;
  Eigen::VectorXd true_intrinsics(sensors::OpenCv5Model::kNumberOfParameters);
  true_intrinsics <<
    785, 640, 400, -3.149e-1, 1.069e-1, 1.616e-4, 1.141e-4, -1.853e-2;
  Pose3d true_extrinsics_left;
  Pose3d true_extrinsics_right;
  true_extrinsics_right.rotation() =
      Eigen::Quaterniond(
          Eigen::AngleAxisd(
              kStereoRotationAngle, Eigen::Vector3d::Random().normalized()));
  true_extrinsics_right.translation() =
      kStereoBaseline * Eigen::Vector3d::Random();

  // Ground truth cameras to be used for synthetic measurement generation.
  sensors::Camera true_camera_left;
  EXPECT_OK(true_camera_left.SetModel(kCameraModel));
  EXPECT_OK(true_camera_left.SetIntrinsics(true_intrinsics));
  true_camera_left.SetExtrinsics(true_extrinsics_left);
  sensors::Camera true_camera_right;
  EXPECT_OK(true_camera_right.SetModel(kCameraModel));
  EXPECT_OK(true_camera_right.SetIntrinsics(true_intrinsics));
  true_camera_right.SetExtrinsics(true_extrinsics_right);
  EXPECT_OK(true_camera_right.SetLatency(kLatency));

  // World model consisting of a single planar object.
  RigidBody planar_target{
    .world_pose_is_constant = true,
    .model_definition_is_constant = true,
  };
  for (int i = 0; i < t_world_points.size(); ++i) {
    planar_target.model_definition[i] = t_world_points[i];
  }
  WorldModel world_model;
  EXPECT_OK(world_model.AddRigidBody(planar_target));

  // Sensorrig trajectory
  Trajectory trajectory_world_sensorrig;
  ASSERT_OK(trajectory_world_sensorrig.AddPoses(poses_world_sensorrig));

  // Generate measurements.
  std::vector<sensors::CameraMeasurement> measurements_left, measurements_right;
  ASSERT_OK_AND_ASSIGN(measurements_left,
      true_camera_left.Project(stamps, trajectory_world_sensorrig,
                               world_model));
  ASSERT_OK_AND_ASSIGN(measurements_right,
      true_camera_right.Project(stamps, trajectory_world_sensorrig,
                                world_model));

  // Create optimization cameras.
  Eigen::VectorXd initial_intrinsics = 1.01 * true_intrinsics;
  initial_intrinsics.tail(5).setZero();
  const Pose3d initial_extrinsics = true_extrinsics_right;
  
  sensors::Camera* camera_left = new sensors::Camera();
  camera_left->SetName("Left");
  EXPECT_OK(camera_left->SetModel(kCameraModel));
  EXPECT_OK(camera_left->SetIntrinsics(initial_intrinsics));
  camera_left->EnableExtrinsicsEstimation(false);
  camera_left->EnableIntrinsicsEstimation(true);
  camera_left->EnableLatencyEstimation(false);
  EXPECT_OK(camera_left->AddMeasurements(measurements_left));
  sensors::Camera* camera_right = new sensors::Camera();
  camera_right->SetName("Right");
  EXPECT_OK(camera_right->SetModel(kCameraModel));
  EXPECT_OK(camera_right->SetIntrinsics(initial_intrinsics));
  camera_right->SetExtrinsics(initial_extrinsics);
  camera_right->EnableExtrinsicsEstimation(true);
  camera_right->EnableIntrinsicsEstimation(true);
  camera_right->EnableLatencyEstimation(true);
  EXPECT_OK(camera_right->AddMeasurements(measurements_right));

  // Construct optimization problem and optimize.
  BatchOptimizer optimizer;
  optimizer.AddSensor(camera_left);
  optimizer.AddSensor(camera_right);
  optimizer.AddWorldModel(world_model);
  optimizer.AddTrajectory(trajectory_world_sensorrig);
  ASSERT_OK_AND_ASSIGN(const auto summary, optimizer.Optimize());

  // Expect near perfect calibration results due to perfect data.
  constexpr double kSmallNumber = 1e-8;
  EXPECT_EQ(summary.termination_type, ceres::CONVERGENCE);
  EXPECT_LT(summary.final_cost, kSmallNumber);
  EXPECT_THAT(true_intrinsics, EigenIsApprox(camera_left->GetIntrinsics(),
                                             kSmallNumber));
  EXPECT_THAT(true_intrinsics, EigenIsApprox(camera_right->GetIntrinsics(),
                                             kSmallNumber));
  EXPECT_THAT(true_extrinsics_right, PoseIsApprox(camera_right->GetExtrinsics(),
                                                  kSmallNumber));
  EXPECT_NEAR(kLatency, camera_right->GetLatency(), kSmallNumber);
  std::cout << summary.FullReport() << std::endl;
}

} // namespace
} // namespace calico
