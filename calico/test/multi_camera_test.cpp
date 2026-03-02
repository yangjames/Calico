#include "calico/sensors/multi_camera.h"

#include "Eigen/Dense"
#include "calico/matchers.h"
#include "calico/sensors/camera_cost_functor.h"
#include "calico/sensors/camera_models.h"
#include "calico/typedefs.h"
#include "calico/world_model.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace calico::sensors {
namespace {

class MultiCameraContainerTest : public ::testing::Test {
 protected:
  const std::string kCameraName = "multi_camera";
  const std::vector<std::string> kImagerNames{"left", "right", "middle"};
  const absl::flat_hash_map<std::string, CameraIntrinsicsModel>
      kImagerToCameraModel = [this]() {
        absl::flat_hash_map<std::string, CameraIntrinsicsModel>
            imager_to_intrinsics_model;
        for (const auto& imager_name : kImagerNames) {
          imager_to_intrinsics_model[imager_name] =
              CameraIntrinsicsModel::kOpenCv5;
        }
        return imager_to_intrinsics_model;
      }();
  const absl::flat_hash_map<std::string, Pose3d> kImagerToExtrinsics =
      [this]() {
        absl::flat_hash_map<std::string, Pose3d> imager_to_extrinsics;
        for (const auto& imager_name : kImagerNames) {
          imager_to_extrinsics.insert(
              {imager_name, Pose3d(Eigen::Quaterniond::UnitRandom(),
                                   Eigen::Vector3d::Random())});
        }
        return imager_to_extrinsics;
      }();
  const absl::flat_hash_map<std::string, Eigen::VectorXd> kImagerToIntrinsics =
      [this]() {
        absl::flat_hash_map<std::string, Eigen::VectorXd> imager_to_intrinsics;
        for (const auto& imager_name : kImagerNames) {
          imager_to_intrinsics.insert(
              {imager_name,
               Eigen::VectorXd::Random(OpenCv5Model::kNumberOfParameters)});
        }
        return imager_to_intrinsics;
      }();
  static constexpr int kNumFeatures = 10;
  static constexpr int kNumModels = 10;
  static constexpr int kNumImages = 10;

  void SetUp() override {
    for (const auto& imager_name : kImagerNames) {
      imager_to_measurements_.insert({imager_name, {}});
      for (int image_id = 0; image_id < kNumImages; ++image_id) {
        for (int model_id = 0; model_id < kNumModels; ++model_id) {
          for (int feature_id = 0; feature_id < kNumFeatures; ++feature_id) {
            imager_to_measurements_[imager_name].push_back(
                CameraMeasurement{.id = {.stamp = static_cast<double>(image_id),
                                         .image_id = image_id,
                                         .model_id = model_id,
                                         .feature_id = feature_id}});
          }
        }
      }
    }
  }

  MultiCamera camera_;
  absl::flat_hash_map<std::string, std::vector<CameraMeasurement>>
      imager_to_measurements_;
};

TEST_F(MultiCameraContainerTest, SettersAndGetters) {
  // Pre-assignment.
  EXPECT_THAT(camera_.GetName(), ::testing::IsEmpty());
  EXPECT_TRUE(camera_.GetModel().empty());
  EXPECT_THAT(camera_.GetSensorExtrinsics(), PoseEq(Pose3d()));
  EXPECT_TRUE(camera_.GetImagerExtrinsics().empty());
  EXPECT_TRUE(camera_.GetIntrinsics().empty());
  // Post-assignment.
  camera_.SetName(kCameraName);
  EXPECT_OK(camera_.SetModel(kImagerToCameraModel));
  camera_.SetImagerExtrinsics(kImagerToExtrinsics);
  EXPECT_OK(camera_.SetIntrinsics(kImagerToIntrinsics));
  EXPECT_EQ(camera_.GetName(), kCameraName);
  for (const auto& [imager, model] : camera_.GetModel()) {
    EXPECT_EQ(model, kImagerToCameraModel.at(imager));
  }
  for (const auto& [imager, extrinsics] : camera_.GetImagerExtrinsics()) {
    EXPECT_THAT(extrinsics, PoseEq(kImagerToExtrinsics.at(imager)));
  }
  for (const auto& [imager, intrinsics] : camera_.GetIntrinsics()) {
    EXPECT_THAT(intrinsics, EigenEq(kImagerToIntrinsics.at(imager)));
  }
}

// TEST_F(CameraContainerTest, AddSingleMeasurementOnlyUniqueAllowed) {
//   const CameraMeasurement measurement{
//       .pixel = Eigen::Vector2d::Random(),
//       .id = {.image_id = 0, .model_id = 1, .feature_id = 2},
//   };
//   camera_.ClearMeasurements();
//   EXPECT_EQ(camera_.NumberOfMeasurements(), 0);
//   EXPECT_OK(camera_.AddMeasurement(measurement));
//   EXPECT_EQ(camera_.NumberOfMeasurements(), 1);
//   // Add the same measurement and expect an error.
//   EXPECT_THAT(camera_.AddMeasurement(measurement),
//               StatusCodeIs(absl::StatusCode::kInvalidArgument));
//   EXPECT_EQ(camera_.NumberOfMeasurements(), 1);
// }

// TEST_F(CameraContainerTest, AddMultipleMeasurementsOnlyUniqueAllowed) {
//   std::vector<CameraMeasurement> measurements = measurements_;
//   camera_.ClearMeasurements();
//   EXPECT_EQ(camera_.NumberOfMeasurements(), 0);
//   EXPECT_OK(camera_.AddMeasurements(measurements));
//   EXPECT_EQ(camera_.NumberOfMeasurements(), measurements.size());
//   const CameraMeasurement redundant_measurement{
//       .id = {.stamp = 0, .image_id = 0, .model_id = 0, .feature_id = 0}};
//   measurements.push_back(redundant_measurement);
//   camera_.ClearMeasurements();
//   EXPECT_THAT(camera_.AddMeasurements(measurements),
//               StatusCodeIs(absl::StatusCode::kInvalidArgument));
//   EXPECT_EQ(camera_.NumberOfMeasurements(), measurements.size() - 1);
// }

// TEST_F(CameraContainerTest, AddCalibrationParametersToProblem) {
//   EXPECT_OK(camera_.SetModel(kCameraModel));
//   EXPECT_OK(camera_.SetIntrinsics(kIntrinsics));
//   camera_.SetExtrinsics(kExtrinsics);
//   ceres::Problem problem;
//   ASSERT_OK_AND_ASSIGN(const int num_parameters,
//                        camera_.AddParametersToProblem(problem));
//   EXPECT_EQ(problem.NumParameters(), num_parameters);
// }

// TEST(CameraProjectionTest, LandmarkInView) {
//   // Construct a scene where a camera is sitting still, hovering 1m above the
//   // origin for 1 second.
//   Eigen::Quaterniond q_world_camera(/*w=*/0.0, /*x=*/1.0, /*y=*/0.0,
//                                     /*z=*/0.0);
//   Eigen::Vector3d t_world_camera(0.0, 0.0, 1.0);
//   Trajectory trajectory;
//   ASSERT_OK(trajectory.FitSpline({
//       {0.0, Pose3d(q_world_camera, t_world_camera)},
//       {1.0, Pose3d(q_world_camera, t_world_camera)},
//   }));
//   // Construct a landmark placed at the origin.
//   Landmark landmark{Eigen::Vector3d::Zero(), 0, true};
//   WorldModel world_model;
//   ASSERT_OK(world_model.AddLandmark(&landmark, /*take_ownership=*/false));
//   // Construct the camera.
//   Camera camera;
//   ASSERT_OK(camera.SetModel(CameraIntrinsicsModel::kOpenCv5));
//   Eigen::VectorXd intrinsics(OpenCv5Model::kNumberOfParameters);
//   intrinsics << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
//   ASSERT_OK(camera.SetIntrinsics(intrinsics));

//   // Project the landmark into the camera.
//   ASSERT_OK_AND_ASSIGN(
//       const std::vector<CameraMeasurement> measurements,
//       camera.Project(std::vector<double>{0.0}, trajectory, world_model));
//   ASSERT_EQ(measurements.size(), 1);
// }

// TEST(CameraProjectionTest, LandmarkOutOfView) {
//   // Construct a scene where a camera is sitting still, hovering 1m above the
//   // origin for 1 second.
//   const Eigen::Quaterniond q_world_camera(/*w=*/0.0, /*x=*/1.0, /*y=*/0.0,
//                                           /*z=*/0.0);
//   const Eigen::Vector3d t_world_camera(0.0, 0.0, 1.0);
//   Trajectory trajectory;
//   ASSERT_OK(trajectory.FitSpline({
//       {0.0, Pose3d(q_world_camera, t_world_camera)},
//       {1.0, Pose3d(q_world_camera, t_world_camera)},
//   }));
//   // Construct a landmark placed behind the camera.
//   Landmark landmark{.point = Eigen::Vector3d(0.0, 0.0, 2.0)};
//   WorldModel world_model;
//   ASSERT_OK(world_model.AddLandmark(&landmark, /*take_ownership=*/false));
//   // Construct the camera.
//   Camera camera;
//   ASSERT_OK(camera.SetModel(CameraIntrinsicsModel::kOpenCv5));
//   Eigen::VectorXd intrinsics(OpenCv5Model::kNumberOfParameters);
//   intrinsics << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
//   ASSERT_OK(camera.SetIntrinsics(intrinsics));
//   // Project the landmark into the camera.
//   ASSERT_OK_AND_ASSIGN(
//       const std::vector<CameraMeasurement> measurements,
//       camera.Project(std::vector<double>{0.0}, trajectory, world_model));
//   ASSERT_EQ(measurements.size(), 0);
// }

// TEST(CameraProjectionTest, RigidBodyInView) {
//   // Construct a scene where a camera is sitting still, hovering 1m above the
//   // origin for 1 second.
//   const Eigen::Quaterniond q_world_camera(/*w=*/0.0, /*x=*/1.0, /*y=*/0.0,
//                                           /*z=*/0.0);
//   const Eigen::Vector3d t_world_camera(0.0, 0.0, 1.0);
//   Trajectory trajectory;
//   ASSERT_OK(trajectory.FitSpline({
//       {0.0, Pose3d(q_world_camera, t_world_camera)},
//       {1.0, Pose3d(q_world_camera, t_world_camera)},
//   }));
//   // Construct a rigidbody placed at the origin.
//   RigidBody rigidbody{.model_definition =
//                           {
//                               {0, Eigen::Vector3d(-0.5, -0.5, 0.0)},
//                               {1, Eigen::Vector3d(-0.5, 0.5, 0.0)},
//                               {2, Eigen::Vector3d(0.5, 0.5, 0.0)},
//                               {3, Eigen::Vector3d(0.5, -0.5, 0.0)},
//                           },
//                       .id = 0};
//   WorldModel world_model;
//   ASSERT_OK(world_model.AddRigidBody(&rigidbody, /*take_ownership=*/false));
//   // Construct the camera.
//   Camera camera;
//   ASSERT_OK(camera.SetModel(CameraIntrinsicsModel::kOpenCv5));
//   Eigen::VectorXd intrinsics(OpenCv5Model::kNumberOfParameters);
//   intrinsics << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
//   ASSERT_OK(camera.SetIntrinsics(intrinsics));
//   // Project the landmark into the camera.
//   ASSERT_OK_AND_ASSIGN(
//       const std::vector<CameraMeasurement> measurements,
//       camera.Project(std::vector<double>{0.0}, trajectory, world_model));
//   ASSERT_EQ(measurements.size(), 4);
// }

// TEST(CameraProjectionTest, RigidBodyOutOfView) {
//   // Construct a scene where a camera is sitting still, hovering 1m above the
//   // origin for 1 second.
//   const Eigen::Quaterniond q_world_camera(/*w=*/0.0, /*x=*/1.0, /*y=*/0.0,
//                                           /*z=*/0.0);
//   const Eigen::Vector3d t_world_camera(0.0, 0.0, 1.0);
//   Trajectory trajectory;
//   ASSERT_OK(trajectory.FitSpline({
//       {0.0, Pose3d(q_world_camera, t_world_camera)},
//       {1.0, Pose3d(q_world_camera, t_world_camera)},
//   }));
//   // Construct a rigidbody placed at the origin.
//   RigidBody rigidbody{
//       .model_definition =
//           {
//               {0, Eigen::Vector3d(-0.5, -0.5, 0.0)},
//               {1, Eigen::Vector3d(-0.5, 0.5, 0.0)},
//               {2, Eigen::Vector3d(0.5, 0.5, 0.0)},
//               {3, Eigen::Vector3d(0.5, -0.5, 0.0)},
//           },
//       .T_world_rigidbody = Pose3d(Eigen::Quaterniond::Identity(),
//                                   Eigen::Vector3d(0.0, 0.0, 2.0)),
//       .id = 0};
//   WorldModel world_model;
//   ASSERT_OK(world_model.AddRigidBody(&rigidbody, /*take_ownership=*/false));
//   // Construct the camera.
//   Camera camera;
//   ASSERT_OK(camera.SetModel(CameraIntrinsicsModel::kOpenCv5));
//   Eigen::VectorXd intrinsics(OpenCv5Model::kNumberOfParameters);
//   intrinsics << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
//   ASSERT_OK(camera.SetIntrinsics(intrinsics));
//   // Project the landmark into the camera.
//   ASSERT_OK_AND_ASSIGN(
//       const std::vector<CameraMeasurement> measurements,
//       camera.Project(std::vector<double>{0.0}, trajectory, world_model));
//   ASSERT_EQ(measurements.size(), 0);
// }
}  // namespace
}  // namespace calico::sensors
