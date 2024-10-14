#include "calico/world_model.h"

#include "calico/matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"


namespace calico {
namespace {

using ::testing::Contains;
using ::testing::Eq;
using ::testing::Key;
using ::testing::SizeIs;


class WorldModelTest : public ::testing::Test {
 protected:
  absl::flat_hash_map<int, Landmark> expected_landmarks {
    {0, Landmark{
          .point = Eigen::Vector3d::Random(),
          .id = 0,
        },
    },
    {1, Landmark{
          .point = Eigen::Vector3d::Random(),
          .id = 1,
        },
    },
  };
  absl::flat_hash_map<int, RigidBody> expected_rigidbodies {
    {0, RigidBody{
          .model_definition = {
            {0, Eigen::Vector3d::Random()},
            {1, Eigen::Vector3d::Random()}
          },
          .T_world_rigidbody = Pose3d(Eigen::Quaterniond::UnitRandom(),
                                      Eigen::Vector3d::Random()),
          .id = 0
        },
    },
    {1, RigidBody{
          .model_definition = {
            {2, Eigen::Vector3d::Random()},
            {3, Eigen::Vector3d::Random()}
          },
          .T_world_rigidbody = Pose3d(Eigen::Quaterniond::UnitRandom(),
                                      Eigen::Vector3d::Random()),
          .id = 1
        },
    },
  };

  WorldModel world_model_;
};

TEST_F(WorldModelTest, RigidBodyAccessors) {
  world_model_.ClearRigidBodies();
  for (auto& [rigidbody_id, rigidbody] : expected_rigidbodies) {
    ASSERT_OK(world_model_.AddRigidBody(&rigidbody, /*take_ownership=*/false));
  }

  // Check that all assigned memory and values are the same.
  for (const auto& [rigidbody_id, actual_rigidbody] : expected_rigidbodies) {
    const auto& actual_model_definition = actual_rigidbody.model_definition;
    const auto& rigidbody = world_model_.rigidbodies().at(rigidbody_id);
    const auto& model_definition = rigidbody->model_definition;
    for (const auto& [feature_id, point] : model_definition) {
      // Compare point.
      const Eigen::Vector3d& actual_point = actual_model_definition.at(feature_id);
      EXPECT_EQ(point.data(), actual_point.data());
      EXPECT_TRUE(point.isApprox(actual_point));
      // Compare pose.
      EXPECT_TRUE(rigidbody->T_world_rigidbody.isApprox(actual_rigidbody.T_world_rigidbody));
      EXPECT_EQ(
        rigidbody->T_world_rigidbody.translation().data(),
        actual_rigidbody.T_world_rigidbody.translation().data()
      );
      EXPECT_EQ(
        rigidbody->T_world_rigidbody.rotation().coeffs().data(),
        actual_rigidbody.T_world_rigidbody.rotation().coeffs().data()
      );
    }
  }
}

TEST_F(WorldModelTest, LandMarkAccessors) {
  world_model_.ClearLandmarks();
  for (auto& [landmark_id, landmark] : expected_landmarks) {
    ASSERT_OK(world_model_.AddLandmark(&landmark, /*take_ownership=*/false));
  }
  for (const auto& [landmark_id, actual_landmark] : expected_landmarks) {
    const auto& landmark = world_model_.landmarks().at(landmark_id);
    const Eigen::Vector3d& actual_point = actual_landmark.point;
    const Eigen::Vector3d& point = landmark->point;
    EXPECT_EQ(point.data(), actual_point.data());
    EXPECT_TRUE(point.isApprox(actual_point));
    EXPECT_EQ(landmark->id, actual_landmark.id);
    EXPECT_EQ(landmark->point_is_constant, actual_landmark.point_is_constant);
  }
}

TEST_F(WorldModelTest, AddLandmark) {
  world_model_.ClearLandmarks();
  EXPECT_EQ(world_model_.NumberOfLandmarks(), 0);
  Landmark* landmark = new Landmark;
  EXPECT_OK(world_model_.AddLandmark(landmark));
  EXPECT_EQ(world_model_.NumberOfLandmarks(), 1);
  EXPECT_EQ(world_model_.AddLandmark(landmark).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(world_model_.NumberOfLandmarks(), 1);
}

TEST_F(WorldModelTest, AddRigidBody) {
  world_model_.ClearRigidBodies();
  EXPECT_EQ(world_model_.NumberOfRigidBodies(), 0);
  RigidBody* rigidbody = new RigidBody;
  EXPECT_OK(world_model_.AddRigidBody(rigidbody));
  EXPECT_EQ(world_model_.NumberOfRigidBodies(), 1);
  EXPECT_EQ(world_model_.AddRigidBody(rigidbody).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(world_model_.NumberOfRigidBodies(), 1);
}

TEST_F(WorldModelTest, SetGravity) {
  const Eigen::Vector3d expected_grav = Eigen::Vector3d::Random();
  world_model_.gravity() = expected_grav;
  EXPECT_THAT(world_model_.gravity(), EigenEq(expected_grav));
}

TEST_F(WorldModelTest, AddParametersToProblem) {
  ceres::Problem problem;
  world_model_.Clear();
  for (auto& [rigidbody_id, rigidbody] : expected_rigidbodies) {
    ASSERT_OK(world_model_.AddRigidBody(&rigidbody, /*take_ownership=*/false));
  }
  for (auto& [landmark_id, landmark] : expected_landmarks) {
    ASSERT_OK(world_model_.AddLandmark(&landmark, /*take_ownership=*/false));
  }
  const int num_parameters_added = world_model_.AddParametersToProblem(problem);
  EXPECT_GT(problem.NumParameters(), 0);
  EXPECT_EQ(num_parameters_added, problem.NumParameters());
  for (const auto& [landmark_id, _] : expected_landmarks) {
    const std::unique_ptr<Landmark>& landmark = world_model_.landmarks().at(landmark_id);
    EXPECT_TRUE(problem.HasParameterBlock(landmark->point.data()));
  }
  for (const auto& [rigidbody_id, _] : expected_rigidbodies) {
    const std::unique_ptr<RigidBody>& rigidbody = world_model_.rigidbodies().at(rigidbody_id);
    for (const auto& [feature_id, point] : rigidbody->model_definition) {
      EXPECT_TRUE(problem.HasParameterBlock(point.data()));
    }
    EXPECT_TRUE(
        problem.HasParameterBlock(
            rigidbody->T_world_rigidbody.rotation().coeffs().data()));
    EXPECT_TRUE(
        problem.HasParameterBlock(
            rigidbody->T_world_rigidbody.translation().data()));
  }
}

} // namespace
} // namespace calico
