#include "calico/world_model.h"

#include "absl/status/status.h"
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
  const absl::flat_hash_map<int, Landmark> expected_landmarks {
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
  const absl::flat_hash_map<int, RigidBody> expected_rigidbodies {
    {0, RigidBody {
        .model_definition = {
          {0, Eigen::Vector3d::Random()},
          {1, Eigen::Vector3d::Random()}
        },
        .T_world_rigidbody = Pose3(Eigen::Quaterniond::UnitRandom(),
                                   Eigen::Vector3d::Random()),
        .id = 0
      }
    },
    {1, RigidBody {
        .model_definition = {
          {2, Eigen::Vector3d::Random()},
          {3, Eigen::Vector3d::Random()}
        },
        .T_world_rigidbody = Pose3(Eigen::Quaterniond::UnitRandom(),
                                   Eigen::Vector3d::Random()),
        .id = 1
      }
    },
  };

  WorldModel world_model_;
};

TEST_F(WorldModelTest, RigidBodyAccessors) {
  world_model_.ClearRigidBodies();
  world_model_.rigidbodies() = expected_rigidbodies;
  const absl::flat_hash_map<int, RigidBody> actual_rigidbodies =
    world_model_.rigidbodies();
  // TODO(yangjames): Write a matcher for this.
  ASSERT_THAT(actual_rigidbodies, SizeIs(expected_rigidbodies.size()));
  for (const auto& [rigidbody_id, expected_rigidbody] : expected_rigidbodies) {
    ASSERT_THAT(actual_rigidbodies, Contains(Key(rigidbody_id)));
    const RigidBody& actual_rigidbody = actual_rigidbodies.at(rigidbody_id);
    EXPECT_THAT(actual_rigidbody.model_definition,
                SizeIs(expected_rigidbody.model_definition.size()));

    for (const auto& [feature_id, expected_point] :
           expected_rigidbody.model_definition) {
      ASSERT_THAT(actual_rigidbody.model_definition, Contains(Key(feature_id)));
      const Eigen::Vector3d& actual_point =
        actual_rigidbody.model_definition.at(feature_id);
      EXPECT_TRUE(actual_point.isApprox(expected_point));
    }
    EXPECT_TRUE(actual_rigidbody.T_world_rigidbody
                .isApprox(expected_rigidbody.T_world_rigidbody));
    EXPECT_EQ(rigidbody_id, actual_rigidbody.id);
    EXPECT_EQ(actual_rigidbody.id, expected_rigidbody.id);
    EXPECT_EQ(actual_rigidbody.world_pose_is_constant,
              expected_rigidbody.world_pose_is_constant);
    EXPECT_EQ(actual_rigidbody.model_definition_is_constant,
              expected_rigidbody.model_definition_is_constant);
  }
}

TEST_F(WorldModelTest, LandMarkAccessors) {
  world_model_.ClearLandmarks();
  world_model_.landmarks() = expected_landmarks;
  const absl::flat_hash_map<int, Landmark> actual_landmarks =
    world_model_.landmarks();
  // TODO(yangjames): Write a matcher for this.
  ASSERT_THAT(actual_landmarks, SizeIs(expected_landmarks.size()));
  for (const auto& [landmark_id, expected_landmark] : expected_landmarks) {
    ASSERT_THAT(actual_landmarks, Contains(Key(landmark_id)));
    const auto& actual_landmark = actual_landmarks.at(landmark_id);
    EXPECT_TRUE(actual_landmark.point.isApprox(expected_landmark.point));
    EXPECT_EQ(actual_landmark.id, expected_landmark.id);
    EXPECT_EQ(actual_landmark.point_is_constant,
              expected_landmark.point_is_constant);
  }
}

TEST_F(WorldModelTest, AddLandmark) {
  world_model_.ClearLandmarks();
  EXPECT_EQ(world_model_.NumberOfLandmarks(), 0);
  const Landmark landmark{};
  EXPECT_EQ(world_model_.AddLandmark(landmark).code(), absl::StatusCode::kOk);
  EXPECT_EQ(world_model_.NumberOfLandmarks(), 1);
  EXPECT_EQ(world_model_.AddLandmark(landmark).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(world_model_.NumberOfLandmarks(), 1);
}

TEST_F(WorldModelTest, AddRigidBody) {
  world_model_.ClearRigidBodies();
  EXPECT_EQ(world_model_.NumberOfRigidBodies(), 0);
  const RigidBody rigidbody{};
  EXPECT_EQ(world_model_.AddRigidBody(rigidbody).code(), absl::StatusCode::kOk);
  EXPECT_EQ(world_model_.NumberOfRigidBodies(), 1);
  EXPECT_EQ(world_model_.AddRigidBody(rigidbody).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(world_model_.NumberOfRigidBodies(), 1);
}

TEST_F(WorldModelTest, AddParametersToProblem) {
  ceres::Problem problem;
  world_model_.Clear();
  world_model_.rigidbodies() = expected_rigidbodies;
  world_model_.landmarks() = expected_landmarks;
  const int num_parameters_added = world_model_.AddParametersToProblem(problem);
  EXPECT_GT(problem.NumParameters(), 0);
  EXPECT_EQ(num_parameters_added, problem.NumParameters());
  for (const auto& [landmark_id, _] : expected_landmarks) {
    Landmark& landmark = world_model_.landmarks().at(landmark_id);
    EXPECT_TRUE(problem.HasParameterBlock(landmark.point.data()));
  }
  for (const auto& [rigidbody_id, _] : expected_rigidbodies) {
    RigidBody& rigidbody = world_model_.rigidbodies().at(rigidbody_id);
    for (const auto& [feature_id, point] : rigidbody.model_definition) {
      EXPECT_TRUE(problem.HasParameterBlock(point.data()));
    }
    EXPECT_TRUE(
        problem.HasParameterBlock(
            rigidbody.T_world_rigidbody.rotation().coeffs().data()));
    EXPECT_TRUE(
        problem.HasParameterBlock(
            rigidbody.T_world_rigidbody.translation().data()));
  }
}

} // namespace
} // namespace calico
