#include "calico/world_model.h"

#include "absl/status/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace calico {
namespace {

class WorldModelTest : public ::testing::Test {
 protected:
  WorldModel world_model_;
};

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

} // namespace
} // namespace calico
