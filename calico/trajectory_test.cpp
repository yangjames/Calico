#include "calico/trajectory.h"

#include "calico/matchers.h"
#include "calico/test_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"


namespace calico {
namespace {

class TrajectoryTest : public ::testing::Test {
 protected:
  absl::flat_hash_map<double, Pose3> trajectory_world_sensorrig;
  std::vector<double> trajectory_key_values;
  void SetUp() override {
    DefaultSyntheticTest test_fixture;
    trajectory_world_sensorrig = test_fixture.TrajectoryAsMap();
    trajectory_key_values = test_fixture.TrajectoryMapKeys();
  }
};

TEST_F(TrajectoryTest, AddPosesWorks) {
  Trajectory trajectory;
  EXPECT_OK(trajectory.AddPoses(trajectory_world_sensorrig));
}

TEST_F(TrajectoryTest, TrajectorySetterAndGetter) {
  Trajectory trajectory;
  const auto& expected_trajectory = trajectory_world_sensorrig;
  trajectory.trajectory() = trajectory_world_sensorrig;
  const auto actual_trajectory = trajectory.trajectory();
  EXPECT_EQ(actual_trajectory.size(), expected_trajectory.size());
  for (const auto& [stamp, actual_pose] : actual_trajectory) {
    const auto& expected_pose = expected_trajectory.at(stamp);
    EXPECT_THAT(actual_pose, PoseEq(expected_pose));
  }
}

} // namespace
} // namespace calico
