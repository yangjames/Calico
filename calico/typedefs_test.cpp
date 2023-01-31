#include "calico/typedefs.h"

#include "gtest/gtest.h"

namespace calico {
namespace {

// Tests Pose3 initialization. Expect an empty pose to be zero translation
// and identity rotation.
TEST(Pose3Test, EmptyConstruction) {
  const Pose3 pose;
  EXPECT_EQ(pose.rotation().w(), 1.0);
  EXPECT_EQ(pose.rotation().x(), 0.0);
  EXPECT_EQ(pose.rotation().y(), 0.0);
  EXPECT_EQ(pose.rotation().z(), 0.0);
  EXPECT_EQ(pose.translation().x(), 0.0);
  EXPECT_EQ(pose.translation().y(), 0.0);
  EXPECT_EQ(pose.translation().z(), 0.0);
}

TEST(Pose3Test, ConstructionWithInitialValues) {
  const Eigen::Vector3d expected_translation = Eigen::Vector3d::Random();
  const Eigen::Quaterniond expected_rotation = Eigen::Quaterniond::UnitRandom();
  const Pose3 pose(expected_rotation, expected_translation);
  EXPECT_EQ(pose.rotation().w(), expected_rotation.w());
  EXPECT_EQ(pose.rotation().x(), expected_rotation.x());
  EXPECT_EQ(pose.rotation().y(), expected_rotation.y());
  EXPECT_EQ(pose.rotation().z(), expected_rotation.z());
  EXPECT_EQ(pose.translation().x(), expected_translation.x());
  EXPECT_EQ(pose.translation().y(), expected_translation.y());
  EXPECT_EQ(pose.translation().z(), expected_translation.z());
}

// Tests assignment of Pose3 member variables. Expect that even if we modify the
// values of a Pose3 object using its accessors, we should expect their
// addresses to remain the same.
TEST(Pose3Test, ValueAssignmentAndMemoryConsistency) {
  Pose3 pose;
  double* translation_ptr = static_cast<double*>(pose.translation().data());
  double* rotation_ptr = static_cast<double*>(pose.rotation().coeffs().data());

  const Eigen::Vector3d expected_translation = Eigen::Vector3d::Random();
  const Eigen::Vector4d random_unit_vector =
      Eigen::Vector4d::Random().normalized();
  const Eigen::Quaterniond expected_rotation(
      random_unit_vector(0), random_unit_vector(1), random_unit_vector(2),
      random_unit_vector(3));
  pose.translation() = expected_translation;
  pose.rotation() = expected_rotation;
  EXPECT_EQ(pose.rotation().w(), expected_rotation.w());
  EXPECT_EQ(pose.rotation().x(), expected_rotation.x());
  EXPECT_EQ(pose.rotation().y(), expected_rotation.y());
  EXPECT_EQ(pose.rotation().z(), expected_rotation.z());
  EXPECT_EQ(pose.translation().x(), expected_translation.x());
  EXPECT_EQ(pose.translation().y(), expected_translation.y());
  EXPECT_EQ(pose.translation().z(), expected_translation.z());
  EXPECT_EQ(translation_ptr, pose.translation().data());
  EXPECT_EQ(rotation_ptr, pose.rotation().coeffs().data());
}

} // namespace
} // namespace calico
