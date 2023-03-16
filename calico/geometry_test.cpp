#include "calico/geometry.h"

#include "calico/matchers.h"
#include "calico/trajectory.h"
#include "calico/test_utils.h"


namespace calico {
namespace {

// Test that our skew symmetric matrix produces the right cross product matrix.
TEST(GeometryTest, TestSkewMatrix) {
  const Eigen::Vector3d v1 = Eigen::Vector3d::Random();
  const Eigen::Vector3d v2 = Eigen::Vector3d::Random();
  const Eigen::Vector3d expected_cross_1_2 = v1.cross(v2);
  const Eigen::Matrix3d V1x = Skew(v1);
  const Eigen::Vector3d actual_cross_1_2 = V1x * v2;
  EXPECT_THAT(actual_cross_1_2, EigenEq(expected_cross_1_2));
}


// Compare ComputeAngularVelocity against numerical differentiation.
TEST(GeometryTest, TestComputeAngularVelocity) {
  // Construct a spline that we can interpolate continuously on.
  DefaultSyntheticTest synthetic_test;
  const auto trajectory_world_body = synthetic_test.TrajectoryAsMap();
  const auto timestamps = synthetic_test.TrajectoryMapKeys();
  Trajectory trajectory;
  ASSERT_OK(trajectory.AddPoses(trajectory_world_body));
  // In small increments, test a subset of angles near knots.
  const double dt = 1e-9;
  for (int i = 0; i < timestamps.size() - 1; ++i) {
    // Interpolate spline at two timestamps close to each other.
    const double stamp_i = timestamps.at(i);
    const double stamp_ii = stamp_i + dt;
    const Eigen::Vector<double, 6> pose_vector_i =
        trajectory.spline().Interpolate({stamp_i}, /*derivative=*/0).value()[0];
    const Eigen::Vector<double, 6> pose_vector_ii =
        trajectory.spline().Interpolate(
            {stamp_ii}, /*derivative=*/0).value()[0];
    const Eigen::Vector3d phi_world_body_i = pose_vector_i.head(3);
    const Eigen::Vector3d phi_world_body_ii = pose_vector_ii.head(3);
    const Eigen::Vector3d phi_body_world_i = -phi_world_body_i;
    const Eigen::Vector3d phi_body_world_ii = -phi_world_body_ii;
    // Convert axis angle to rotation matrices.
    const Eigen::Matrix3d R_world_body_i = Eigen::AngleAxisd(
        phi_world_body_i.norm(), phi_world_body_i.normalized())
        .toRotationMatrix();
    const Eigen::Matrix3d R_world_body_ii = Eigen::AngleAxisd(
        phi_world_body_ii.norm(), phi_world_body_ii.normalized())
        .toRotationMatrix();
    const Eigen::Matrix3d R_body_world_i = Eigen::AngleAxisd(
        phi_body_world_i.norm(), phi_body_world_i.normalized())
        .toRotationMatrix();
    const Eigen::Matrix3d R_body_world_ii = Eigen::AngleAxisd(
        phi_body_world_ii.norm(), phi_body_world_ii.normalized())
        .toRotationMatrix();
    // Compute rotation deltas for world-from-body and body-from-world rotations.
    const Eigen::Matrix3d dR_world_body =
        R_world_body_ii * R_world_body_i.transpose();
    const Eigen::Matrix3d dR_body_world =
        R_body_world_ii * R_body_world_i.transpose();
    // Convert rotation deltas into rates in the log-space.
    const double dt_inv = 1.0 / dt;
    const Eigen::AngleAxisd dphi_world_body(dR_world_body);
    const Eigen::AngleAxisd dphi_body_world(dR_body_world);
    const Eigen::Vector3d expected_omega_world_body =
        dt_inv * (dphi_world_body.angle() * dphi_world_body.axis());
    const Eigen::Vector3d expected_omega_body_world =
        dt_inv * (dphi_body_world.angle() * dphi_body_world.axis());
    // Compute analytical time derivative.
    const Eigen::Vector<double, 6> pose_dot_vector_i =
        trajectory.spline().Interpolate({stamp_i}, /*derivative=*/1).value()[0];
    const Eigen::Vector3d phi_dot_world_body_i = pose_dot_vector_i.head(3);
    const Eigen::Vector3d phi_dot_body_world_i = -phi_dot_world_body_i;
    const Eigen::Vector3d actual_omega_world_body =
        ComputeAngularVelocity(phi_world_body_i, phi_dot_world_body_i);
    const Eigen::Vector3d actual_omega_body_world =
        ComputeAngularVelocity(phi_body_world_i, phi_dot_body_world_i);
    // Compare.
    constexpr double kSmallNumber = 1e-5;
    const Eigen::Vector3d error_world_body =
        actual_omega_world_body - expected_omega_world_body;
    const Eigen::Vector3d error_body_world =
        actual_omega_body_world - expected_omega_body_world;
    EXPECT_LT(error_world_body.norm(), kSmallNumber);
    EXPECT_LT(error_body_world.norm(), kSmallNumber);
  }
}


} // namespace
} // namespace calico
