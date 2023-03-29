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

TEST(GeometryTest, TestiSkew) {
  const Eigen::Vector3d expected_v = Eigen::Vector3d::Random();
  const Eigen::Vector3d actual_v = iSkew(Skew(expected_v));
  EXPECT_THAT(actual_v, EigenEq(expected_v));
}

TEST(GeometryTest, TestSO3) {
  // Test small angles.
  constexpr double kSmallNumber = 1e-7;
  const double dtheta = 1e-12;
  for (int i = 0; i < 10; ++i) {
    Eigen::Vector3d axis = Eigen::Vector3d::Random();
    axis.normalize();
    const double theta = pow(10, i) * dtheta;
    const Eigen::Vector3d expected_phi = theta * axis;
    const Eigen::Vector3d actual_phi = LnSO3(ExpSO3(expected_phi));
    const Eigen::Vector3d error = actual_phi - expected_phi;
    EXPECT_LT(error.norm(), kSmallNumber);
  }
}

// Compare analytically compute angular velocity/acceleration and compare against
// numerical differentiation.
TEST(GeometryTest, TestComputeRodriguesFormulaJacobians) {
  // Construct a spline that we can interpolate continuously on.
  DefaultSyntheticTest synthetic_test;
  const auto trajectory_world_body = synthetic_test.TrajectoryAsMap();
  const auto timestamps = synthetic_test.TrajectoryMapKeys();
  Trajectory trajectory;
  ASSERT_OK(trajectory.AddPoses(trajectory_world_body));
  // In small increments, test a subset of angles near knots.
  const double dt = 1e-6;
  for (int i = 0; i < timestamps.size() - 2; ++i) {
    // Interpolate spline at two timestamps close to each other.
    const double stamp_i = timestamps.at(i);
    const double stamp_ii = stamp_i + dt;
    const double stamp_iii = stamp_ii + dt;
    const Eigen::Vector<double, 6> pose_vector_i =
        trajectory.spline().Interpolate({stamp_i}, /*derivative=*/0).value()[0];
    const Eigen::Vector<double, 6> pose_vector_ii =
        trajectory.spline().Interpolate(
            {stamp_ii}, /*derivative=*/0).value()[0];
    const Eigen::Vector<double, 6> pose_vector_iii =
        trajectory.spline().Interpolate(
            {stamp_iii}, /*derivative=*/0).value()[0];
    const Eigen::Vector3d phi_world_body_i = pose_vector_i.head(3);
    const Eigen::Vector3d phi_world_body_ii = pose_vector_ii.head(3);
    const Eigen::Vector3d phi_world_body_iii = pose_vector_iii.head(3);
    const Eigen::Vector3d phi_body_world_i = -phi_world_body_i;
    const Eigen::Vector3d phi_body_world_ii = -phi_world_body_ii;
    const Eigen::Vector3d phi_body_world_iii = -phi_world_body_iii;
    // Convert axis angle to rotation matrices.
    const Eigen::Matrix3d R_world_body_i = Eigen::AngleAxisd(
        phi_world_body_i.norm(), phi_world_body_i.normalized())
        .toRotationMatrix();
    const Eigen::Matrix3d R_world_body_ii = Eigen::AngleAxisd(
        phi_world_body_ii.norm(), phi_world_body_ii.normalized())
        .toRotationMatrix();
    const Eigen::Matrix3d R_world_body_iii = Eigen::AngleAxisd(
        phi_world_body_iii.norm(), phi_world_body_iii.normalized())
        .toRotationMatrix();
    const Eigen::Matrix3d R_body_world_i = Eigen::AngleAxisd(
        phi_body_world_i.norm(), phi_body_world_i.normalized())
        .toRotationMatrix();
    const Eigen::Matrix3d R_body_world_ii = Eigen::AngleAxisd(
        phi_body_world_ii.norm(), phi_body_world_ii.normalized())
        .toRotationMatrix();
    const Eigen::Matrix3d R_body_world_iii = Eigen::AngleAxisd(
        phi_body_world_iii.norm(), phi_body_world_iii.normalized())
        .toRotationMatrix();
    // Compute rotation deltas for world-from-body and body-from-world rotations.
    const Eigen::Matrix3d dR_world_body_i =
        R_world_body_ii * R_world_body_i.transpose();
    const Eigen::Matrix3d dR_world_body_ii =
        R_world_body_iii * R_world_body_ii.transpose();
    const Eigen::Matrix3d dR_body_world_i =
        R_body_world_ii * R_body_world_i.transpose();
    const Eigen::Matrix3d dR_body_world_ii =
        R_body_world_iii * R_body_world_ii.transpose();
    // Convert rotation deltas into rates in the log-space.
    const double dt_inv = 1.0 / dt;
    const Eigen::AngleAxisd dphi_world_body_i(dR_world_body_i);
    const Eigen::AngleAxisd dphi_body_world_i(dR_body_world_i);
    const Eigen::AngleAxisd dphi_world_body_ii(dR_world_body_ii);
    const Eigen::AngleAxisd dphi_body_world_ii(dR_body_world_ii);
    const Eigen::Vector3d expected_omega_world_body_i =
        dt_inv * (dphi_world_body_i.angle() * dphi_world_body_i.axis());
    const Eigen::Vector3d expected_omega_body_world_i =
        dt_inv * (dphi_body_world_i.angle() * dphi_body_world_i.axis());
    const Eigen::Vector3d expected_omega_world_body_ii =
        dt_inv * (dphi_world_body_ii.angle() * dphi_world_body_ii.axis());
    const Eigen::Vector3d expected_omega_body_world_ii =
        dt_inv * (dphi_body_world_ii.angle() * dphi_body_world_ii.axis());
    const Eigen::Vector3d expected_alpha_world_body =
        dt_inv * (expected_omega_world_body_ii - expected_omega_world_body_i);
    const Eigen::Vector3d expected_alpha_body_world =
        dt_inv * (expected_omega_body_world_ii - expected_omega_body_world_i);
    // Compute analytical first and second time derivatives.
    const Eigen::Vector<double, 6> pose_dot_vector_i =
        trajectory.spline().Interpolate({stamp_i}, /*derivative=*/1).value()[0];
    const Eigen::Vector<double, 6> pose_ddot_vector_i =
        trajectory.spline().Interpolate({stamp_i}, /*derivative=*/2).value()[0];
    const Eigen::Vector3d phi_dot_world_body_i = pose_dot_vector_i.head(3);
    const Eigen::Vector3d phi_dot_body_world_i = -phi_dot_world_body_i;
    const Eigen::Vector3d phi_ddot_world_body_i = pose_ddot_vector_i.head(3);
    const Eigen::Vector3d phi_ddot_body_world_i = -phi_ddot_world_body_i;
    // Compute angular velocity.
    const Eigen::Matrix3d J_world_body = ExpSO3Jacobian(phi_world_body_i);
    const Eigen::Vector3d actual_omega_world_body =
        J_world_body * phi_dot_world_body_i;
    const Eigen::Matrix3d J_body_world = ExpSO3Jacobian(phi_body_world_i);
    const Eigen::Vector3d actual_omega_body_world =
        J_body_world * phi_dot_body_world_i;
    // Compute angular acceleration.
    const Eigen::Matrix3d J_dot_world_body =
        ExpSO3JacobianDot(phi_world_body_i, phi_dot_world_body_i);
    const Eigen::Matrix3d J_dot_body_world =
      ExpSO3JacobianDot(phi_body_world_i, phi_dot_body_world_i);
    const Eigen::Vector3d actual_alpha_world_body =
      J_dot_world_body * phi_dot_world_body_i +
      J_world_body * phi_ddot_world_body_i;
    const Eigen::Vector3d actual_alpha_body_world =
      J_dot_body_world * phi_dot_body_world_i +
      J_body_world * phi_ddot_body_world_i;
    // Compare.
    constexpr double kSmallErrorOmega = 1e-4;
    constexpr double kSmallErrorAlpha = 1e-2;
    const Eigen::Vector3d error_omega_world_body =
        actual_omega_world_body - expected_omega_world_body_i;
    const Eigen::Vector3d error_omega_body_world =
        actual_omega_body_world - expected_omega_body_world_i;
    const Eigen::Vector3d error_alpha_world_body =
        actual_alpha_world_body - expected_alpha_world_body;
    const Eigen::Vector3d error_alpha_body_world =
        actual_alpha_body_world - expected_alpha_body_world;
    EXPECT_LT(error_omega_world_body.norm(), kSmallErrorOmega);
    EXPECT_LT(error_omega_body_world.norm(), kSmallErrorOmega);
    EXPECT_LT(error_alpha_world_body.norm(), kSmallErrorAlpha);
    EXPECT_LT(error_alpha_body_world.norm(), kSmallErrorAlpha);
  }
}


} // namespace
} // namespace calico
