#ifndef CALICO_MATCHERS_H_
#define CALICO_MATCHERS_H_

#include <iostream>

#include "calico/geometry.h"
#include "calico/statusor_macros.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "Eigen/Dense"
#include "calico/typedefs.h"
#include "gmock/gmock.h"


namespace calico {

// Matcher for a pose object with tolerance.
MATCHER_P2(PoseIsApprox, expected_pose, tolerance, "") {
  const Eigen::Matrix3d R1 = arg.rotation().toRotationMatrix();
  const Eigen::Matrix3d R2 = expected_pose.rotation().toRotationMatrix();
  const Eigen::Matrix3d dR = R1 * R2.transpose();
  const Eigen::Vector3d rot_err = LnSO3(dR);
  const Eigen::Vector3d pos_err =
      arg.translation() - expected_pose.translation();
  return (rot_err.norm() < tolerance && pos_err.norm() < tolerance);
}

// Matcher for a pose object with double precision floating point tolerance.
MATCHER_P(PoseEq, expected_pose, "") {
  return (arg.rotation().isApprox(expected_pose.rotation()) &&
          arg.translation().isApprox(expected_pose.translation()));
}

// Matcher for an Eigen vector with double precision floating point tolerance.
MATCHER_P(EigenEq, expected_vector, "") {
  *result_listener << arg.transpose() << " vs. " << expected_vector.transpose();
  return (arg.isApprox(expected_vector));
}

// Matcher for an Eigen vector with given tolerance.
MATCHER_P2(EigenIsApprox, expected_vector, tolerance, "") {
  *result_listener << arg.transpose() << " vs. " << expected_vector.transpose();
  return ((arg - expected_vector).norm() < tolerance);
}

MATCHER_P(StatusIs, status, "") {
  return arg.status().code() == status;
}

MATCHER_P(StatusCodeIs, status, "") {
  return arg.code() == status;
}

} // namespace calico
#endif // CALICO_MATCHERS_H_
