#ifndef CALICO_MATCHERS_H_
#define CALICO_MATCHERS_H_

#include <iostream>

#include "calico/statusor_macros.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "Eigen/Dense"
#include "calico/typedefs.h"
#include "gmock/gmock.h"


namespace calico {

// Matcher for a pose object with tolerance.
MATCHER_P2(PoseIsApprox, expected_pose, tolerance, "") {
  return (arg.rotation().isApprox(expected_pose.rotation(), tolerance) &&
          arg.translation().isApprox(expected_pose.translation(), tolerance));
}

// Matcher for a pose object with double precision floating point tolerance.
MATCHER_P(PoseEq, expected_pose, "") {
  return (arg.rotation().isApprox(expected_pose.rotation()) &&
          arg.translation().isApprox(expected_pose.translation()));
}

// Matcher for an Eigen vector with double precision floating point tolerance.
MATCHER_P(EigenEq, expected_vector, "") {
  return (arg.isApprox(expected_vector));
}

// Matcher for an Eigen vector with given tolerance.
MATCHER_P2(EigenIsApprox, expected_vector, tolerance, "") {
  return (arg.isApprox(expected_vector, tolerance));
}

MATCHER_P(ImageSizeEq, expected_image_size, "") {
  return (arg.width == expected_image_size.width &&
          arg.height == expected_image_size.height);
}

MATCHER_P(StatusIs, status, "") {
  return arg.status().code() == status;
}


} // namespace calico
#endif // CALICO_MATCHERS_H_
