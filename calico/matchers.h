#ifndef CALICO_MATCHERS_H_
#define CALICO_MATCHERS_H_

#include "Eigen/Dense"
#include "calico/typedefs.h"
#include "gmock/gmock.h"


namespace calico {

MATCHER_P2(PoseIsApprox, expected_pose, tolerance, "") {
  return (arg.rotation().isApprox(expected_pose.rotation(), tolerance) &&
          arg.translation().isApprox(expected_pose.translation(), tolerance));
}

MATCHER_P(PoseEq, expected_pose, "") {
  return (arg.rotation().isApprox(expected_pose.rotation()) &&
          arg.translation().isApprox(expected_pose.translation()));
}

MATCHER_P(EigenEq, expected_vector, "") {
  return (arg.isApprox(expected_vector));
}

MATCHER_P(ImageSizeEq, expected_image_size, "") {
  return (arg.width == expected_image_size.width &&
          arg.height == expected_image_size.height);
}


} // namespace calico
#endif // CALICO_MATCHERS_H_
