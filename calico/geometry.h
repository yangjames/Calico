#ifndef CALICO_GEOMETRY_H_
#define CALICO_GEOMETRY_H_

#include "Eigen/Dense"

namespace calico {


// Convert a 3-vector to a skew-symmetric cross-product matrix.
template <typename T>
inline Eigen::Matrix3<T> Skew(const Eigen::Vector3<T>& v) {
  Eigen::Matrix3<T> V;
  V.setZero();
  V(0, 1) = -v.z();
  V(1, 0) = v.z();
  V(0, 2) = v.y();
  V(2, 0) = -v.y();
  V(1, 2) = -v.x();
  V(2, 1) = v.x();
  return V;
}

// Compute the angular velocity of a rigid body. Current orientation `phi`
// is an axis-angle vector such that `angle = phi.norm()`. `phi_dot` is the time
// derivative of `phi`. This method will use these two quantities to transform
// the axis-angle rate vector via the appopriate SO(3) manifold Jacobian.
template <typename T>
inline Eigen::Vector3<T> ComputeAngularVelocity(const Eigen::Vector3<T>& phi,
                                                const Eigen::Vector3<T>& phi_dot) {
  const T theta_sq = phi.squaredNorm();
  Eigen::Matrix3<T> J;
  J.setIdentity();
  if (theta_sq != static_cast<T>(0.0)) {
    const T theta = sqrt(theta_sq);
    const T theta_fo = theta_sq * theta_sq;
    T c1, c2;
    // If small angle, compute the first 3 terms of the Taylor expansion.
    if (abs(theta) < static_cast<T>(1e-7)) {
      c1 = static_cast<T>(0.5) - theta_sq * static_cast<T>(1.0 / 24.0) +
        theta_fo * static_cast<T>(1.0 / 720.0);
      c2 = static_cast<T>(1.0 / 6.0) - theta_sq * static_cast<T>(1.0 / 120.0) +
        theta_fo * static_cast<T>(1.0 / 5040.0);
    } else {
      const T inv_theta_sq = static_cast<T>(1.0) / theta_sq;
      c1 = (static_cast<T>(1.0) - cos(theta)) * inv_theta_sq;
      c2 = (static_cast<T>(1.0) - sin(theta) / theta) * inv_theta_sq;
    }
    Eigen::Matrix3<T> phi_x = Skew(phi);
    J += c1 * phi_x + c2 * phi_x * phi_x;
  }
  const Eigen::Vector3<T> omega = J * phi_dot;
  return omega;
}

                           
} // namespace calico

#endif // CALICO_GEOMETRY_H_
