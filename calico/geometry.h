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

// Converts a skew symmetric matrix back into a vector.
template <typename T>
inline Eigen::Vector3<T> iSkew(const Eigen::Matrix3<T>& V) {
  Eigen::Vector3<T> v;
  v.x() = V(2, 1) - V(1, 2);
  v.y() = V(0, 2) - V(2, 0);
  v.z() = V(1, 0) - V(0, 1);
  return 0.5 * v;
}

// Fourth order Taylor expansion of sin(theta).
template <typename T>
inline T SmallAngleSin(const T theta) {
  const T theta_sq = theta * theta;
  return theta * (T(1.0) - theta_sq *
                  (T(1.0 / 6.0) + theta_sq *
                   (T(1.0 / 120.0) - theta_sq * T(1.0 / 5040.0))));
}

// Fourth order Taylor expansion of 1 - cos(theta).
template <typename T>
inline T SmallAngleOneMinusCos(const T theta) {
  const T theta_sq = theta * theta;
  return theta_sq * (T(0.5) - theta_sq *
                     (T(1.0 / 24.0) + theta_sq *
                      (T(1.0/720.0) - theta_sq * T(1.0 / 40320.0))));
}

// Convert axis-angle to rotation matrix.
template <typename T>
inline Eigen::Matrix3<T> ExpSO3(const Eigen::Vector3<T>& phi) {
  constexpr T kSmallAngle = 1e-7;
  const T theta = phi.norm();
  if (theta == T(0)) {
    return Eigen::Matrix3<T>::Identity();
  }
  T sin_term, cos_term;
  if (theta < kSmallAngle) {
    sin_term = SmallAngleSin(theta);
    cos_term = SmallAngleOneMinusCos(theta);
  } else {
    sin_term = sin(theta);
    cos_term = T(1.0) - cos(theta);
  }
  const Eigen::Vector3<T> phi_hat = phi / theta;
  const Eigen::Matrix3<T> Phi = Skew(phi_hat);
  Eigen::Matrix3<T> R =
      Eigen::Matrix3<T>::Identity() + sin_term * Phi + cos_term * Phi * Phi;
  return R;
}

// Convert rotation matrix to axis-angle. Algorithm pulled from this page:
// https://math.stackexchange.com/questions/83874/efficient-and-accurate-numerical-implementation-of-the-inverse-rodrigues-rotatio
template <typename T>
inline Eigen::Vector3<T> LnSO3(const Eigen::Matrix3<T>& R) {
  constexpr T kInvSqrt2 = T(1.0 / sqrt(2.0));
  constexpr T kSmallAngle = 1e-7;
  const T tr = R.trace();
  if (tr == T(3.0)) {
    return Eigen::Vector3<T>::Zero();
  }
  Eigen::Vector3<T> phi = iSkew(R);
  const T cos_theta = T(0.5) * (tr - T(1.0));
  const T sin_theta = phi.norm();

  if (cos_theta >= kInvSqrt2) {
    const T theta = asin(sin_theta);
    const T sin_term = theta < kSmallAngle ? SmallAngleSin(theta) : sin(theta);
    phi *= (theta / sin_term);
  } else if (cos_theta > -kInvSqrt2) {
    const T theta = acos(cos_theta);
    const T sin_term = theta < kSmallAngle ? SmallAngleSin(theta) : sin(theta);
    phi *= (theta / sin_term);
  } else {
    const Eigen::Vector3<T> diag = R.diagonal().array() - cos_theta;
    const T dx2 = diag.x() * diag.x();
    const T dy2 = diag.y() * diag.y();
    const T dz2 = diag.z() * diag.z();
    Eigen::Vector3<T> axis;
    if ((dx2 > dz2) && (dx2 > dy2)){
      axis.x() = diag.x();
      axis.y() = T(0.5) * (R(0, 1) + R(1, 0));
      axis.z() = T(0.5) * (R(0, 2) + R(2, 0));
    } else if (dy2 > dz2) {
      axis.x() = T(0.5) * (R(1, 0) + R(0, 1));
      axis.y() = diag.y();
      axis.z() = T(0.5) * (R(1, 2) + R(2, 1));
    } else{
      axis.x() = T(0.5) * (R(2, 0) + R(0, 2));
      axis.y() = T(0.5) * (R(2, 1) + R(1, 2));
      axis.z() = diag.z();
    }
    if ((phi.dot(axis)) < T(0)) {
      axis *= T(-1.0);
    }
    const T theta = M_PI - asin(sin_theta);
    phi = theta * axis.normalized();
  }
  return phi;
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
