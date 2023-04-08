#ifndef CALICO_GEOMETRY_H_
#define CALICO_GEOMETRY_H_

#include "calico/typedefs.h"
#include "Eigen/Dense"


namespace calico {

/// Convert a 3-vector to a skew-symmetric cross-product matrix.
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

/// Converts a skew symmetric matrix back into a vector.
template <typename T>
inline Eigen::Vector3<T> iSkew(const Eigen::Matrix3<T>& V) {
  Eigen::Vector3<T> v;
  v.x() = V(2, 1) - V(1, 2);
  v.y() = V(0, 2) - V(2, 0);
  v.z() = V(1, 0) - V(0, 1);
  return 0.5 * v;
}

/// Fourth order Taylor expansion of sin(theta).
template <typename T>
inline T SmallAngleSin(const T theta) {
  const T theta_sq = theta * theta;
  return theta * (T(1.0) - theta_sq *
                  (T(1.0 / 6.0) + theta_sq *
                   (T(1.0 / 120.0) - theta_sq * T(1.0 / 5040.0))));
}

/// Fourth order Taylor expansion of cos(theta).
template <typename T>
inline T SmallAngleCos(const T theta) {
  const T theta_sq = theta * theta;
  return T(1.0) - theta_sq * (T(0.5) - theta_sq *
                              (T(1.0 / 24.0) + theta_sq *
                               (T(1.0/720.0) - theta_sq * T(1.0 / 40320.0))));
}

/// Convert axis-angle to rotation matrix.
template <typename T>
inline Eigen::Matrix3<T> ExpSO3(const Eigen::Vector3<T>& phi) {
  constexpr T kSmallAngle = 1e-7;
  const T theta = phi.norm();
  if (theta == T(0)) {
    return Eigen::Matrix3<T>::Identity();
  }
  T sin_theta, one_m_cos_theta;
  if (theta < kSmallAngle) {
    sin_theta = SmallAngleSin(theta);
    one_m_cos_theta = T(1.0) - SmallAngleCos(theta);
  } else {
    sin_theta = sin(theta);
    one_m_cos_theta = T(1.0) - cos(theta);
  }
  const Eigen::Vector3<T> phi_hat = phi / theta;
  const Eigen::Matrix3<T> Phi = Skew(phi_hat);
  Eigen::Matrix3<T> R =
      Eigen::Matrix3<T>::Identity() + sin_theta * Phi +
      one_m_cos_theta * Phi * Phi;
  return R;
}

/// Convert rotation matrix to axis-angle. Algorithm pulled from this page:
/// https://math.stackexchange.com/questions/83874/efficient-and-accurate-numerical-implementation-of-the-inverse-rodrigues-rotatio
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
    const T sin_theta = theta < kSmallAngle ? SmallAngleSin(theta) : sin(theta);
    phi *= (theta / sin_theta);
  } else if (cos_theta > -kInvSqrt2) {
    const T theta = acos(cos_theta);
    const T sin_theta = theta < kSmallAngle ? SmallAngleSin(theta) : sin(theta);
    phi *= (theta / sin_theta);
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

/// Compute the manifold Jacobian matrix of the Rodrigues formula, i.e.:
/// \f[
/// \mathbf{R}\left(\boldsymbol{\Phi}\right) = \exp\left(
///   \mathbf{I} + \frac{\sin(\theta)}{\theta} \left[\boldsymbol{\Phi}\right]_\times +
///   \frac{1 - \cos(\theta)}{\theta^2}{\left[\boldsymbol{\Phi}\right]_\times}^2
/// \right)\\
/// \frac{\partial \exp\left(\left[\boldsymbol{\Phi}\right]_\times\right)}{\partial \boldsymbol{\Phi}} =
///   \mathbf{I} + \frac{1-\cos(\theta)}{\theta^2} \left[\boldsymbol{\Phi}\right]_\times +
///   \frac{\theta-\sin(\theta)}{\theta^3} {\left[\boldsymbol{\Phi}\right]_\times}^2\\
/// \theta = \boldsymbol{\Phi}^T\boldsymbol{\Phi}
/// \f]
template <typename T>
inline Eigen::Matrix3<T> ExpSO3Jacobian(
    const Eigen::Vector3<T>& phi) {
  const T theta_sq = phi.squaredNorm();
  Eigen::Matrix3<T> J;
  J.setIdentity();
  if (theta_sq == static_cast<T>(0.0)) {
    return J;
  }
  const T theta = sqrt(theta_sq);
  T one_m_cos_theta, sin_theta;
  if (theta < T(1e-7)) {
    sin_theta = SmallAngleSin(theta);
    one_m_cos_theta = T(1.0) - SmallAngleCos(theta);
  } else {
    sin_theta = sin(theta);
    one_m_cos_theta = T(1.0) - cos(theta);
  }
  const T inv_theta = T(1.0) / theta;
  const Eigen::Vector3<T> phi_hat = inv_theta * phi;
  const Eigen::Matrix3<T> phi_hat_x = Skew(phi_hat);
  J += inv_theta * (one_m_cos_theta * phi_hat_x +
                    (theta - sin_theta) * phi_hat_x * phi_hat_x);
  return J;
}

/// Compute the manifold Hessian matrix of the Rodrigues formula, i.e.:
/// \f[
///   \mathbf{R}(\boldsymbol{\Phi}) =
///     \exp\left(\left[\boldsymbol{\Phi}\right]_\times\right) =
///     \mathbf{I} + \frac{\sin(\theta)}{\theta}\left[\boldsymbol{\Phi}\right]_\times +
///     \frac{1-\cos(\theta)}{\theta^2} {\left[\boldsymbol{\Phi}\right]_\times}^2\\
///  \mathbf{H} = \frac{\partial^2 \exp\left(\left[\boldsymbol{\Phi}\right]_\times\right)}
///    {\partial \boldsymbol{\Phi}^2}
/// \f]
template <typename T>
inline std::vector<Eigen::Matrix3<T>> ExpSO3Hessian(
    const Eigen::Vector3<T>& phi) {
  const std::vector<Eigen::Matrix3<T>> G = {
    Skew(Eigen::Vector3<T>(T(1), T(0), T(0))),
    Skew(Eigen::Vector3<T>(T(0), T(1), T(0))),
    Skew(Eigen::Vector3<T>(T(0), T(0), T(1)))
  };

  std::vector<Eigen::Matrix3<T>> H(3);
  std::fill(H.begin(), H.end(), Eigen::Matrix3<T>::Zero());
  const T theta_sq = phi.squaredNorm();
  if (theta_sq == T(0.0)) {
    return H;
  }
  const T theta = sqrt(theta_sq);
  T ct, st;
  if (theta < T(1e-7)) {
    ct = SmallAngleCos(theta);
    st = SmallAngleSin(theta);
  } else {
    ct = cos(theta);
    st = sin(theta);
  }
  const T inv_theta = T(1.0) / theta;
  const T inv_theta_sq = inv_theta * inv_theta;
  const Eigen::Vector3<T> phi_hat = inv_theta * phi;
  const Eigen::Matrix3<T> phi_hat_x = Skew(phi_hat);
  const T c0 = ct - st * inv_theta;
  const T c1 = (T(1.0) - ct) * inv_theta_sq;
  const T c2 = T(3.0) * inv_theta_sq * st - inv_theta * (ct - T(2.0));
  const T c3 = inv_theta_sq * (theta - st);
  for (int i = 0; i < 3; ++i) {
    H[i] = (c0 * phi_hat(i)) * phi_hat_x + c1 * G[i] +
        (c2 * phi_hat(i)) * (phi_hat_x * phi_hat_x) +
        c3 * (G[i] * phi_hat_x + phi_hat_x * G[i]);
  }
  return H;
}

/// Compute the time derivative of the Rodrigues formula manifold Jacobian.
template <typename T>
inline Eigen::Matrix3<T> ExpSO3JacobianDot(
    const Eigen::Vector3<T>& phi, const Eigen::Vector3<T>& phi_dot) {
  const std::vector<Eigen::Matrix3<T>> H = ExpSO3Hessian(phi);
  Eigen::Matrix3<T> Jdot;
  for (int i = 0; i < H.size(); ++i) {
    Jdot.col(i) = H.at(i) * phi_dot;
  }
  return Jdot;
}

} // namespace calico

#endif // CALICO_GEOMETRY_H_
