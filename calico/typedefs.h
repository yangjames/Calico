#ifndef CALICO_TYPEDEFS_H_
#define CALICO_TYPEDEFS_H_

#include <iostream>

#include "Eigen/Dense"


namespace Eigen {

template <typename T>
using Vector2 = Matrix<T, 2, 1>;

template <typename T>
using Vector3 = Matrix<T, 3, 1>;

template <typename T>
using Vector4 = Matrix<T, 4, 1>;

template <typename T>
using VectorX = Matrix<T, Dynamic, 1>;

template <typename T>
using MatrixX = Matrix<T, Dynamic, Dynamic>;

template <typename T, int N>
using Vector = Matrix<T, N, 1>;

} // namespace Eigen

namespace calico {

template <typename T>
class Pose3 {
 public:
  Pose3() {
    q_.setIdentity();
    t_.setZero();
  }

  Pose3(const Eigen::Quaternion<T>& q,
        const Eigen::Vector3<T>& t)
    : q_(q), t_(t) {}
  
  // Rotation accessors.
  Eigen::Quaternion<T>& rotation() {
    return q_;
  }
  const Eigen::Quaternion<T>& rotation() const {
    return q_;
  }

  // Translation accessors.
  Eigen::Vector3<T>& translation() {
    return t_;
  }
  const Eigen::Vector3<T>& translation() const {
    return t_;
  }

  // Rotation setter/getter as 4-vectors for python bindings.
  void SetRotation(const Eigen::Vector4<T>& q) {
    const Eigen::Vector4<T> qn = q.normalized();
    q_.w() = qn(0);
    q_.x() = qn(1);
    q_.y() = qn(2);
    q_.z() = qn(3);
  }
  Eigen::Vector4<T> GetRotation() const {
    return Eigen::Vector4<T>(q_.w(), q_.x(), q_.y(), q_.z());
  }
  // Translation setter/getter for python bindings.
  void SetTranslation(const Eigen::Vector3<T>& t) {
    t_ = t;
  }
  Eigen::Vector3<T> GetTranslation() const {
    return t_;
  }


  Pose3 operator*(const Pose3<T>& T_b_a) const {
    // this = T_c_b
    const Eigen::Quaternion<T>& q_c_b = this->rotation();
    const Eigen::Quaternion<T>& q_b_a = T_b_a.rotation();
    const Eigen::Vector3<T>& t_c_b = this->translation();
    const Eigen::Vector3<T>& t_b_a = T_b_a.translation();
    const Eigen::Quaternion<T> q_c_a = q_c_b * q_b_a;
    const Eigen::Vector3<T> t_c_a = q_c_b * t_b_a + t_c_b;
    return Pose3(q_c_a, t_c_a);
  }

  Eigen::Vector3<T> operator*(const Eigen::Vector3<T>& p) const {
    return this->rotation()*p + this->translation();
  }

  Pose3<T> inverse() const {
    const Eigen::Quaternion<T> q_inv = this->rotation().conjugate();
    const Eigen::Vector3<T> t_inv = -(q_inv * this->translation());
    return Pose3(q_inv, t_inv);
  }

  bool isApprox(const Pose3<T>& pose) const {
    return (pose.rotation().isApprox(q_) && pose.translation().isApprox(t_));
  }

  friend std::ostream& operator<<(std::ostream& os, const Pose3<T>& pose) {
    os
      << "q: " << pose.rotation().coeffs().transpose() <<
      ", t: " << pose.translation().transpose();
    return os;
  }

 private:
  Eigen::Quaternion<T> q_;
  Eigen::Vector3<T> t_;
};

using Pose3d = Pose3<double>;


} // namespace calico

#endif // CALICO_TYPEDEFS_H_
