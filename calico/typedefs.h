#ifndef CALICO_TYPEDEFS_H_
#define CALICO_TYPEDEFS_H_

#include <iostream>

#include "Eigen/Dense"


namespace Eigen {

template <typename T>
using Vector2 = Matrix<T,2,1>;

template <typename T>
using Vector3 = Matrix<T,3,1>;

template <typename T>
using VectorX = Matrix<T,Dynamic,1>;

} // namespace Eigen

namespace calico {

class Pose3 {
 public:
  Pose3() {
    q_.setIdentity();
    t_.setZero();
  }

  Pose3(const Eigen::Quaterniond& q,
        const Eigen::Vector3d& t)
    : q_(q), t_(t) {}
  
  Eigen::Quaterniond& rotation() {
    return q_;
  }

  const Eigen::Quaterniond& rotation() const {
    return q_;
  }

  Eigen::Vector3d& translation() {
    return t_;
  }

  const Eigen::Vector3d& translation() const {
    return t_;
  }

  Pose3 operator*(const Pose3& T_b_a) const {
    // this = T_c_b
    const Eigen::Quaterniond& q_c_b = this->rotation();
    const Eigen::Quaterniond& q_b_a = T_b_a.rotation();
    const Eigen::Vector3d& t_c_b = this->translation();
    const Eigen::Vector3d& t_b_a = T_b_a.translation();
    const Eigen::Quaterniond q_c_a = q_c_b * q_b_a;
    const Eigen::Vector3d t_c_a = q_c_b * t_b_a + t_c_b;
    return Pose3(q_c_a, t_c_a);
  }

  Eigen::Vector3d operator*(const Eigen::Vector3d& p) const {
    return this->rotation()*p + this->translation();
  }

  Pose3 inverse() const {
    const Eigen::Quaterniond q_inv = this->rotation().conjugate();
    const Eigen::Vector3d t_inv = -(q_inv * this->translation());
    return Pose3(q_inv, t_inv);
  }

  bool isApprox(const Pose3& pose) const {
    return (pose.rotation().isApprox(q_) && pose.translation().isApprox(t_));
  }

  friend std::ostream& operator<<(std::ostream& os, const Pose3& pose) {
    os
      << "q: " << pose.rotation().coeffs().transpose() <<
      ", t: " << pose.translation().transpose();
    return os;
  }

 private:
  Eigen::Quaterniond q_;
  Eigen::Vector3d t_;
};

} // namespace calico

#endif // CALICO_TYPEDEFS_H_
