#ifndef CALICO_TYPEDEFS_H_
#define CALICO_TYPEDEFS_H_

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

 private:
  Eigen::Quaterniond q_;
  Eigen::Vector3d t_;
};

} // namespace calico

#endif // CALICO_TYPEDEFS_H_
