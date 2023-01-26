#ifndef CALICO_TYPEDEFS_H_
#define CALICO_TYPEDEFS_H_

#include "Eigen/Dense"

namespace Eigen {

template<typename T>
using Vector2 = Matrix<T,2,1>;

template<typename T>
using Vector3 = Matrix<T,3,1>;

template<typename T>
using VectorX = Matrix<T,Dynamic,1>;

} // namespace Eigen

#endif // CALICO_TYPEDEFS_H_
