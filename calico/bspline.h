#ifndef CALICO_BSPLINE_H_
#define CALICO_BSPLINE_H_

#include "calico/typedefs.h"

#include "absl/status/statusor.h"
#include "Eigen/Dense"


namespace calico {

// Generic N-DOF spline fitter. This class implements the following paper:
// "General Matrix Representations for B-Splines", K. Qin.
// https://xiaoxingchen.github.io/2020/03/02/bspline_in_so3/general_matrix_representation_for_bsplines.pdf
template <typename T, int N>
class BSpline {
 public:

  // Define a container for an `N`-DOF B-spline with type `T`.
  ~BSpline() = default;

  // Fits an N-DOF uniform B-spline fitted to given timestamps
  // and N-dimensional data. User also specifies the spline order and the knot
  // frequency of the spline.
  absl::Status FitToData(const std::vector<T>& time,
                         const std::vector<Eigen::Vector<T,N>>& data,
                         int spline_order, double knot_frequency);

  // Interpolate the spline at given times for the given derivative. If no
  // derivative is specified, it defaults to direct interpolation.
  absl::StatusOr<std::vector<Eigen::Vector<T,N>>>
  Interpolate(const std::vector<T>& times, int derivative = 0) const;

  // Returns the number of control points in the spline.
  int NumberOfControlPoints() const;

  // Returns the degree of the spline, i.e. SplineOrder() - 1. For example, if
  // the spline is defined as `y(t) = a0 + a1*t + a2*t^2`, spline degree is 2
  // which is the highest degree of time,.
  int SplineDegree() const;

  // Returns the spline order.
  int SplineOrder() const;

 private:
  // Data/parameters for fitting spline.
  int spline_order_;
  double knot_frequency_;
  std::vector<Eigen::Vector<T,N>> data_;
  std::vector<T> time_;

  // Derived properties of the spline.
  int spline_degree_;
  int num_knots_;
  int num_valid_knots_;
  int num_control_points_;
  int num_valid_segments_;
  std::vector<T> knots_;
  std::vector<T> valid_knots_;
  Eigen::MatrixX<T> derivative_coeffs_;
  std::vector<Eigen::MatrixX<T>> Mi_;
  Eigen::MatrixX<T> control_points_;

  // Convenience function for getting the index of the active control point for
  // a queried time.
  int GetControlPointIndex(T query_time) const;

  // Convenience function for getting the basis matrix for a specific spline
  // index and derivative.
  Eigen::MatrixX<T> GetBasisMatrix(int spline_idx, int derivative) const;

  // Convenience function for computing the power rule coefficients. These
  // coefficients are used when computing derivatives of the spline.
  void ComputePowerRuleCoefficients();

  // Convenience function for computing a knot vector.
  void ComputeKnotVector();

  // Convenience function for computing the basis matrices for this spline.
  void ComputeBasisMatrices();

  // Methods implementing the recursive algorithm for computing the basis
  // matrix per Theorem 1 of "General Matrix Representations for B-Splines".
  //   M_k = [ M_km1 ] * A + [  0^T  ] * B
  //       = [  0^T  ]       [ M_km1 ]
  //   
  //   A = [ 1-d_0,i-k+2   d_0,i-k+2                              ]
  //       [              1-d_0,i-k+3  d_0,i-k+3                  ]
  //       [                  ...                                 ]
  //       [                                       1-d_0,i  d_0,i ]
  //    
  //   B = [ -d_1,i-k+2   d_1,i-k+2                               ]
  //       [             -d_1,i-k+3   d_0,i-k+3                   ]
  //       [                    ...                               ]
  //       [                                      -d_1,i  d_1,i   ]
  // where k is the spline order and i is the spline segment number.
  Eigen::MatrixX<T> M(int k, int i);
  T d_0(int k, int i, int j);
  T d_1(int k, int i, int j);

  // Fit spline to data.
  void FitSpline();

  // Check that inputs provided for spline fit are valid.
  absl::Status CheckDataForSplineFit(
      const std::vector<T>& time,
      const std::vector<Eigen::Vector<T,N>>& data, int spline_order,
      double knot_frequency);
};

} // namespace calico

#include "calico/bspline.hpp"
#endif // CALICO_BSPLINE_H_
