#ifndef CALICO_BSPLINE_H_
#define CALICO_BSPLINE_H_

#include "calico/typedefs.h"

#include "absl/status/statusor.h"
#include "ceres/problem.h"
#include "Eigen/Dense"


namespace calico {

// Generic N-DOF spline fitter. This class implements the following paper:
// "General Matrix Representations for B-Splines", K. Qin.
// https://xiaoxingchen.github.io/2020/03/02/bspline_in_so3/general_matrix_representation_for_bsplines.pdf
template <int N, typename T = double>
class BSpline {
 public:

  ~BSpline() = default;

  // Add this spline's control points to a ceres problem. Returns the number of
  // parameters added, which should be N * number of control points.
  int AddParametersToProblem(ceres::Problem& problem);

  // Fits an N-DOF uniform B-spline fitted to given timestamps
  // and N-dimensional data. User also specifies the spline order and the knot
  // frequency of the spline.
  absl::Status FitToData(const std::vector<T>& time,
                         const std::vector<Eigen::Vector<T,N>>& data,
                         int spline_order, T knot_frequency);

  // Convenience function for getting the index of the active control point for
  // a queried time.
  int GetControlPointIndex(T query_time) const;

  // Convenience function for getting the index of the appropriate beginning
  // knot for a spline segment with index control_point_index.
  int GetKnotIndexFromControlPointIndex(int control_point_index) const;

  // Interpolate the spline at given times for the given derivative. If no
  // derivative is specified, it defaults to direct interpolation.
  absl::StatusOr<std::vector<Eigen::Vector<T,N>>>
  Interpolate(const std::vector<T>& times, int derivative = 0) const;

  // Setter/getter for knot vector.
  const std::vector<T>& knots() const { return knots_; }
  std::vector<T>& knots() { return knots_; }

  // Setter/getter for control points.
  const Eigen::MatrixXd& control_points() const {
    return control_points_;
  }
  Eigen::MatrixXd& control_points() {
    return control_points_;
  }

  // Setter/getter for basis matrices.
  const std::vector<Eigen::MatrixX<T>>& basis_matrices() const {
    return Mi_;
  }
  std::vector<Eigen::MatrixX<T>>& basis_matrices() {
    return Mi_;
  }

  static Eigen::Vector<T, N> Evaluate(
      const Eigen::Ref<const Eigen::MatrixX<T>>& control_points_set, T knot0,
      T knot1, const Eigen::MatrixX<T>& basis_matrix, T stamp, int derivative);

 private:
  // Data/parameters for fitting spline.
  int spline_order_;
  T knot_frequency_;
  std::vector<Eigen::Vector<T,N>> data_;
  std::vector<T> time_;

  // Derived properties of the spline.
  int spline_degree_;
  std::vector<T> knots_;
  std::vector<T> valid_knots_;
  Eigen::MatrixXd derivative_coeffs_;
  std::vector<Eigen::MatrixXd> Mi_;
  Eigen::MatrixXd control_points_; // num_control_pointsxN matrix.


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
      T knot_frequency);
};

} // namespace calico

#include "calico/bspline.hpp"
#endif // CALICO_BSPLINE_H_
