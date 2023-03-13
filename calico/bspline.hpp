#include <algorithm>

#include "calico/statusor_macros.h"
#include "ceres/problem.h"

namespace calico {


template <int N, typename T>
int BSpline<N, T>::AddParametersToProblem(ceres::Problem& problem) {
  problem.AddParameterBlock(control_points_.data(), control_points_.size());
  return control_points_.size();
}

template <int N, typename T>
absl::Status BSpline<N, T>::FitToData(
    const std::vector<T>& time,
    const std::vector<Eigen::Vector<T,N>>& data, int spline_order,
    T knot_frequency) {
  RETURN_IF_ERROR(
      CheckDataForSplineFit(time, data, spline_order, knot_frequency));
  time_ = time;
  data_ = data;
  spline_order_ = spline_order;
  knot_frequency_ = knot_frequency;
  // Derived information about the spline.
  spline_degree_ = spline_order_ - 1;
  ComputeKnotVector();
  ComputeBasisMatrices();
  // Fit the spline.
  FitSpline();
  return absl::OkStatus();
}

template<int N, typename T>
Eigen::Vector<T, N> BSpline<N, T>::Evaluate(
    const Eigen::Ref<const Eigen::MatrixX<T>>& control_points_set, T knot0,
    T knot1, const Eigen::MatrixX<T>& basis_matrix, T stamp,
    int derivative) {
  // Compute u.
  const T dt = knot1 - knot0;
  const T dt_inv = static_cast<T>(1.0) / dt;
  const T u = (stamp - knot0) * dt_inv;
  // Derivative of u with respect to t raised to n'th derivative power per
  // chain rule.
  T dnu_dtn = static_cast<T>(1.0);
  for (int j = 0; j < derivative; ++j) {
    dnu_dtn *= dt_inv;
  }
  // Construct U vector.
  Eigen::Matrix<T, 1, Eigen::Dynamic> derivative_coeffs(
      control_points_set.rows());
  derivative_coeffs.setOnes();
  derivative_coeffs.head(derivative).setZero();
  Eigen::Matrix<T, 1, Eigen::Dynamic> U(control_points_set.rows());
  U.setOnes();
  for (int i = derivative; i < derivative_coeffs.size(); ++i) {
    T coeff = static_cast<T>(1.0);
    for (int j = i - derivative; j < i; ++j) {
      coeff *= (j + 1);
    }
    derivative_coeffs(i) = coeff;
    U(i) = (i > derivative) ? (u * U(i-1)) : U(i);
  }
  U = U.array() * derivative_coeffs.array() * dnu_dtn;
  // Evaluate spline.
  return (U * basis_matrix * control_points_set).transpose();
}

template <int N, typename T>
absl::StatusOr<std::vector<Eigen::Vector<T, N>>> BSpline<N, T>::Interpolate(
    const std::vector<T>& times, int derivative) const {
  if (derivative < 0 || derivative > spline_degree_) {
    return absl::InvalidArgumentError("Invalid derivative for interpolation.");
  }
  for (const T& t : times) {
    if (t < valid_knots_.front() || t > valid_knots_.back()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Cannot interpolate ", t, ". Value is not within valid knots."));
    }
   }
  int num_interp = times.size();
  std::vector<Eigen::Vector<T, N>> y(num_interp);
  for (int i = 0; i < num_interp; ++i) {
    const int spline_idx = GetControlPointIndex(times[i]);
    const int knot_idx = GetKnotIndexFromControlPointIndex(spline_idx);
    const Eigen::MatrixX<T> UM = GetSplineBasis(spline_idx, times[i], derivative);
    const Eigen::MatrixX<T> control_points =
        control_points_.block(spline_idx, 0, spline_order_, N);
    y[i] = Evaluate(control_points, knots_[knot_idx], knots_[knot_idx + 1],
                    Mi_[spline_idx], times[i], derivative);
  }
  return y;
}

template <int N, typename T>
Eigen::MatrixX<T> BSpline<N, T>::GetSplineBasis(
    int spline_idx, T stamp, int derivative) const {
  // Compute u.
  const int knot_idx = GetKnotIndexFromControlPointIndex(spline_idx);
  const T& knot0 = knots_.at(knot_idx);
  const T& knot1 = knots_.at(knot_idx + 1);
  const T dt = knot1 - knot0;
  const T dt_inv = static_cast<T>(1.0) / dt;
  const T u = (stamp - knot0) * dt_inv;
  // Derivative of u with respect to t raised to n'th derivative power per
  // chain rule.
  T dnu_dtn = static_cast<T>(1.0);
  for (int j = 0; j < derivative; ++j) {
    dnu_dtn *= dt_inv;
  }
  // Construct U vector.
  Eigen::Matrix<T, 1, Eigen::Dynamic> derivative_coeffs(spline_order_);
  derivative_coeffs.setOnes();
  derivative_coeffs.head(derivative).setZero();
  Eigen::Matrix<T, 1, Eigen::Dynamic> U(spline_order_);
  U.setOnes();
  for (int i = derivative; i < derivative_coeffs.size(); ++i) {
    T coeff = static_cast<T>(1.0);
    for (int j = i - derivative; j < i; ++j) {
      coeff *= (j + 1);
    }
    derivative_coeffs(i) = coeff;
    U(i) = (i > derivative) ? (u * U(i-1)) : U(i);
  }
  U = U.array() * derivative_coeffs.array() * dnu_dtn;
  const Eigen::MatrixX<T>& basis_matrix = Mi_.at(spline_idx);
  return U * basis_matrix;
}


template <int N, typename T>
int BSpline<N, T>::GetControlPointIndex(T query_time) const {
  int spline_idx = -1;
  if (query_time == valid_knots_.back()) {
    spline_idx = valid_knots_.size() - 2;
  }
  else if (query_time < valid_knots_.back()) {
    auto lower =
      std::upper_bound(valid_knots_.begin(), valid_knots_.end(), query_time);
    spline_idx = lower - valid_knots_.begin() - 1;
  }
  return spline_idx;
}

template <int N, typename T>
int BSpline<N, T>::GetKnotIndexFromControlPointIndex(
    int control_point_index) const {
  return control_point_index + spline_degree_;
}

template <int N, typename T>
void BSpline<N, T>::ComputeKnotVector() {
  const T duration = time_.back() - time_.front();
  const T dt = 1.0 / knot_frequency_;
  const int num_valid_knots = 1 + std::ceil(duration * knot_frequency_);
  const int num_knots = num_valid_knots + 2 * spline_degree_;

  knots_.resize(num_knots);
  valid_knots_.resize(num_valid_knots);
  for (int i = -spline_degree_; i < num_knots - spline_degree_; ++i) {
    const T knot_value = time_.front() + dt * i;
    knots_[i + spline_degree_] = knot_value;
    int valid_knot_index = i;
    if (valid_knot_index > -1 && valid_knot_index < num_valid_knots) {
      valid_knots_[valid_knot_index] = knot_value;
    }
  }
}

template <int N, typename T>
void BSpline<N, T>::ComputeBasisMatrices() {
  const int num_valid_segments = valid_knots_.size() - 1;
  Mi_.resize(num_valid_segments);
  for (int i = 0; i < num_valid_segments; ++i) {
    Mi_[i] = M(spline_order_, i + spline_degree_);
  }
}

template<int N, typename T>
Eigen::MatrixX<T> BSpline<N, T>::M(int k, int i) {
  Eigen::MatrixX<T> M_k;
  if (k == 1) {
    M_k.resize(k, k);
    M_k(0, 0) = k;
    return M_k;
  }

  Eigen::MatrixX<T> M_km1 = M(k - 1, i);
  const int num_rows = M_km1.rows();
  const int num_cols = M_km1.cols();
  Eigen::MatrixX<T> M1(num_rows + 1, num_cols), M2(num_rows + 1, num_cols);
  M1.setZero();
  M2.setZero();
  M1.block(0, 0, num_rows, num_cols) = M_km1;
  M2.block(1, 0, num_rows, num_cols) = M_km1;

  Eigen::MatrixX<T> A(k - 1,k);
  Eigen::MatrixX<T> B(k - 1,k);
  A.setZero();
  B.setZero();
  for (int index = 0; index < k - 1; ++index) {
    int j = i - k + 2 + index;
    const T d0 = d_0(k, i, j);
    const T d1 = d_1(k, i, j);
    A(index, index) = 1.0 - d0;
    A(index, index + 1) = d0;
    B(index, index) = -d1;
    B(index, index + 1) = d1;
  }
  M_k = M1 * A + M2 * B;
  return M_k;
}

template <int N, typename T>
T BSpline<N, T>::d_0(int k, int i, int j) {
  const T den = knots_[j + k - 1] - knots_[j];
  if (den <= 0.0) {
    return 0.0;
  }
  const T num = knots_[i] - knots_[j];
  return num / den;
}

template <int N, typename T>
T BSpline<N, T>::d_1(int k, int i, int j) {
  const T den = knots_[j + k - 1] - knots_[j];
  if (den <= 0.0) {
    return 0.0;
  }
  const T num = knots_[i+1] - knots_[i];
  return num / den;
}

template <int N, typename T>
void BSpline<N, T>::FitSpline() {
  const int num_data = time_.size();
  int num_control_points = knots_.size() - spline_order_;
  Eigen::MatrixX<T> X(num_data, num_control_points);
  X.setZero();
  for (int j = 0; j < num_data; j++) {
    const T t = time_[j];
    int spline_index = -1;

    if (t == valid_knots_.back()) {
      spline_index = Mi_.size() - 1;
    }
    else if (t == valid_knots_.front()) {
      spline_index = 0;
    }
    else if (t < valid_knots_.back()) {
      auto lower = std::upper_bound(valid_knots_.begin(), valid_knots_.end(), t);
      spline_index = lower - valid_knots_.begin() - 1;
    }
    const int knot_index = GetKnotIndexFromControlPointIndex(spline_index);
    // Grab the intermediate knot values
    const T ti = knots_[knot_index];
    const T tii = knots_[knot_index + 1];
    // Construct U vector
    Eigen::VectorX<T> U(spline_order_);
    U.setOnes();
    const T u = (t - ti) / (tii - ti);
    for (int i = 1; i < spline_order_; ++i) {
      U(i) = u * U(i - 1);
    }
    // Construct U*M
    Eigen::MatrixX<T> M = Mi_[spline_index];
    Eigen::MatrixX<T> UM = U.transpose() * M;
    X.block(j, spline_index, UM.rows(), UM.cols()) = UM;
  }

  Eigen::Matrix<T,Eigen::Dynamic,N> data(num_data, N);
  for (int i = 0; i < num_data; ++i) {
    data.row(i) = data_[i].transpose();
  }
  // TODO(yangjames): X is highly sparse, and X'X is banded, symmetric, positive
  // definite. Figure out how to sparsify the solve step for the control points
  // as this gets very expensive with more knots.
  const Eigen::MatrixX<T> XtX = X.transpose() * X;
  const Eigen::MatrixX<T> Xtd = X.transpose() * data;
  control_points_ =
      XtX.colPivHouseholderQr().solve(Xtd);
}

template <int N, typename T>
absl::Status BSpline<N, T>::CheckDataForSplineFit(
    const std::vector<T>& time,
    const std::vector<Eigen::Vector<T, N>>& data,
    int spline_order, T knot_frequency) {
  // Assert that data and time are properly sized
  if (!time.size()) {
    return absl::InvalidArgumentError("Attempted to fit data on empty time vector.");
  }
  if (data.empty()) {
    return absl::InvalidArgumentError("Attempted to fit on empty data.");
  }
  if (time.size() != data.size()) {
    return absl::InvalidArgumentError("Data and time vectors are not the same size.");
  }
  // Check that time is strictly increasing
  auto iter = std::adjacent_find(time_.begin(), time_.end(), std::greater<T>());
  if (iter != time_.end()) {
    return absl::InvalidArgumentError("Time vector is not monotonically increasing.");
  }
  // Check that spline order is greater than 2
  if (spline_order < 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Spline order must be greater than 2. Got ", spline_order));
  }
  // Check that knot frequency is greater than 0
  if (knot_frequency <= 0) {
    return absl::InvalidArgumentError("Knot frequency must be greater than 0.");
  }
  return absl::OkStatus();
}

} // namespace calico
