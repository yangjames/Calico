#include <algorithm>

#include "calico/statusor_macros.h"


namespace calico {

template <typename T, int N>
absl::Status BSpline<T, N>::FitToData(
    const Eigen::VectorX<T>& time, const std::vector<Eigen::Vector<T,N>>& data,
    int spline_order, double knot_frequency) {
  RETURN_IF_ERROR(
      CheckDataForSplineFit(time, data, spline_order, knot_frequency));
  time_ = time;
  data_ = data;
  spline_order_ = spline_order;
  knot_frequency_ = knot_frequency;
  // Derived information about the spline.
  spline_degree_ = spline_order_ - 1;
  ComputePowerRuleCoefficients();
  ComputeKnotVector();
  ComputeBasisMatrices();
  // Fit the spline.
  FitSpline();
  return absl::OkStatus();
}

template <typename T, int N>
absl::StatusOr<std::vector<Eigen::Vector<T, N>>> BSpline<T, N>::Interpolate(
    const Eigen::VectorX<T>& times, int derivative) const {
  if (derivative < 0 || derivative > spline_degree_) {
    return absl::InvalidArgumentError("Invalid derivative for interpolation.");
  }
  return std::vector<Eigen::Vector<T, N>>{};
}


template<typename T, int N>
void BSpline<T, N>::ComputePowerRuleCoefficients() {
  // Resize the matrix and initialize to zeros
  derivative_coeffs_.resize(spline_degree_, spline_order_);
  derivative_coeffs_.setZero();

  // Populate each coefficient
  for (int deriv = 0; deriv < spline_degree_; ++deriv) {
    for (int order = deriv; order < spline_order_; ++order) {
      T pwr = 1;
      for (int k = order-deriv; k < order; ++k) {
        pwr *= (k + 1);
      }
      derivative_coeffs_(deriv, order) = pwr;
    }
  }
}

template <typename T, int N>
void BSpline<T,N>::ComputeKnotVector() {
  const T duration = (time_.tail(1) - time_.head(1))(0);
  const T dt = 1.0 / knot_frequency_;

  num_knots_ = 1 + ceil(duration * knot_frequency_) + 2 * spline_degree_;
  num_control_points_ = num_knots_ - spline_order_;
  num_valid_knots_ = num_knots_ - 2 * spline_degree_;

  knots_.resize(num_knots_);
  valid_knots_.resize(num_valid_knots_);
  for (int i = -spline_degree_; i < num_knots_ - spline_degree_; ++i) {
    const T knot_value = time_.head(1)(0) + dt * i;
    knots_[i + spline_degree_] = knot_value;

    int valid_knot_index = i;
    if (valid_knot_index > -1 && valid_knot_index < num_valid_knots_) {
      valid_knots_[valid_knot_index] = knot_value;
    }
  }
}

template <typename T, int N>
void BSpline<T,N>::ComputeBasisMatrices() {
  num_valid_segments_ = num_knots_ - 2 * spline_order_ + 1;
  Mi_.resize(num_valid_segments_);
  for (int i = 0; i < num_valid_segments_; ++i) {
    Mi_[i] = M(spline_order_, i + spline_degree_);
  }
}

template<typename T, int N>
Eigen::MatrixX<T> BSpline<T,N>::M(int k, int i) {
  Eigen::MatrixX<T> M_k;
  if (k == 1) {
    M_k.resize(k, k);
    M_k(0, 0) = k;
    return M_k;
  }

  Eigen::MatrixX<T> M_km1 = M(k - 1, i);
  int num_rows = M_km1.rows();
  int num_cols = M_km1.cols();
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
    T d0 = d_0(k, i, j);
    T d1 = d_1(k, i, j);
    A(index, index)   = 1.0 - d0;
    A(index, index + 1) = d0;
    B(index, index)   = -d1;
    B(index, index + 1) = d1;
  }
  M_k = M1 * A + M2 * B;
  return M_k;
}

template <typename T, int N>
T BSpline<T,N>::d_0(int k, int i, int j) {
  T den = knots_[j + k - 1] - knots_[j];
  if (den <= 0.0) {
    return 0.0;
  }
  T num = knots_[i] - knots_[j];
  return num / den;
}

template <typename T, int N>
T BSpline<T,N>::d_1(int k, int i, int j) {
  T den = knots_[j + k - 1] - knots_[j];
  if (den <= 0.0) {
    return 0.0;
  }
  T num = knots_[i+1] - knots_[i];
  return num / den;
}

template <typename T, int N>
void BSpline<T,N>::FitSpline() {
  const int num_data = time_.size();
  Eigen::MatrixX<T> X(num_data, num_control_points_);
  X.setZero();
  for (int j = 0; j < num_data; j++) {
    const T t = time_[j];
    int spline_index = -1;

    if (t == valid_knots_.back()) {
      spline_index = num_valid_segments_ - 1;
    }
    else if (t == valid_knots_.front()) {
      spline_index = 0;
    }
    else if (t < valid_knots_.back()) {
      auto lower = std::upper_bound(valid_knots_.begin(), valid_knots_.end(), t);
      spline_index = lower - valid_knots_.begin() - 1;
    }
    const int knot_index = spline_index + spline_degree_;
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
  // definite. Figure out how to sparsify the solve step for control_points_ as
  // this gets very expensive with more knots.
  Eigen::MatrixX<T> XtX = X.transpose() * X;
  Eigen::MatrixX<T> Xtd = X.transpose() * data;
  control_points_ = XtX.colPivHouseholderQr().solve(Xtd).transpose();
}

template<typename T, int N>
absl::Status BSpline<T,N>::CheckDataForSplineFit(
    const Eigen::VectorX<T>& time, const std::vector<Eigen::Vector<T, N>>& data,
    int spline_order, double knot_frequency) {
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
  for (int i = 1; i < time.size(); ++i) {
    if (time(i - 1) >= time(i)) {
      return absl::InvalidArgumentError("Time vector is not monotonically increasing.");
    }
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
