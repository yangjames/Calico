#include <algorithm>

#include "calico/statusor_macros.h"
#include "ceres/problem.h"

namespace calico {

template <int N>
int BSpline<N>::AddParametersToProblem(ceres::Problem& problem) {
  for (Eigen::Vector<double, N>& control_point : control_points_) {
    problem.AddParameterBlock(control_point.data(), N);
  }
}

template <int N>
absl::Status BSpline<N>::FitToData(
    const std::vector<double>& time, const std::vector<Eigen::Vector<double,N>>& data,
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

template <int N>
absl::StatusOr<std::vector<Eigen::Vector<double, N>>> BSpline<N>::Interpolate(
    const std::vector<double>& times, int derivative) const {
  if (derivative < 0 || derivative > spline_degree_) {
    return absl::InvalidArgumentError("Invalid derivative for interpolation.");
  }
  for (const double& t : times) {
    if (t < valid_knots_.front() || t > valid_knots_.back()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Cannot interpolate ", t, ". Value is not within valid knots."));
    }
  }
  int num_interp = times.size();
  std::vector<Eigen::Vector<double, N>> y(num_interp);
  for (int i = 0; i < num_interp; ++i) {
    const int spline_idx = GetControlPointIndex(times[i]);
    const int knot_idx = spline_idx + spline_degree_;
    // Compute u.
    const double& ti = knots_[knot_idx];
    const double& tii = knots_[knot_idx + 1];
    const double dt = tii - ti;
    const double dt_inv = 1.0 / dt;
    const double u = (times[i] - ti) * dt_inv;
    // Derivative of u with respect to t raised to n'th derivative power per
    // chain rule.
    double dnu_dtn = 1.0;
    for (int j = 0; j < derivative; ++j) {
      dnu_dtn *= dt_inv;
    }
    // Construct U vector
    Eigen::Matrix<double, 1, Eigen::Dynamic> U(spline_order_);
    U.setOnes();
    for (int j = derivative + 1; j <  spline_order_; ++j) {
      U(j) = u * U(j - 1);
    }
    U = U.array() * derivative_coeffs_.row(derivative).array() * dnu_dtn;
    // Manually multiply here: y = U*M*V, V = control points.This is necessary
    // because control points are stored as N-dimensional vectors rather than
    // one giant NxM matrix in order to make it easier to index into for cost
    // function construction.
    const Eigen::MatrixXd UM = U*Mi_[spline_idx];
    y[i].setZero();
    for (int j = 0; j < spline_order_; ++j) {
      const int control_point_idx = spline_idx + j;
      y[i].x() += control_points_[control_point_idx].x() * UM(j);
      y[i].y() += control_points_[control_point_idx].y() * UM(j);
      y[i].z() += control_points_[control_point_idx].z() * UM(j);
    }
  }
  return y;
}

template<int N>
int BSpline<N>::GetControlPointIndex(double query_time) const {
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

template<int N>
void BSpline<N>::ComputePowerRuleCoefficients() {
  // Resize the matrix and initialize to zeros
  derivative_coeffs_.resize(spline_degree_, spline_order_);
  derivative_coeffs_.setZero();

  // Populate each coefficient
  for (int deriv = 0; deriv < spline_degree_; ++deriv) {
    for (int order = deriv; order < spline_order_; ++order) {
      double pwr = 1.0;
      for (int k = order-deriv; k < order; ++k) {
        pwr *= (k + 1);
      }
      derivative_coeffs_(deriv, order) = pwr;
    }
  }
}

template <int N>
void BSpline<N>::ComputeKnotVector() {
  const double duration = time_.back() - time_.front();
  const double dt = 1.0 / knot_frequency_;
  const int num_valid_knots = 1 + std::ceil(duration * knot_frequency_);
  const int num_knots = num_valid_knots + 2 * spline_degree_;

  knots_.resize(num_knots);
  valid_knots_.resize(num_valid_knots);
  for (int i = -spline_degree_; i < num_knots - spline_degree_; ++i) {
    const double knot_value = time_.front() + dt * i;
    knots_[i + spline_degree_] = knot_value;

    int valid_knot_index = i;
    if (valid_knot_index > -1 && valid_knot_index < num_valid_knots) {
      valid_knots_[valid_knot_index] = knot_value;
    }
  }
}

template <int N>
void BSpline<N>::ComputeBasisMatrices() {
  const int num_valid_segments = valid_knots_.size() - 1;
  Mi_.resize(num_valid_segments);
  for (int i = 0; i < num_valid_segments; ++i) {
    Mi_[i] = M(spline_order_, i + spline_degree_);
  }
}

template<int N>
Eigen::MatrixXd BSpline<N>::M(int k, int i) {
  Eigen::MatrixXd M_k;
  if (k == 1) {
    M_k.resize(k, k);
    M_k(0, 0) = k;
    return M_k;
  }

  Eigen::MatrixXd M_km1 = M(k - 1, i);
  const int num_rows = M_km1.rows();
  const int num_cols = M_km1.cols();
  Eigen::MatrixXd M1(num_rows + 1, num_cols), M2(num_rows + 1, num_cols);
  M1.setZero();
  M2.setZero();
  M1.block(0, 0, num_rows, num_cols) = M_km1;
  M2.block(1, 0, num_rows, num_cols) = M_km1;

  Eigen::MatrixXd A(k - 1,k);
  Eigen::MatrixXd B(k - 1,k);
  A.setZero();
  B.setZero();
  for (int index = 0; index < k - 1; ++index) {
    int j = i - k + 2 + index;
    const double d0 = d_0(k, i, j);
    const double d1 = d_1(k, i, j);
    A(index, index)   = 1.0 - d0;
    A(index, index + 1) = d0;
    B(index, index)   = -d1;
    B(index, index + 1) = d1;
  }
  M_k = M1 * A + M2 * B;
  return M_k;
}

template <int N>
double BSpline<N>::d_0(int k, int i, int j) {
  const double den = knots_[j + k - 1] - knots_[j];
  if (den <= 0.0) {
    return 0.0;
  }
  const double num = knots_[i] - knots_[j];
  return num / den;
}

template <int N>
double BSpline<N>::d_1(int k, int i, int j) {
  const double den = knots_[j + k - 1] - knots_[j];
  if (den <= 0.0) {
    return 0.0;
  }
  const double num = knots_[i+1] - knots_[i];
  return num / den;
}

template <int N>
void BSpline<N>::FitSpline() {
  const int num_data = time_.size();
  int num_control_points = knots_.size() - spline_order_;
  Eigen::MatrixXd X(num_data, num_control_points);
  X.setZero();
  for (int j = 0; j < num_data; j++) {
    const double t = time_[j];
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
    const int knot_index = spline_index + spline_degree_;
    // Grab the intermediate knot values
    const double ti = knots_[knot_index];
    const double tii = knots_[knot_index + 1];
    // Construct U vector
    Eigen::VectorXd U(spline_order_);
    U.setOnes();
    const double u = (t - ti) / (tii - ti);
    for (int i = 1; i < spline_order_; ++i) {
      U(i) = u * U(i - 1);
    }
    // Construct U*M
    Eigen::MatrixXd M = Mi_[spline_index];
    Eigen::MatrixXd UM = U.transpose() * M;
    X.block(j, spline_index, UM.rows(), UM.cols()) = UM;
  }

  Eigen::Matrix<double,Eigen::Dynamic,N> data(num_data, N);
  for (int i = 0; i < num_data; ++i) {
    data.row(i) = data_[i].transpose();
  }
  // TODO(yangjames): X is highly sparse, and X'X is banded, symmetric, positive
  // definite. Figure out how to sparsify the solve step for control_points_ as
  // this gets very expensive with more knots.
  const Eigen::MatrixXd XtX = X.transpose() * X;
  const Eigen::MatrixXd Xtd = X.transpose() * data;
  Eigen::MatrixXd control_points =
      XtX.colPivHouseholderQr().solve(Xtd).transpose();
  for (int i = 0; i < control_points.cols(); ++i) {
    control_points_.push_back(control_points.col(i));
  }
}

template<int N>
absl::Status BSpline<N>::CheckDataForSplineFit(
    const std::vector<double>& time,
    const std::vector<Eigen::Vector<double, N>>& data,
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
  auto iter = std::adjacent_find(time_.begin(), time_.end(), std::greater<double>());
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
