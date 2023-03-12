#include "calico/bspline.h"

#include "calico/matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace calico {
namespace {

class BSplineTest : public ::testing::Test {
 protected:
  static constexpr int kNumPoints = 101;
  static constexpr double kDt = 0.1;
  static constexpr int kSplineOrder = 6;
  static constexpr int kKnotFrequency = 5;
  std::vector<double> time;
  std::vector<Eigen::Vector3d> data;

  void SetUp() override {
    time.resize(kNumPoints);
    data.resize(kNumPoints);
    for (int i = 0; i < kNumPoints; ++i) {
      double t = kDt * i;
      double x = std::cos(t);
      double y = std::sin(1.5 * t);
      double z = t * std::cos(t);
      Eigen::Vector3d v(x, y, z);
      time[i] = t;
      data[i] = v;
    }
  }
};

TEST_F(BSplineTest, InvalidDerivatives) {
  BSpline<3> spline;
  EXPECT_OK(spline.FitToData(time, data, kSplineOrder, kKnotFrequency));
  std::vector<double> interp_time{0};
  EXPECT_THAT(spline.Interpolate(interp_time, -1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(spline.Interpolate(interp_time, kSplineOrder),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(BSplineTest, InvalidInterpolationTime) {
  BSpline<3> spline;
  EXPECT_OK(spline.FitToData(time, data, kSplineOrder, kKnotFrequency));
  std::vector<double> interp_time{-1};
  EXPECT_THAT(spline.Interpolate(interp_time, -1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(BSplineTest, InterpolationPrecision3DOF) {
  constexpr int kNumInterpPoints = 201;
  constexpr double kSmallValueD0 = 1e-6;
  constexpr double kSmallValueD1 = 1e-5;
  constexpr double kSmallValueD2 = 1e-4;
  constexpr double kSmallValueD3 = 1e-2;
  const double dt = (time.back() - time.front()) / kNumInterpPoints;
  std::vector<double> interp_times(kNumInterpPoints);
  for (int i = 0; i < kNumInterpPoints; ++i) {
    interp_times[i] = dt * i;
  }
  
  BSpline<3> spline;
  EXPECT_OK(spline.FitToData(time, data, kSplineOrder, kKnotFrequency));

  // Check first three derivatives.
  ASSERT_OK_AND_ASSIGN(const std::vector<Eigen::Vector3d> y,
                       spline.Interpolate(interp_times, /*derivative=*/ 0));
  ASSERT_OK_AND_ASSIGN(const std::vector<Eigen::Vector3d> yp,
                       spline.Interpolate(interp_times, /*derivative=*/ 1));
  ASSERT_OK_AND_ASSIGN(const std::vector<Eigen::Vector3d> y2p,
                       spline.Interpolate(interp_times, /*derivative=*/ 2));
  ASSERT_OK_AND_ASSIGN(const std::vector<Eigen::Vector3d> y3p,
                       spline.Interpolate(interp_times, /*derivative=*/ 3));
  for (int i = 0; i < kNumInterpPoints; ++i) {
    const double& t = interp_times[i];
    const Eigen::Vector3d expected_value_y(std::cos(t), std::sin(1.5 * t),
                                           t * std::cos(t));
    const Eigen::Vector3d expected_value_yp(
        -std::sin(t), 1.5 * std::cos(1.5 * t), std::cos(t) - t * std::sin(t));
    const Eigen::Vector3d expected_value_y2p(
        -std::cos(t), -2.25 * std::sin(1.5 * t),
        -2.0 *std::sin(t) - t * std::cos(t));
    const Eigen::Vector3d expected_value_y3p(
        std::sin(t), -3.375 * std::cos(1.5 * t),
        t * std::sin(t) - 3.0 * std::cos(t));
                                             
    EXPECT_THAT(expected_value_y, EigenIsApprox(y[i], kSmallValueD0));
    EXPECT_THAT(expected_value_yp, EigenIsApprox(yp[i], kSmallValueD1));
    EXPECT_THAT(expected_value_y2p, EigenIsApprox(y2p[i], kSmallValueD2));
    EXPECT_THAT(expected_value_y3p, EigenIsApprox(y3p[i], kSmallValueD3));
  }
}

}
} // namespace calico
