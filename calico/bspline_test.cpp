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
  std::vector<double> time;
  std::vector<Eigen::Vector3d> data;

  void SetUp() override {
    time.resize(kNumPoints);
    data.resize(kNumPoints);
    for (int i = 0; i < kNumPoints; ++i) {
      double t = kDt * i;
      double ct = std::cos(t);
      Eigen::Vector3d v(ct, ct, ct);
      time[i] = t;
      data[i] = v;
    }
  }
};

TEST_F(BSplineTest, InvalidDerivatives) {
  constexpr int kSplineOrder = 6;
  constexpr int kKnotFrequency = 5;
  BSpline<double, 3> spline;
  EXPECT_OK(spline.FitToData(time, data, kSplineOrder, kKnotFrequency));
  std::vector<double> interp_time{0};
  EXPECT_THAT(spline.Interpolate(interp_time, -1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(spline.Interpolate(interp_time, kSplineOrder),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(BSplineTest, InvalidInterpolationTime) {
  constexpr int kSplineOrder = 6;
  constexpr int kKnotFrequency = 5;
  BSpline<double, 3> spline;
  EXPECT_OK(spline.FitToData(time, data, kSplineOrder, kKnotFrequency));
  std::vector<double> interp_time{-1};
  EXPECT_THAT(spline.Interpolate(interp_time, -1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}
} // namespace calico
