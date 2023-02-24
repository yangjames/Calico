#include "calico/bspline.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace calico {
namespace {

TEST(BSplineTest, InvalidDerivative) {
  constexpr int kNumPoints = 101;
  constexpr double kDt = 0.1;

  Eigen::VectorXd time(kNumPoints);
  std::vector<Eigen::Vector3d> data(kNumPoints);

  for (int i = 0; i < kNumPoints; ++i) {
    double t = kDt * i;
    double ct = std::cos(t);
    Eigen::Vector3d v(ct, ct, ct);
    time(i) = t;
    data[i] = v;
  }
  BSpline<double, 3> spline(time, data, 6, 5);
  EXPECT_
}

}
} // namespace calico
