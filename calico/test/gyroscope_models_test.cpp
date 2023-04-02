#include "calico/sensors/gyroscope_models.h"

#include "calico/matchers.h"
#include "calico/test_utils.h"
#include "Eigen/Dense"
#include "gmock/gmock.h"
#include "gtest/gtest.h"


namespace calico::sensors {
namespace {

struct GyroscopeModelTestCase {
  std::string test_name;
  GyroscopeIntrinsicsModel gyroscope_model;
  int parameter_size;
  Eigen::Vector3d sample_omega;
};

using GyroscopeModelTest = ::testing::TestWithParam<GyroscopeModelTestCase>;

TEST_P(GyroscopeModelTest, TestProjectionAndUnprojection) {
  const GyroscopeModelTestCase& test_case = GetParam();
  const auto gyroscope_model = GyroscopeModel::Create(
      test_case.gyroscope_model);
  ASSERT_NE(gyroscope_model, nullptr);
  EXPECT_EQ(gyroscope_model->GetType(), test_case.gyroscope_model);
  EXPECT_EQ(gyroscope_model->NumberOfParameters(), test_case.parameter_size);
  Eigen::VectorXd intrinsics(test_case.parameter_size);
  intrinsics.setRandom();
  ASSERT_OK_AND_ASSIGN(const Eigen::Vector3d measurement,
      gyroscope_model->Project(intrinsics, test_case.sample_omega));
  ASSERT_OK_AND_ASSIGN(const Eigen::Vector3d actual_omega,
      gyroscope_model->Unproject(intrinsics, measurement));
  constexpr double kSmallNumber = 1e-9;
  EXPECT_THAT(actual_omega, EigenIsApprox(test_case.sample_omega, kSmallNumber));
}

INSTANTIATE_TEST_SUITE_P(
    GyroscopeModelTests, GyroscopeModelTest,
    testing::ValuesIn<GyroscopeModelTestCase>({
        {
          "GyroscopeScaleOnly",
          GyroscopeIntrinsicsModel::kGyroscopeScaleOnly,
          GyroscopeScaleOnlyModel::kNumberOfParameters,
          Eigen::Vector3d::Random(),
        },
        {
          "GyroscopeScaleAndBias",
          GyroscopeIntrinsicsModel::kGyroscopeScaleAndBias,
          GyroscopeScaleAndBiasModel::kNumberOfParameters,
          Eigen::Vector3d::Random(),
        },
        {
          "GyroscopeVectorNav",
          GyroscopeIntrinsicsModel::kGyroscopeVectorNav,
          GyroscopeVectorNavModel::kNumberOfParameters,
          Eigen::Vector3d::Random(),
        }
      }),
    [](const testing::TestParamInfo<GyroscopeModelTest::ParamType>& info
         ) {
      return info.param.test_name;
    });

} // namespace
} // namespace calico::sensors
