#include "calico/sensors/accelerometer_models.h"

#include "calico/matchers.h"
#include "calico/test_utils.h"
#include "Eigen/Dense"
#include "gmock/gmock.h"
#include "gtest/gtest.h"


namespace calico::sensors {
namespace {

struct AccelerometerModelTestCase {
  std::string test_name;
  AccelerometerIntrinsicsModel accelerometer_model;
  int parameter_size;
  Eigen::Vector3d sample_acceleration;
};

using AccelerometerModelTest =
    ::testing::TestWithParam<AccelerometerModelTestCase>;

TEST_P(AccelerometerModelTest, TestProjectionAndUnprojection) {
  const AccelerometerModelTestCase& test_case = GetParam();
  const auto accelerometer_model = AccelerometerModel::Create(
      test_case.accelerometer_model);
  ASSERT_NE(accelerometer_model, nullptr);
  EXPECT_EQ(accelerometer_model->GetType(), test_case.accelerometer_model);
  EXPECT_EQ(accelerometer_model->NumberOfParameters(), test_case.parameter_size);
  Eigen::VectorXd intrinsics(test_case.parameter_size);
  intrinsics.setRandom();
  ASSERT_OK_AND_ASSIGN(const Eigen::Vector3d measurement,
      accelerometer_model->Project(intrinsics, test_case.sample_acceleration));
  ASSERT_OK_AND_ASSIGN(const Eigen::Vector3d actual_acceleration,
      accelerometer_model->Unproject(intrinsics, measurement));
  constexpr double kSmallNumber = 1e-9;
  EXPECT_THAT(actual_acceleration,
              EigenIsApprox(test_case.sample_acceleration, kSmallNumber));
}

INSTANTIATE_TEST_SUITE_P(
    AccelerometerModelTests, AccelerometerModelTest,
    testing::ValuesIn<AccelerometerModelTestCase>({
        {
          "ScaleOnly",
          AccelerometerIntrinsicsModel::kAccelerometerScaleOnly,
          AccelerometerScaleOnlyModel::kNumberOfParameters,
          Eigen::Vector3d::Random(),
        },
        {
          "ScaleAndBias",
          AccelerometerIntrinsicsModel::kAccelerometerScaleAndBias,
          AccelerometerScaleAndBiasModel::kNumberOfParameters,
          Eigen::Vector3d::Random(),
        },
      }),
    [](const testing::TestParamInfo<AccelerometerModelTest::ParamType>& info
         ) {
      return info.param.test_name;
    });

} // namespace
} // namespace calico::sensors
