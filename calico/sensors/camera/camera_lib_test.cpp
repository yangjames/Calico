#include "calico/sensors/camera/camera_models.h"

#include "gtest/gtest.h"

namespace calico::sensors::camera {
namespace {

// Creation of camera models.
struct CameraModelCreationTestCase {
  std::string test_name;
  CameraIntrinsicsModel camera_model;
  int parameter_size;
};

using CameraModelCreationTest =
    ::testing::TestWithParam<CameraModelCreationTestCase>;

TEST_P(CameraModelCreationTest, TestModelGetters) {
  const CameraModelCreationTestCase& test_case = GetParam();
  const auto camera_model_statusor =
      CameraModel::Create(test_case.camera_model);
  ASSERT_TRUE(camera_model_statusor.ok());
  EXPECT_EQ(camera_model_statusor.value()->GetType(), test_case.camera_model);
  EXPECT_EQ(camera_model_statusor.value()->GetParameterSize(),
            test_case.parameter_size);
}

INSTANTIATE_TEST_SUITE_P(
    CameraModelCreationTests, CameraModelCreationTest,
    testing::ValuesIn<CameraModelCreationTestCase>({
        {
          "OpenCv5",
          CameraIntrinsicsModel::kOpenCv5,
          OpenCv5Model::kNumberOfParameters
        },
      }),
    [](const testing::TestParamInfo<CameraModelCreationTest::ParamType>& info) {
      return info.param.test_name;
    });

} // namespace
} // namespace calico::sensors::camera
