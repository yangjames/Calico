#include "calico/sensors/camera_models.h"

#include "Eigen/Dense"
#include "gmock/gmock.h"
#include "gtest/gtest.h"


namespace calico::sensors {
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
  const auto camera_model =
      CameraModel::Create(test_case.camera_model);
  EXPECT_NE(camera_model, nullptr);
  EXPECT_EQ(camera_model->GetType(), test_case.camera_model);
  EXPECT_EQ(camera_model->NumberOfParameters(),
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

class CameraModelTest : public ::testing::Test {
 protected:
  static constexpr double kSamplePlaneWidth = 1.5;
  static constexpr double kSamplePlaneHeight = 1.5;
  static constexpr double kDelta = 0.25;
  static constexpr int kNumXPoints = static_cast<int>(kSamplePlaneWidth / kDelta) + 1;
  static constexpr int kNumYPoints = static_cast<int>(kSamplePlaneHeight / kDelta) + 1;
  Eigen::Matrix3d R_world_camera;
  Eigen::Vector3d t_world_camera;
  std::vector<Eigen::Vector3d> t_world_points;

  // Set up a synthetic world with a plane of points measuring 1.5x1.5m on the
  // ground with a camera facing downward at 1m above the center of the plane.
  void SetUp() override {
    R_world_camera <<
      1, 0, 0,
      0, -1, 0,
      0, 0, -1;
    t_world_camera << kSamplePlaneWidth / 2.0, kSamplePlaneHeight / 2.0, 1.0;
    for (int i = 0; i < kNumXPoints; ++i) {
      for (int j = 0; j < kNumYPoints; ++j) {
        const double x = i * kDelta;
        const double y = j * kDelta;
        t_world_points.push_back(Eigen::Vector3d(x, y, 0.0));
      }
    }
  }
};

TEST_F(CameraModelTest, OpenCv5ModelProjectionAndUnprojection) {
  constexpr double kSmallError = 1e-10;
  Eigen::VectorXd intrinsics(OpenCv5Model::kNumberOfParameters);
  intrinsics <<
    785, 640, 400, -3.149e-1, 1.069e-1, 1.616e-4, 1.141e-4, -1.853e-2;
  for (const Eigen::Vector3d& t_world_point : t_world_points) {
    // Transform camera point into world coordinates.
    const Eigen::Vector3d t_camera_point =
      R_world_camera.transpose() * (t_world_point - t_world_camera);
    // Forward projection.
    absl::StatusOr<Eigen::Vector2d> pixel =
      OpenCv5Model::ProjectPoint(intrinsics, t_camera_point);
    ASSERT_EQ(pixel.status().code(), absl::StatusCode::kOk);
    // Invert forward projection.
    absl::StatusOr<Eigen::Vector3d> unprojected_point =
      OpenCv5Model::UnprojectPixel(intrinsics, *pixel);
    ASSERT_EQ(unprojected_point.status().code(), absl::StatusCode::kOk);
    // Compare results.
    const Eigen::Vector3d t_camera_point_metric =
      t_camera_point / t_camera_point.z();
    EXPECT_TRUE(t_camera_point_metric.isApprox(
        *unprojected_point, kSmallError));
  }
}

} // namespace
} // namespace calico::sensors
