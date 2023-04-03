#include "calico/sensors/camera_models.h"

#include "calico/matchers.h"
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
  const auto camera_model = CameraModel::Create(test_case.camera_model);
  EXPECT_NE(camera_model, nullptr);
  EXPECT_EQ(camera_model->GetType(), test_case.camera_model);
  EXPECT_EQ(camera_model->NumberOfParameters(), test_case.parameter_size);
}

INSTANTIATE_TEST_SUITE_P(
    CameraModelCreationTests, CameraModelCreationTest,
    testing::ValuesIn<CameraModelCreationTestCase>({
        {
          "OpenCv5",
          CameraIntrinsicsModel::kOpenCv5,
          OpenCv5Model::kNumberOfParameters,
        },
        {
          "KannalaBrandt",
          CameraIntrinsicsModel::kKannalaBrandt,
          KannalaBrandtModel::kNumberOfParameters,
        },
        {
          "DoubleSphere",
          CameraIntrinsicsModel::kDoubleSphere,
          DoubleSphereModel::kNumberOfParameters,
        },
        {
          "FieldOfView",
          CameraIntrinsicsModel::kFieldOfView,
          FieldOfViewModel::kNumberOfParameters,
        },
      }),
    [](const testing::TestParamInfo<CameraModelCreationTest::ParamType>& info) {
      return info.param.test_name;
    });

class CameraModelTest : public ::testing::Test {
 protected:
  static constexpr double kSamplePlaneWidth = 1.5;
  static constexpr double kSamplePlaneHeight = 1.5;
  static constexpr double kDelta = 0.025;
  static constexpr int kNumXPoints =
      static_cast<int>(kSamplePlaneWidth / kDelta) + 1;
  static constexpr int kNumYPoints =
      static_cast<int>(kSamplePlaneHeight / kDelta) + 1;
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
    ASSERT_OK_AND_ASSIGN(const Eigen::Vector2d pixel,
        OpenCv5Model::ProjectPoint(intrinsics, t_camera_point));
    // Invert forward projection.
    ASSERT_OK_AND_ASSIGN(const Eigen::Vector3d unprojected_point,
        OpenCv5Model::UnprojectPixel(intrinsics, pixel));
    // Compare results.
    const Eigen::Vector3d bearing_vector = t_camera_point.normalized();
    EXPECT_THAT(bearing_vector, EigenIsApprox(unprojected_point, kSmallError));
  }
}

// TODO(yangjames): Unproject is not that great for KB model :( it takes 100
// Newton iterations to get down to 1e-9, converges way too slowly. Figure out
// why this is.
TEST_F(CameraModelTest, KannalaBrandtModelProjectionAndUnprojection) {
  constexpr double kSmallError = 1e-9;
  Eigen::VectorXd intrinsics(KannalaBrandtModel::kNumberOfParameters);
  intrinsics <<
    785, 640, 400, -3.149e-1, 1.069e-1, 1.616e-4, 1.141e-4;
  for (const Eigen::Vector3d& t_world_point : t_world_points) {
    // Transform camera point into world coordinates.
    const Eigen::Vector3d t_camera_point =
      R_world_camera.transpose() * (t_world_point - t_world_camera);
    // Forward projection.
    ASSERT_OK_AND_ASSIGN(const Eigen::Vector2d pixel,
        KannalaBrandtModel::ProjectPoint(intrinsics, t_camera_point));
    // Invert forward projection.
    ASSERT_OK_AND_ASSIGN(const Eigen::Vector3d unprojected_point,
        KannalaBrandtModel::UnprojectPixel(intrinsics, pixel));
    // Compare results.
    const Eigen::Vector3d bearing_vector = t_camera_point.normalized();
    EXPECT_THAT(bearing_vector, EigenIsApprox(unprojected_point, kSmallError));
  }
}

TEST_F(CameraModelTest, DoubleSphereModelProjectionAndUnprojection) {
  constexpr double kSmallError = 1e-12;
  Eigen::VectorXd intrinsics(DoubleSphereModel::kNumberOfParameters);
  intrinsics <<
    785, 640, 400, 0.5, 0.5;
  for (const Eigen::Vector3d& t_world_point : t_world_points) {
    // Transform camera point into world coordinates.
    const Eigen::Vector3d t_camera_point =
      R_world_camera.transpose() * (t_world_point - t_world_camera);
    // Forward projection.
    ASSERT_OK_AND_ASSIGN(const Eigen::Vector2d pixel,
        DoubleSphereModel::ProjectPoint(intrinsics, t_camera_point));
    // Invert forward projection.
    ASSERT_OK_AND_ASSIGN(const Eigen::Vector3d unprojected_point,
        DoubleSphereModel::UnprojectPixel(intrinsics, pixel));
    // Compare results.
    const Eigen::Vector3d bearing_vector = t_camera_point.normalized();
    EXPECT_THAT(bearing_vector, EigenIsApprox(unprojected_point, kSmallError));
  }
}

TEST_F(CameraModelTest, FieldOfViewModelProjectionAndUnprojection) {
  constexpr double kSmallError = 1e-12;
  Eigen::VectorXd intrinsics(FieldOfViewModel::kNumberOfParameters);
  intrinsics <<
    785, 640, 400, 0.05;
  for (const Eigen::Vector3d& t_world_point : t_world_points) {
    // Transform camera point into world coordinates.
    const Eigen::Vector3d t_camera_point =
      R_world_camera.transpose() * (t_world_point - t_world_camera);
    // Forward projection.
    ASSERT_OK_AND_ASSIGN(const Eigen::Vector2d pixel,
        FieldOfViewModel::ProjectPoint(intrinsics, t_camera_point));
    // Invert forward projection.
    ASSERT_OK_AND_ASSIGN(const Eigen::Vector3d unprojected_point,
        FieldOfViewModel::UnprojectPixel(intrinsics, pixel));
    // Compare results.
    const Eigen::Vector3d bearing_vector = t_camera_point.normalized();
    EXPECT_THAT(bearing_vector, EigenIsApprox(unprojected_point, kSmallError));
  }
}

} // namespace
} // namespace calico::sensors
