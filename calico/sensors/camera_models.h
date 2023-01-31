#ifndef CALICO_SENSORS_CAMERA_MODELS_H_
#define CALICO_SENSORS_CAMERA_MODELS_H_

#include "calico/typedefs.h"

#include <memory>

#include "absl/status/statusor.h"
#include "Eigen/Dense"


namespace calico::sensors {

// Camera model types.
enum class CameraIntrinsicsModel : int {
  kNone,
  kPinhole,
  kOpenCv5,
  kOpenCv8,
  kKannalaBrandt,
  kFov,
};

// Base class for camera models.
class CameraModel {
 public:
  virtual ~CameraModel() = default;

  // Project a point resolved in the camera frame into the pixel space.
  // Top level call invokes the derived class's implementation.
  template <typename T>
  absl::StatusOr<Eigen::Vector2<T>> ProjectPoint(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& point) const;

  // Unproject a pixel and return the corresponding unit ray pointing from
  // the camera origin to the pixel location. Multiplying this unit vector
  // by the point's distance gives you the location of that point relative
  // to the camera resolved in the camera frame.
  virtual absl::StatusOr<Eigen::Vector3d> UnprojectPixel(
      const Eigen::VectorXd& intrinsics,
      const Eigen::Vector2d& pixel) const = 0;

  // Getter for camera model type.
  virtual CameraIntrinsicsModel GetType() const  = 0;

  // Getter for the number of parameters for this camera model.
  virtual int NumberOfParameters() const = 0;

  // TODO(yangjames): This method should return an absl::StatusOr. Figure out
  //                  to support this using unique_ptr's, or find macros that
  //                  already implement this feature (i.e. ASSIGN_OR_RETURN).
  // Factory method for creating a camera model with `camera_model` type.
  // This method will return a nullptr if an unsupported CameraIntrinsicsModel
  // is passed in.
  static std::unique_ptr<CameraModel> Create(
      CameraIntrinsicsModel camera_model);
};

// 5-parameter Brown-Conrady projection model as presented in OpenCV. This model
// assumes an isotropic pinhole model, i.e. `fx == fy = f`.
// Parameters are in the following order:
//   [f, cx, cy, k1, k2, p1, p2, k3]
// See https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html for more details.
class OpenCv5Model : public CameraModel {
 public:
  static constexpr int kNumberOfParameters = 8;
  static constexpr CameraIntrinsicsModel kModelType =
      CameraIntrinsicsModel::kOpenCv5;

  OpenCv5Model() = default;
  ~OpenCv5Model() override = default;
  OpenCv5Model& operator=(const OpenCv5Model&) = default;

  template <typename T>
  absl::StatusOr<Eigen::Vector2<T>> ProjectPoint(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& point) const {
    if (point.z() <= T(0.0)) {
      return absl::InvalidArgumentError(
          "Camera point is behind the camera. Cannot project.");
    }
    if (intrinsics.size() != kNumberOfParameters) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid number of intrinsics parameters provided. Expected ",
          kNumberOfParameters, ". Got ", intrinsics.size()));
    }
    // Prepare values.
    const T x = point.x() / point.z();
    const T y = point.y() / point.z();
    const T r2 = point.squaredNorm();
    const T& f = intrinsics(0);
    const T& cx = intrinsics(1);
    const T& cy = intrinsics(2);
    const T& k1 = intrinsics(3);
    const T& k2 = intrinsics(4);
    const T& p1 = intrinsics(5);
    const T& p2 = intrinsics(6);
    const T& k3 = intrinsics(7);
    const T s = 1 + r2*(k1 + r2*(k2 + r2*k3));
    Eigen::Vector2<T> projection(x, y);
    // Apply radial distortion.
    projection *= s;
    // Apply tangential distortion.
    projection.x() += T(2.0)*p1*x*y + p2*(r2 + T(2.0)*x*x);
    projection.y() += T(2.0)*p2*x*y + p1*(r2 + T(2.0)*y*y);
    // Apply pinhole parameters.
    projection *= f;
    projection.x() += cx;
    projection.y() += cy;
    return projection;
  }

  absl::StatusOr<Eigen::Vector3d> UnprojectPixel(
      const Eigen::VectorXd& intrinsics,
      const Eigen::Vector2d& pixel) const final {
    return absl::UnimplementedError(absl::StrCat(
            "UnprojectPixel not yet implemented for camera model ",
            GetType(), "."));
  }

  CameraIntrinsicsModel GetType() const final {
    return kModelType;
  }

  int NumberOfParameters() const final {
    return kNumberOfParameters;
  }
};
} // namespace calico::sensors

#endif // CALICO_SENSORS_CAMERA_MODELS_H_
