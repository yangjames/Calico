#ifndef CALICO_SENSORS_CAMERA_MODELS_H_
#define CALICO_SENSORS_CAMERA_MODELS_H_

#include "calico/typedefs.h"

#include <memory>

#include "absl/status/statusor.h"
#include "calico/statusor_macros.h"
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

  // Unproject a pixel and return the corresponding metric plane point. In order
  // to get the unprojected point in pixel space, apply the pinhole paraeters to
  // the unprojectd results.
  // Top level call invokes the derived class's implementation.
  template <typename T>
  absl::StatusOr<Eigen::Vector3<T>> UnprojectPixel(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector2<T>& pixel) const;

  // Getter for camera model type.
  virtual CameraIntrinsicsModel GetType() const  = 0;

  // Getter for the number of parameters for this camera model.
  virtual int NumberOfParameters() const = 0;

  // TODO(yangjames): This method should return an absl::StatusOr. Figure out
  //   how to support this using unique_ptr's, or find macros that already
  //   implement this feature (i.e. ASSIGN_OR_RETURN).
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
  static absl::StatusOr<Eigen::Vector2<T>> ProjectPoint(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& point) {
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
    const T& f = intrinsics(0);
    const T& cx = intrinsics(1);
    const T& cy = intrinsics(2);
    const T& k1 = intrinsics(3);
    const T& k2 = intrinsics(4);
    const T& p1 = intrinsics(5);
    const T& p2 = intrinsics(6);
    const T& k3 = intrinsics(7);
    Eigen::Vector2<T> projection(point.x() / point.z(), point.y() / point.z());
    const T x = projection.x();
    const T y = projection.y();
    const T r2 = projection.squaredNorm();
    const T s = T(1.0) + r2 * (k1 + r2 * (k2 + r2 * k3));
    // Apply radial distortion.
    projection *= s;
    // Apply tangential distortion.
    projection.x() += T(2.0) * p1 * x * y + p2 * (r2 + T(2.0) * x * x);
    projection.y() += T(2.0) * p2 * x * y + p1 * (r2 + T(2.0) * y * y);
    // Apply pinhole parameters.
    projection *= f;
    projection.x() += cx;
    projection.y() += cy;
    return projection;
  }

  template <typename T>
  static absl::StatusOr<Eigen::Vector3<T>> UnprojectPixel(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector2<T>& pixel,
      int max_iterations = 30) {
    constexpr T kSmallValue = T(1e-14);
    if (intrinsics.size() != kNumberOfParameters) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Expected ", kNumberOfParameters, " parameters for intrinsics. Got ",
          intrinsics.size()));
    }
    const T& f = intrinsics(0);
    const T& cx = intrinsics(1);
    const T& cy = intrinsics(2);
    const T& k1 = intrinsics(3);
    const T& k2 = intrinsics(4);
    const T& p1 = intrinsics(5);
    const T& p2 = intrinsics(6);
    const T& k3 = intrinsics(7);
    const T inv_f = T(1.0) / f;
    // Invert pinhole model from distorted point.
    const T xd0 = inv_f * (pixel.x() - cx);
    const T yd0 = inv_f * (pixel.y() - cy);
    // Set distorted pixel as initial guess.
    T x = xd0;
    T y = yd0;
    // Run Newton's method on the residual of distorted pixel.
    for (int i = 0; i < max_iterations; ++i) {
      const T x2 = x * x;
      const T y2 = y * y;
      const T r2 = x2 + y2;
      const T s = T(1.0) + r2 * (k1 + r2 * (k2 + r2 * k3));
      // Compute residual.
      const T err_x = xd0 - (s * x + 2 * p1 * x * y + p2 * (r2 + 2 * x2));
      const T err_y = yd0 - (s * y + 2 * p2 * x * y + p1 * (r2 + 2 * y2));
      // If residual is small enough, exit early.
      if (std::abs(err_x) + std::abs(err_y) < kSmallValue) {
        break;
      }
      // Compute Jacobian:
      // J = [a, b]
      //     [b, c]
      const T ds = 2 * (k1 + r2 * (2 * k2 + 3 * k3 * r2));
      const T a = ds * x2 + s + 2 * (p1 * y + 3 * p2 * x);
      const T b = ds * x * y + 2 * (p1 * x + p2 * y);
      const T c = ds * y2 + s + 2 * (p2 * x + 3 * p1 * y);
      // Invert Jacobian;
      const T det = T(1.0) / (a * c - b * b);
      const T alpha = det * c;
      const T beta = -det * b;
      const T gamma = det * a;
      // Apply update.
      x = x + (alpha * err_x + beta * err_y);
      y = y + (beta * err_x + gamma * err_y);
    }
    return Eigen::Vector3<T>(x, y, 1);
  }

  CameraIntrinsicsModel GetType() const final {
    return kModelType;
  }

  int NumberOfParameters() const final {
    return kNumberOfParameters;
  }
};

// 4-parameter Kannala-Brandt projection model as presented in OpenCV, also known
// as the "fisheye" model. This model assumes an isotropic pinhole model, i.e.
// `fx == fy = f`. Parameters are in the following order:
//   [f, cx, cy, k1, k2, k3, k4]
// See https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html for more details.
class KannalaBrandtModel : public CameraModel {
 public:
  static constexpr int kNumberOfParameters = 7;
  static constexpr CameraIntrinsicsModel kModelType =
      CameraIntrinsicsModel::kKannalaBrandt;

  KannalaBrandtModel() = default;
  ~KannalaBrandtModel() override = default;
  KannalaBrandtModel& operator=(const KannalaBrandtModel&) = default;

  template <typename T>
  static absl::StatusOr<Eigen::Vector2<T>> ProjectPoint(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& point) {
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
    const T& f = intrinsics(0);
    const T& cx = intrinsics(1);
    const T& cy = intrinsics(2);
    const T& k1 = intrinsics(3);
    const T& k2 = intrinsics(4);
    const T& k3 = intrinsics(5);
    const T& k4 = intrinsics(6);

    Eigen::Vector2<T> projection(point.x() / point.z(), point.y() / point.z());
    const T r = projection.norm();
    T s;
    if (r < T(1e-9)) {
      const T r2 = r * r;
      s = T(1.0) + r2 * (k1 - T(1.0/3.0) + r2 * (-k1 + k2 + 0.2));
    } else {
      const T theta = atan(r);
      const T theta2 = theta * theta;
      const T theta_d = theta * (T(1.0) + theta2 *
                                 (k1 + theta2 *
                                  (k2 + theta2 * (k3 + theta2 * k4))));
      s = theta_d / r;
    }
    // Apply radial distortion.
    projection *= s;
    // Apply pinhole parameters.
    projection *= f;
    projection.x() += cx;
    projection.y() += cy;
    return projection;
  }

  // TODO(yangjames): Unproject is not that great for KB model :( it takes 100
  // Newton iterations to get down to 1e-9, converges way too slowly. Figure out
  // why this is.
  template <typename T>
  static absl::StatusOr<Eigen::Vector3<T>> UnprojectPixel(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector2<T>& pixel,
      int max_iterations = 100) {
    constexpr T kSmallValue = T(1e-14);
    constexpr T kSmallR = T(1e-9);
    if (intrinsics.size() != kNumberOfParameters) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Expected ", kNumberOfParameters, " parameters for intrinsics. Got ",
          intrinsics.size()));
    }
    const T& f = intrinsics(0);
    const T& cx = intrinsics(1);
    const T& cy = intrinsics(2);
    const T& k1 = intrinsics(3);
    const T& k2 = intrinsics(4);
    const T& k3 = intrinsics(5);
    const T& k4 = intrinsics(6);
    const T inv_f = T(1.0) / f;
    // Invert pinhole model from distorted point.
    const T xd0 = inv_f * (pixel.x() - cx);
    const T yd0 = inv_f * (pixel.y() - cy);
    // If pixel is already close to the origin, no need to continue.
    if (xd0 * xd0 + yd0 * yd0 < kSmallValue) {
      return Eigen::Vector3<T>(xd0, yd0, 1.0);
    }
    // Set distorted pixel as initial guess.
    T x = xd0;
    T y = yd0;
    // Run Newton's method on the residual of distorted pixel.
    for (int i = 0; i < max_iterations; ++i) {
      const T r2 = x * x + y * y;
      const T r = sqrt(r2);
      T s;
      if (r < kSmallR) {
        const T r2 = r * r;
        s = T(1.0) + r2 * (k1 - T(1.0/3.0) + r2 * (-k1 + k2 + 0.2));
      } else {
        const T theta = atan(r);
        const T theta2 = theta * theta;
        const T theta_d = theta * (T(1.0) + theta2 *
                                   (k1 + theta2 *
                                    (k2 + theta2 * (k3 + theta2 * k4))));
        s = theta_d / r;
      }
      const T err_x = xd0 - s * x;
      const T err_y = yd0 - s * y;
      // If residual is small enough, exit early.
      if (std::abs(err_x) + std::abs(err_y) < kSmallValue) {
        break;
      }
      // Compute Jacobian:
      // J = [a, b]
      //     [b, c]
      T a, b, c;
      // If small r, use 3rd order Taylor expansion instead.
      if (r < kSmallR) {
        const T q = T(2.0) * (k1 - T(1.0 / 3.0)) + T(4.0) * r2 * (-k1 + k2 + T(0.2));
        const T dsdx = q * x;
        const T dsdy = q * y;
        a = dsdx * x + s;
        b = q * x * y;
        c = dsdy * y + s;
      } else {
        const T theta = atan(r);
        const T theta2 = theta * theta;
        const T theta_d = theta * (1.0 + theta2 *
                                   (k1 + theta2 *
                                    (k2 + theta2 * (k3 + theta2 * k4))));
        const T s = theta_d / r;

        const T drdx = x / r;
        const T drdy = y / r;
        const T inv_r = T(1.0) / r;
        const T dthetadr = T(1.0) / (T(1.0) + r2);
        const T dthetaddtheta =
          T(1.0) + theta2 * (T(3.0) * k1 + theta2 *
                             (T(5.0) * k2 + theta2 *
                              (T(7.0) * k3 + theta2 * T(9.0) * k4)));
        const T dthetaddr = dthetaddtheta * dthetadr;
        const T dsdr = dthetaddr*inv_r + s * inv_r;
        a = dsdr * drdx * x + s;
        b = dsdr * x * y / r;
        c = dsdr * drdy * y + s;
      }
      // Invert Jacobian;
      const T det = T(1.0) / (a * c - b * b);
      if (std::isnan(det)) {
        std::cout << "r: " << r << std::endl;
        std::cout << "a, b, c: " << a << ", " << b << ", " << c << std::endl;
        std::cout << "det: " << det << std::endl;
      }
      const T alpha = det * c;
      const T beta = -det * b;
      const T gamma = det * a;
      // Apply update.
      x = x + (alpha * err_x + beta * err_y);
      y = y + (beta * err_x + gamma * err_y);
    }
    return Eigen::Vector3<T>(x, y, 1);
  }

  CameraIntrinsicsModel GetType() const final {
    return kModelType;
  }

  int NumberOfParameters() const final {
    return kNumberOfParameters;
  }
};

template <typename T>
absl::StatusOr<Eigen::Vector2<T>> CameraModel::ProjectPoint(
    const Eigen::VectorX<T>& intrinsics,
    const Eigen::Vector3<T>& point) const {
  if (const auto derived = dynamic_cast<const OpenCv5Model*>(this)) {
    return derived->ProjectPoint(intrinsics, point);
  }
  if (const auto derived = dynamic_cast<const KannalaBrandtModel*>(this)) {
    return derived->ProjectPoint(intrinsics, point);
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "ProjectPoint for camera model ", this->GetType(), " not supported."));
}

template <typename T>
absl::StatusOr<Eigen::Vector3<T>> CameraModel::UnprojectPixel(
    const Eigen::VectorX<T>& intrinsics,
    const Eigen::Vector2<T>& pixel) const {
  if (const auto derived = dynamic_cast<const OpenCv5Model*>(this)) {
    return derived->UnprojectPixel(intrinsics, pixel);
  }
  if (const auto derived = dynamic_cast<const KannalaBrandtModel*>(this)) {
    return derived->UnprojectPixel(intrinsics, pixel);
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "UnprojectPixel for camera model ", this->GetType(), " not supported."));
}

} // namespace calico::sensors

#endif // CALICO_SENSORS_CAMERA_MODELS_H_
