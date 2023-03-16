#ifndef CALICO_SENSORS_GYROSCOPE_MODELS_H_
#define CALICO_SENSORS_GYROSCOPE_MODELS_H_

#include "calico/typedefs.h"

#include <memory>

#include "absl/status/statusor.h"
#include "Eigen/Dense"


namespace calico::sensors {

// Gyroscope model types.
enum class GyroscopeIntrinsicsModel : int {
  kNone,
  kScaleOnly,
  kScaleAndBias,
  kVectorNav,
};

// Base class for gyroscope models.
class GyroscopeModel {
 public:
  virtual ~GyroscopeModel() = default;

  // Project an angular velocity vector through the intrinsics model.
  // Top level call invokes the derived class's implementation.
  template <typename T>
  absl::StatusOr<Eigen::Vector3<T>> Project(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& omega_sensor_world) const;

  // Invert the intrinsics model to get an angular velocity vector.
  // Top level call invokes the derived class's implementation.
  template <typename T>
  absl::StatusOr<Eigen::Vector3<T>> Unproject(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& measurement) const;

  // Getter for gyroscope model type.
  virtual GyroscopeIntrinsicsModel GetType() const  = 0;

  // Getter for the number of parameters for this model.
  virtual int NumberOfParameters() const = 0;

  // TODO(yangjames): This method should return an absl::StatusOr. Figure out
  //   how to support this using unique_ptr's, or find macros that already
  //   implement this feature (i.e. ASSIGN_OR_RETURN).
  // Factory method for creating a gyroscope model with `gyroscope_model` type.
  // This method will return a nullptr if an unsupported
  // GyroscopeIntrinsicsModel is passed in.
  static std::unique_ptr<GyroscopeModel> Create(
      GyroscopeIntrinsicsModel gyroscope_model);
};

// 1-parameter scale-only intrinsics model.
class ScaleOnlyModel : public GyroscopeModel {
 public:
  static constexpr int kNumberOfParameters = 1;
  static constexpr GyroscopeIntrinsicsModel kModelType =
      GyroscopeIntrinsicsModel::kScaleOnly;

  ScaleOnlyModel() = default;
  ~ScaleOnlyModel() override = default;
  ScaleOnlyModel& operator=(const ScaleOnlyModel&) = default;

  template <typename T>
  static absl::StatusOr<Eigen::Vector3<T>> Project(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& omega_sensor_world) {
    const T& scale = intrinsics(0);
    return scale * omega_sensor_world;
  }

  template <typename T>
  static absl::StatusOr<Eigen::Vector3<T>> Unproject(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& measurement) {
    const T& inv_scale = static_cast<T>(1.0) / intrinsics(0);
    return inv_scale * measurement;
  }

  GyroscopeIntrinsicsModel GetType() const final {
    return kModelType;
  }

  int NumberOfParameters() const final {
    return kNumberOfParameters;
  }
};

// 4-parameter scale + bias intrinsics model.
// Parameter order:
//   [s, bx, by, bz]
class ScaleAndBiasModel : public GyroscopeModel {
 public:
  static constexpr int kNumberOfParameters = 4;
  static constexpr GyroscopeIntrinsicsModel kModelType =
      GyroscopeIntrinsicsModel::kScaleAndBias;

  ScaleAndBiasModel() = default;
  ~ScaleAndBiasModel() override = default;
  ScaleAndBiasModel& operator=(const ScaleAndBiasModel&) = default;

  template <typename T>
  static absl::StatusOr<Eigen::Vector3<T>> Project(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& omega_sensor_world) {
    if (intrinsics.size() != kNumberOfParameters) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid gyroscope intrinsics size. Expected ", kNumberOfParameters,
          ", got ", intrinsics.size()));
    }
    const T& scale = intrinsics(0);
    const Eigen::Ref<const Eigen::Vector3<T>> bias =
        intrinsics.segment(1, 3);
    return scale * omega_sensor_world + bias;
  }

  template <typename T>
  static absl::StatusOr<Eigen::Vector3<T>> Unproject(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& measurement) {
    if (intrinsics.size() != kNumberOfParameters) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid gyroscope intrinsics size. Expected ", kNumberOfParameters,
          ", got ", intrinsics.size()));
    }
    const T& inv_scale = static_cast<T>(1.0) / intrinsics(0);
    const Eigen::Ref<const Eigen::Vector3<T>> bias =
        intrinsics.segment(1, 3);
    return inv_scale * (measurement - bias);
  }

  GyroscopeIntrinsicsModel GetType() const final {
    return kModelType;
  }

  int NumberOfParameters() const final {
    return kNumberOfParameters;
  }
};

template <typename T>
absl::StatusOr<Eigen::Vector3<T>> GyroscopeModel::Project(
    const Eigen::VectorX<T>& intrinsics,
    const Eigen::Vector3<T>& omega_sensor_world) const {
  if (const auto derived = dynamic_cast<const ScaleOnlyModel*>(this)) {
    return derived->Project(intrinsics, omega_sensor_world);
  }
  if (const auto derived = dynamic_cast<const ScaleAndBiasModel*>(this)) {
    return derived->Project(intrinsics, omega_sensor_world);
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Project for gyroscope model ", this->GetType(), " not supported."));
}

template <typename T>
absl::StatusOr<Eigen::Vector3<T>> GyroscopeModel::Unproject(
    const Eigen::VectorX<T>& intrinsics,
    const Eigen::Vector3<T>& measurement) const {
  if (const auto derived = dynamic_cast<const ScaleOnlyModel*>(this)) {
    return derived->Unproject(intrinsics, measurement);
  }
  if (const auto derived = dynamic_cast<const ScaleAndBiasModel*>(this)) {
    return derived->Unproject(intrinsics, measurement);
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Unproject for gyroscope model ", this->GetType(), " not supported."));
}

} // namespace calico::sensors

#endif // CALICO_SENSORS_GYROSCOPE_MODELS_H_
