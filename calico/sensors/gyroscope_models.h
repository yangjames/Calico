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
  kScale,
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
class ScaleModel : public GyroscopeModel {
 public:
  static constexpr int kNumberOfParameters = 8;
  static constexpr GyroscopeIntrinsicsModel kModelType =
      GyroscopeIntrinsicsModel::kScale;

  ScaleModel() = default;
  ~ScaleModel() override = default;
  ScaleModel& operator=(const ScaleModel&) = default;

  template <typename T>
  static absl::StatusOr<Eigen::Vector2<T>> Project(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& omega_sensor_world) {
    return Eigen::Vector3<T>();
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
    const Eigen::Vector3<T>& point) const {
  if (const auto derived = dynamic_cast<const ScaleModel*>(this)) {
    return derived->ProjectPoint(intrinsics, point);
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Project for gyroscope model ", this->GetType(), " not supported."));
}

template <typename T>
absl::StatusOr<Eigen::Vector3<T>> GyroscopeModel::Unproject(
    const Eigen::VectorX<T>& intrinsics,
    const Eigen::Vector3<T>& omega_sensor_world) const {
  if (const auto derived = dynamic_cast<const ScaleModel*>(this)) {
    return derived->Unproject(intrinsics, omega_sensor_world);
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Unproject for gyroscope model ", this->GetType(), " not supported."));
}

} // namespace calico::sensors

#endif // CALICO_SENSORS_GYROSCOPE_MODELS_H_
