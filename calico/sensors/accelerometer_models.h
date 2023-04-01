#ifndef CALICO_SENSORS_ACCELEROMETER_MODELS_H_
#define CALICO_SENSORS_ACCELEROMETER_MODELS_H_

#include "calico/typedefs.h"

#include <memory>

#include "absl/status/statusor.h"
#include "Eigen/Dense"


namespace calico::sensors {

// Accelerometer model types.
enum class AccelerometerIntrinsicsModel : int {
  kNone,
  kAccelerometerScaleOnly,
  kAccelerometerScaleAndBias,
};

// Base class for accelerometer models.
class AccelerometerModel {
 public:
  virtual ~AccelerometerModel() = default;

  // Project a linear acceleration vector through the intrinsics model.
  // Top level call invokes the derived class's implementation.
  template <typename T>
  absl::StatusOr<Eigen::Vector3<T>> Project(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& ddt_sensor_world_sensor) const;

  // Invert the intrinsics model to get an angular velocity vector.
  // Top level call invokes the derived class's implementation.
  template <typename T>
  absl::StatusOr<Eigen::Vector3<T>> Unproject(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& measurement) const;

  // Getter for accelerometer model type.
  virtual AccelerometerIntrinsicsModel GetType() const  = 0;

  // Getter for the number of parameters for this model.
  virtual int NumberOfParameters() const = 0;

  // TODO(yangjames): This method should return an absl::StatusOr. Figure out
  //   how to support this using unique_ptr's, or find macros that already
  //   implement this feature (i.e. ASSIGN_OR_RETURN).
  // Factory method for creating a accelerometer model with `accelerometer_model`
  // type. This method will return a nullptr if an unsupported
  // AccelerometerIntrinsicsModel is passed in.
  static std::unique_ptr<AccelerometerModel> Create(
      AccelerometerIntrinsicsModel accelerometer_model);
};

// 1-parameter scale-only intrinsics model.
class AccelerometerScaleOnlyModel : public AccelerometerModel {
 public:
  static constexpr int kNumberOfParameters = 1;
  static constexpr AccelerometerIntrinsicsModel kModelType =
      AccelerometerIntrinsicsModel::kAccelerometerScaleOnly;

  AccelerometerScaleOnlyModel() = default;
  ~AccelerometerScaleOnlyModel() override = default;
  AccelerometerScaleOnlyModel& operator=(
      const AccelerometerScaleOnlyModel&) = default;

  template <typename T>
  static absl::StatusOr<Eigen::Vector3<T>> Project(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& ddt_sensor_world_sensor) {
    const T& scale = intrinsics(0);
    return scale * ddt_sensor_world_sensor;
  }

  template <typename T>
  static absl::StatusOr<Eigen::Vector3<T>> Unproject(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& measurement) {
    const T& inv_scale = static_cast<T>(1.0) / intrinsics(0);
    return inv_scale * measurement;
  }

  AccelerometerIntrinsicsModel GetType() const final {
    return kModelType;
  }

  int NumberOfParameters() const final {
    return kNumberOfParameters;
  }
};

// 4-parameter scale + bias intrinsics model.
// Parameter order:
//   [s, bx, by, bz]
class AccelerometerScaleAndBiasModel : public AccelerometerModel {
 public:
  static constexpr int kNumberOfParameters = 4;
  static constexpr AccelerometerIntrinsicsModel kModelType =
      AccelerometerIntrinsicsModel::kAccelerometerScaleAndBias;

  AccelerometerScaleAndBiasModel() = default;
  ~AccelerometerScaleAndBiasModel() override = default;
  AccelerometerScaleAndBiasModel& operator=(
      const AccelerometerScaleAndBiasModel&) = default;

  template <typename T>
  static absl::StatusOr<Eigen::Vector3<T>> Project(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& ddt_sensor_world_sensor) {
    if (intrinsics.size() != kNumberOfParameters) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid accelerometer intrinsics size. Expected ",
          kNumberOfParameters, ", got ", intrinsics.size()));
    }
    const T& scale = intrinsics(0);
    const Eigen::Ref<const Eigen::Vector3<T>> bias =
        intrinsics.segment(1, 3);
    return scale * ddt_sensor_world_sensor + bias;
  }

  template <typename T>
  static absl::StatusOr<Eigen::Vector3<T>> Unproject(
      const Eigen::VectorX<T>& intrinsics,
      const Eigen::Vector3<T>& measurement) {
    if (intrinsics.size() != kNumberOfParameters) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid accelerometer intrinsics size. Expected ",
          kNumberOfParameters, ", got ", intrinsics.size()));
    }
    const T& inv_scale = static_cast<T>(1.0) / intrinsics(0);
    const Eigen::Ref<const Eigen::Vector3<T>> bias =
        intrinsics.segment(1, 3);
    return inv_scale * (measurement - bias);
  }

  AccelerometerIntrinsicsModel GetType() const final {
    return kModelType;
  }

  int NumberOfParameters() const final {
    return kNumberOfParameters;
  }
};

template <typename T>
absl::StatusOr<Eigen::Vector3<T>> AccelerometerModel::Project(
    const Eigen::VectorX<T>& intrinsics,
    const Eigen::Vector3<T>& ddt_sensor_world_sensor) const {
  if (const auto derived =
      dynamic_cast<const AccelerometerScaleOnlyModel*>(this)) {
    return derived->Project(intrinsics, ddt_sensor_world_sensor);
  }
  if (const auto derived =
      dynamic_cast<const AccelerometerScaleAndBiasModel*>(this)) {
    return derived->Project(intrinsics, ddt_sensor_world_sensor);
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Project for accelerometer model ", this->GetType(), " not supported."));
}

template <typename T>
absl::StatusOr<Eigen::Vector3<T>> AccelerometerModel::Unproject(
    const Eigen::VectorX<T>& intrinsics,
    const Eigen::Vector3<T>& measurement) const {
  if (const auto derived =
      dynamic_cast<const AccelerometerScaleOnlyModel*>(this)) {
    return derived->Unproject(intrinsics, measurement);
  }
  if (const auto derived =
      dynamic_cast<const AccelerometerScaleAndBiasModel*>(this)) {
    return derived->Unproject(intrinsics, measurement);
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Unproject for accelerometer model ", this->GetType(), " not supported."));
}

} // namespace calico::sensors

#endif // CALICO_SENSORS_ACCELEROMETER_MODELS_H_
