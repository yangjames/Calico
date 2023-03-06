#ifndef CALICO_SENSORS_SENSOR_BASE_H_
#define CALICO_SENSORS_SENSOR_BASE_H_

#include "calico/typedefs.h"
#include "calico/world_model.h"

#include "Eigen/Dense"

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ceres/problem.h"


namespace calico::sensors {

// Types of loss functions.
enum class LossFunctionType {
  kNone,
  kHuber,
  kCauchy,
};

// Optimization options for a sensor.
struct SensorOptimizationOptions {
  LossFunctionType loss_function_type;
  double loss_function_scale;
  bool intrinsics_fixed;
  bool extrinsics_fixed;
};

// Base class for sensors. For the sake of readability, we make this a purely
// virtual class, so setters and getters must be implemented at the derived
// class level.
class Sensor {
 public:
  virtual ~Sensor() = default;

  // Setter/getter for name.
  virtual void SetName(absl::string_view name) = 0;
  virtual const std::string& GetName() const = 0;

  // Setter/getter for extrinsics parameters.
  virtual void SetExtrinsics(const Pose3& T_sensorrig_sensor) = 0;
  virtual const Pose3& GetExtrinsics() const = 0;

  // Setter/getter for intrinsics parameters.
  virtual absl::Status SetIntrinsics(const Eigen::VectorXd& intrinsics) = 0;
  virtual const Eigen::VectorXd& GetIntrinsics() const = 0;

  // Enable or disable extrinsics parameters.
  virtual void EnableExtrinsicsParameters(bool enable) = 0;

  // Enable or disable intrinsics.
  virtual void EnableIntrinsicsParameters(bool enable) = 0;

  // Method for adding this sensor's calibration parameters to a problem.
  // Returns the number of parameters added.
  virtual absl::StatusOr<int> AddParametersToProblem(
      ceres::Problem& problem) = 0;

  // Method for adding this sensor's residuals to a problem.
  // TODO(yangjames): Replace sensorrig_trajectory with a BSpline.
  virtual absl::StatusOr<int> AddResidualsToProblem(
      ceres::Problem& problem,
      absl::flat_hash_map<double, Pose3>& sensorrig_trajectory,
      WorldModel& world_model) = 0;
};
} // namespace calico::sensors

#endif // CALICO_SENSORS_SENSOR_BASE_H_
