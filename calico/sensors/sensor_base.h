#ifndef CALICO_SENSORS_SENSOR_BASE_H_
#define CALICO_SENSORS_SENSOR_BASE_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "calico/optimization_utils.h"
#include "calico/trajectory.h"
#include "calico/typedefs.h"
#include "calico/world_model.h"
#include "ceres/problem.h"
#include "Eigen/Dense"


/// Sensors namespace
namespace calico::sensors {


/// Base class for sensors. For the sake of readability, we make this a purely
/// virtual class, so setters and getters must be implemented at the derived
/// class level.
class Sensor {
 public:
  virtual ~Sensor() = default;

  /// Setter for name.
  virtual void SetName(const std::string& name) = 0;

  /// Getter for name.
  virtual const std::string& GetName() const = 0;

  /// Setter for extrinsics parameters.
  virtual void SetExtrinsics(const Pose3d& T_sensorrig_sensor) = 0;

  /// Getter for extrinsics parameters.
  virtual const Pose3d& GetExtrinsics() const = 0;

  /// Setter for intrinsics parameters.
  virtual absl::Status SetIntrinsics(const Eigen::VectorXd& intrinsics) = 0;

  /// Getter for intrinsics parameters.
  virtual const Eigen::VectorXd& GetIntrinsics() const = 0;

  /// Setter for sensor latency.
  virtual absl::Status SetLatency(double latency) = 0;

  /// Getter for sensor latency.
  virtual double GetLatency() const = 0;

  /// Enable or disable extrinsics estimation.
  virtual void EnableExtrinsicsEstimation(bool enable) = 0;

  /// Enable or disable intrinsics estimation.
  virtual void EnableIntrinsicsEstimation(bool enable) = 0;

  /// Enable or disable latency estimation.
  virtual void EnableLatencyEstimation(bool enable) = 0;

  /// Update residuals for this sensor.

  /// This will only apply to measurements not marked as outliers.\n\n
  /// **Note: This method is meant to be invoked by BatchOptimizer ONLY. It is
  /// not recommended that you invoke this method manually.**
  virtual absl::Status UpdateResiduals(ceres::Problem& problem) = 0;

  /// Clear all stored info about residuals.
  virtual void ClearResidualInfo() = 0;

  /// Setter for loss function and scale.
  virtual void SetLossFunction(
      utils::LossFunctionType loss, double scale = 1.0) = 0;

  /// Add this sensor's calibration parameters to the ceres problem.

  /// Returns the number of parameters added to the problem, which should be
  /// intrinsics + extrinsics + latency. If the sensor's model hasn't been set
  /// yet, it will return an invalid argument error.
  virtual absl::StatusOr<int> AddParametersToProblem(
      ceres::Problem& problem) = 0;

  /// Contribue this sensor's residuals to the ceres problem.

  /// `sensorrig_trajectory` is the world-from-sensorrig trajectory
  /// \f$\mathbf{T}^w_r(t)\f$.
  virtual absl::StatusOr<int> AddResidualsToProblem(
      ceres::Problem& problem,
      Trajectory& sensorrig_trajectory,
      WorldModel& world_model) = 0;
};
} // namespace calico::sensors

#endif // CALICO_SENSORS_SENSOR_BASE_H_
