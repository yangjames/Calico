#ifndef CALICO_SENSORS_SENSOR_BASE_H_
#define CALICO_SENSORS_SENSOR_BASE_H_

#include "Eigen/Dense"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "calico/optimization_utils.h"
#include "calico/trajectory.h"
#include "calico/typedefs.h"
#include "calico/world_model.h"
#include "ceres/problem.h"

/// Sensors namespace
namespace calico::sensors {

/// Base class for sensors. For the sake of readability, we make this a purely
/// virtual class, so setters and getters must be implemented at the derived
/// class level.
class Sensor {
 public:
  virtual ~Sensor() = default;

  /// Update residuals for this sensor.

  /// This will only apply to measurements not marked as outliers.\n\n
  /// **Note: This method is meant to be invoked by BatchOptimizer ONLY. It is
  /// not recommended that you invoke this method manually.**
  virtual absl::Status UpdateResiduals(ceres::Problem& problem) = 0;

  /// Clear all stored info about residuals.
  virtual void ClearResidualInfo() = 0;

  /// Setter for loss function and scale.
  virtual void SetLossFunction(utils::LossFunctionType loss,
                               double scale = 1.0) = 0;

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
      ceres::Problem& problem, Trajectory& sensorrig_trajectory,
      WorldModel& world_model) = 0;

  /// Set the measurement noise \f$\sigma\f$.

  /// This value is used to weight the
  /// sensor's residuals such that:
  /// \f\[
  ///   \boldsymbol{\Sigma} = \sigma^2\mathbf{I}\\
  ///   \boldsymbol{\epsilon} = \boldsymbol{\Sigma}^{-1/2}\left(\mathbf{y} -
  ///   \mathbf{\hat{y}}\left(\mathbf{x}, \boldsymbol{\beta}\right)\right)\\
  ///   \mathbf{J} =
  ///   \frac{\partial\boldsymbol{\epsilon}}{\partial\delta\boldsymbol{\beta}}\\
  ///   \delta\boldsymbol{\beta} =
  ///   \left(\mathbf{J}^T\mathbf{J}\right)^{-1}\mathbf{J}^T\boldsymbol{\epsilon}
  /// \f\]
  virtual absl::Status SetMeasurementNoise(double sigma) = 0;
};
}  // namespace calico::sensors

#endif  // CALICO_SENSORS_SENSOR_BASE_H_
