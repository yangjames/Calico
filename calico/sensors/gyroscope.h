#ifndef CALICO_SENSORS_GYROSCOPE_H_
#define CALICO_SENSORS_GYROSCOPE_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "calico/sensors/sensor_base.h"
#include "calico/sensors/gyroscope_models.h"
#include "calico/trajectory.h"
#include "calico/typedefs.h"
#include "ceres/problem.h"
#include "Eigen/Dense"


namespace calico::sensors {

/// Gyroscope observation id type for a gyroscope measurement.
/// This object is hashable by `absl::Hash` for use as a key in
/// `absl::flat_hash_map` or `absl::flat_hash_set`.
struct GyroscopeObservationId {
  /// Timestamp in seconds.
  double stamp;
  /// Sequence number of this measurement.
  int sequence;

  template <typename H>
  friend H AbslHashValue(H h, const GyroscopeObservationId& id) {
    return H::combine(std::move(h), id.stamp, id.sequence);
  }
  friend bool operator==(const GyroscopeObservationId& lhs,
                         const GyroscopeObservationId& rhs) {
    return (lhs.stamp == rhs.stamp &&
            lhs.sequence == rhs.sequence);
  }
};

/// Gyroscope measurement type.
struct GyroscopeMeasurement {
  /// \brief Raw uncalibrated measurement value from a gyroscope.
  Eigen::Vector3d measurement;
  /// \brief Id of this observation.
  GyroscopeObservationId id;
};

/// Gyroscope class.
class Gyroscope : public Sensor {
 public:
  explicit Gyroscope() = default;
  Gyroscope(const Gyroscope&) = delete;
  Gyroscope& operator=(const Gyroscope&) = delete;
  ~Gyroscope() = default;

  void SetName(const std::string& name) final;
  const std::string& GetName() const final;
  void SetExtrinsics(const Pose3d& T_sensorrig_sensor) final;
  const Pose3d& GetExtrinsics() const final;
  absl::Status SetIntrinsics(const Eigen::VectorXd& intrinsics) final;
  const Eigen::VectorXd& GetIntrinsics() const final;
  absl::Status SetLatency(double latency) final;
  double GetLatency() const final;
  void EnableExtrinsicsEstimation(bool enable) final;
  void EnableIntrinsicsEstimation(bool enable) final;
  void EnableLatencyEstimation(bool enable) final;
  void SetLossFunction(utils::LossFunctionType loss, double scale) final;
  absl::StatusOr<int> AddParametersToProblem(ceres::Problem& problem) final;
  absl::StatusOr<int> AddResidualsToProblem(
      ceres::Problem & problem,
      Trajectory& sensorrig_trajectory,
      WorldModel& world_model) final;
  absl::Status SetMeasurementNoise(double sigma) final;
  absl::Status UpdateResiduals(ceres::Problem& problem) final;
  void ClearResidualInfo() final;

  /// Compute synthetic gyroscope measurements at given a sensorrig trajectory.

  /// This method interpolates the sensorrig trajectory at given timestamps and
  /// generates synthetic measurements as would be observed by the actual sensor
  /// at those timestamps.\n\n
  /// `interp_times` is a vector of timestamps in seconds at which
  /// `sensorrig_trajectory` will be interpolated. No assumptions are made about
  /// timestamp uniqueness or order.\n\n
  /// `sensorrig_trajectory` is the world-from-sensorrig trajectory
  /// \f$\mathbf{T}^w_r(t)\f$.\n\n
  absl::StatusOr<std::vector<GyroscopeMeasurement>> Project(
      const std::vector<double>& interp_times,
      const Trajectory& sensorrig_trajectory,
      const WorldModel& world_model) const;

  /// Setter for gyroscope model.
  absl::Status SetModel(GyroscopeIntrinsicsModel gyroscope_model);

  /// Getter for gyroscope model.
  GyroscopeIntrinsicsModel GetModel() const;

  /// Add a gyroscope measurement to the measurement list.

  /// Returns an error if the measurement's id is duplicated without adding.
  absl::Status AddMeasurement(const GyroscopeMeasurement& measurement);

  /// Add multiple measurements to the measurement list.

  /// Returns an error status if any measurements are duplicates. This method
  /// will add the entire vector, but skips any duplicates.
  /// **Note: If this method encounters any duplicates, it will STILL attempt to
  /// add the entire vector. If it returns an error status, it means that all
  /// unique measurements have been added, but duplicates have been skipped.**
  absl::Status AddMeasurements(
      const std::vector<GyroscopeMeasurement>& measurements);

  /// Clear all measurements.
  void ClearMeasurements();

  /// Get current number of measurements stored.
  int NumberOfMeasurements() const;

 private:
  std::string name_;
  bool intrinsics_enabled_;
  bool extrinsics_enabled_;
  bool latency_enabled_;
  std::unique_ptr<GyroscopeModel> gyroscope_model_;
  Pose3d T_sensorrig_sensor_;
  Eigen::VectorXd intrinsics_;
  double latency_ = 0.0;
  double sigma_ = 1.0;
  utils::LossFunctionType loss_function_;
  double loss_scale_ = 1.0;
  absl::flat_hash_map<GyroscopeObservationId, GyroscopeMeasurement>
      id_to_measurement_;
  absl::flat_hash_map<GyroscopeObservationId, Eigen::Vector3d> id_to_residual_;
  absl::flat_hash_map<GyroscopeObservationId, ceres::ResidualBlockId>
      id_to_residual_id_;
};

} // namespace calico::sensors

#endif // CALICO_SENSORS_GYROSCOPE_H_
