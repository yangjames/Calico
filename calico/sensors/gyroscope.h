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

// Gyroscope observation id.
struct GyroObservationId {
  double stamp;
  int sequence;

  template <typename H>
  friend H AbslHashValue(H h, const GyroObservationId& id) {
    return H::combine(std::move(h), id.stamp, id.sequence);
  }
  friend bool operator==(const GyroObservationId& lhs,
                         const GyroObservationId& rhs) {
    return (lhs.stamp == rhs.stamp &&
            lhs.sequence == rhs.sequence);
  }
};

// Gyroscope measurement type.
struct GyroscopeMeasurement {
  Eigen::Vector3d measurement;
  GryoObservationId id;
};


class Gyroscope : public Sensor {
 public:
  explicit Gyroscope() = default;
  Gyroscope(const Gyroscope&) = delete;
  Gyroscope& operator=(const Gyroscope&) = delete;
  ~Gyroscope() = default;

  // Setter/getter for name.
  void SetName(absl::string_view name) final;
  const std::string& GetName() const final;

  // Setter/getter for extrinsics parameters.
  void SetExtrinsics(const Pose3d& T_sensorrig_sensor) final;
  const Pose3d& GetExtrinsics() const final;

  // Setter/getter for intrinsics parameters.
  absl::Status SetIntrinsics(const Eigen::VectorXd& intrinsics) final;
  const Eigen::VectorXd& GetIntrinsics() const final;

  // Setter/getter for sensor latency.
  absl::Status SetLatency(double latency) final;
  double GetLatency() const final;

  // Enable flags for intrinsics, extrinsics, and latency.
  void EnableExtrinsicsEstimation(bool enable) final;
  void EnableIntrinsicsEstimation(bool enable) final;
  void EnableLatencyEstimation(bool enable) final;

  // Add this gyroscope's parameters to the ceres problem. Returns the number of
  // parameters added to the problem, which should be intrinsics + extrinsics +
  // latency. If the gyroscope model hasn't been set yet, it will return an
  // invalid argument error.
  absl::StatusOr<int> AddParametersToProblem(ceres::Problem& problem) final;

  // Contribue this gyroscope's residuals to the ceres problem.
  absl::StatusOr<int> AddResidualsToProblem(
      ceres::Problem & problem,
      Trajectory& sensorrig_trajectory,
      WorldModel& world_model) final;

  // Compute synthetic gyroscope measurements at given a sensor rig trajectory.
  absl::StatusOr<std::vector<GyroscopeMeasurement>> Project(
      const std::vector<double>& interp_times,
      const Trajectory& sensorrig_trajectory) const;

  // Setter/getter for gyroscope model.
  absl::Status SetModel(GyroscopeIntrinsicsModel gyroscope_model);
  GyroscopeIntrinsicsModel GetModel() const;

  // Add a gyroscope measurement to the measurement list. Returns an error if the
  // measurement's id is duplicated without adding.
  absl::Status AddMeasurement(const GyroscopeMeasurement& measurement);

  // Add multiple measurements to the measurement list. Returns an error status
  // if any measurements are duplicates. This method will add the entire vector,
  // but skips any duplicates.
  absl::Status AddMeasurements(
      const std::vector<GyroscopeMeasurement>& measurements);

  // Remove a measurement with a specific observation id. Returns an error if
  // the id was not associated with a measurement.
  absl::Status RemoveMeasurementById(GyroObservationId id);

  // Remove multiple measurements by their observation ids. Returns an error if
  // it attempts to remove an id that was not associated with a measurement.
  // This method will remove the entire vector, but skip invalid entries.
  absl::Status RemoveMeasurementsById(
      const std::vector<GyroObservationId>& ids);

  // Clear all measurements.
  void ClearMeasurements();

  // Get current number of measurements stored.
  int NumberOfMeasurements() const;

 private:
  std::string name_;
  bool intrinsics_enabled_;
  bool extrinsics_enabled_;
  bool latency_enabled_;
  std::unique_ptr<GyroscopeModel> gyroscope_model_;
  Pose3d T_sensorrig_sensor_;
  Eigen::VectorXd intrinsics_;
  double latency_;
  absl::flat_hash_map<GyroObservationId, GyroscopeMeasurement>
      id_to_measurement_;
};

} // namespace calico::sensors

#endif // CALICO_SENSORS_GYROSCOPE_H_
