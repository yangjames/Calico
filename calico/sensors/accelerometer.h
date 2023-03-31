#ifndef CALICO_SENSORS_ACCELEROMETER_H_
#define CALICO_SENSORS_ACCELEROMETER_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "calico/sensors/sensor_base.h"
#include "calico/sensors/accelerometer_models.h"
#include "calico/trajectory.h"
#include "calico/typedefs.h"
#include "ceres/problem.h"
#include "Eigen/Dense"


namespace calico::sensors {

// Accelerometer observation id.
struct AccelerometerObservationId {
  double stamp;
  int sequence;

  template <typename H>
  friend H AbslHashValue(H h, const AccelerometerObservationId& id) {
    return H::combine(std::move(h), id.stamp, id.sequence);
  }
  friend bool operator==(const AccelerometerObservationId& lhs,
                         const AccelerometerObservationId& rhs) {
    return (lhs.stamp == rhs.stamp &&
            lhs.sequence == rhs.sequence);
  }
};

// Accelerometer measurement type.
struct AccelerometerMeasurement {
  Eigen::Vector3d measurement;
  AccelerometerObservationId id;
};


class Accelerometer : public Sensor {
 public:
  explicit Accelerometer() = default;
  Accelerometer(const Accelerometer&) = delete;
  Accelerometer& operator=(const Accelerometer&) = delete;
  ~Accelerometer() = default;

  // Setter/getter for name.
  void SetName(const std::string& name) final;
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

  // Set loss function type.
  void SetLossFunction(utils::LossFunctionType loss, double scale) final;

  // Add this accelerometer's parameters to the ceres problem. Returns the number of
  // parameters added to the problem, which should be intrinsics + extrinsics +
  // latency. If the accelerometer model hasn't been set yet, it will return an
  // invalid argument error.
  absl::StatusOr<int> AddParametersToProblem(ceres::Problem& problem) final;

  // Contribue this accelerometer's residuals to the ceres problem.
  absl::StatusOr<int> AddResidualsToProblem(
      ceres::Problem & problem,
      Trajectory& sensorrig_trajectory,
      WorldModel& world_model) final;

  // Update residuals for this sensor.
  absl::Status UpdateResiduals(ceres::Problem& problem) final;

  // Clear all residual information.
  void ClearResidualInfo() final;

  // Compute synthetic accelerometer measurements at given a sensor rig trajectory.
  absl::StatusOr<std::vector<AccelerometerMeasurement>> Project(
      const std::vector<double>& interp_times,
      const Trajectory& sensorrig_trajectory,
      const WorldModel& world_model) const;

  // Setter/getter for accelerometer model.
  absl::Status SetModel(AccelerometerIntrinsicsModel accelerometer_model);
  AccelerometerIntrinsicsModel GetModel() const;

  // Add a accelerometer measurement to the measurement list. Returns an error if the
  // measurement's id is duplicated without adding.
  absl::Status AddMeasurement(const AccelerometerMeasurement& measurement);

  // Add multiple measurements to the measurement list. Returns an error status
  // if any measurements are duplicates. This method will add the entire vector,
  // but skips any duplicates.
  absl::Status AddMeasurements(
      const std::vector<AccelerometerMeasurement>& measurements);

  // Clear all measurements.
  void ClearMeasurements();

  // Get current number of measurements stored.
  int NumberOfMeasurements() const;

 private:
  std::string name_;
  bool intrinsics_enabled_;
  bool extrinsics_enabled_;
  bool latency_enabled_;
  std::unique_ptr<AccelerometerModel> accelerometer_model_;
  Pose3d T_sensorrig_sensor_;
  Eigen::VectorXd intrinsics_;
  double latency_;
  utils::LossFunctionType loss_function_;
  double loss_scale_;
  absl::flat_hash_map<AccelerometerObservationId, AccelerometerMeasurement>
      id_to_measurement_;
  absl::flat_hash_map<AccelerometerObservationId, Eigen::Vector3d> id_to_residual_;
  absl::flat_hash_map<AccelerometerObservationId, ceres::ResidualBlockId>
      id_to_residual_id_;
};

} // namespace calico::sensors

#endif // CALICO_SENSORS_ACCELEROMETER_H_
