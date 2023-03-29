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
struct GyroscopeObservationId {
  double stamp;
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

// Gyroscope measurement type.
struct GyroscopeMeasurement {
  Eigen::Vector3d measurement;
  GyroscopeObservationId id;
};


class Gyroscope : public Sensor {
 public:
  explicit Gyroscope() = default;
  Gyroscope(const Gyroscope&) = delete;
  Gyroscope& operator=(const Gyroscope&) = delete;
  ~Gyroscope() = default;

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

  // Update residuals for this sensor. This is mean to be only invoked by the
  // BatchOptimizer class.
  absl::Status UpdateResiduals(ceres::Problem& problem) final;

  // Clear all residual information.
  void ClearResidualInfo() final;

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
  utils::LossFunctionType loss_function_;
  double loss_scale_;
  absl::flat_hash_map<GyroscopeObservationId, GyroscopeMeasurement>
      id_to_measurement_;
  absl::flat_hash_map<GyroscopeObservationId, Eigen::Vector3d> id_to_residual_;
  absl::flat_hash_map<GyroscopeObservationId, ceres::ResidualBlockId>
      id_to_residual_id_;
};

} // namespace calico::sensors

#endif // CALICO_SENSORS_GYROSCOPE_H_
