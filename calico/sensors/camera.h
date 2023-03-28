#ifndef CALICO_SENSORS_CAMERA_H_
#define CALICO_SENSORS_CAMERA_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "calico/sensors/sensor_base.h"
#include "calico/sensors/camera_models.h"
#include "calico/trajectory.h"
#include "calico/typedefs.h"
#include "ceres/problem.h"
#include "Eigen/Dense"


namespace calico::sensors {

// CameraObservationId type for a camera measurement. This object is hashable by
// `absl::Hash` for use as a key in `absl::flat_hash_map` or
// `absl::flat_hash_set`.
struct CameraObservationId {
  double stamp;
  int image_id;
  int model_id;
  int feature_id;

  template <typename H>
  friend H AbslHashValue(H h, const CameraObservationId& id) {
    return H::combine(std::move(h), id.stamp, id.image_id, id.model_id,
                      id.feature_id);
  }
  friend bool operator==(const CameraObservationId& lhs,
                         const CameraObservationId& rhs) {
    return (lhs.stamp == rhs.stamp &&
            lhs.image_id == rhs.image_id &&
            lhs.model_id == rhs.model_id &&
            lhs.feature_id == rhs.feature_id);
  }
};

// Camera measurement type.
struct CameraMeasurement {
  Eigen::Vector2d pixel;
  CameraObservationId id;
};


class Camera : public Sensor {
 public:
  explicit Camera() = default;
  Camera(const Camera&) = delete;
  Camera& operator=(const Camera&) = delete;
  ~Camera() = default;

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

  // Add this camera's parameters to the ceres problem. Returns the number of
  // parameters added to the problem, which should be intrinsics + extrinsics.
  // If the camera model hasn't been set yet, it will return an invalid
  // argument error.
  absl::StatusOr<int> AddParametersToProblem(ceres::Problem& problem) final;

  // Contribue this camera's residuals to the ceres problem.
  absl::StatusOr<int> AddResidualsToProblem(
      ceres::Problem & problem,
      Trajectory& sensorrig_trajectory,
      WorldModel& world_model) final;

  // Update residuals for this sensor.
  absl::Status UpdateResiduals(ceres::Problem& problem) final;

  // Clear all residual information.
  void ClearResidualInfo() final;

  // Compute the project of a world model through a kinematic chain. This
  // method returns only valid synthetic measurements as would be observed by
  // the actual sensor, complying with physicality such as features being in
  // front of the camera and within image bounds.
  absl::StatusOr<std::vector<CameraMeasurement>> Project(
      const std::vector<double>& interp_times,
      const Trajectory& sensorrig_trajectory,
      const WorldModel& world_model) const;

  // Setter/getter for camera model.
  absl::Status SetModel(CameraIntrinsicsModel camera_model);
  CameraIntrinsicsModel GetModel() const;

  // Add a camera measurement to the measurement list. Returns an error if the
  // measurement's id is duplicated without adding.
  absl::Status AddMeasurement(const CameraMeasurement& measurement);

  // Add multiple measurements to the measurement list. Returns an error status
  // if any measurements are duplicates. This method will add the entire vector,
  // but skips any duplicates.
  absl::Status AddMeasurements(
      const std::vector<CameraMeasurement>& measurements);

  // Getter for observation id to residual map. If this sensor hasn't been
  // optimized, the returned map will be empty.
  const absl::flat_hash_map<CameraObservationId, Eigen::Vector2d>&
  GetMeasurementIdToResidual() const;

  // Getter for observation id to measurement map.
  const absl::flat_hash_map<CameraObservationId, CameraMeasurement>&
  GetMeasurementIdToMeasurement() const;

  // Returns a vector of measurement-residual pairs. Returns an error if
  // residuals and measurements are not the same size, or if one contains an
  // observation ID that the other doesn't.
  absl::StatusOr<std::vector<std::pair<CameraMeasurement, Eigen::Vector2d>>>
  GetMeasurementResidualPairs() const;

  // Clear all measurements.
  void ClearMeasurements();

  // Get current number of measurements stored.
  int NumberOfMeasurements() const;

 private:
  std::string name_;
  bool intrinsics_enabled_;
  bool extrinsics_enabled_;
  bool latency_enabled_;
  std::unique_ptr<CameraModel> camera_model_;
  Pose3d T_sensorrig_sensor_;
  Eigen::VectorXd intrinsics_;
  double latency_;
  utils::LossFunctionType loss_function_;
  double loss_scale_;
  absl::flat_hash_map<CameraObservationId, CameraMeasurement>
      id_to_measurement_;
  absl::flat_hash_map<CameraObservationId, Eigen::Vector2d> id_to_residual_;
  absl::flat_hash_map<CameraObservationId, ceres::ResidualBlockId>
      id_to_residual_id_;
};

} // namespace calico::sensors

#endif // CALICO_SENSORS_CAMERA_H_
