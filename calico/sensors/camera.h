#ifndef CALICO_SENSORS_CAMERA_H_
#define CALICO_SENSORS_CAMERA_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "calico/sensors/sensor_base.h"
#include "calico/sensors/camera_models.h"
#include "calico/trajectory.h"
#include "calico/typedefs.h"
#include "ceres/problem.h"
#include "Eigen/Dense"


namespace calico::sensors {

/// Camera observation id type for a camera measurement.
/// This object is hashable by `absl::Hash` for use as a key in
/// `absl::flat_hash_map` or `absl::flat_hash_set`.
struct CameraObservationId {
  /// Timestamp in seconds.
  double stamp;
  /// Image id.
  int image_id;
  /// \brief RigidBody model id. Equivalent to `RigidBody.id` field of a
  /// RigidBody or `kLandmarkFrameId` if this measurement corersponds to a
  /// landmark.
  int model_id;
  /// \brief Feature id. Equivalent to the key field of
  /// `RigidBody.model_definition` of RigidBody or `Landmark.id` field of
  /// Landmark.
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
  /// \brief Pixel location of the observed feature.
  Eigen::Vector2d pixel;
  /// \brief Id of this observation.
  CameraObservationId id;
};


/// Camera class. 
class Camera : public Sensor {
 public:
  explicit Camera() = default;
  Camera(const Camera&) = delete;
  Camera& operator=(const Camera&) = delete;
  ~Camera() = default;

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

  /// Compute synthetic camera measurements given a Trajectory and WorldModel.

  /// This method projects the world model through the kinematic chain at given
  /// timestamps. This method returns only valid synthetic measurements as would
  /// be observed by the actual sensor, complying with physicality such as
  /// features being in front of the camera. Returns measurements in the order
  /// of the interpolation timestamps.\n\n
  /// `interp_times` is a vector of timestamps in seconds at which
  /// `sensorrig_trajectory` will be interpolated. No assumptions are made about
  /// timestamp uniqueness or order.\n\n
  /// `sensorrig_trajectory` is the world-from-sensorrig trajectory
  /// \f$\mathbf{T}^w_r(t)\f$.\n\n
  absl::StatusOr<std::vector<CameraMeasurement>> Project(
      const std::vector<double>& interp_times,
      const Trajectory& sensorrig_trajectory,
      const WorldModel& world_model) const;

  /// Setter for the camera model.
  absl::Status SetModel(CameraIntrinsicsModel camera_model);

  /// Getter for the camera model.
  CameraIntrinsicsModel GetModel() const;

  /// Add a single camera measurement to the measurement list.

  /// Returns an error if the measurement's id is duplicated without adding.
  absl::Status AddMeasurement(const CameraMeasurement& measurement);

  /// Add multiple measurements to the measurement list.

  /// Returns an error status if any measurements are duplicates within its
  /// internally managed set of measurements.\n\n
  /// **Note: If this method encounters any duplicates, it will STILL attempt to
  /// add the entire vector. If it returns an error status, it means that all
  /// unique measurements have been added, but duplicates have been skipped.**
  absl::Status AddMeasurements(
      const std::vector<CameraMeasurement>& measurements);

  /// Getter for all measurements. Returns a map of observation ids to
  /// measurements. Will be empty if there are no measurements.
  const absl::flat_hash_map<CameraObservationId, CameraMeasurement>&
  GetMeasurementIdToMeasurement() const;

  /// Returns a vector of measurement-residual pairs.

  /// Only returns for measurements that have residuals. Returns an error if
  /// there are more residuals than measurements, or if there are no
  /// measurements.\n\n
  /// **Note: This method will only return residuals for measurements that have
  /// NOT been marked as outliers.**
  absl::StatusOr<std::vector<std::pair<CameraMeasurement, Eigen::Vector2d>>>
  GetMeasurementResidualPairs() const;
  
  /// Tag a single measurement as an outlier by its measurement ID.

  /// Camera class keeps track of an outliers list internally. If passed a
  /// measurement ID that does not correspond with any measurement tracked by
  /// this camera, an InvalidArgument status is returned.
  absl::Status MarkOutlierById(const CameraObservationId& id);

  /// Tag multiple measurements as outliers by measurement ID.

  /// Camera class keeps track of an outliers list internally. If passed a
  /// measurement ID that does not correspond with any measurement tracked by
  /// this camera, an InvalidArgument status is returned.
  absl::Status MarkOutliersById(const std::vector<CameraObservationId>& ids);

  /// Clear outliers list.
  void ClearOutliersList();

  /// Clear all measurements.

  /// This will also clear any internally stored residuals and marked outliers.
  void ClearMeasurements();

  /// Get current number of measurements stored.
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
  double sigma_;
  utils::LossFunctionType loss_function_;
  double loss_scale_;
  absl::flat_hash_map<CameraObservationId, CameraMeasurement>
      id_to_measurement_;
  absl::flat_hash_map<CameraObservationId, Eigen::Vector2d> id_to_residual_;
  absl::flat_hash_map<CameraObservationId, ceres::ResidualBlockId>
      id_to_residual_id_;
  absl::flat_hash_set<CameraObservationId> outlier_ids_;
};

} // namespace calico::sensors

#endif // CALICO_SENSORS_CAMERA_H_
