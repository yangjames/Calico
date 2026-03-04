#ifndef CALICO_SENSORS_MULTI_CAMERA_H_
#define CALICO_SENSORS_MULTI_CAMERA_H_

#include <string>
#include <vector>

#include "Eigen/Dense"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "calico/sensors/camera.h"
#include "calico/sensors/camera_models.h"
#include "calico/sensors/sensor_base.h"
#include "calico/trajectory.h"
#include "calico/typedefs.h"
#include "ceres/problem.h"

namespace calico::sensors {

/// Multi-camera class. This class can be used to calibrate multiple imagers on
/// a single module, for example, stereo cameras.
class MultiCamera : public Sensor {
 public:
  MultiCamera() = delete;  // Require instantiation with imager names.
  explicit MultiCamera(const absl::flat_hash_set<std::string>& imagers) {
    imagers_ = imagers;
    for (const auto& imager : imagers) {
      imager_to_intrinsics_enabled_[imager] = false;
      imager_to_extrinsics_enabled_[imager] = false;
      imager_to_latency_enabled_[imager] = false;
      imager_to_camera_model_[imager] = nullptr;
      imager_to_pose_sensor_from_imager_[imager] = Pose3d();
      imager_to_intrinsics_[imager] = Eigen::VectorXd(0);
      imager_to_latency_[imager] = 0.0;
      imager_to_id_to_measurement_.insert({imager, {}});
      imager_to_id_to_residual_.insert({imager, {}});
      imager_to_id_to_residual_id_.insert({imager, {}});
      imager_to_outlier_ids_.insert({imager, {}});
    }
  }
  MultiCamera(const MultiCamera&) = delete;
  MultiCamera& operator=(const MultiCamera&) = delete;
  ~MultiCamera() = default;

  // /// Top-level sensor module name.
  void SetName(const std::string& sensor_name) { sensor_name_ = sensor_name; }
  const std::string& GetName() const { return sensor_name_; }

  /// Setter for the camera model.
  absl::Status SetModel(const std::string& imager,
                        CameraIntrinsicsModel camera_model);

  /// Getter for the camera model.
  absl::StatusOr<CameraIntrinsicsModel> GetModel(
      const std::string& imager) const;

  /// Sets extrinsics for the camera module itself. Each sensor will internally
  /// have an additional extrinsics offset relative to this one. This can be set
  /// to a constant CAD value.
  void SetSensorExtrinsics(const Pose3d& pose_sensorrig_from_sensor) {
    pose_sensorrig_from_sensor_ = pose_sensorrig_from_sensor;
  }
  const Pose3d& GetSensorExtrinsics() const {
    return pose_sensorrig_from_sensor_;
  }

  /// Set the extrinsics for each internal sensor. This transform will be
  /// applied on top of pose_sensorrig_from_sensor which is held constant during
  /// optimization.
  absl::Status SetImagerExtrinsics(const std::string& imager,
                                   const Pose3d& pose_sensor_from_imager);
  absl::StatusOr<Pose3d> GetImagerExtrinsics(const std::string& imager) const;

  absl::Status SetIntrinsics(const std::string& imager,
                             const Eigen::VectorXd& intrinsics);
  absl::StatusOr<Eigen::VectorXd> GetIntrinsics(
      const std::string& imager) const;

  absl::Status SetLatency(const std::string& imager, double latency) {
    imager_to_latency_[imager] = latency;
    return absl::OkStatus();
  }
  double GetLatency(const std::string& imager) const {
    return imager_to_latency_.at(imager);
  }
  void EnableExtrinsicsEstimation(const std::string& imager, bool enable) {
    imager_to_extrinsics_enabled_[imager] = enable;
  }

  void EnableIntrinsicsEstimation(const std::string& imager, bool enable) {
    imager_to_intrinsics_enabled_[imager] = enable;
  }
  void EnableLatencyEstimation(const std::string& imager, bool enable) {
    imager_to_latency_enabled_[imager] = enable;
  }
  void SetLossFunction(utils::LossFunctionType loss, double scale) final {
    loss_function_ = loss;
    loss_scale_ = scale;
  }
  absl::StatusOr<int> AddParametersToProblem(ceres::Problem& problem) final;
  absl::StatusOr<int> AddResidualsToProblem(ceres::Problem& problem,
                                            Trajectory& sensorrig_trajectory,
                                            WorldModel& world_model) final {
    return absl::UnimplementedError(
        "AddResidualsToProblem not implemented for MultiCamera.");
  }
  absl::Status SetMeasurementNoise(double sigma) final {
    if (sigma > 0) {
      sigma_ = sigma;
      return absl::OkStatus();
    }
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot set ", GetName(), " measurement noise to ", sigma,
                     ". Measurement noise must be positive."));
  }
  absl::Status UpdateResiduals(ceres::Problem& problem) final {
    return absl::UnimplementedError(
        "UpdateResiduals not implemented for MultiCamera.");
  }
  void ClearResidualInfo() final {
    for (auto& [_, id_to_residual_] : imager_to_id_to_residual_) {
      id_to_residual_.clear();
    }
    for (auto& [_, id_to_residual_id_] : imager_to_id_to_residual_id_) {
      id_to_residual_id_.clear();
    }
  }

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
  absl::StatusOr<
      absl::flat_hash_map<std::string, std::vector<CameraMeasurement>>>
  Project(const std::vector<double>& interp_times,
          const Trajectory& sensorrig_trajectory,
          const WorldModel& world_model) const;

  /// Add a single camera measurement to the measurement list.
  /// Returns an error if the measurement's id is duplicated without adding.
  absl::Status AddMeasurement(const std::string& imager,
                              const CameraMeasurement& measurement);

  /// Add multiple measurements to the measurement list.
  /// Returns an error status if any measurements are duplicates within its
  /// internally managed set of measurements.\n\n
  /// **Note: If this method encounters any duplicates, it will STILL attempt to
  /// add the entire vector. If it returns an error status, it means that all
  /// unique measurements have been added, but duplicates have been skipped.**
  absl::Status AddMeasurements(
      const std::string& imager,
      const std::vector<CameraMeasurement>& measurements);

  /// Getter for all measurements. Returns a map of observation ids to
  /// measurements. Will be empty if there are no measurements.
  const absl::flat_hash_map<
      std::string, absl::flat_hash_map<CameraObservationId, CameraMeasurement>>&
  GetMeasurementIdToMeasurement() const {
    return imager_to_id_to_measurement_;
  }

  /// Returns a vector of measurement-residual pairs.

  /// Only returns for measurements that have residuals. Returns an error if
  /// there are more residuals than measurements, or if there are no
  /// measurements.\n\n
  /// **Note: This method will only return residuals for measurements that have
  /// NOT been marked as outliers.**
  absl::StatusOr<absl::flat_hash_map<
      std::string, std::vector<std::pair<CameraMeasurement, Eigen::Vector2d>>>>
  GetMeasurementResidualPairs() const;

  // /// Tag a single measurement as an outlier by its measurement ID.

  // /// Camera class keeps track of an outliers list internally. If passed a
  // /// measurement ID that does not correspond with any measurement tracked by
  // /// this camera, an InvalidArgument status is returned.
  // absl::Status MarkOutlierById(const CameraObservationId& id);

  // /// Tag multiple measurements as outliers by measurement ID.

  // /// Camera class keeps track of an outliers list internally. If passed a
  // /// measurement ID that does not correspond with any measurement tracked by
  // /// this camera, an InvalidArgument status is returned.
  // absl::Status MarkOutliersById(const std::vector<CameraObservationId>& ids);

  // /// Clear outliers list.
  // void ClearOutliersList();

  // /// Clear all measurements.

  /// This will also clear any internally stored residuals and marked outliers.
  void ClearMeasurements();

  /// Get current number of measurements stored for a given imager.
  int NumberOfMeasurements(const std::string& imager) const;

 private:
  std::string sensor_name_;
  absl::flat_hash_set<std::string> imagers_;

  Pose3d pose_sensorrig_from_sensor_;
  absl::flat_hash_map<std::string, bool> imager_to_intrinsics_enabled_;
  absl::flat_hash_map<std::string, bool> imager_to_extrinsics_enabled_;
  absl::flat_hash_map<std::string, bool> imager_to_latency_enabled_;
  absl::flat_hash_map<std::string, std::unique_ptr<CameraModel>>
      imager_to_camera_model_;
  absl::flat_hash_map<std::string, Pose3d> imager_to_pose_sensor_from_imager_;
  absl::flat_hash_map<std::string, Eigen::VectorXd> imager_to_intrinsics_;
  absl::flat_hash_map<std::string, double> imager_to_latency_;
  double sigma_ = 1.0;
  utils::LossFunctionType loss_function_;
  double loss_scale_ = 1.0;
  absl::flat_hash_map<
      std::string, absl::flat_hash_map<CameraObservationId, CameraMeasurement>>
      imager_to_id_to_measurement_;
  absl::flat_hash_map<std::string,
                      absl::flat_hash_map<CameraObservationId, Eigen::Vector2d>>
      imager_to_id_to_residual_;
  absl::flat_hash_map<std::string, absl::flat_hash_map<CameraObservationId,
                                                       ceres::ResidualBlockId>>
      imager_to_id_to_residual_id_;
  absl::flat_hash_map<std::string, absl::flat_hash_set<CameraObservationId>>
      imager_to_outlier_ids_;
};

}  // namespace calico::sensors

#endif  // CALICO_SENSORS_CAMERA_H_
