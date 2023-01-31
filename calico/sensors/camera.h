#ifndef CALICO_SENSORS_CAMERA_H_
#define CALICO_SENSORS_CAMERA_H_

#include "calico/typedefs.h"
#include "calico/sensors/sensor_base.h"
#include "calico/sensors/camera_models.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "ceres/problem.h"
#include "Eigen/Dense"


namespace calico::sensors {

// ObservationId type for a camera measurement. This object is hashable by
// `absl::Hash` for use as a key in `absl::flat_hash_map` or
// `absl::flat_hash_set`.
struct ObservationId {
  int image_id;
  int model_id;
  int feature_id;

  template <typename H>
  friend H AbslHashValue(H h, const ObservationId& id) {
    return H::combine(std::move(h), id.image_id, id.model_id, id.feature_id);
  }
  friend bool operator==(const ObservationId& lhs, const ObservationId& rhs) {
    return (lhs.image_id == rhs.image_id &&
            lhs.model_id == rhs.model_id &&
            lhs.feature_id == rhs.feature_id);
  }
};

// Camera measurement type.
struct CameraMeasurement {
  Eigen::Vector2d pixel;
  ObservationId id;
  double stamp;
};

class Camera : public Sensor {
 public:
  explicit Camera() = default;
  Camera(const Camera&) = delete;
  Camera& operator=(const Camera&) = delete;
  ~Camera() = default;

  // Setter/getter for name.
  void SetName(absl::string_view name) final;
  const std::string& GetName() const final;

  // Setter/getter for extrinsics parameters.
  void SetExtrinsics(const Pose3& T_sensorrig_sensor) final;
  const Pose3& GetExtrinsics() const final;

  // Setter/getter for intrinsics parameters.
  absl::Status SetIntrinsics(const Eigen::VectorXd& intrinsics) final;
  const Eigen::VectorXd& GetIntrinsics() const final;

  // Setter/getter for camera model.
  absl::Status SetCameraModel(CameraIntrinsicsModel camera_model);
  CameraIntrinsicsModel GetCameraModel() const;

  // Add a camera measurement to the measurement list. Returns an error if the
  // measurement's id is duplicated without adding.
  absl::Status AddMeasurement(const CameraMeasurement& measurement);

  // Add multiple measurements to the measurement list. Returns an error status
  // if any measurements are duplicates. This method will add the entire vector,
  // but skips any duplicates.
  absl::Status AddMeasurements(
      const std::vector<CameraMeasurement>& measurements);

  // Remove a measurement with a specific observation id. Returns an error if
  // the id was not associated with a measurement.
  absl::Status RemoveMeasurementById(const ObservationId& id);

  // Remove multiple measurements by their observation ids. Returns an error if
  // it attempts to remove an id that was not associated with a measurement.
  // This method will remove the entire vector, but skip invalid entries.
  absl::Status RemoveMeasurementsById(const std::vector<ObservationId>& ids);

  // Clear all measurements.
  void ClearMeasurements();

  // Get current number of measurements stored.
  int NumberOfMeasurements() const;

 private:
  std::string name_;
  int image_width_;
  int image_height_;
  std::unique_ptr<CameraModel> camera_model_;
  Pose3 T_sensorrig_sensor_;
  Eigen::VectorXd intrinsics_;
  absl::flat_hash_map<ObservationId, CameraMeasurement> id_to_measurement_;
};

} // namespace calico::sensors

#endif // CALICO_SENSORS_CAMERA_H_
