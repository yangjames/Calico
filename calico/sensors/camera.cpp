#include "calico/sensors/camera.h"

namespace calico::sensors {

void Camera::SetName(absl::string_view name) {
  name_ = name;
}
const std::string& Camera::GetName() const { return name_; }

void Camera::SetExtrinsics(const Pose3& T_sensorrig_sensor) {
  T_sensorrig_sensor_ = T_sensorrig_sensor;
}
    
const Pose3& Camera::GetExtrinsics() const {
  return T_sensorrig_sensor_;
}

absl::Status Camera::SetIntrinsics(const Eigen::VectorXd& intrinsics) {
  if (!camera_model_) {
    return absl::InvalidArgumentError("Camera model has not been set!");
  }
  if (intrinsics.size() != camera_model_->NumberOfParameters()) {
    return absl::InvalidArgumentError(
        absl::StrCat(
            "Tried to set intrinsics of size ", intrinsics.size(),
            " for camera ", GetName(), ". Expected intrinsics size of ",
            camera_model_->NumberOfParameters()));
  }
  intrinsics_ = intrinsics;
  return absl::OkStatus();
}

const Eigen::VectorXd& Camera::GetIntrinsics() const {
  return intrinsics_;
}

absl::Status Camera::SetCameraModel(CameraIntrinsicsModel camera_model) {
  camera_model_ = CameraModel::Create(camera_model);
  if (camera_model_) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Could not create camera model for type ", camera_model,
      ". It is likely not yet implemented."));
}

CameraIntrinsicsModel Camera::GetCameraModel() const {
  return camera_model_ ?
    camera_model_->GetType() : CameraIntrinsicsModel::kNone;
}

absl::Status Camera::AddMeasurement(const CameraMeasurement& measurement) {
  if (id_to_measurement_.contains(measurement.id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Tried to add redundant measurement - Image id: ",
                     measurement.id.image_id, ", model id: ",
                     measurement.id.model_id, ", feature id: ",
                     measurement.id.feature_id));
  }
  id_to_measurement_[measurement.id] = measurement;
  return absl::OkStatus();
}

absl::Status Camera::AddMeasurements(
    const std::vector<CameraMeasurement>& measurements) {
  std::string message;
  for (const auto& measurement : measurements) {
    absl::Status status = AddMeasurement(measurement);
    if (!status.ok()) {
      message += std::string(status.message()) + "\n";
    }
  }
  if (message.empty()) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(message);
}

absl::Status Camera::RemoveMeasurementById(const ObservationId& id) {
  if (id_to_measurement_.erase(id)) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Attempted to remove invalid mesaurement - Image id: ",
      id.image_id, ", model id: ", id.model_id, ", feature_id: ",
      id.feature_id));
}

absl::Status Camera::RemoveMeasurementsById(
    const std::vector<ObservationId>& ids) {
  std::string message;
  for (const auto& id : ids) {
    absl::Status status = RemoveMeasurementById(id);
    if (!status.ok()) {
      message += std::string(status.message()) + "\n";
    }
  }
  if (message.empty()) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(message);
}

void Camera::ClearMeasurements() {
  id_to_measurement_.clear();
}

int Camera::NumberOfMeasurements() const {
  return id_to_measurement_.size();
}

} // namespace calico::sensors::camera
