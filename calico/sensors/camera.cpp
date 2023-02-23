#include "calico/sensors/camera.h"

#include "calico/sensors/camera_cost_functor.h"
#include "calico/optimization_utils.h"


namespace calico::sensors {

absl::StatusOr<int> Camera::AddParametersToProblem(ceres::Problem& problem) {
  int num_parameters_added = 0;
  if (!camera_model_) {
    return absl::FailedPreconditionError(
        "Cannot add camera parameters. Camera model is not yet defined.");
  }
  problem.AddParameterBlock(intrinsics_.data(), intrinsics_.size());
  num_parameters_added += intrinsics_.size();
  num_parameters_added += utils::AddPoseToProblem(problem, T_sensorrig_sensor_);

  if (!intrinsics_enabled_) {
    problem.SetParameterBlockConstant(intrinsics_.data());
  }
  if (!extrinsics_enabled_) {
    utils::SetPoseConstantInProblem(problem, T_sensorrig_sensor_);
  }
  return num_parameters_added;
}

absl::StatusOr<int> Camera::AddResidualsToProblem(
    ceres::Problem& problem,
    absl::flat_hash_map<int, Pose3>& sensorrig_trajectory,
    WorldModel& world_model) {
  int num_residuals_added = 0;
  for (const auto& [observation_id, measurement] : id_to_measurement_) {
    const int rigidbody_id = observation_id.model_id;
    if (!world_model.rigidbodies().contains(rigidbody_id)) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Attempted to create cost function from an observation for a "
          "rigidbody with id ", rigidbody_id, " that does not exist in "
          "the world model."));
    }
    // Get the right rigidbody reference from the world model.
    RigidBody& rigidbody_ref = world_model.rigidbodies().at(rigidbody_id);
    Eigen::Vector3d& t_model_point =
      rigidbody_ref.model_definition.at(observation_id.feature_id);
    Pose3& T_world_sensorrig = sensorrig_trajectory.at(observation_id.image_id);
    // Construct a cost function and supply parameters for this residual.
    std::vector<double*> parameters;
    ceres::CostFunction* cost_function =
        CameraCostFunctor::CreateCostFunction(
            measurement.pixel, camera_model_->GetType(), intrinsics_,
            T_sensorrig_sensor_, t_model_point, rigidbody_ref.T_world_rigidbody,
            T_world_sensorrig, parameters);
    const auto residual_block_id = problem.AddResidualBlock(
        cost_function, /*loss_function=*/nullptr, parameters);
    num_residuals_added += 1;
  }
  return num_residuals_added;
}

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

void Camera::EnableExtrinsicsParameters(bool enable) {
  extrinsics_enabled_ = enable;
}

void Camera::EnableIntrinsicsParameters(bool enable) {
  intrinsics_enabled_ = enable;
}

absl::Status Camera::SetImageSize(const ImageSize& image_size) {
  if (!(image_size.width > 0 && image_size.height > 0)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid image size of width - ", image_size.width, ", height - ",
        image_size.height, ". Width and height must be positive values."));
  }
  image_size_ = image_size;
  return absl::OkStatus();
}

ImageSize Camera::GetImageSize() const {
  return image_size_;
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
