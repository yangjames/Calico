#include "calico/sensors/camera.h"

#include "calico/sensors/camera_cost_functor.h"
#include "calico/optimization_utils.h"


namespace calico::sensors {

void Camera::SetName(const std::string& name) {
  name_ = name;
}
const std::string& Camera::GetName() const { return name_; }

void Camera::SetExtrinsics(const Pose3d& T_sensorrig_sensor) {
  T_sensorrig_sensor_ = T_sensorrig_sensor;
}

const Pose3d& Camera::GetExtrinsics() const {
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

absl::Status Camera::SetLatency(double latency) {
  latency_ = latency;
  return absl::OkStatus();
}

double Camera::GetLatency() const {
  return latency_;
}

void Camera::EnableExtrinsicsEstimation(bool enable) {
  extrinsics_enabled_ = enable;
}

void Camera::EnableIntrinsicsEstimation(bool enable) {
  intrinsics_enabled_ = enable;
}

void Camera::EnableLatencyEstimation(bool enable) {
  latency_enabled_ = enable;
}

absl::Status Camera::SetMeasurementNoise(double sigma) {
  if (sigma <= 0.0) {
    return absl::InvalidArgumentError("Sigma must be greater than 0.");
  }
  sigma_ = sigma;
  return absl::OkStatus();
}

absl::Status Camera::UpdateResiduals(ceres::Problem& problem) {
  for (const auto [measurement_id, residual_id] : id_to_residual_id_) {
    Eigen::Vector2d residual;
    if (!problem.EvaluateResidualBlock(residual_id,
        /*apply_loss_function=*/false, nullptr, residual.data(), nullptr)) {
      return absl::InternalError("Failed to update residual for camera " + name_);
    }
    id_to_residual_[measurement_id] = residual;
  }
  return absl::OkStatus();
}

void Camera::ClearResidualInfo() {
  id_to_residual_id_.clear();
  id_to_residual_.clear();
}

void Camera::SetLossFunction(utils::LossFunctionType loss, double scale) {
  loss_function_ = loss;
  loss_scale_ = scale;
}

absl::StatusOr<int> Camera::AddParametersToProblem(ceres::Problem& problem) {
  int num_parameters_added = 0;
  if (!camera_model_) {
    return absl::FailedPreconditionError(
        "Cannot add camera parameters. Camera model is not yet defined.");
  }
  problem.AddParameterBlock(intrinsics_.data(), intrinsics_.size());
  num_parameters_added += intrinsics_.size();
  num_parameters_added += utils::AddPoseToProblem(problem, T_sensorrig_sensor_);
  problem.AddParameterBlock(&latency_, 1);
  ++num_parameters_added;
  if (!intrinsics_enabled_) {
    problem.SetParameterBlockConstant(intrinsics_.data());
  }
  if (!extrinsics_enabled_) {
    utils::SetPoseConstantInProblem(problem, T_sensorrig_sensor_);
  }
  if (!latency_enabled_) {
    problem.SetParameterBlockConstant(&latency_);
  }
  return num_parameters_added;
}

absl::StatusOr<int> Camera::AddResidualsToProblem(
    ceres::Problem& problem,
    Trajectory& sensorrig_trajectory,
    WorldModel& world_model) {
  int num_residuals_added = 0;
  for (const auto& [observation_id, measurement] : id_to_measurement_) {
    // Skip measurements marked as outliers.
    if (outlier_ids_.contains(observation_id)) {
      continue;
    }
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
    // Construct a cost function and supply parameters for this residual.
    std::vector<double*> parameters;

    ceres::CostFunction* cost_function =
        CameraCostFunctor::CreateCostFunction(
            measurement.pixel, camera_model_->GetType(), intrinsics_,
            T_sensorrig_sensor_, latency_, t_model_point,
            rigidbody_ref.T_world_rigidbody, sensorrig_trajectory,
            observation_id.stamp, parameters);
    ceres::LossFunction* loss_function = CreateLossFunction(
        loss_function_, loss_scale_);
    const auto residual_block_id = problem.AddResidualBlock(
        cost_function, loss_function, parameters);
    id_to_residual_id_[observation_id] = residual_block_id;
    num_residuals_added += 1;
  }
  return num_residuals_added;
}

absl::StatusOr<std::vector<CameraMeasurement>> Camera::Project(
    const std::vector<double>& interp_times,
    const Trajectory& sensorrig_trajectory,
    const WorldModel& world_model) const {
  std::vector<Pose3d> poses_world_sensorrig;
  ASSIGN_OR_RETURN(poses_world_sensorrig,
                   sensorrig_trajectory.Interpolate(interp_times));
  std::vector<CameraMeasurement> measurements;
  int image_id = 0;
  for (int i = 0; i < interp_times.size(); ++i) {
    const Pose3d& T_world_sensorrig = poses_world_sensorrig.at(i);
    const double& stamp = interp_times.at(i);
    const Pose3d T_camera_world =
        (T_world_sensorrig * T_sensorrig_sensor_).inverse();
    // Project all landmarks.
    for (const auto& [landmark_id, landmark] : world_model.landmarks()) {
      const Eigen::Vector3d point_camera = T_camera_world * landmark.point;
      if (point_camera.z() <= 0) {
        continue;
      }
      const absl::StatusOr<Eigen::Vector2d> projection =
          camera_model_->ProjectPoint(intrinsics_, point_camera);
      measurements.push_back({
          *projection, {
            stamp + latency_,
            image_id,
            kLandmarkFrameId,
            landmark_id
          }});
    }
    // Project all rigid bodies.
    for (const auto& [rigidbody_id, rigidbody] : world_model.rigidbodies()) {
      const Pose3d T_camera_rigidbody =
          T_camera_world * rigidbody.T_world_rigidbody;
      for (const auto& [point_id, point] : rigidbody.model_definition) {
        const Eigen::Vector3d point_camera = T_camera_rigidbody * point;
        const absl::StatusOr<Eigen::Vector2d> projection =
          camera_model_->ProjectPoint(intrinsics_, point_camera);
        measurements.push_back({
            *projection, {
              stamp + latency_,
              image_id,
              rigidbody_id,
              point_id
            }});
      }
    }
    ++image_id;
  }
  return measurements;
}

absl::Status Camera::SetModel(CameraIntrinsicsModel camera_model) {
  camera_model_ = CameraModel::Create(camera_model);
  intrinsics_ = Eigen::VectorXd::Zero(camera_model_->NumberOfParameters());
  if (camera_model_) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Could not create camera model for type ", camera_model,
      ". It is likely not yet implemented."));
}

CameraIntrinsicsModel Camera::GetModel() const {
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

const absl::flat_hash_map<CameraObservationId, CameraMeasurement>&
Camera::GetMeasurementIdToMeasurement() const {
  return id_to_measurement_;
}

absl::StatusOr<std::vector<std::pair<CameraMeasurement, Eigen::Vector2d>>>
Camera::GetMeasurementResidualPairs() const {
  if (id_to_residual_.size() > id_to_measurement_.size()) {
    return absl::InternalError(
        "There are more residuals than measurements.");
  }
  if (id_to_measurement_.empty()) {
    return absl::FailedPreconditionError(
        "Measurements are empty. Nothing to return.");
  }
  std::vector<std::pair<CameraMeasurement, Eigen::Vector2d>> pairs;
  for (const auto [id, residual] : id_to_residual_) {
    auto it = id_to_measurement_.find(id);
    if (it != id_to_measurement_.end()) {
      pairs.push_back({it->second, residual});
    } else {
      return absl::InternalError(
          "Found a residual that doesn't correspond to any measurement.");
    }
  }
  return pairs;
}

absl::Status Camera::MarkOutlierById(const CameraObservationId& id) {
  if (!id_to_measurement_.contains(id)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Attempted to add id that is not within the measurement set.",
        "\nStamp - ", id.stamp, ", image_id - ", id.image_id,
        "model_id - ", id.model_id, "feature_id - ", id.feature_id));
  }
  outlier_ids_.insert(id);
  return absl::OkStatus();
}

absl::Status Camera::MarkOutliersById(const std::vector<CameraObservationId>& ids) {
  for (const auto& id : ids) {
    RETURN_IF_ERROR(MarkOutlierById(id));
  }
  return absl::OkStatus();
}

void Camera::ClearOutliersList() {
  outlier_ids_.clear();
}

void Camera::ClearMeasurements() {
  id_to_measurement_.clear();
  id_to_residual_id_.clear();
  id_to_residual_.clear();
  outlier_ids_.clear();
}

int Camera::NumberOfMeasurements() const {
  return id_to_measurement_.size();
}

} // namespace calico::sensors
