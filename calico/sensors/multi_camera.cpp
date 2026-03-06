#include "calico/sensors/multi_camera.h"

#include <memory>

#include "calico/optimization_utils.h"
#include "calico/sensors/imager_cost_functor.h"

namespace calico::sensors {

absl::Status MultiCamera::SetModel(const std::string& imager,
                                   CameraIntrinsicsModel camera_model) {
  if (!imagers_.contains(imager)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Imager ", imager, " not found in multi-camera set."));
  }
  imager_to_camera_model_[imager] = CameraModel::Create(camera_model);
  imager_to_intrinsics_[imager] = Eigen::VectorXd::Zero(
      imager_to_camera_model_[imager]->NumberOfParameters());
  if (!imager_to_camera_model_[imager]) {
    return absl::InvalidArgumentError(
        absl::StrCat("Could not create camera model for type ", camera_model,
                     ". It is likely not yet implemented."));
  }
  return absl::OkStatus();
}

absl::StatusOr<CameraIntrinsicsModel> MultiCamera::GetModel(
    const std::string& imager) const {
  if (!imagers_.contains(imager)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Imager ", imager, " not found in multi-camera set."));
  }
  const auto& model = imager_to_camera_model_.at(imager);
  if (!model) {
    return CameraIntrinsicsModel::kNone;
  }
  return imager_to_camera_model_.at(imager)->GetType();
}

absl::Status MultiCamera::SetImagerExtrinsics(
    const std::string& imager, const Pose3d& pose_sensor_from_imager) {
  if (!imagers_.contains(imager)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Imager ", imager, " not found in MultiCamera."));
  }
  imager_to_pose_sensor_from_imager_[imager] = pose_sensor_from_imager;
  return absl::OkStatus();
}

absl::StatusOr<Pose3d> MultiCamera::GetImagerExtrinsics(
    const std::string& imager) const {
  if (!imagers_.contains(imager)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Imager ", imager, " not found in MultiCamera."));
  }
  return imager_to_pose_sensor_from_imager_.at(imager);
}

absl::Status MultiCamera::SetIntrinsics(const std::string& imager,
                                        const Eigen::VectorXd& intrinsics) {
  if (!imagers_.contains(imager)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Imager ", imager, " not found in multi-camera set."));
  }

  const auto& camera_model = imager_to_camera_model_[imager];
  if (!camera_model) {
    return absl::InvalidArgumentError("Camera model has not been set!");
  }
  if (intrinsics.size() != camera_model->NumberOfParameters()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Tried to set intrinsics of size ", intrinsics.size(),
                     " for camera ", GetName(), ".Expected intrinsics size of ",
                     camera_model->NumberOfParameters()));
  }
  imager_to_intrinsics_[imager] = intrinsics;
  return absl::OkStatus();
}

absl::Status MultiCamera::UpdateResiduals(ceres::Problem& problem) {
  for (const auto& imager : imagers_) {
    auto& id_to_residual_ = imager_to_id_to_residual_.at(imager);
    const auto& id_to_residual_id_ = imager_to_id_to_residual_id_.at(imager);
    for (const auto [measurement_id, residual_id] : id_to_residual_id_) {
      Eigen::Vector2d residual;
      if (!problem.EvaluateResidualBlock(residual_id,
                                         /*apply_loss_function=*/false, nullptr,
                                         residual.data(), nullptr)) {
        return absl::InternalError("Failed to update residual for camera " +
                                   sensor_name_);
      }
      id_to_residual_[measurement_id] = residual;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<int> MultiCamera::AddParametersToProblem(
    ceres::Problem& problem) {
  int num_parameters_added = 0;
  // Add parameters.
  for (const auto& imager : imagers_) {
    if (!imager_to_camera_model_[imager]) {
      return absl::FailedPreconditionError(
          absl::StrCat("Cannot add parameters for imager ", imager,
                       ". Camera model is not yet defined."));
    }
    problem.AddParameterBlock(imager_to_intrinsics_[imager].data(),
                              imager_to_intrinsics_[imager].size());
    num_parameters_added += imager_to_intrinsics_[imager].size();
    num_parameters_added += utils::AddPoseToProblem(
        problem, imager_to_pose_sensor_from_imager_[imager]);
    problem.AddParameterBlock(&imager_to_latency_[imager], 1);
    ++num_parameters_added;
  }
  num_parameters_added +=
      utils::AddPoseToProblem(problem, pose_sensorrig_from_sensor_);

  // Set constant parameters if not enabled for estimation.
  for (const auto& imager : imagers_) {
    if (!imager_to_intrinsics_enabled_.at(imager))
      problem.SetParameterBlockConstant(imager_to_intrinsics_[imager].data());
    if (!imager_to_extrinsics_enabled_.at(imager))
      utils::SetPoseConstantInProblem(
          problem, imager_to_pose_sensor_from_imager_[imager]);
    if (!imager_to_latency_enabled_.at(imager))
      problem.SetParameterBlockConstant(&imager_to_latency_[imager]);
  }
  if (!sensor_extrinsics_enabled_) {
    utils::SetPoseConstantInProblem(problem, pose_sensorrig_from_sensor_);
  }
  return num_parameters_added;
}

absl::StatusOr<int> MultiCamera::AddResidualsToProblem(
    ceres::Problem& problem, Trajectory& sensorrig_trajectory,
    WorldModel& world_model) {
  int num_residuals_added = 0;
  for (const auto& [imager, id_to_measurement_] :
       imager_to_id_to_measurement_) {
    const auto& outlier_ids_ = imager_to_outlier_ids_.at(imager);
    auto& id_to_residual_id_ = imager_to_id_to_residual_id_.at(imager);
    for (const auto& [observation_id, measurement] : id_to_measurement_) {
      // Skip measurements marked as outliers.
      if (outlier_ids_.contains(observation_id)) {
        continue;
      }
      const int rigidbody_id = observation_id.model_id;
      if (!world_model.rigidbodies().contains(rigidbody_id)) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Attempted to create cost function from an observation for a "
            "rigidbody with id ",
            rigidbody_id,
            " that does not exist in "
            "the world model."));
      }
      // Get the right rigidbody reference from the world model.
      std::unique_ptr<RigidBody>& rigidbody_ref =
          world_model.rigidbodies().at(rigidbody_id);
      Eigen::Vector3d& t_model_point =
          rigidbody_ref->model_definition.at(observation_id.feature_id);
      // Construct a cost function and supply parameters for this residual.
      std::vector<double*> parameters;

      ceres::CostFunction* cost_function =
          ImagerCostFunctor::CreateCostFunction(
              measurement.pixel, sigma_,
              imager_to_camera_model_.at(imager)->GetType(),
              imager_to_intrinsics_.at(imager),
              imager_to_pose_sensor_from_imager_.at(imager),
              pose_sensorrig_from_sensor_, imager_to_latency_.at(imager),
              t_model_point, rigidbody_ref->T_world_rigidbody,
              sensorrig_trajectory, observation_id.stamp, parameters);
      ceres::LossFunction* loss_function =
          CreateLossFunction(loss_function_, loss_scale_);
      const auto residual_block_id =
          problem.AddResidualBlock(cost_function, loss_function, parameters);
      id_to_residual_id_[observation_id] = residual_block_id;
      num_residuals_added += 1;
    }
  }
  return num_residuals_added;
}

absl::StatusOr<absl::flat_hash_map<std::string, std::vector<CameraMeasurement>>>
MultiCamera::Project(const std::vector<double>& interp_times,
                     const Trajectory& sensorrig_trajectory,
                     const WorldModel& world_model) const {
  std::vector<Pose3d> poses_world_from_sensorrig;
  ASSIGN_OR_RETURN(poses_world_from_sensorrig,
                   sensorrig_trajectory.Interpolate(interp_times));
  absl::flat_hash_map<std::string, std::vector<CameraMeasurement>>
      imager_to_measurements;
  int image_id = 0;
  for (int i = 0; i < interp_times.size(); ++i) {
    const Pose3d& pose_world_from_sensorrig = poses_world_from_sensorrig.at(i);
    const double& stamp = interp_times.at(i);
    const Pose3d pose_world_from_sensor =
        pose_world_from_sensorrig * pose_sensorrig_from_sensor_;
    for (const auto& imager : imagers_) {
      auto& measurements = imager_to_measurements[imager];
      auto& camera_model = imager_to_camera_model_.at(imager);
      const Pose3d& pose_sensor_from_imager =
          imager_to_pose_sensor_from_imager_.at(imager);
      const Pose3d pose_camera_from_world =
          (pose_world_from_sensor * pose_sensor_from_imager).inverse();
      const double latency = imager_to_latency_.at(imager);
      const Eigen::VectorXd& intrinsics = imager_to_intrinsics_.at(imager);
      // Project all landmarks.
      for (const auto& [landmark_id, landmark] : world_model.landmarks()) {
        const Eigen::Vector3d point_camera =
            pose_camera_from_world * landmark->point;
        if (point_camera.z() <= 0) {
          continue;
        }
        const absl::StatusOr<Eigen::Vector2d> projection =
            camera_model->ProjectPoint(intrinsics, point_camera);
        measurements.push_back(
            {*projection,
             {stamp + latency, image_id, kLandmarkFrameId, landmark_id}});
      }
      // Project all rigid bodies.
      for (const auto& [rigidbody_id, rigidbody] : world_model.rigidbodies()) {
        const Pose3d pose_camera_rigidbody =
            pose_camera_from_world * rigidbody->T_world_rigidbody;
        for (const auto& [point_id, point] : rigidbody->model_definition) {
          const Eigen::Vector3d point_camera = pose_camera_rigidbody * point;
          if (point_camera.z() <= 0) {
            continue;
          }
          const absl::StatusOr<Eigen::Vector2d> projection =
              camera_model->ProjectPoint(intrinsics, point_camera);
          measurements.push_back(
              {*projection,
               {stamp + latency, image_id, rigidbody_id, point_id}});
        }
      }
    }
    ++image_id;
  }
  return imager_to_measurements;
}

absl::Status MultiCamera::AddMeasurement(const std::string& imager,
                                         const CameraMeasurement& measurement) {
  if (!imager_to_id_to_measurement_.contains(imager)) {
    imager_to_id_to_measurement_.insert({imager, {}});
  }
  auto& id_to_measurement = imager_to_id_to_measurement_.at(imager);
  if (id_to_measurement.contains(measurement.id)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Tried to add redundant measurement - Image id: ",
        measurement.id.image_id, ", model id: ", measurement.id.model_id,
        ", feature id: ", measurement.id.feature_id));
  }
  id_to_measurement[measurement.id] = measurement;
  return absl::OkStatus();
}

absl::Status MultiCamera::AddMeasurements(
    const std::string& imager,
    const std::vector<CameraMeasurement>& measurements) {
  std::string message;
  for (const auto& measurement : measurements) {
    absl::Status status = AddMeasurement(imager, measurement);
    if (!status.ok()) {
      message += std::string(status.message()) + "\n";
    }
  }
  if (message.empty()) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(message);
}

absl::StatusOr<absl::flat_hash_map<
    std::string, std::vector<std::pair<CameraMeasurement, Eigen::Vector2d>>>>
MultiCamera::GetMeasurementResidualPairs() const {
  for (const auto& imager : imagers_) {
    if (imager_to_id_to_residual_.at(imager).size() >
        imager_to_id_to_measurement_.at(imager).size()) {
      return absl::InternalError("There are more residuals than measurements.");
    }
    if (imager_to_id_to_measurement_.at(imager).empty()) {
      return absl::FailedPreconditionError(
          "Measurements are empty. Nothing to return.");
    }
  }

  absl::flat_hash_map<
      std::string, std::vector<std::pair<CameraMeasurement, Eigen::Vector2d>>>
      imager_to_pairs;
  for (const auto& imager : imagers_) {
    auto& pairs = imager_to_pairs[imager];
    const auto& id_to_measurement = imager_to_id_to_measurement_.at(imager);
    const auto& id_to_residual = imager_to_id_to_residual_.at(imager);
    for (const auto [id, residual] : id_to_residual) {
      auto it = id_to_measurement.find(id);
      if (it != id_to_measurement.end()) {
        pairs.push_back({it->second, residual});
      } else {
        return absl::InternalError(
            "Found a residual that doesn't correspond to any measurement.");
      }
    }
  }
  return imager_to_pairs;
}

// absl::Status Camera::MarkOutlierById(const CameraObservationId& id) {
//   if (!id_to_measurement_.contains(id)) {
//     return absl::InvalidArgumentError(absl::StrCat(
//         "Attempted to add id that is not within the measurement set.",
//         "\nStamp - ", id.stamp, ", image_id - ", id.image_id, "model_id -
//         ", id.model_id, "feature_id - ", id.feature_id));
//   }
//   outlier_ids_.insert(id);
//   return absl::OkStatus();
// }

// absl::Status Camera::MarkOutliersById(
//     const std::vector<CameraObservationId>& ids) {
//   for (const auto& id : ids) {
//     RETURN_IF_ERROR(MarkOutlierById(id));
//   }
//   return absl::OkStatus();
// }

// void Camera::ClearOutliersList() { outlier_ids_.clear(); }

void MultiCamera::ClearMeasurements() {
  for (auto& [_, id_to_measurement] : imager_to_id_to_measurement_) {
    id_to_measurement.clear();
  }
  for (auto& [_, id_to_residual_id] : imager_to_id_to_residual_id_) {
    id_to_residual_id.clear();
  }
  for (auto& [_, id_to_residual] : imager_to_id_to_residual_) {
    id_to_residual.clear();
  }
  for (auto& [_, outlier_ids] : imager_to_outlier_ids_) {
    outlier_ids.clear();
  }
}

int MultiCamera::NumberOfMeasurements(const std::string& imager) const {
  if (imager_to_id_to_measurement_.contains(imager)) {
    return imager_to_id_to_measurement_.at(imager).size();
  }
  return 0;
}

}  // namespace calico::sensors
