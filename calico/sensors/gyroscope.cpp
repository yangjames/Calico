#include "calico/sensors/gyroscope.h"

#include "calico/optimization_utils.h"
#include "calico/sensors/gyroscope_cost_functor.h"
#include "calico/statusor_macros.h"


namespace calico::sensors {

absl::StatusOr<int> Gyroscope::AddParametersToProblem(ceres::Problem& problem) {
  int num_parameters_added = 0;
  if (!gyroscope_model_) {
    return absl::FailedPreconditionError(
        "Cannot add gyroscope parameters. Gyroscope model is not yet defined.");
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

absl::StatusOr<int> Gyroscope::AddResidualsToProblem(
    ceres::Problem& problem,
    Trajectory& sensorrig_trajectory,
    WorldModel& world_model) {
  int num_residuals_added = 0;
  for (const auto& [observation_id, measurement] : id_to_measurement_) {
    // Construct a cost function and supply parameters for this residual.
    std::vector<double*> parameters;
    ceres::CostFunction* cost_function =
        GyroscopeCostFunctor::CreateCostFunction(
            measurement.measurement, gyroscope_model_->GetType(), intrinsics_,
            T_sensorrig_sensor_, latency_, sensorrig_trajectory,
            observation_id.stamp, parameters);
    const auto residual_block_id = problem.AddResidualBlock(
        cost_function, /*loss_function=*/nullptr, parameters);
    ++num_residuals_added;
  }
  return num_residuals_added;
}

absl::StatusOr<std::vector<GyroscopeMeasurement>> Gyroscope::Project(
    const std::vector<double>& interp_times,
    const Trajectory& sensorrig_trajectory) const {
  std::vector<Eigen::Vector<double, 6>> pose_vectors;
  ASSIGN_OR_RETURN(pose_vectors, sensorrig_trajectory.spline().Interpolate(
      interp_times, /*derivative=*/0));
  std::vector<Eigen::Vector<double, 6>> pose_dot_vectors;
  ASSIGN_OR_RETURN(pose_dot_vectors, sensorrig_trajectory.spline().Interpolate(
      interp_times, /*derivative=*/1));
  std::vector<GyroscopeMeasurement> measurements(interp_times.size());
  for (int i = 0; i < interp_times.size(); ++i) {
    const Eigen::Vector3d phi_sensorrig_world = -pose_vectors.at(i).head(3);
    const Eigen::Vector3d phi_dot_sensorrig_world =
        -pose_dot_vectors.at(i).head(3);
    const Eigen::Vector3d omega_sensorrig_world =
        ComputeAngularVelocity(phi_sensorrig_world, phi_dot_sensorrig_world);
    const Eigen::Vector3d omega_gyroscope_world =
        T_sensorrig_sensor_.rotation().inverse() * omega_sensorrig_world;
    const double& stamp = interp_times.at(i);
    Eigen::Vector3d projection;
    ASSIGN_OR_RETURN(projection, gyroscope_model_->Project(
        intrinsics_, omega_gyroscope_world));
    measurements[i] = GyroscopeMeasurement{projection, {stamp + latency_, i}};
  }
  return measurements;
}

absl::Status Gyroscope::SetLatency(double latency) {
  latency_ = latency;
  return absl::OkStatus();
}

double Gyroscope::GetLatency() const {
  return latency_;
}

void Gyroscope::SetName(absl::string_view name) {
  name_ = name;
}
const std::string& Gyroscope::GetName() const { return name_; }

void Gyroscope::SetExtrinsics(const Pose3d& T_sensorrig_sensor) {
  T_sensorrig_sensor_ = T_sensorrig_sensor;
}
    
const Pose3d& Gyroscope::GetExtrinsics() const {
  return T_sensorrig_sensor_;
}

absl::Status Gyroscope::SetIntrinsics(const Eigen::VectorXd& intrinsics) {
  if (!gyroscope_model_) {
    return absl::InvalidArgumentError("Gyroscope model has not been set!");
  }
  if (intrinsics.size() != gyroscope_model_->NumberOfParameters()) {
    return absl::InvalidArgumentError(
        absl::StrCat(
            "Tried to set intrinsics of size ", intrinsics.size(),
            " for gyroscope ", GetName(), ". Expected intrinsics size of ",
            gyroscope_model_->NumberOfParameters()));
  }
  intrinsics_ = intrinsics;
  return absl::OkStatus();
}

const Eigen::VectorXd& Gyroscope::GetIntrinsics() const {
  return intrinsics_;
}

void Gyroscope::EnableExtrinsicsEstimation(bool enable) {
  extrinsics_enabled_ = enable;
}

void Gyroscope::EnableIntrinsicsEstimation(bool enable) {
  intrinsics_enabled_ = enable;
}

void Gyroscope::EnableLatencyEstimation(bool enable) {
  latency_enabled_ = enable;
}

absl::Status Gyroscope::SetModel(GyroscopeIntrinsicsModel gyroscope_model) {
  gyroscope_model_ = GyroscopeModel::Create(gyroscope_model);
  intrinsics_ = Eigen::VectorXd::Zero(gyroscope_model_->NumberOfParameters());
  if (gyroscope_model_) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Could not create gyroscope model for type ", gyroscope_model,
      ". It is likely not yet implemented."));
}

GyroscopeIntrinsicsModel Gyroscope::GetModel() const {
  return gyroscope_model_ ?
    gyroscope_model_->GetType() : GyroscopeIntrinsicsModel::kNone;
}

absl::Status Gyroscope::AddMeasurement(const GyroscopeMeasurement& measurement) {
  if (id_to_measurement_.contains(measurement.id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Tried to add redundant measurement - Sequence: ",
                     measurement.id.sequence, ", stamp: ",
                     measurement.id.stamp));
  }
  id_to_measurement_[measurement.id] = measurement;
  return absl::OkStatus();
}

absl::Status Gyroscope::AddMeasurements(
    const std::vector<GyroscopeMeasurement>& measurements) {
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

absl::Status Gyroscope::RemoveMeasurementById(const GyroscopeObservationId& id) {
  if (id_to_measurement_.erase(id)) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Attempted to remove invalid mesaurement - Sequence: ", id.sequence,
      ", stamp: ", id.stamp));
}

absl::Status Gyroscope::RemoveMeasurementsById(
    const std::vector<GyroscopeObservationId>& ids) {
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

void Gyroscope::ClearMeasurements() {
  id_to_measurement_.clear();
}

int Gyroscope::NumberOfMeasurements() const {
  return id_to_measurement_.size();
}

} // namespace calico::sensors
