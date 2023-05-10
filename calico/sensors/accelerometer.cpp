#include "calico/sensors/accelerometer.h"

#include "calico/optimization_utils.h"
#include "calico/sensors/accelerometer_cost_functor.h"
#include "calico/statusor_macros.h"


namespace calico::sensors {

absl::StatusOr<int> Accelerometer::AddParametersToProblem(
    ceres::Problem& problem) {
  int num_parameters_added = 0;
  if (!accelerometer_model_) {
    return absl::FailedPreconditionError(
        "Cannot add accelerometer parameters. Accelerometer model is not yet "
        "defined.");
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

absl::StatusOr<int> Accelerometer::AddResidualsToProblem(
    ceres::Problem& problem,
    Trajectory& sensorrig_trajectory,
    WorldModel& world_model) {
  int num_residuals_added = 0;
  for (const auto& [observation_id, measurement] : id_to_measurement_) {
    // Construct a cost function and supply parameters for this residual.
    std::vector<double*> parameters;
    ceres::CostFunction* cost_function =
        AccelerometerCostFunctor::CreateCostFunction(
            measurement.measurement, sigma_, accelerometer_model_->GetType(),
            intrinsics_, T_sensorrig_sensor_, latency_, world_model.gravity(),
            sensorrig_trajectory, observation_id.stamp, parameters);
    ceres::LossFunction* loss_function = CreateLossFunction(
        loss_function_, loss_scale_);
    const auto residual_block_id = problem.AddResidualBlock(
        cost_function, loss_function, parameters);
    id_to_residual_id_[observation_id] = residual_block_id;
    ++num_residuals_added;
  }
  return num_residuals_added;
}

absl::Status Accelerometer::UpdateResiduals(ceres::Problem& problem) {
  for (const auto [measurement_id, residual_id] : id_to_residual_id_) {
    Eigen::Vector3d residual;
    if (!problem.EvaluateResidualBlock(residual_id,
        /*apply_loss_function=*/false, nullptr, residual.data(), nullptr)) {
      return absl::InternalError("Failed to update residual for accelerometer " +
          name_);
    }
    id_to_residual_[measurement_id] = residual;
  }
  return absl::OkStatus();
}

void Accelerometer::ClearResidualInfo() {
  id_to_residual_id_.clear();
  id_to_residual_.clear();
}

absl::StatusOr<std::vector<AccelerometerMeasurement>> Accelerometer::Project(
    const std::vector<double>& interp_times,
    const Trajectory& sensorrig_trajectory,
    const WorldModel& world_model) const {
  std::vector<Eigen::Vector<double, 6>> pose_vectors;
  ASSIGN_OR_RETURN(pose_vectors, sensorrig_trajectory.spline().Interpolate(
      interp_times, /*derivative=*/0));
  std::vector<Eigen::Vector<double, 6>> pose_dot_vectors;
  ASSIGN_OR_RETURN(pose_dot_vectors, sensorrig_trajectory.spline().Interpolate(
      interp_times, /*derivative=*/1));
  std::vector<Eigen::Vector<double, 6>> pose_ddot_vectors;
  ASSIGN_OR_RETURN(pose_ddot_vectors, sensorrig_trajectory.spline().Interpolate(
      interp_times, /*derivative=*/2));
  std::vector<AccelerometerMeasurement> measurements(interp_times.size());
  for (int i = 0; i < interp_times.size(); ++i) {
    const Eigen::Vector3d phi_sensorrig_world = -pose_vectors.at(i).head(3);
    const Eigen::Vector3d phi_dot_sensorrig_world =
        -pose_dot_vectors.at(i).head(3);
    const Eigen::Vector3d phi_ddot_sensorrig_world =
        -pose_ddot_vectors.at(i).head(3);
    double q_sensorrig_world_array[4];
    ceres::AngleAxisToQuaternion(phi_sensorrig_world.data(),
                                 q_sensorrig_world_array);
    const Eigen::Quaterniond q_sensorrig_world(
        q_sensorrig_world_array[0], q_sensorrig_world_array[1],
        q_sensorrig_world_array[2], q_sensorrig_world_array[3]);
    const Eigen::Vector3d ddt_world_sensorrig = pose_ddot_vectors.at(i).tail(3);
    const Eigen::Matrix3d J = ExpSO3Jacobian(phi_sensorrig_world);
    const Eigen::Matrix3d Jdot = ExpSO3JacobianDot(phi_sensorrig_world,
                                                   phi_dot_sensorrig_world);
    const Eigen::Vector3d omega_sensorrig_world = J * phi_dot_sensorrig_world;
    const Eigen::Vector3d alpha_sensorrig_world =
        Jdot * phi_dot_sensorrig_world + J * phi_ddot_sensorrig_world;
    const Eigen::Matrix3d Alpha = -Skew(alpha_sensorrig_world);
    const Eigen::Matrix3d Omega = -Skew(omega_sensorrig_world);
    const Eigen::Vector3d ddt_accelerometer_world_accelerometer =
        T_sensorrig_sensor_.rotation().inverse() * (
            q_sensorrig_world * (ddt_world_sensorrig - world_model.gravity()) +
            (Omega * Omega + Alpha) * T_sensorrig_sensor_.translation());
    const double& stamp = interp_times.at(i);
    Eigen::Vector3d projection;
    ASSIGN_OR_RETURN(projection, accelerometer_model_->Project(
        intrinsics_, ddt_accelerometer_world_accelerometer));
    measurements[i] = AccelerometerMeasurement{
        projection, {stamp + latency_, i}};
  }
  return measurements;
}

absl::Status Accelerometer::SetLatency(double latency) {
  latency_ = latency;
  return absl::OkStatus();
}

double Accelerometer::GetLatency() const {
  return latency_;
}

void Accelerometer::SetName(const std::string& name) {
  name_ = name;
}
const std::string& Accelerometer::GetName() const { return name_; }

void Accelerometer::SetExtrinsics(const Pose3d& T_sensorrig_sensor) {
  T_sensorrig_sensor_ = T_sensorrig_sensor;
}
    
const Pose3d& Accelerometer::GetExtrinsics() const {
  return T_sensorrig_sensor_;
}

absl::Status Accelerometer::SetIntrinsics(const Eigen::VectorXd& intrinsics) {
  if (!accelerometer_model_) {
    return absl::InvalidArgumentError("Accelerometer model has not been set!");
  }
  if (intrinsics.size() != accelerometer_model_->NumberOfParameters()) {
    return absl::InvalidArgumentError(
        absl::StrCat(
            "Tried to set intrinsics of size ", intrinsics.size(),
            " for accelerometer ", GetName(), ". Expected intrinsics size of ",
            accelerometer_model_->NumberOfParameters()));
  }
  intrinsics_ = intrinsics;
  return absl::OkStatus();
}

const Eigen::VectorXd& Accelerometer::GetIntrinsics() const {
  return intrinsics_;
}

void Accelerometer::EnableExtrinsicsEstimation(bool enable) {
  extrinsics_enabled_ = enable;
}

void Accelerometer::EnableIntrinsicsEstimation(bool enable) {
  intrinsics_enabled_ = enable;
}

void Accelerometer::EnableLatencyEstimation(bool enable) {
  latency_enabled_ = enable;
}

absl::Status Accelerometer::SetMeasurementNoise(double sigma) {
  if (sigma <= 0.0) {
    return absl::InvalidArgumentError("Sigma must be greater than 0.");
  }
  sigma_ = sigma;
  return absl::OkStatus();
}

void Accelerometer::SetLossFunction(utils::LossFunctionType loss, double scale) {
  loss_function_ = loss;
  loss_scale_ = scale;
}

absl::Status Accelerometer::SetModel(AccelerometerIntrinsicsModel accelerometer_model) {
  accelerometer_model_ = AccelerometerModel::Create(accelerometer_model);
  intrinsics_ = Eigen::VectorXd::Zero(accelerometer_model_->NumberOfParameters());
  if (accelerometer_model_) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Could not create accelerometer model for type ", accelerometer_model,
      ". It is likely not yet implemented."));
}

AccelerometerIntrinsicsModel Accelerometer::GetModel() const {
  return accelerometer_model_ ?
    accelerometer_model_->GetType() : AccelerometerIntrinsicsModel::kNone;
}

absl::Status Accelerometer::AddMeasurement(const AccelerometerMeasurement& measurement) {
  if (id_to_measurement_.contains(measurement.id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Tried to add redundant measurement - Sequence: ",
                     measurement.id.sequence, ", stamp: ",
                     measurement.id.stamp));
  }
  id_to_measurement_[measurement.id] = measurement;
  return absl::OkStatus();
}

absl::Status Accelerometer::AddMeasurements(
    const std::vector<AccelerometerMeasurement>& measurements) {
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

void Accelerometer::ClearMeasurements() {
  id_to_measurement_.clear();
}

int Accelerometer::NumberOfMeasurements() const {
  return id_to_measurement_.size();
}

} // namespace calico::sensors
