#include "calico/batch_optimizer.h"

#include <thread>

#include "calico/statusor_macros.h"


namespace calico {

ceres::Solver::Options DefaultSolverOptions() {
  ceres::Solver::Options default_options;
  default_options.linear_solver_type = ceres::DENSE_SCHUR;
  default_options.minimizer_progress_to_stdout = true;
  default_options.function_tolerance = 1e-8;
  default_options.parameter_tolerance = 1e-10;
  return default_options;
}

BatchOptimizer::~BatchOptimizer() {
  // If we don't own an object, release the pointer so it doesn't get
  // de-allocated.
  for (int i = 0; i < sensors_.size(); ++i) {
    if (!own_sensors_[i]) {
      sensors_.at(i).release();
    }
  }
  if (!own_trajectory_world_body_) {
    trajectory_world_body_.release();
  }
  if (!own_world_model_) {
    world_model_.release();
  }
}

void BatchOptimizer::AddSensor(
    sensors::Sensor* sensor, bool take_ownership) {
  sensors_.push_back(std::unique_ptr<sensors::Sensor>(sensor));
  own_sensors_.push_back(take_ownership);
}

void BatchOptimizer::AddWorldModel(
    WorldModel* world_model, bool take_ownership) {
  world_model_ = std::unique_ptr<WorldModel>(world_model);
  own_world_model_ = take_ownership;
}

void BatchOptimizer::AddTrajectory(
    Trajectory* trajectory_world_body, bool take_ownership) {
  trajectory_world_body_ = std::unique_ptr<Trajectory>(trajectory_world_body);
  own_trajectory_world_body_ = take_ownership;
}

absl::StatusOr<ceres::Solver::Summary> BatchOptimizer::Optimize(
    const ceres::Solver::Options& options) {
  int num_parameters = 0;
  int num_residuals = 0;
  ceres::Problem problem;
  // Add world model and trajectory to problem.
  num_parameters += world_model_->AddParametersToProblem(problem);
  num_parameters += trajectory_world_body_->AddParametersToProblem(problem);
  for (std::unique_ptr<sensors::Sensor>& sensor : sensors_) {
    sensor->ClearResidualInfo();
    ASSIGN_OR_RETURN(const auto num_parameters_added,
                     sensor->AddParametersToProblem(problem));
    num_parameters += num_parameters_added;
    ASSIGN_OR_RETURN(const auto num_residuals_added,
        sensor->AddResidualsToProblem(problem, *trajectory_world_body_,
                                      *world_model_));
    num_residuals += num_residuals_added;
  }
  // Run solver.
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Update residuals for every sensor.
  for (std::unique_ptr<sensors::Sensor>& sensor : sensors_) {
    RETURN_IF_ERROR(sensor->UpdateResiduals(problem));
  }

  return summary;
}

} // namespace calico
