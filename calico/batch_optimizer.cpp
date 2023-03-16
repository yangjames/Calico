#include "calico/batch_optimizer.h"

#include "calico/statusor_macros.h"


namespace calico {

void BatchOptimizer::AddSensor(
    sensors::Sensor* sensor) {
  sensors_.push_back(std::unique_ptr<sensors::Sensor>(sensor));
}

void BatchOptimizer::AddWorldModel(WorldModel* world_model) {
  world_model_ = std::unique_ptr<WorldModel>(world_model);
}

void BatchOptimizer::AddTrajectory(Trajectory* trajectory_world_body) {
  trajectory_world_body_ = std::unique_ptr<Trajectory>(trajectory_world_body);
}

absl::StatusOr<ceres::Solver::Summary> BatchOptimizer::Optimize() {
  int num_parameters = 0;
  int num_residuals = 0;
  ceres::Problem problem;
  // Add world model and trajectory to problem.
  num_parameters += world_model_->AddParametersToProblem(problem);
  num_parameters += trajectory_world_body_->AddParametersToProblem(problem);
  for (std::unique_ptr<sensors::Sensor>& sensor : sensors_) {
    ASSIGN_OR_RETURN(const auto num_parameters_added,
                     sensor->AddParametersToProblem(problem));
    num_parameters += num_parameters_added;
    ASSIGN_OR_RETURN(const auto num_residuals_added,
        sensor->AddResidualsToProblem(problem, *trajectory_world_body_,
                                      *world_model_));
    num_residuals += num_residuals_added;
  }
  // Run solver.
  // TODO: Make optimizer options configurable
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.preconditioner_type = ceres::JACOBI;
  options.minimizer_progress_to_stdout = true;
  options.parameter_tolerance = 1e-12;
  options.num_threads = 4;
  options.max_num_iterations = 50;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  return summary;
}

} // namespace calico
