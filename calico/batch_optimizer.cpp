#include "calico/batch_optimizer.h"

#include "calico/statusor_macros.h"


namespace calico {

void BatchOptimizer::AddSensor(
    sensors::Sensor* sensor) {
  sensors_.push_back(std::unique_ptr<sensors::Sensor>(sensor));
}

void BatchOptimizer::AddWorldModel(const WorldModel& world_model) {
  world_model_ = world_model;
}

void BatchOptimizer::AddTrajectory(
    const absl::flat_hash_map<int, Pose3>& trajectory_world_body) {
  trajectory_world_body_ = trajectory_world_body;
}

absl::StatusOr<ceres::Solver::Summary> BatchOptimizer::Optimize() {
  ceres::Problem problem;
  // Add world model and trajectory to problem.
  int num_parameters = 0;
  int num_residuals = 0;
  num_parameters += world_model_.AddParametersToProblem(problem);
  for (std::unique_ptr<sensors::Sensor>& sensor : sensors_) {
    ASSIGN_OR_RETURN(const auto num_parameters_added,
                     sensor->AddParametersToProblem(problem));
    num_parameters += num_parameters_added;
    ASSIGN_OR_RETURN(const auto num_residuals_added,
        sensor->AddResidualsToProblem(problem, trajectory_world_body_, world_model_));
    num_residuals += num_residuals_added;
  }
  // Run solver.
  // TODO: Make optimizer options configurable
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.preconditioner_type = ceres::JACOBI;
  options.minimizer_progress_to_stdout = true;
  options.num_threads = 1;
  options.max_num_iterations = 50;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  return summary;
}

} // namespace calico
