#ifndef CALICO_BATCH_OPTIMIZER_H_
#define CALICO_BATCH_OPTIMIZER_H_

#include <memory>

#include "calico/sensors/sensor_base.h"
#include "calico/trajectory.h"
#include "calico/world_model.h"
#include "ceres/solver.h"
#include "absl/status/statusor.h"


namespace calico {

// Default solver options.
ceres::Solver::Options DefaultSolverOptions();

class BatchOptimizer {
 public:

  ~BatchOptimizer();
  void AddSensor(sensors::Sensor* sensor, bool take_ownership = true);
  void AddWorldModel(WorldModel* world_model, bool take_ownership = true);
  void AddTrajectory(Trajectory* trajectory_world_body,
                     bool take_ownership = true);
  absl::StatusOr<ceres::Solver::Summary> Optimize(
      const ceres::Solver::Options& options = DefaultSolverOptions());

 private:
  bool own_trajectory_world_body_;
  bool own_world_model_;
  std::vector<bool> own_sensors_;
  std::vector<std::unique_ptr<sensors::Sensor>> sensors_;
  std::unique_ptr<WorldModel> world_model_;
  std::unique_ptr<Trajectory> trajectory_world_body_;
};

} // namespace calico

#endif // CALICO_BATCH_OPTIMIZER_H_
