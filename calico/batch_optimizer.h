#ifndef CALICO_BATCH_OPTIMIZER_H_
#define CALICO_BATCH_OPTIMIZER_H_

#include <memory>

#include "calico/sensors/sensor_base.h"
#include "calico/trajectory.h"
#include "calico/world_model.h"
#include "ceres/solver.h"
#include "absl/status/statusor.h"


namespace calico {

class BatchOptimizer {
 public:
  void AddSensor(sensors::Sensor* sensor);
  void AddWorldModel(WorldModel* world_model);
  void AddTrajectory(Trajectory* trajectory_world_body);
  absl::StatusOr<ceres::Solver::Summary> Optimize();

 private:
  std::vector<std::unique_ptr<sensors::Sensor>> sensors_;
  std::unique_ptr<WorldModel> world_model_;
  std::unique_ptr<Trajectory> trajectory_world_body_;
};

} // namespace calico

#endif // CALICO_BATCH_OPTIMIZER_H_
