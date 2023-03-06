#ifndef CALICO_BATCH_OPTIMIZER_H_
#define CALICO_BATCH_OPTIMIZER_H_

#include <memory>

#include "calico/sensors/camera.h"
#include "calico/sensors/sensor_base.h"
#include "calico/world_model.h"
#include "ceres/solver.h"
#include "absl/status/statusor.h"


namespace calico {

class BatchOptimizer {
 public:
  void AddSensor(sensors::Sensor* sensor);
  void AddWorldModel(const WorldModel& world_model);
  void AddTrajectory(
      const absl::flat_hash_map<double, Pose3>& trajectory_world_body);
  absl::StatusOr<ceres::Solver::Summary> Optimize();

 private:
  std::vector<std::unique_ptr<sensors::Sensor>> sensors_;
  WorldModel world_model_;
  absl::flat_hash_map<double, Pose3> trajectory_world_body_;
};

} // namespace calico

#endif // CALICO_BATCH_OPTIMIZER_H_
