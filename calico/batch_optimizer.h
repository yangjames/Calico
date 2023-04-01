#ifndef CALICO_BATCH_OPTIMIZER_H_
#define CALICO_BATCH_OPTIMIZER_H_

#include <memory>

#include "calico/sensors/sensor_base.h"
#include "calico/trajectory.h"
#include "calico/world_model.h"
#include "ceres/solver.h"
#include "absl/status/statusor.h"


/// Primary calico namespace.
namespace calico {

/// Default solver options.
ceres::Solver::Options DefaultSolverOptions();

/// Batch optimizer class. Takes `Sensor`, `WorldModel`, and `Trajectory`
/// objects and adds their parameters to a least-squares optimization problem.
/// Sensor classes also contribute residuals which are functions of `WorldModel`
/// and `Trajectory`.
class BatchOptimizer {
 public:

  ~BatchOptimizer();

  /// Add a sensor to the optimizer. `sensor` is a raw pointer, and by default,
  /// `BatchOptimizer` will take ownership of that pointer, so there is no need
  /// to de-allocate once BatchOptimizer goes out of scope.\n\n
  /// **NOTE: If this pointer is owned by a smart pointer or must
  /// persist out of the BatchOptimizer scope, pass in `take_ownership=false`.**
  void AddSensor(sensors::Sensor* sensor, bool take_ownership = true);

  /// Add a WorldModel to the optimizer. `world_model` is a raw pointer, and by
  /// default, `BatchOptimizer` will take ownership of that pointer, so there is
  /// no need to de-allocate once BatchOptimizer goes out of scope.\n\n
  /// **NOTE: If this pointer is owned by a smart pointer or must persist out of
  /// the BatchOptimizer scope, pass in `take_ownership=false`.**
  void AddWorldModel(WorldModel* world_model, bool take_ownership = true);

  /// Add a Trajectory to the optimizer. `trajectory_world_sensorrig` is a raw
  /// pointer, and by default, `BatchOptimizer` will take ownership of that
  /// pointer, so there is no need to de-allocate once BatchOptimizer goes out
  /// of scope.\n\n
  /// **NOTE: If this pointer is owned by a smart pointer or must persist out of
  /// the BatchOptimizer scope, pass in `take_ownership=false`.**\n\n
  /// `trajectory_world_sensorrig` is the time-parameterized pose of the
  /// sensor rig w.r.t. the world origin resolved in the world frame
  /// \f[
  /// \mathbf{T}^w_r(t) = \left[\begin{matrix}\mathbf{R}^w_r & \mathbf{t}^w_{wr}
  /// \\\mathbf{0}&1\end{matrix}\right]
  /// \f].
  void AddTrajectory(Trajectory* trajectory_world_sensorrig,
                     bool take_ownership = true);

  /// Run the optimization with solver options. By default, `options` is
  /// `DefaultSolverOptions()`.\n\n
  /// `options` is an unmodified [`ceres::Solver::Options` object]
  /// (http://ceres-solver.org/nnls_solving.html#solver-options). Click the link
  /// to see all configurable options.
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
