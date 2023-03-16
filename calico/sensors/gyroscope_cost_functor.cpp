#include "calico/sensors/gyroscope_cost_functor.h"

#include "ceres/dynamic_autodiff_cost_function.h"


namespace calico::sensors {

GyroscopeCostFunctor::GyroscopeCostFunctor(
    GyroscopeIntrinsicsModel gyroscope_model, const Eigen::Vector3d& measurement,
    double stamp, const Trajectory& trajectory_world_sensorrig)
  : measurement_(measurement) {
  gyroscope_model_ = GyroscopeModel::Create(gyroscope_model);
  trajectory_evaluation_params_
      = trajectory_world_sensorrig.GetEvaluationParams(stamp);
}

ceres::CostFunction* GyroscopeCostFunctor::CreateCostFunction(
    const Eigen::Vector3d& measurement,
    GyroscopeIntrinsicsModel gyroscope_model, Eigen::VectorXd& intrinsics,
    Pose3d& extrinsics, double& latency, Trajectory& trajectory_world_sensorrig,
    double stamp, std::vector<double*>& parameters) {
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<GyroscopeCostFunctor>(
          new GyroscopeCostFunctor(gyroscope_model, measurement, stamp,
                                   trajectory_world_sensorrig));
  // intrinsics
  parameters.push_back(intrinsics.data());
  cost_function->AddParameterBlock(intrinsics.size());
  // extrinsics rotation q_sensrorig_gyroscope
  parameters.push_back(extrinsics.rotation().coeffs().data());
  cost_function->AddParameterBlock(extrinsics.rotation().coeffs().size());
  // extrinsics translation t_sensorrig_gyroscope
  parameters.push_back(extrinsics.translation().data());
  cost_function->AddParameterBlock(extrinsics.translation().size());
  // latency
  parameters.push_back(&latency);
  cost_function->AddParameterBlock(1);
  // trajectory spline control points.
  parameters.push_back(
      trajectory_world_sensorrig.spline().control_points().data());
  cost_function->AddParameterBlock(
      trajectory_world_sensorrig.spline().control_points().size());
  // Residual
  cost_function->SetNumResiduals(kGyroscopeResidualSize);
  return cost_function;
}

} // namespace calico::sensors
