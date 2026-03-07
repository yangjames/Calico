#include "calico/sensors/gyroscope_cost_functor.h"

#include "ceres/dynamic_autodiff_cost_function.h"


namespace calico::sensors {

GyroscopeCostFunctor::GyroscopeCostFunctor(
    GyroscopeIntrinsicsModel gyroscope_model, const Eigen::Vector3d& measurement,
    double sigma, double stamp, const Trajectory& trajectory_world_sensorrig)
  : measurement_(measurement) {
  gyroscope_model_ = GyroscopeModel::Create(gyroscope_model);
  position_trajectory_evaluation_params_
      = trajectory_world_sensorrig.GetEvaluationParamsPosition(stamp);
  rotation_trajectory_evaluation_params_
      = trajectory_world_sensorrig.GetEvaluationParamsRotation(stamp);
  information_ = (sigma > 0.0) ? (1.0 / sigma) : 1.0;
}

ceres::CostFunction* GyroscopeCostFunctor::CreateCostFunction(
    const Eigen::Vector3d& measurement, double sigma,
    GyroscopeIntrinsicsModel gyroscope_model, Eigen::VectorXd& intrinsics,
    Pose3d& extrinsics, double& latency, Trajectory& trajectory_world_sensorrig,
    double stamp, std::vector<double*>& parameters) {
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<GyroscopeCostFunctor>(
          new GyroscopeCostFunctor(gyroscope_model, measurement, sigma, stamp,
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
  const int rotation_spline_order =
      trajectory_world_sensorrig.rotation_spline().GetSplineOrder();
  const int rotation_idx =
      trajectory_world_sensorrig.rotation_spline().GetSplineIndex(stamp);
  for (int i = 0; i < rotation_spline_order; ++i) {
    parameters.push_back(
        trajectory_world_sensorrig.rotation_spline().control_points().at(rotation_idx + i).data());
    cost_function->AddParameterBlock(
        trajectory_world_sensorrig.rotation_spline().control_points().at(rotation_idx + i).size());
  }
  const int position_idx =
      trajectory_world_sensorrig.position_spline().GetSplineIndex(stamp);
  const int position_spline_order =
      trajectory_world_sensorrig.position_spline().GetSplineOrder();
  for (int i = 0; i < position_spline_order; ++i) {
    parameters.push_back(
        trajectory_world_sensorrig.position_spline().control_points().at(position_idx + i).data());
    cost_function->AddParameterBlock(
        trajectory_world_sensorrig.position_spline().control_points().at(position_idx + i).size());
  }
  // Residual
  cost_function->SetNumResiduals(kGyroscopeResidualSize);
  return cost_function;
}

} // namespace calico::sensors
