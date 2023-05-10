#include "calico/sensors/accelerometer_cost_functor.h"

#include "ceres/dynamic_autodiff_cost_function.h"


namespace calico::sensors {

AccelerometerCostFunctor::AccelerometerCostFunctor(
    AccelerometerIntrinsicsModel accelerometer_model, const Eigen::Vector3d& measurement,
    double sigma, double stamp, const Trajectory& trajectory_world_sensorrig)
  : measurement_(measurement) {
  accelerometer_model_ = AccelerometerModel::Create(accelerometer_model);
  trajectory_evaluation_params_
      = trajectory_world_sensorrig.GetEvaluationParams(stamp);
  information_ = (sigma > 0.0) ? (1.0 / sigma) : 1.0;
}

ceres::CostFunction* AccelerometerCostFunctor::CreateCostFunction(
    const Eigen::Vector3d& measurement, double sigma,
    AccelerometerIntrinsicsModel accelerometer_model,
    Eigen::VectorXd& intrinsics, Pose3d& extrinsics, double& latency,
    Eigen::Vector3d& gravity, Trajectory& trajectory_world_sensorrig,
    double stamp, std::vector<double*>& parameters) {
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<AccelerometerCostFunctor>(
          new AccelerometerCostFunctor(accelerometer_model, measurement, sigma,
                                       stamp, trajectory_world_sensorrig));
  // intrinsics
  parameters.push_back(intrinsics.data());
  cost_function->AddParameterBlock(intrinsics.size());
  // extrinsics rotation q_sensrorig_accelerometer
  parameters.push_back(extrinsics.rotation().coeffs().data());
  cost_function->AddParameterBlock(extrinsics.rotation().coeffs().size());
  // extrinsics translation t_sensorrig_accelerometer
  parameters.push_back(extrinsics.translation().data());
  cost_function->AddParameterBlock(extrinsics.translation().size());
  // latency
  parameters.push_back(&latency);
  cost_function->AddParameterBlock(1);
  // gravity
  parameters.push_back(gravity.data());
  cost_function->AddParameterBlock(gravity.size());
  // trajectory spline control points.
  const int idx = trajectory_world_sensorrig.spline().GetSplineIndex(stamp);
  const int spline_order = trajectory_world_sensorrig.spline().GetSplineOrder();
  for (int i = 0; i < spline_order; ++i) {
    parameters.push_back(
        trajectory_world_sensorrig.spline().control_points().at(idx + i).data());
    cost_function->AddParameterBlock(
        trajectory_world_sensorrig.spline().control_points().at(idx + i).size());
  }
  // Residual
  cost_function->SetNumResiduals(kAccelerometerResidualSize);
  return cost_function;
}

} // namespace calico::sensors
