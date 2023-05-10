#include "calico/sensors/camera_cost_functor.h"

#include "ceres/dynamic_autodiff_cost_function.h"


namespace calico::sensors {

CameraCostFunctor::CameraCostFunctor(
    CameraIntrinsicsModel camera_model, const Eigen::Vector2d& pixel,
    double sigma, double stamp, const Trajectory& trajectory_world_sensorrig)
  : pixel_(pixel) {
  camera_model_ = CameraModel::Create(camera_model);
  trajectory_evaluation_params_
      = trajectory_world_sensorrig.GetEvaluationParams(stamp);
  information_ = (sigma > 0.0) ? (1.0 / sigma) : 1.0;
}

ceres::CostFunction* CameraCostFunctor::CreateCostFunction(
    const Eigen::Vector2d& pixel, double sigma,
    CameraIntrinsicsModel camera_model, Eigen::VectorXd& intrinsics,
    Pose3d& extrinsics, double& latency, Eigen::Vector3d& t_model_point,
    Pose3d& T_world_model, Trajectory& trajectory_world_sensorrig, double stamp,
    std::vector<double*>& parameters) {
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<CameraCostFunctor>(
          new CameraCostFunctor(camera_model, pixel, sigma, stamp,
                                trajectory_world_sensorrig));
  // intrinsics
  parameters.push_back(intrinsics.data());
  cost_function->AddParameterBlock(intrinsics.size());
  // extrinsics rotation q_sensrorig_camera
  parameters.push_back(extrinsics.rotation().coeffs().data());
  cost_function->AddParameterBlock(extrinsics.rotation().coeffs().size());
  // extrinsics translation t_sensorrig_camera
  parameters.push_back(extrinsics.translation().data());
  cost_function->AddParameterBlock(extrinsics.translation().size());
  // latency
  parameters.push_back(&latency);
  cost_function->AddParameterBlock(1);
  // model point position t_model_point
  parameters.push_back(t_model_point.data());
  cost_function->AddParameterBlock(t_model_point.size());
  // model world pose rotation q_world_model
  Eigen::Quaterniond& q_world_model = T_world_model.rotation();
  parameters.push_back(q_world_model.coeffs().data());
  cost_function->AddParameterBlock(q_world_model.coeffs().size());
  // model world pose translation t_world_model
  Eigen::Vector3d& t_world_model = T_world_model.translation();
  parameters.push_back(t_world_model.data());
  cost_function->AddParameterBlock(t_world_model.size());
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
  cost_function->SetNumResiduals(kCameraResidualSize);
  return cost_function;
}

} // namespace calico::sensors
