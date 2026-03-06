#include "calico/sensors/imager_cost_functor.h"

#include <vector>

#include "ceres/dynamic_autodiff_cost_function.h"

namespace calico::sensors {

ImagerCostFunctor::ImagerCostFunctor(
    CameraIntrinsicsModel camera_model, const Eigen::Vector2d& pixel,
    double sigma, double stamp, const Trajectory& trajectory_world_sensorrig)
    : pixel_(pixel) {
  camera_model_ = CameraModel::Create(camera_model);
  position_trajectory_evaluation_params_ =
      trajectory_world_sensorrig.GetEvaluationParamsPosition(stamp);
  rotation_trajectory_evaluation_params_ =
      trajectory_world_sensorrig.GetEvaluationParamsRotation(stamp);
  information_ = (sigma > 0.0) ? (1.0 / sigma) : 1.0;
}

ceres::CostFunction* ImagerCostFunctor::CreateCostFunction(
    const Eigen::Vector2d& pixel, double sigma,
    CameraIntrinsicsModel camera_model, Eigen::VectorXd& intrinsics,
    Pose3d& imager_extrinsics, Pose3d& sensor_extrinsics, double& latency,
    Eigen::Vector3d& t_model_point, Pose3d& T_world_model,
    Trajectory& trajectory_world_sensorrig, double stamp,
    std::vector<double*>& parameters) {
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<ImagerCostFunctor>(
          new ImagerCostFunctor(camera_model, pixel, sigma, stamp,
                                trajectory_world_sensorrig));
  // intrinsics
  parameters.push_back(intrinsics.data());
  cost_function->AddParameterBlock(intrinsics.size());
  // imager extrinsics rotation q_sensor_imager
  parameters.push_back(imager_extrinsics.rotation().coeffs().data());
  cost_function->AddParameterBlock(
      imager_extrinsics.rotation().coeffs().size());
  // imager extrinsics translation t_sensor_imager
  parameters.push_back(imager_extrinsics.translation().data());
  cost_function->AddParameterBlock(imager_extrinsics.translation().size());
  // sensor extrinsics rotation q_sensorrig_sensor
  parameters.push_back(sensor_extrinsics.rotation().coeffs().data());
  cost_function->AddParameterBlock(
      sensor_extrinsics.rotation().coeffs().size());
  // sensor extrinsics translation t_sensorrig_sensor
  parameters.push_back(sensor_extrinsics.translation().data());
  cost_function->AddParameterBlock(sensor_extrinsics.translation().size());
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
  const int rotation_idx =
      trajectory_world_sensorrig.rotation_spline().GetSplineIndex(stamp);
  const int rotation_spline_order =
      trajectory_world_sensorrig.rotation_spline().GetSplineOrder();
  for (int i = 0; i < rotation_spline_order; ++i) {
    parameters.push_back(trajectory_world_sensorrig.rotation_spline()
                             .control_points()
                             .at(rotation_idx + i)
                             .data());
    cost_function->AddParameterBlock(
        trajectory_world_sensorrig.rotation_spline()
            .control_points()
            .at(rotation_idx + i)
            .size());
  }
  const int position_idx =
      trajectory_world_sensorrig.position_spline().GetSplineIndex(stamp);
  const int position_spline_order =
      trajectory_world_sensorrig.position_spline().GetSplineOrder();
  for (int i = 0; i < position_spline_order; ++i) {
    parameters.push_back(trajectory_world_sensorrig.position_spline()
                             .control_points()
                             .at(position_idx + i)
                             .data());
    cost_function->AddParameterBlock(
        trajectory_world_sensorrig.position_spline()
            .control_points()
            .at(position_idx + i)
            .size());
  }
  // Residual
  cost_function->SetNumResiduals(kCameraResidualSize);
  return cost_function;
}

}  // namespace calico::sensors
