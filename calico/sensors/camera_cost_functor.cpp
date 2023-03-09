#include "calico/sensors/camera_cost_functor.h"

#include "ceres/dynamic_autodiff_cost_function.h"


namespace calico::sensors {

CameraCostFunctor::CameraCostFunctor(
    const CameraIntrinsicsModel camera_model, const Eigen::Vector2d& pixel,
    double stamp, const TrajectorySegment<double>& sp_T_world_sensorrig)
  : pixel_(pixel), stamp_(stamp) {
  camera_model_ = CameraModel::Create(camera_model);
  basis_matrix_ = sp_T_world_sensorrig.basis_matrix;
  knot0_ = sp_T_world_sensorrig.knot0;
  knot1_ = sp_T_world_sensorrig.knot1;
}

ceres::CostFunction* CameraCostFunctor::CreateCostFunction(
    const Eigen::Vector2d& pixel, CameraIntrinsicsModel camera_model,
    Eigen::VectorXd& intrinsics, Pose3d& extrinsics,
    Eigen::Vector3d& t_model_point, Pose3d& T_world_model,
    TrajectorySegment<double>& sp_T_world_sensorrig, double stamp,
    std::vector<double*>& parameters) {
  auto* cost_function =
      new ceres::DynamicAutoDiffCostFunction<CameraCostFunctor>(
          new CameraCostFunctor(camera_model, pixel, stamp,
                                sp_T_world_sensorrig));
  // intrinsics
  parameters.push_back(intrinsics.data());
  cost_function->AddParameterBlock(intrinsics.size());
  // extrinsics rotation q_sensrorig_camera
  parameters.push_back(extrinsics.rotation().coeffs().data());
  cost_function->AddParameterBlock(extrinsics.rotation().coeffs().size());
  // extrinsics translation t_sensorrig_camera
  parameters.push_back(extrinsics.translation().data());
  cost_function->AddParameterBlock(extrinsics.translation().size());
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
  // sp_T_world_sensorrig
  for (Eigen::Vector3d& rotation_control_point :
           sp_T_world_sensorrig.rotation_control_points) {
    parameters.push_back(rotation_control_point.data());
    cost_function->AddParameterBlock(rotation_control_point.size());
  }
  for (Eigen::Vector3d& position_control_point :
           sp_T_world_sensorrig.position_control_points) {
    parameters.push_back(position_control_point.data());
    cost_function->AddParameterBlock(position_control_point.size());
  }
  // Residual
  cost_function->SetNumResiduals(kCameraResidualSize);
  return cost_function;
}

} // namespace calico::sensors
