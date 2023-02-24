#include "calico/sensors/camera_cost_functor.h"

#include "ceres/dynamic_autodiff_cost_function.h"


namespace calico::sensors {

CameraCostFunctor::CameraCostFunctor(const CameraIntrinsicsModel camera_model,
                                     const Eigen::Vector2d& pixel)
  : pixel_(pixel) {
  camera_model_ = CameraModel::Create(camera_model);
}

ceres::CostFunction* CameraCostFunctor::CreateCostFunction(
    const Eigen::Vector2d& pixel, CameraIntrinsicsModel camera_model,
    Eigen::VectorXd& intrinsics, Pose3& extrinsics,
    Eigen::Vector3d& t_model_point, Pose3& T_world_model,
    Pose3& T_world_sensorrig, std::vector<double*>& parameters) {
  auto* cost_function =
    new ceres::DynamicAutoDiffCostFunction<CameraCostFunctor>(
        new CameraCostFunctor(camera_model, pixel));
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
  // TODO(yangjames): Replace this with B-Spline coefficients.
  // q_world_sensorrig
  parameters.push_back(T_world_sensorrig.rotation().coeffs().data());
  cost_function->AddParameterBlock(
      T_world_sensorrig.rotation().coeffs().size());
  // t_world_sensorrig
  parameters.push_back(T_world_sensorrig.translation().data());
  cost_function->AddParameterBlock(T_world_sensorrig.translation().size());
  // Residual
  cost_function->SetNumResiduals(kCameraResidualSize);
  return cost_function;
}

} // namespace calico::sensors
