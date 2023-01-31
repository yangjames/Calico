#ifndef CALICO_SENSORS_CAMERA_COST_FUNCTOR_H_
#define CALICO_SENSORS_CAMERA_COST_FUNCTOR_H_

#include "calico/sensors/camera_models.h"

namespace calico::sensors {


// Enum listing the positions of parameters for a camera cost function.
enum class CameraParameterIndices : int {
  // Camera intrinsics.
  kIntrinsicsIndex = 0,
  // Extrinsic parameters of the camera relative to its sensor rig.
  kExtrinsicsRotationIndex = 1,
  kExtrinsicsTranslationIndex = 2,
  // Parameters related to some detected "model" object in the world:
  //   1. The point resolved in the model frame.
  //   2. The rotation portion of the model pose resolved in the world
  //      frame.
  //   3. The translation portion of the model pose resolved in the world
  //      frame.
  kModelPointIndex = 3,
  kModelRotationIndex = 4,
  kModelTranslationIndex = 5,
  // Pose of the sensor rig for a given measurement.
  kSensorRigRotationIndex = 6,
  kSensorRigTranslationIndex = 7,
};

// Generic auto-differentiation camera cost functor. Residuals will be based on
// how the camera model is initialized.
class CameraCostFunctor {
 public:
  explicit CameraCostFunctor(const CameraIntrinsicsModel camera_model,
                             const Eigen::Vector2d& pixel)
    : pixel_(pixel) {
    camera_model_ = CameraModel::Create(camera_model);
  }

  // Parameters to the cost function:
  //   intrinsics:
  //     All parameters in the intrinsics model as an Eigen column vector.
  //     Order of the parameters will need to be in agreement with the model
  //     being used.
  //   q_sensorrig_camera:
  //     Rotation from sensorrig frame to camera frame as a quaternion.
  //   t_sensorrig_camera:
  //     Position of camera relative to sensorrig origin resolved in the
  //     sensorrig frame.
  //   q_world_model:
  //     Rotation from world frame to model frame as a quaternion.
  //   t_world_model:
  //     Position of model relative to world origin resolved in the world frame.
  //   q_world_sensorrig:
  //     Rotation from world frame to sensorrig frame.
  //   t_world_sensorrig:
  //     Position of sensorrig relative to world origin resolved in the world
  //     frame.
  template <typename T>
  bool operator()(const T* const parameters, T* residuals) {
    // Parse intrinsics.
    const T* intrinsics_ptr =
      static_cast<const T*>(&(parameters[static_cast<int>(
          CameraParameterIndices::kIntrinsicsIndex)][0]));
    const int parameter_size = camera_model_->GetParameterSize();
    const Eigen::VectorN<T> intrinsics = Eigen::Map<const Eigen::VectorN<T>>(
        intrinsics_ptr, parameter_size);
    // Parse extrinsics.
    const Eigen::Map<const Eigen::Quaternion<T>> q_sensorrig_camera(
        &(parameters[static_cast<int>(kExtrinsicsRotationIndex)][0]));
    const Eigen::Map<const Eigen::Vector3<T>> t_sensorrig_camera(
        &(parameters[static_cast<int>(kExtrinsicsTranslationIndex)][0]));
    // Parse model point and model pose resolved in the world frame.
    const Eigen::Map<const Eigen::Vector3<T>> t_model_point(
        &(parameters[static_cast<int>(kModelPointIndex)][0]));
    const Eigen::Map<const Eigen::Quaternion<T>> q_world_model(
        &(parameters[static_cast<int>(kModelRotationIndex)][0]));
    const Eigen::Map<const Eigen::Vector3<T>> t_world_model(
        &(parameters[static_cast<int>(kModelTranslationIndex)][0]));
    // Parse sensor rig pose resolved in the world frame.
    const Eigen::Map<const Eigen::Quaternion<T>> q_world_sensorrig(
        &(parameters[static_cast<int>(kSensorRigRotationIndex)][0]));
    const Eigen::Map<const Eigen::Vector3<T>> t_world_sensorrig(
        &(parameters[static_cast<int>(kSensorRigTranslationIndex)][0]));
    // Resolve the model point in the camera frame.
    const Eigen::Vector3<T> t_world_camera =
        t_world_sensorrig + q_world_sensorrig * t_sensorrig_camera;
    const Eigen::Quaternion<T> q_world_camera =
        q_world_sensorrig * q_sensorrig_camera;
    const Eigen::Vector3<T> t_world_point =
        t_world_model + q_world_model * t_model_point;
    const Eigen::Vector3<T> t_camera_point =
        q_world_camera.conjugate() * (t_world_point - t_world_camera);
    // Project the point through the camera model.
    const absl::StatusOr<Eigen::Vector2<T>> projection =
        camera_model_->ProjectPoint(intrinsics, t_camera_point);
    // Assign the residual, or return boolean indicating success/failure.
    if (projection.ok()) {
      Eigen::Map<Eigen::Vector2<T>> error(residual);
      const Eigen::Vector2<T> pixel = pixel_.template cast<T>();
      error = pixel - *projection;
      return true;
    }
    return false;
  }

 private:
  Eigen::Vector2d pixel_;
  std::unique_ptr<CameraModel> camera_model_;
}
} // namespace calico::sensors

#endif // CALICO_SENSORS_CAMERA_COST_FUNCTOR_H_
