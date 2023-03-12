#ifndef CALICO_SENSORS_CAMERA_COST_FUNCTOR_H_
#define CALICO_SENSORS_CAMERA_COST_FUNCTOR_H_

#include "calico/sensors/camera_models.h"
#include "calico/trajectory.h"
#include "ceres/cost_function.h"


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
  // Rotation and position control points of the entire trajectory spline as an
  // Nx6 matrix.
  kSensorRigPoseSplineControlPointsIndex = 6,
};

// Generic auto-differentiation camera cost functor. Residuals will be based on
// how the camera model is initialized.
class CameraCostFunctor {
 public:
  static constexpr int kCameraResidualSize = 2;
  explicit CameraCostFunctor(
      const CameraIntrinsicsModel camera_model, const Eigen::Vector2d& pixel,
      double stamp, const Trajectory& sp_T_world_sensorrig);

  // Convenience function for creating a camera cost function.
  static ceres::CostFunction* CreateCostFunction(
      const Eigen::Vector2d& pixel, CameraIntrinsicsModel camera_model,
      Eigen::VectorXd& intrinsics, Pose3d& extrinsics,
      Eigen::Vector3d& t_model_point, Pose3d& T_world_model,
      Trajectory& trajectory_world_sensorrig, double stamp,
      std::vector<double*>& parameters);

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
  //   t_model_point:
  //     Position of the point in the model resolved int he model frame.
  //   t_world_model:
  //     Position of model relative to world origin resolved in the world frame.
  //   cp0_rot thorugh cp5_rot:
  //     Rotation spline control points for this particular spline segment.
  //   cp0_pos through cp5_pos:
  //     Position spline control points for this particular spline segment.
  template <typename T>
  bool operator()(T const* const* parameters, T* residual) {
    // Parse intrinsics.
    const T* intrinsics_ptr =
      static_cast<const T*>(&(parameters[static_cast<int>(
          CameraParameterIndices::kIntrinsicsIndex)][0]));
    const int parameter_size = camera_model_->NumberOfParameters();
    const Eigen::VectorX<T> intrinsics = Eigen::Map<const Eigen::VectorX<T>>(
        intrinsics_ptr, parameter_size);
    // Parse extrinsics.
    const Eigen::Map<const Eigen::Quaternion<T>> q_sensorrig_camera(
        &(parameters[static_cast<int>(
            CameraParameterIndices::kExtrinsicsRotationIndex)][0]));
    const Eigen::Map<const Eigen::Vector3<T>> t_sensorrig_camera(
        &(parameters[static_cast<int>(
            CameraParameterIndices::kExtrinsicsTranslationIndex)][0]));
    // Parse model point and model pose resolved in the world frame.
    const Eigen::Map<const Eigen::Vector3<T>> t_model_point(
        &(parameters[static_cast<int>(
            CameraParameterIndices::kModelPointIndex)][0]));
    const Eigen::Map<const Eigen::Quaternion<T>> q_world_model(
        &(parameters[static_cast<int>(
            CameraParameterIndices::kModelRotationIndex)][0]));
    const Eigen::Map<const Eigen::Vector3<T>> t_world_model(
        &(parameters[static_cast<int>(
            CameraParameterIndices::kModelTranslationIndex)][0]));
    // Parse sensor rig rotation spline resolved in the world frame.
    const int num_control_points = trajectory_evaluation_params_.num_control_points;
    const Eigen::Map<const Eigen::MatrixX<T>> all_control_points(
        &(parameters[static_cast<int>(
            CameraParameterIndices::kSensorRigPoseSplineControlPointsIndex)][0]),
        num_control_points, 6);
    const T knot0 = static_cast<T>(trajectory_evaluation_params_.knot0);
    const T knot1 = static_cast<T>(trajectory_evaluation_params_.knot1);
    const T stamp = static_cast<T>(trajectory_evaluation_params_.stamp);
    const Eigen::MatrixX<T> basis_matrix =
        trajectory_evaluation_params_.basis_matrix.template cast<T>();
    const Eigen::MatrixX<T> control_points =
        all_control_points.block(trajectory_evaluation_params_.spline_index, 0,
                                 Trajectory::kSplineOrder, 6);
    const Eigen::Vector<T, 6> pose_vector = BSpline<6, T>::Evaluate(
        control_points, knot0, knot1, basis_matrix, stamp, /*derivative=*/0);
    const Pose3<T> T_world_sensorrig = Trajectory::VectorToPose3(pose_vector);
    // Resolve the model point in the camera frame.
    const Pose3<T> T_sensorrig_camera(q_sensorrig_camera, t_sensorrig_camera);
    const Pose3<T> T_world_camera = T_world_sensorrig * T_sensorrig_camera;
    const Pose3<T> T_world_model(q_world_model, t_world_model);
    const Pose3<T> T_camera_model = T_world_camera.inverse() * T_world_model;
    const Eigen::Vector3<T> t_camera_point = T_camera_model * t_model_point;
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
  TrajectoryEvaluationParams trajectory_evaluation_params_;
};
} // namespace calico::sensors

#endif // CALICO_SENSORS_CAMERA_COST_FUNCTOR_H_
