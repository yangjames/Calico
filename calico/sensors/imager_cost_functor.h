#ifndef CALICO_SENSORS_IMAGER_COST_FUNCTOR_H_
#define CALICO_SENSORS_IMAGER_COST_FUNCTOR_H_

#include <memory>
#include <vector>

#include "Eigen/Dense"
#include "calico/sensors/camera_models.h"
#include "calico/trajectory.h"
#include "ceres/cost_function.h"

namespace calico::sensors {

// Enum listing the positions of parameters for a camera cost function.
enum class ImagerParameterIndices : int {
  // Camera intrinsics.
  kIntrinsicsIndex = 0,
  // Extrinsic parameters of the imager relative to its sensor module.
  kImagerExtrinsicsRotationIndex = 1,
  kImagerExtrinsicsTranslationIndex = 2,
  // Extrinsic parameters of the sensor module relative to the sensor rig.
  kSensorExtrinsicsRotationIndex = 3,
  kSensorExtrinsicsTranslationIndex = 4,
  // Sensor latency.
  kLatencyIndex = 5,
  // Parameters related to some detected "model" object in the world:
  //   1. The point resolved in the model frame.
  //   2. The rotation portion of the model pose resolved in the world
  //      frame.
  //   3. The translation portion of the model pose resolved in the world
  //      frame.
  kModelPointIndex = 6,
  kModelRotationIndex = 7,
  kModelTranslationIndex = 8,
  // Rotation and position control points of the associated spline segment as
  // two Nx3 matrices (rotation then position) where N is the spline order.
  kSensorRigPoseSplineControlPointsIndex = 9,
};

// Generic auto-differentiation camera cost functor. Residuals will be based on
// how the camera model is initialized.
class ImagerCostFunctor {
 public:
  static constexpr int kCameraResidualSize = 2;
  ImagerCostFunctor(CameraIntrinsicsModel camera_model,
                    const Eigen::Vector2d& pixel, double sigma, double stamp,
                    const Trajectory& sp_T_world_sensorrig);

  // Convenience function for creating a camera cost function.
  static ceres::CostFunction* CreateCostFunction(
      const Eigen::Vector2d& pixel, double sigma,
      CameraIntrinsicsModel camera_model, Eigen::VectorXd& intrinsics,
      Pose3d& imager_extrinsics, Pose3d& sensor_extrinsics, double& latency,
      Eigen::Vector3d& t_model_point, Pose3d& T_world_model,
      Trajectory& trajectory_world_sensorrig, double stamp,
      std::vector<double*>& parameters);

  // Parameters to the cost function:
  //   intrinsics:
  //     All parameters in the intrinsics model as an Eigen column vector.
  //     Order of the parameters will need to be in agreement with the model
  //     being used.
  //   q_sensor_imager:
  //     Rotation from sensor frame to camera frame as a quaternion.
  //   t_sensor_imager:
  //     Position of camera relative to sensor origin resolved in the
  //     sensor frame.
  //   q_sensorrig_sensor:
  //     Rotation from sensor rig frame to sensor frame as a quaternion.
  //   t_sensorrig_sensor:
  //     Position of sensor relative to sensor rig origin resolved in the
  //     sensor rig frame.
  //   latency:
  //     Sensor latency in seconds.
  //   q_world_model:
  //     Rotation from world frame to model frame as a quaternion.
  //   t_model_point:
  //     Position of the point in the model resolved in the model frame.
  //   t_world_model:
  //     Position of model relative to world origin resolved in the world frame.
  //   control_points:
  //     Control points for the entire pose trajectory.
  template <typename T>
  bool operator()(T const* const* parameters, T* residual) {
    // Parse intrinsics.
    const T* intrinsics_ptr = static_cast<const T*>(
        &(parameters[static_cast<int>(ImagerParameterIndices::kIntrinsicsIndex)]
                    [0]));
    const int parameter_size = camera_model_->NumberOfParameters();
    const Eigen::VectorX<T> intrinsics =
        Eigen::Map<const Eigen::VectorX<T>>(intrinsics_ptr, parameter_size);
    // Parse imager extrinsics.
    const Eigen::Map<const Eigen::Quaternion<T>> q_sensor_imager(
        &(parameters[static_cast<int>(
            ImagerParameterIndices::kImagerExtrinsicsRotationIndex)][0]));
    const Eigen::Map<const Eigen::Vector3<T>> t_sensor_imager(
        &(parameters[static_cast<int>(
            ImagerParameterIndices::kImagerExtrinsicsTranslationIndex)][0]));
    // Parse sensor extrinsics.
    const Eigen::Map<const Eigen::Quaternion<T>> q_sensorrig_sensor(
        &(parameters[static_cast<int>(
            ImagerParameterIndices::kSensorExtrinsicsRotationIndex)][0]));
    const Eigen::Map<const Eigen::Vector3<T>> t_sensorrig_sensor(
        &(parameters[static_cast<int>(
            ImagerParameterIndices::kSensorExtrinsicsTranslationIndex)][0]));
    // Parse latency.
    const T latency =
        parameters[static_cast<int>(ImagerParameterIndices::kLatencyIndex)][0];
    // Parse model point and model pose resolved in the world frame.
    const Eigen::Map<const Eigen::Vector3<T>> t_model_point(
        &(parameters[static_cast<int>(ImagerParameterIndices::kModelPointIndex)]
                    [0]));
    const Eigen::Map<const Eigen::Quaternion<T>> q_world_model(
        &(parameters[static_cast<int>(
            ImagerParameterIndices::kModelRotationIndex)][0]));
    const Eigen::Map<const Eigen::Vector3<T>> t_world_model(
        &(parameters[static_cast<int>(
            ImagerParameterIndices::kModelTranslationIndex)][0]));
    // Parse sensor rig spline resolved in the world frame.
    const int num_rotation_control_points =
        rotation_trajectory_evaluation_params_.num_control_points;
    Eigen::MatrixX<T> rotation_control_points(num_rotation_control_points, 3);
    for (int i = 0; i < num_rotation_control_points; ++i) {
      rotation_control_points.row(i) = Eigen::Map<const Eigen::Vector3<T>>(
          &(parameters[static_cast<int>(
                           ImagerParameterIndices::
                               kSensorRigPoseSplineControlPointsIndex) +
                       i][0]));
    }
    const int num_position_control_points =
        position_trajectory_evaluation_params_.num_control_points;
    Eigen::MatrixX<T> position_control_points(num_position_control_points, 3);
    for (int i = 0; i < num_position_control_points; ++i) {
      position_control_points.row(i) = Eigen::Map<const Eigen::Vector3<T>>(
          &(parameters[static_cast<int>(
                           ImagerParameterIndices::
                               kSensorRigPoseSplineControlPointsIndex) +
                       i + num_rotation_control_points][0]));
    }

    const Eigen::MatrixX<T> rotation_basis_matrix =
        rotation_trajectory_evaluation_params_.basis_matrix.template cast<T>();
    const T rotation_knot0 =
        static_cast<T>(rotation_trajectory_evaluation_params_.knot0);
    const T rotation_knot1 =
        static_cast<T>(rotation_trajectory_evaluation_params_.knot1);
    const T rotation_stamp =
        static_cast<T>(rotation_trajectory_evaluation_params_.stamp) - latency;

    const Eigen::MatrixX<T> position_basis_matrix =
        position_trajectory_evaluation_params_.basis_matrix.template cast<T>();
    const T position_knot0 =
        static_cast<T>(position_trajectory_evaluation_params_.knot0);
    const T position_knot1 =
        static_cast<T>(position_trajectory_evaluation_params_.knot1);
    const T position_stamp =
        static_cast<T>(position_trajectory_evaluation_params_.stamp) - latency;
    // Evaluate the pose.
    const Eigen::Vector3<T> phi_sensorrig_world = -BSpline<3, T>::Evaluate(
        rotation_control_points, rotation_knot0, rotation_knot1,
        rotation_basis_matrix, rotation_stamp, 0);
    T q_sensorrig_world_array[4];
    ceres::AngleAxisToQuaternion(phi_sensorrig_world.data(),
                                 q_sensorrig_world_array);
    const Eigen::Quaternion<T> q_sensorrig_world(
        q_sensorrig_world_array[0], q_sensorrig_world_array[1],
        q_sensorrig_world_array[2], q_sensorrig_world_array[3]);
    const Eigen::Vector3<T> t_world_sensorrig = BSpline<3, T>::Evaluate(
        position_control_points, position_knot0, position_knot1,
        position_basis_matrix, position_stamp, 0);

    // Resolve the model point in the camera frame.
    const Eigen::Quaternion<T> q_sensorrig_imager =
        q_sensorrig_sensor * q_sensor_imager;
    const Eigen::Vector3<T> t_sensorrig_imager =
        t_sensorrig_sensor + q_sensorrig_sensor * t_sensor_imager;
    const Eigen::Quaternion<T> q_imager_model =
        q_sensorrig_imager.inverse() * q_sensorrig_world * q_world_model;
    const Eigen::Vector3<T> t_world_imager =
        t_world_sensorrig + q_sensorrig_world.inverse() * t_sensorrig_imager;
    const Eigen::Vector3<T> t_model_imager =
        q_world_model.inverse() * (t_world_imager - t_world_model);
    const Eigen::Vector3<T> t_camera_point =
        q_imager_model * (t_model_point - t_model_imager);
    // Project the point through the camera model.
    const absl::StatusOr<Eigen::Vector2<T>> projection =
        camera_model_->ProjectPoint(intrinsics, t_camera_point);
    // Assign the residual, or return boolean indicating success/failure.
    if (projection.ok()) {
      Eigen::Map<Eigen::Vector2<T>> error(residual);
      const Eigen::Vector2<T> pixel = pixel_.template cast<T>();
      error = (pixel - *projection) * static_cast<T>(information_);
      return true;
    }
    return false;
  }

 private:
  Eigen::Vector2d pixel_;
  double information_;
  std::unique_ptr<CameraModel> camera_model_;
  TrajectoryEvaluationParams position_trajectory_evaluation_params_;
  TrajectoryEvaluationParams rotation_trajectory_evaluation_params_;
};
}  // namespace calico::sensors

#endif  // CALICO_SENSORS_IMAGER_COST_FUNCTOR_H_
