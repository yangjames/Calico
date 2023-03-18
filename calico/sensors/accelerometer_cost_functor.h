#ifndef CALICO_SENSORS_ACCELEROMETER_COST_FUNCTOR_H_
#define CALICO_SENSORS_ACCELEROMETER_COST_FUNCTOR_H_

#include "calico/geometry.h"
#include "calico/sensors/accelerometer_models.h"
#include "calico/trajectory.h"
#include "ceres/cost_function.h"


namespace calico::sensors {

// Enum listing the positions of parameters for a accelerometer cost function.
enum class AccelerometerParameterIndices : int {
  // Accelerometer intrinsics.
  kIntrinsicsIndex = 0,
  // Extrinsics parameters of the accelerometer relative to its sensor rig.
  kExtrinsicsRotationIndex = 1,
  kExtrinsicsTranslationIndex = 2,
  // Sensor latency.
  kLatencyIndex = 3,
  // Gravity vector.
  kGravityIndex = 4,
  // Rotation and position control points of the entire trajectory spline as an
  // Nx6 matrix.
  kSensorRigPoseSplineControlPointsIndex = 5
};

// Generic auto-differentiation accelerometer cost functor. Residuals will be based on
// how the accelerometer model is initialized.
class AccelerometerCostFunctor {
 public:
  static constexpr int kAccelerometerResidualSize = 3;
  explicit AccelerometerCostFunctor(
      AccelerometerIntrinsicsModel accelerometer_model,
      const Eigen::Vector3d& measurement, double stamp,
      const Trajectory& sp_T_world_sensorrig);

  // Convenience function for creating a accelerometer cost function.
  static ceres::CostFunction* CreateCostFunction(
      const Eigen::Vector3d& measurement,
      AccelerometerIntrinsicsModel accelerometer_model,
      Eigen::VectorXd& intrinsics, Pose3d& extrinsics, double& latency,
      Eigen::Vector3d& gravity, Trajectory& trajectory_world_sensorrig, 
      double stamp, std::vector<double*>& parameters);

  // Parameters to the cost function:
  //   intrinsics:
  //     All parameters in the intrinsics model as an Eigen column vector.
  //     Order of the parameters will need to be in agreement with the model
  //     being used.
  //   q_sensorrig_accelerometer:
  //     Rotation from sensorrig frame to accelerometer frame as a quaternion.
  //   t_sensorrig_accelerometer:
  //     Position of accelerometer relative to sensorrig origin resolved in the
  //     sensorrig frame.
  //   latency:
  //     Sensor latency in seconds.
  //   gravity:
  //     Gravity vector resolved in the world frame.
  //   control_points:
  //     Control points for the entire pose trajectory.
  template <typename T>
    bool operator()(T const* const* parameters, T* residual) {
    // Parse intrinsics.
    const T* intrinsics_ptr =
      static_cast<const T*>(&(parameters[static_cast<int>(
          AccelerometerParameterIndices::kIntrinsicsIndex)][0]));
    const int parameter_size = accelerometer_model_->NumberOfParameters();
    const Eigen::VectorX<T> intrinsics = Eigen::Map<const Eigen::VectorX<T>>(
        intrinsics_ptr, parameter_size);
    // Parse extrinsics.
    const Eigen::Map<const Eigen::Quaternion<T>> q_sensorrig_accelerometer(
        &(parameters[static_cast<int>(
            AccelerometerParameterIndices::kExtrinsicsRotationIndex)][0]));
    const Eigen::Map<const Eigen::Vector3<T>> t_sensorrig_accelerometer(
        &(parameters[static_cast<int>(
            AccelerometerParameterIndices::kExtrinsicsTranslationIndex)][0]));
    // Parse latency.
    const T latency =
        parameters[static_cast<int>(
            AccelerometerParameterIndices::kLatencyIndex)][0];
    // Parse gravity.
    const Eigen::Map<const Eigen::Vector3<T>> gravity(
        &(parameters[static_cast<int>(
            AccelerometerParameterIndices::kGravityIndex)][0]));
    // Parse sensor rig spline resolved in the world frame.
    const int& num_control_points =
        trajectory_evaluation_params_.num_control_points;
    const Eigen::Map<const Eigen::MatrixX<T>> all_control_points(
        &(parameters[static_cast<int>(
            AccelerometerParameterIndices::kSensorRigPoseSplineControlPointsIndex)]
          [0]), num_control_points, 6);
    const Eigen::Ref<const Eigen::MatrixX<T>> control_points =
        all_control_points.block(trajectory_evaluation_params_.spline_index, 0,
                                 Trajectory::kSplineOrder, 6);
    const Eigen::MatrixX<T> basis_matrix =
        trajectory_evaluation_params_.basis_matrix.template cast<T>();
    const T knot0 = static_cast<T>(trajectory_evaluation_params_.knot0);
    const T knot1 = static_cast<T>(trajectory_evaluation_params_.knot1);
    const T stamp =
        static_cast<T>(trajectory_evaluation_params_.stamp) - latency;
    // Evaluate the pose, pose rate, and pose acceleration.
    const Eigen::Vector<T, 6> pose_vector =
        BSpline<Trajectory::kSplineOrder, T>::Evaluate(
            control_points, knot0, knot1, basis_matrix, stamp, /*derivative=*/0);
    const Eigen::Vector<T, 6> pose_dot_vector =
        BSpline<Trajectory::kSplineOrder, T>::Evaluate(
            control_points, knot0, knot1, basis_matrix, stamp, /*derivative=*/1);
    const Eigen::Vector<T, 6> pose_ddot_vector =
        BSpline<Trajectory::kSplineOrder, T>::Evaluate(
            control_points, knot0, knot1, basis_matrix, stamp, /*derivative=*/2);
    // Compute the kinematics of the accelerometer.
    const Eigen::Vector3<T> phi_sensorrig_world = -pose_vector.head(3);
    const Eigen::Vector3<T> phi_dot_sensorrig_world = -pose_dot_vector.head(3);
    const Eigen::Vector3<T> phi_ddot_sensorrig_world = -pose_ddot_vector.head(3);
    T q_sensorrig_world_array[4];
    ceres::AngleAxisToQuaternion(phi_sensorrig_world.data(),
                                 q_sensorrig_world_array);
    const Eigen::Quaternion<T> q_sensorrig_world(
        q_sensorrig_world_array[0], q_sensorrig_world_array[1],
        q_sensorrig_world_array[2], q_sensorrig_world_array[3]);
    const Eigen::Ref<const Eigen::Vector3<T>> ddt_world_sensorrig =
        pose_ddot_vector.tail(3);
    const Eigen::Matrix3<T> J = ExpSO3Jacobian(phi_sensorrig_world);
    const Eigen::Vector3<T> omega_sensorrig_world = J * phi_dot_sensorrig_world;
    const Eigen::Vector3<T> omega_accelerometer_world =
        q_sensorrig_accelerometer.inverse() * omega_sensorrig_world;
    const Eigen::Matrix3<T> Jdot = ExpSO3JacobianDot(phi_sensorrig_world,
                                                     phi_dot_sensorrig_world);
    const Eigen::Vector3<T> alpha_sensorrig_world =
        Jdot * phi_dot_sensorrig_world + J * phi_ddot_sensorrig_world;
    const Eigen::Vector3<T> alpha_accelerometer_world =
        q_sensorrig_accelerometer.inverse() * alpha_sensorrig_world;
    const Eigen::Matrix3<T> Alpha = Skew(alpha_accelerometer_world);
    const Eigen::Matrix3<T> Omega = Skew(omega_accelerometer_world);
    const Eigen::Vector3<T> ddt_accelerometer_world_accelerometer =
        q_sensorrig_accelerometer.inverse() *
        (q_sensorrig_world * (ddt_world_sensorrig - gravity) +
        (Omega * Omega + Alpha) * t_sensorrig_accelerometer);
    // Project the sensor angular velocity through the accelerometer model.
    const absl::StatusOr<Eigen::Vector3<T>> projection =
        accelerometer_model_->Project(
            intrinsics, ddt_accelerometer_world_accelerometer);
    if (projection.ok()) {
      Eigen::Map<Eigen::Vector3<T>> error(residual);
      const Eigen::Vector3<T> measurement = measurement_.template cast<T>();
      error = measurement - *projection;
      return true;
    }
    return false;
  }

 private:
  Eigen::Vector3d measurement_;
  std::unique_ptr<AccelerometerModel> accelerometer_model_;
  TrajectoryEvaluationParams trajectory_evaluation_params_;
};
} // namespace calico::sensors
#endif // CALICO_SENSORS_ACCELEROMETER_COST_FUNCTOR_H_
