#ifndef CALICO_SENSORS_GYROSCOPE_COST_FUNCTOR_H_
#define CALICO_SENSORS_GYROSCOPE_COST_FUNCTOR_H_

#include "calico/geometry.h"
#include "calico/sensors/gyroscope_models.h"
#include "calico/trajectory.h"
#include "ceres/cost_function.h"


namespace calico::sensors {

// Enum listing the positions of parameters for a gyroscope cost function.
enum class GyroscopeParameterIndices : int {
  // Gyroscope intrinsics.
  kIntrinsicsIndex = 0,
  // Extrinsics parameters of the gyroscope relative to its sensor rig.
  kExtrinsicsRotationIndex = 1,
  kExtrinsicsTranslationIndex = 2,
  // Sensor latency.
  kLatencyIndex = 3,
  // Rotation and position control points of the appropriate spline segment as an
  // Nx6 matrix.
  kSensorRigPoseSplineControlPointsIndex = 4
};

// Generic auto-differentiation gyroscope cost functor. Residuals will be based on
// how the gyroscope model is initialized.
class GyroscopeCostFunctor {
 public:
  static constexpr int kGyroscopeResidualSize = 3;
  explicit GyroscopeCostFunctor(
      GyroscopeIntrinsicsModel gyroscope_model,
      const Eigen::Vector3d& measurement, double stamp,
      const Trajectory& sp_T_world_sensorrig);

  // Convenience function for creating a gyroscope cost function.
  static ceres::CostFunction* CreateCostFunction(
      const Eigen::Vector3d& measurement,
      GyroscopeIntrinsicsModel gyroscope_model, Eigen::VectorXd& intrinsics,
      Pose3d& extrinsics, double& latency,
      Trajectory& trajectory_world_sensorrig, double stamp,
      std::vector<double*>& parameters);

  // Parameters to the cost function:
  //   intrinsics:
  //     All parameters in the intrinsics model as an Eigen column vector.
  //     Order of the parameters will need to be in agreement with the model
  //     being used.
  //   q_sensorrig_gyroscope:
  //     Rotation from sensorrig frame to gyroscope frame as a quaternion.
  //   t_sensorrig_gyroscope:
  //     Position of gyroscope relative to sensorrig origin resolved in the
  //     sensorrig frame.
  //   latency:
  //     Sensor latency in seconds.
  //   control_points:
  //     Control points for the entire pose trajectory.
  template <typename T>
    bool operator()(T const* const* parameters, T* residual) {
    // Parse intrinsics.
    const T* intrinsics_ptr =
      static_cast<const T*>(&(parameters[static_cast<int>(
          GyroscopeParameterIndices::kIntrinsicsIndex)][0]));
    const int parameter_size = gyroscope_model_->NumberOfParameters();
    const Eigen::VectorX<T> intrinsics = Eigen::Map<const Eigen::VectorX<T>>(
        intrinsics_ptr, parameter_size);
    // Parse extrinsics.
    const Eigen::Map<const Eigen::Quaternion<T>> q_sensorrig_gyroscope(
        &(parameters[static_cast<int>(
            GyroscopeParameterIndices::kExtrinsicsRotationIndex)][0]));
    const Eigen::Map<const Eigen::Vector3<T>> t_sensorrig_gyroscope(
        &(parameters[static_cast<int>(
            GyroscopeParameterIndices::kExtrinsicsTranslationIndex)][0]));
    // Parse latency.
    const T latency =
        parameters[static_cast<int>(
            GyroscopeParameterIndices::kLatencyIndex)][0];
    // Parse sensor rig spline resolved in the world frame.
    const int& num_control_points =
        trajectory_evaluation_params_.num_control_points;
   Eigen::MatrixX<T> control_points(num_control_points, 6);
    for (int i = 0; i < num_control_points; ++i) {
      control_points.row(i) = Eigen::Map<const Eigen::Vector<T, 6>>(
          &(parameters[static_cast<int>(
              GyroscopeParameterIndices::kSensorRigPoseSplineControlPointsIndex
              ) + i][0]));
    }
    const Eigen::MatrixX<T> basis_matrix =
        trajectory_evaluation_params_.basis_matrix.template cast<T>();
    const T knot0 = static_cast<T>(trajectory_evaluation_params_.knot0);
    const T knot1 = static_cast<T>(trajectory_evaluation_params_.knot1);
    const T stamp =
        static_cast<T>(trajectory_evaluation_params_.stamp) - latency;
    // Evaluate the pose and pose rate.
    const Eigen::Vector<T, 6> pose_vector =
        BSpline<Trajectory::kSplineOrder, T>::Evaluate(
            control_points, knot0, knot1, basis_matrix, stamp,
            /*derivative=*/0);
    const Eigen::Vector<T, 6> pose_dot_vector =
        BSpline<Trajectory::kSplineOrder, T>::Evaluate(
            control_points, knot0, knot1, basis_matrix, stamp,
            /*derivative=*/1);
    // Compute the angular velocity of the gyroscope.
    // TODO(yangjames): Also evaluate acceleration for g-sensitivity
    //                  calculations.
    const Eigen::Vector3<T> phi_sensorrig_world = -pose_vector.head(3);
    const Eigen::Vector3<T> phi_dot_sensorrig_world = -pose_dot_vector.head(3);
    const Eigen::Matrix3<T> J = ExpSO3Jacobian(phi_sensorrig_world);
    const Eigen::Vector3<T> omega_sensorrig_world = J * phi_dot_sensorrig_world;
    const Eigen::Vector3<T> omega_gyroscope_world =
        -(q_sensorrig_gyroscope.inverse() * omega_sensorrig_world);
    // Project the sensor angular velocity through the gyroscope model.
    const absl::StatusOr<Eigen::Vector3<T>> projection =
        gyroscope_model_->Project(intrinsics, omega_gyroscope_world);
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
  std::unique_ptr<GyroscopeModel> gyroscope_model_;
  TrajectoryEvaluationParams trajectory_evaluation_params_;
};
} // namespace calico::sensors
#endif // CALICO_SENSORS_GYROSCOPE_COST_FUNCTOR_H_
