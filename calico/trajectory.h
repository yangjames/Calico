#ifndef CALICO_TRAJECTORY_H_
#define CALICO_TRAJECTORY_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "calico/bspline.h"
#include "calico/typedefs.h"
#include "ceres/problem.h"
#include "ceres/rotation.h"


namespace calico {

struct TrajectoryEvaluationParams {
  int spline_index;
  double knot0;
  double knot1;
  double stamp;
  int num_control_points;
  Eigen::MatrixXd basis_matrix;
};

/// Trajectory class. Takes in discrete world-from-sensorrig poses and encodes
/// them as a 6-DOF B-Spline.
class Trajectory {
 public:
  /// Default spline order of 6.
  static constexpr int kDefaultSplineOrder = 6;

  /// Default knot frequency at 10Hz.
  static constexpr double kDefaultKnotFrequency = 10;

  /// Add discrete poses representing a time-parameterized trajectory. A new
  /// spline will be fit to these poses with a specified knot frequency and
  /// spline order. Returns `absl::OkStatus()` if successful, an error otherwise.
  /// \n\n
  /// `poses_world_body` is a map between a `double` timestamp in seconds and
  /// `Pose3d` world-from-sensorrig 6-DOF pose:
  /// \f[
  /// \mathbf{T}^w_r(t) = \left[\begin{matrix}\mathbf{R}^w_r&\mathbf{t}^w_{wr}\\
  /// \mathbf{0}&1\end{matrix}\right]
  /// \f]
  /// `knot_frequency` is the knot rate of the spline. If `poses_world_body`
  /// contains timestamps spanning from \f$t_0\f$ to \f$t_f\f$ seconds, this
  /// function will yield a spline with \f$n_{knots} = \left(t_f - t_0\right)
  /// \left(\text{knot_frequency}\right)\f$\n\n
  /// `spline_order` is the order of the spline. A spline order of \f$n\f$
  /// implies that the pose at any given time is:
  /// \f[
  /// \mathbf{T}^w_r(t) = \left[\begin{matrix}\boldsymbol{\Phi}^w_r(t)\\\
  /// \mathbf{t}^w_{wr}(t)\end{matrix}\right] = \mathbf{c}_{i,0} +
  /// \mathbf{c}_{i,1}t + ... + \mathbf{c}_{i,n}t^n
  /// \f]
  /// where \f$\mathbf{c}_{i,j}\f$ is the j'th control point for the i'th spline
  /// segment.
  absl::Status FitSpline(
      const absl::flat_hash_map<double, Pose3d>& poses_world_sensorrig,
      double knot_frequency = kDefaultKnotFrequency,
      int spline_order = kDefaultSplineOrder);

  /// Add internal parameters to a ceres problem. Any internal parameters set to
  /// constant are marked as such in the problem. Returns the total number of
  /// parameters added to the problem.
  int AddParametersToProblem(ceres::Problem& problem);

  /// Accessors for the spline.
  const BSpline<6>& spline() const { return spline_pose_world_body_; }
  BSpline<6>& spline() { return spline_pose_world_body_; }

  /// Interpolate the trajectory at given timestamps. Returns a vector of
  /// world-from-sensorrig poses evaluated in the order of the timestamps.
  /// Returns an error if something went wrong during interpolation. For example,
  /// attempting to interpolate on out-of-bounds timestmaps.
  /// \n\n
  /// `interp_times` is the timestamps to query the spline. No assumptions are
  /// made about order.
  absl::StatusOr<std::vector<Pose3d>>
  Interpolate(const std::vector<double>& interp_times) const;

  /// Convert a 6-DOF vector to Pose3 type. First three elements of the vector
  /// are expected to be an \f$SO(3)\f$ log map vector, and the last three are the
  /// position values.\n\n
  /// \f[
  /// \text{vector}=\left[\begin{matrix}\boldsymbol{\Phi}\\\mathbf{t}\end{matrix}\right]\\
  /// \boldsymbol{\Phi} =
  ///  \left[\begin{matrix}\Phi_x\\\Phi_y\\\Phi_z\end{matrix}\right]\\
  /// \mathbf{R} = \exp\left([\boldsymbol{\Phi}]_\times\right)\\
  /// \left[\boldsymbol{\Phi}\right]_\times =
  /// \left[\begin{matrix}0 &-\Phi_z&\Phi_y\\
  ///           \Phi_z&0&-\Phi_x\\
  ///           -\Phi_y&\Phi_x&0\end{matrix}\right]
  /// \f]
  template <typename T>
  static Pose3<T> VectorToPose3(const Eigen::Vector<T, 6>& vector) {
    const Eigen::Vector3<T> phi = vector.head(3);
    const Eigen::Vector3<T> pos = vector.tail(3);
    Eigen::Vector4<T> q;
    ceres::AngleAxisToQuaternion(phi.data(), q.data());
    const Eigen::Quaternion<T> rot(q(0), q(1), q(2), q(3));
    return Pose3<T>(rot, pos);
  }

  /// Setter/getter for the internal map between timestamp and pose.
  const absl::flat_hash_map<double, Pose3d>& trajectory() const;
  absl::flat_hash_map<double, Pose3d>& trajectory();

  /// Get the parameters needed to evaluate the spline for a given timestamp.
  TrajectoryEvaluationParams GetEvaluationParams(double stamp) const;

 private:
  absl::flat_hash_map<double, Pose3d> pose_id_to_pose_world_body_;
  BSpline<6> spline_pose_world_body_;

  // Convenience function for unwrapping discrete axis-angle vectors in order to
  // get a more continuous signal.
  void UnwrapPhaseLogMap(std::vector<Eigen::Vector3d>& phi);

};

} // namespace calico

#endif //CALICO_TRAJECTORY_H_
