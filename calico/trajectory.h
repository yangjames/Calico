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

class Trajectory {
 public:
  static constexpr int kSplineOrder = 6;
  static constexpr double kKnotFrequency = 10;

  // Add discrete poses representing a time-parameterized trajectory. A new
  // spline will be fit to these poses.
  absl::Status AddPoses(
      const absl::flat_hash_map<double, Pose3d>& poses_world_body);

  // Add internal parameters to a ceres problem. Any internal parameters set to
  // constant are marked as such in the problem. Returns the total number of
  // parameters added to the problem.
  int AddParametersToProblem(ceres::Problem& problem);

  // Setter/getter for the spline.
  const BSpline<6>& spline() const { return spline_pose_world_body_; }
  BSpline<6>& spline() { return spline_pose_world_body_; }

  // Setter/getter for the internal map between timestamp and pose.
  const absl::flat_hash_map<double, Pose3d>& trajectory() const;
  absl::flat_hash_map<double, Pose3d>& trajectory();

  // Get the parameters needed to evaluate the spline for a given timestamp.
  TrajectoryEvaluationParams GetEvaluationParams(double stamp) const;

  // Interpolate the trajectory at given timestamps.
  absl::StatusOr<std::vector<Pose3d>>
  Interpolate(const std::vector<double>& interp_times) const;

  // Dump the trajectory to binary.
  // void WriteToFile(absl::string_view fname) const;

  // Convert a 6-DOF vector to Pose3 type. First three elements of the vector
  // are expected to be an SO(3) log map vector, and the last three are the
  // position values.
  // vector = [phi] = [phi_x phi_y phi_z t_x t_y t_z]^T
  //          [pos]
  // R = exp([phi]_x)
  // [phi]_x = [0 -phi_z phi_y]
  //           [phi_z 0 -phi_x]
  //           [-phi_y phi_x 0]
  template <typename T>
  static Pose3<T> VectorToPose3(const Eigen::Vector<T, 6>& vector) {
    const Eigen::Vector3<T> phi = vector.head(3);
    const Eigen::Vector3<T> pos = vector.tail(3);
    Eigen::Vector4<T> q;
    ceres::AngleAxisToQuaternion(phi.data(), q.data());
    const Eigen::Quaternion<T> rot(q(0), q(1), q(2), q(3));
    return Pose3<T>(rot, pos);
  }

 private:
  absl::flat_hash_map<double, Pose3d> pose_id_to_pose_world_body_;
  BSpline<6> spline_pose_world_body_;

  // Fit a 6-DOF spline through a set of stamped poses.
  absl::Status FitSpline(
      const absl::flat_hash_map<double, Pose3d>& poses_world_body);

  // Convenience function for unwrapping discrete axis-angle vectors in order to
  // get a more continuous signal.
  void UnwrapPhaseLogMap(std::vector<Eigen::Vector3d>& phi);

};

} // namespace calico

#endif //CALICO_TRAJECTORY_H_
