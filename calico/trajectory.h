#ifndef CALICO_TRAJECTORY_H_
#define CALICO_TRAJECTORY_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "calico/bspline.h"
#include "calico/typedefs.h"
#include "ceres/problem.h"
#include "ceres/rotation.h"


namespace calico {

template <typename T>
struct TrajectorySegment {
  std::vector<Eigen::Vector3<T>> rotation_control_points;
  std::vector<Eigen::Vector3<T>> position_control_points;
  T knot0;
  T knot1;
  Eigen::MatrixX<T> basis_matrix;
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

  // Setter/getter for the internal map between timestamp and pose.
  const absl::flat_hash_map<double, Pose3d>& trajectory() const;
  absl::flat_hash_map<double, Pose3d>& trajectory();

  /*
  // Getter for a spline segment object associated with a specific timestamp.
  TrajectorySegment GetSplineSegment(double stamp) const;
  */

  // Interpolate the trajectory at given timestamps.
  absl::StatusOr<std::vector<Pose3d>>
  Interpolate(const std::vector<double>& interp_times);

  // Dump the trajectory to binary.
  void WriteToFile(absl::string_view fname) const;

  template <typename T>
  static Pose3<T> Evaluate(const TrajectorySegment<T> segment, T stamp) {
    const std::vector<Eigen::Vector3<T>>& rotation_control_points =
      segment.rotation_control_points;
    const std::vector<Eigen::Vector3<T>>& position_control_points =
      segment.position_control_points;
    const T& knot0 = segment.knot0;
    const T& knot1 = segment.knot1;
    const Eigen::MatrixX<T>& basis_matrix = segment.basis_matrix;
    const Eigen::Vector3<T> phi = BSpline<3, T>::Evaluate(
        rotation_control_points, knot0, knot1, basis_matrix, stamp);
    const Eigen::Vector3<T> pos = BSpline<3, T>::Evaluate(
        position_control_points, knot0, knot1, basis_matrix, stamp);
    Eigen::Vector4<T> q;
    ceres::AngleAxisToQuaternion(phi.data(), q.data());
    const Eigen::Quaternion<T> rot(q(0), q(1), q(2), q(3));
    return Pose3<T>(rot, pos);
  }

 private:
  absl::flat_hash_map<double, Pose3d> pose_id_to_pose_world_body_;
  BSpline<3> phi_world_sensorrig_;
  BSpline<3> t_world_sensorrig_;

  // Fit a 6-DOF spline through a set of stamped poses.
  absl::Status FitSpline(
      const absl::flat_hash_map<double, Pose3d>& poses_world_body);

  // Convenience function for unwrapping discrete axis-angle vectors in order to
  // get a more continuous signal.
  void UnwrapPhaseLogMap(std::vector<Eigen::Vector3d>& phi);

};

} // namespace calico

#endif //CALICO_TRAJECTORY_H_
