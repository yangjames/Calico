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
  T knot0;
  T knot1;
  Eigen::MatrixX<T> basis_matrix;
  std::vector<const T*> rotation_control_points;
  std::vector<const T*> position_control_points;
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

  // Getter for a spline segment object associated with a specific timestamp.
  TrajectorySegment<double> GetTrajectorySegment(double stamp) const;

  // Interpolate the trajectory at given timestamps.
  absl::StatusOr<std::vector<Pose3d>>
  Interpolate(const std::vector<double>& interp_times);

  // Dump the trajectory to binary.
  void WriteToFile(absl::string_view fname) const;

  // Evaluate a trajectory segment at a given timestamp and derivative order.
  template <typename T>
  static absl::StatusOr<Eigen::Vector<T, 6>> Evaluate(
      const TrajectorySegment<T> segment, T stamp, int derivative) {
    int num_points = segment.rotation_control_points.size();
    if (num_points != segment.position_control_points.size()) {
      return absl::InvalidArgumentError("Rotation control points and position "
                                        "control points have different sizes.");
    }
    if ((segment.basis_matrix.rows() != num_points) ||
        (segment.basis_matrix.cols() != num_points)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid basis matrix size. Expected ", num_points, "x", num_points,
          ", got ", segment.basis_matrix.rows(), "x",
          segment.basis_matrix.cols()));
    }
    std::vector<Eigen::Vector3<T>> rotation_control_points(num_points);
    std::vector<Eigen::Vector3<T>> position_control_points(num_points);
    for (int i = 0; i < rotation_control_points.size(); ++i) {
      rotation_control_points[i] =
          Eigen::Map<const Eigen::Vector3<T>>(segment.rotation_control_points.at(i));
      position_control_points[i] =
          Eigen::Map<const Eigen::Vector3<T>>(segment.position_control_points.at(i));
    }
    const T& knot0 = segment.knot0;
    const T& knot1 = segment.knot1;
    const Eigen::MatrixX<T>& basis_matrix = segment.basis_matrix;
    const Eigen::Vector3<T> phi = BSpline<3, T>::Evaluate(
        rotation_control_points, knot0, knot1, basis_matrix, stamp, derivative);
    const Eigen::Vector3<T> pos = BSpline<3, T>::Evaluate(
       position_control_points, knot0, knot1, basis_matrix, stamp, derivative);
    Eigen::Vector<T, 6> evaluation_vector;
    evaluation_vector << phi, pos;
    return evaluation_vector;
  }

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
