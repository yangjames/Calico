#ifndef CALICO_TRAJECTORY_H_
#define CALICO_TRAJECTORY_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "calico/bspline.h"
#include "calico/typedefs.h"
#include "ceres/problem.h"


namespace calico {

class Trajectory {
 public:

  // Add poses.
  absl::Status AddPoses(const absl::flat_hash_map<double, Pose3>& poses_world_body);

  // Setter/getter for the internal map between timestamp and pose.
  const absl::flat_hash_map<double, Pose3>& trajectory() const;
  absl::flat_hash_map<double, Pose3>& trajectory();

  // Add internal parameters to a ceres problem. Any internal parameters set to
  // constant are marked as such in the problem. Returns the total number of
  // parameters added to the problem.
  int AddParametersToProblem(ceres::Problem& problem);

  /*
  absl::Status FitSpline(const std::vector<Pose3>& poses_world_body,
                         const std::vector<double>& timestamps);

  // Interpolate the trajectory at given timestamps.
  std::vector<Pose3> Interpolate(const std::vector<double>& interp_times);

  // Convenience function for unwrapping discrete Euler angles in order to get
  // a more continuous signal.
  static void UnwrapEuler(std::vector<Eigen::Vector3d>& euler);
  */


 private:

  absl::flat_hash_map<double, Pose3> pose_id_to_pose_world_body_;

  /*
  static constexpr int kSplineOrder = 6;
  static constexpr int kKnotFrequency = 10;
  BSpline<3> phi_world_sensorrig_;
  BSpline<3> t_world_sensorrig_;
  */
};

} // namespace calico

#endif //CALICO_TRAJECTORY_H_
