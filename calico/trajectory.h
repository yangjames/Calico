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

  // Add discrete poses representing a time-parameterized trajectory. A new
  // spline will be fit to these poses.
  absl::Status AddPoses(
      const absl::flat_hash_map<double, Pose3>& poses_world_body);

  // Setter/getter for the internal map between timestamp and pose.
  const absl::flat_hash_map<double, Pose3>& trajectory() const;
  absl::flat_hash_map<double, Pose3>& trajectory();

  // Add internal parameters to a ceres problem. Any internal parameters set to
  // constant are marked as such in the problem. Returns the total number of
  // parameters added to the problem.
  int AddParametersToProblem(ceres::Problem& problem);

  // Dump the trajectory to binary.
  void WriteToFile(absl::string_view fname) const;

  // Getter for a spline segment object associated with a specific timestamp.
  //SplineSegment GetSplineSegment(double stamp) const;

  /*
  // Interpolate the trajectory at given timestamps.
  std::vector<Pose3> Interpolate(const std::vector<double>& interp_times);

  */

 private:
  absl::flat_hash_map<double, Pose3> pose_id_to_pose_world_body_;
  static constexpr int kSplineOrder = 6;
  static constexpr double kKnotFrequency = 10;
  BSpline<3> phi_world_sensorrig_;
  BSpline<3> t_world_sensorrig_;

  // Fit a 6-DOF spline through a set of stamped poses.
  absl::Status FitSpline(
      const absl::flat_hash_map<double, Pose3>& poses_world_body);

  // Convenience function for unwrapping discrete axis-angle vectors in order to
  // get a more continuous signal.
  void UnwrapPhaseLogMap(std::vector<Eigen::Vector3d>& phi);

};

} // namespace calico

#endif //CALICO_TRAJECTORY_H_
