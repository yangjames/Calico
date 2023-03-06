#include "calico/trajectory.h"

#include "calico/optimization_utils.h"
#include "calico/statusor_macros.h"
#include "Eigen/Geometry"


namespace calico {

absl::Status Trajectory::AddPoses(
    const absl::flat_hash_map<double, Pose3>& poses_world_body) {
  pose_id_to_pose_world_body_ = poses_world_body;
  return absl::OkStatus();
}

const absl::flat_hash_map<double, Pose3>& Trajectory::trajectory() const {
  return pose_id_to_pose_world_body_;
}

absl::flat_hash_map<double, Pose3>& Trajectory::trajectory() {
  return pose_id_to_pose_world_body_;
}
  /*
absl::Status Trajectory::FitSpline(
    const std::vector<Pose3>& poses_world_sensorrig,
    const std::vector<double>& timestamps) {
  if (poses_world_sensorrig.size() != timestamps.size()) {
    return absl::InvalidArgumentError(
        "Poses and timestamps must be the same size.");
  }
  const int num_poses = poses_world_sensorrig.size();
  std::vector<Eigen::Vector3d> phi_world_sensorrig(num_poses);
  std::vector<Eigen::Vector3d> t_world_sensorrig(num_poses);
  for (int i = 0; i < num_poses; ++i) {
    const Pose3& T_world_sensorrig = poses_world_sensorrig[i];
    phi_world_sensorrig[i] = T_world_sensorrig.rotation().eulerAngles(2, 1, 0);
    t_world_sensorrig[i] = T_world_sensorrig.translation();
  }
  UnwrapEuler(phi_world_sensorrig);

  RETURN_IF_ERROR(phi_world_sensorrig_.FitToData(
      timestamps, phi_world_sensorrig, kSplineOrder, kKnotFrequency));
  RETURN_IF_ERROR(t_world_sensorrig_.FitToData(
      timestamps, t_world_sensorrig, kSplineOrder, kKnotFrequency));
  return absl::OkStatus();
}

void Trajectory::UnwrapEuler(std::vector<Eigen::Vector3d>& euler) {}

} // namespace calico

*/
int Trajectory::AddParametersToProblem(ceres::Problem& problem) {
  int num_parameters = 0;
  for (auto& [pose_id, pose] : pose_id_to_pose_world_body_) {
    num_parameters += utils::AddPoseToProblem(problem, pose);
  }
  /*
  int num_parameters = phi_world_sensorrig_.AddParametersToProblem(problem);
  num_parameters += t_world_sensorrig_.AddParametersToProblem(problem);
  */
  return num_parameters;
}

} // namespace calico
