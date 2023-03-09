#include "calico/trajectory.h"

#include <fstream>
#include <iostream>

#include "calico/optimization_utils.h"
#include "calico/statusor_macros.h"
#include "Eigen/Geometry"


namespace calico {

absl::Status Trajectory::AddPoses(
    const absl::flat_hash_map<double, Pose3d>& poses_world_body) {
  pose_id_to_pose_world_body_ = poses_world_body;
  return FitSpline(poses_world_body);
}

const absl::flat_hash_map<double, Pose3d>& Trajectory::trajectory() const {
  return pose_id_to_pose_world_body_;
}

absl::flat_hash_map<double, Pose3d>& Trajectory::trajectory() {
  return pose_id_to_pose_world_body_;
}

absl::Status Trajectory::FitSpline(
    const absl::flat_hash_map<double, Pose3d>& poses_world_sensorrig) {
  // Grab all sorted timestamps from the map.
  std::vector<double> stamps;
  for (const auto& [stamp, _] : poses_world_sensorrig) {
    stamps.push_back(stamp);
  }
  std::sort(stamps.begin(), stamps.end());
  // Convert each pose into two 3-vectors, rotation and position.
  const int num_poses = poses_world_sensorrig.size();
  std::vector<Eigen::Vector3d> phi_world_sensorrig(num_poses);
  std::vector<Eigen::Vector3d> t_world_sensorrig(num_poses);
  int i = 0;
  for (const auto& stamp : stamps) {
    const Pose3d& T_world_sensorrig = poses_world_sensorrig.at(stamp);
    Eigen::AngleAxisd vec(T_world_sensorrig.rotation());
    phi_world_sensorrig[i] = vec.axis() * vec.angle();
    t_world_sensorrig[i] = T_world_sensorrig.translation();
    ++i;
  }
  UnwrapPhaseLogMap(phi_world_sensorrig);

  RETURN_IF_ERROR(phi_world_sensorrig_.FitToData(
      stamps, phi_world_sensorrig, kSplineOrder, kKnotFrequency));
  RETURN_IF_ERROR(t_world_sensorrig_.FitToData(
      stamps, t_world_sensorrig, kSplineOrder, kKnotFrequency));
  return absl::OkStatus();
}


void Trajectory::UnwrapPhaseLogMap(std::vector<Eigen::Vector3d>& phi) {
  for (int i = 1; i < phi.size(); ++i) {
    const Eigen::Vector3d& v1 = phi[i];
    double theta = v1.norm();
    if (theta == 0) {
      continue;
    }
    const Eigen::Vector3d& v0 = phi[i-1];
    double k = std::round((v1.transpose() * v0 - theta * theta) /
                          (2.0 * M_PI * theta));
    phi[i] *= (1.0 + 2.0 * M_PI * k / theta);
  }
}

int Trajectory::AddParametersToProblem(ceres::Problem& problem) {
  /*
  int num_parameters = 0;
  for (auto& [pose_id, pose] : pose_id_to_pose_world_body_) {
    num_parameters += utils::AddPoseToProblem(problem, pose);
  }
  */
  int num_parameters = phi_world_sensorrig_.AddParametersToProblem(problem);
  num_parameters += t_world_sensorrig_.AddParametersToProblem(problem);
  return num_parameters;
}

absl::StatusOr<std::vector<Pose3d>>
Trajectory::Interpolate(const std::vector<double>& interp_times) {
  std::vector<Eigen::Vector3d> phi_interp;
  ASSIGN_OR_RETURN(phi_interp, phi_world_sensorrig_.Interpolate(interp_times));
  std::vector<Eigen::Vector3d> pos_interp;
  ASSIGN_OR_RETURN(pos_interp, t_world_sensorrig_.Interpolate(interp_times));
  std::vector<Pose3d> interpolated_poses(interp_times.size());
  for (int i = 0; i < interp_times.size(); ++i) {
    const Eigen::Vector3d& pos = pos_interp[i];
    const Eigen::Vector3d& phi = phi_interp[i];
    Eigen::Vector4d q;
    ceres::AngleAxisToQuaternion(phi.data(), q.data());
    const Eigen::Quaterniond rot(q(0), q(1), q(2), q(3));
    interpolated_poses[i] = Pose3d(rot, pos);
  }
  return interpolated_poses;
}

void Trajectory::WriteToFile(absl::string_view fname) const {
  std::ofstream file(std::string(fname), std::ios::out | std::ios::binary);
  file.write((char*)&kSplineOrder, sizeof(int));
  file.write((char*)&kKnotFrequency, sizeof(double));
  for (const auto& spline :
           std::vector{phi_world_sensorrig_, t_world_sensorrig_}) {
    const auto& control_points = spline.control_points();
    const auto& knots = spline.knots();
    const int num_control_points = control_points.size();
    const int num_knots = knots.size();
    file.write((char*)&num_control_points, sizeof(int));
    file.write((char*)&num_knots, sizeof(int));
    for (const auto& control_point : control_points) {
      file.write((char*) control_point.data(), sizeof(double)*3);
    }
    for (const auto& knot : knots) {
      file.write((char*) &knot, sizeof(double));
    }
  }
}

} // namespace calico
