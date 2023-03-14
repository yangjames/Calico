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

int Trajectory::AddParametersToProblem(ceres::Problem& problem) {
  return spline_pose_world_body_.AddParametersToProblem(problem);
}

const absl::flat_hash_map<double, Pose3d>& Trajectory::trajectory() const {
  return pose_id_to_pose_world_body_;
}

absl::flat_hash_map<double, Pose3d>& Trajectory::trajectory() {
  return pose_id_to_pose_world_body_;
}

TrajectoryEvaluationParams Trajectory::GetEvaluationParams(double stamp) const {
  const int control_point_idx =
      spline_pose_world_body_.GetControlPointIndex(stamp);
  const int knot_idx =
      spline_pose_world_body_.GetKnotIndexFromControlPointIndex(
          control_point_idx);
  const int num_control_points =
      spline_pose_world_body_.control_points().rows();
  return TrajectoryEvaluationParams {
    .spline_index = control_point_idx,
    .knot0 = spline_pose_world_body_.knots().at(knot_idx),
    .knot1 = spline_pose_world_body_.knots().at(knot_idx + 1),
    .stamp = stamp,
    .num_control_points = num_control_points,
    .basis_matrix =
        spline_pose_world_body_.basis_matrices().at(control_point_idx),
  };
}

absl::Status Trajectory::FitSpline(
    const absl::flat_hash_map<double, Pose3d>& poses_world_body) {
  // Grab all sorted timestamps from the map.
  std::vector<double> stamps;
  for (const auto& [stamp, _] : poses_world_body) {
    stamps.push_back(stamp);
  }
  std::sort(stamps.begin(), stamps.end());
  // Convert each pose into two 3-vectors, rotation and position.
  const int num_poses = poses_world_body.size();
  std::vector<Eigen::Vector3d> phi_world_body(num_poses);
  std::vector<Eigen::Vector3d> t_world_body(num_poses);
  int i = 0;
  for (const auto& stamp : stamps) {
    const Pose3d& T_world_body = poses_world_body.at(stamp);
    Eigen::AngleAxisd vec(T_world_body.rotation());
    phi_world_body[i] = vec.axis() * vec.angle();
    t_world_body[i] = T_world_body.translation();
    ++i;
  }
  UnwrapPhaseLogMap(phi_world_body);

  std::vector<Eigen::Vector<double, 6>> pose_world_body(num_poses);
  for (int i = 0; i < pose_world_body.size(); ++i) {
    pose_world_body[i].head(3) = phi_world_body[i];
    pose_world_body[i].tail(3) = t_world_body[i];
  }

  RETURN_IF_ERROR(spline_pose_world_body_.FitToData(
      stamps, pose_world_body, kSplineOrder, kKnotFrequency));
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

absl::StatusOr<std::vector<Pose3d>>
Trajectory::Interpolate(const std::vector<double>& interp_times) const {
  std::vector<Eigen::Vector<double, 6>> pose_vectors_interp;
  ASSIGN_OR_RETURN(pose_vectors_interp,
                   spline_pose_world_body_.Interpolate(interp_times));
  std::vector<Pose3d> interpolated_poses(interp_times.size());
  for (int i = 0; i < interp_times.size(); ++i) {
    interpolated_poses[i] = VectorToPose3(pose_vectors_interp[i]);
  }
  return interpolated_poses;
}

// void Trajectory::WriteToFile(absl::string_view fname) const {
//   std::ofstream file(std::string(fname), std::ios::out | std::ios::binary);
//   file.write((char*)&kSplineOrder, sizeof(int));
//   file.write((char*)&kKnotFrequency, sizeof(double));
//   for (const auto& spline :
//            std::vector{phi_world_body_, t_world_body_}) {
//     const auto& control_points = spline.control_points();
//     const auto& knots = spline.knots();
//     const int num_control_points = control_points.size();
//     const int num_knots = knots.size();
//     file.write((char*)&num_control_points, sizeof(int));
//     file.write((char*)&num_knots, sizeof(int));
//     for (const auto& control_point : control_points) {
//       file.write((char*) control_point.data(), sizeof(double)*3);
//     }
//     for (const auto& knot : knots) {
//       file.write((char*) &knot, sizeof(double));
//     }
//   }
// }

} // namespace calico
