#include "calico/world_model.h"

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "calico/optimization_utils.h"

namespace calico {

  WorldModel::WorldModel() {
    gravity_.setZero();
    gravity_.z() = kGravityDefaultZ;
    gravity_enabled_ = false;
  }

absl::Status WorldModel::AddLandmark(const Landmark& landmark) {
  if (landmark_id_to_landmark_.contains(landmark.id)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Landmark with id ", landmark.id, " already exists in world model."));
  }
  landmark_id_to_landmark_[landmark.id] = landmark;
  return absl::OkStatus();
}

absl::Status WorldModel::AddRigidBody(const RigidBody& rigidbody) {
  if (rigidbody_id_to_rigidbody_.contains(rigidbody.id)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Rigid body with id ", rigidbody.id, "already exists in world model."));
  }
  rigidbody_id_to_rigidbody_[rigidbody.id] = rigidbody;
  return absl::OkStatus();
}

int WorldModel::AddParametersToProblem(ceres::Problem& problem) {
  int num_parameters_added = 0;
  // Add all landmarks to problem.
  for (auto& [_, landmark] : landmark_id_to_landmark_) {
    problem.AddParameterBlock(landmark.point.data(), landmark.point.size());
    num_parameters_added += landmark.point.size();
    // Set this landmark as constant if flagged.
    if (landmark.point_is_constant) {
      problem.SetParameterBlockConstant(landmark.point.data());
    }
  }
  // Add all rigidbodies to problem.
  for (auto& [_, rigidbody] : rigidbody_id_to_rigidbody_) {
    for (auto& [_, point] : rigidbody.model_definition) {
      problem.AddParameterBlock(point.data(), point.size());
      num_parameters_added += point.size();
    }
    num_parameters_added += utils::AddPoseToProblem(
        problem, rigidbody.T_world_rigidbody);
    // Set this rigidbody's model definition as constant if flagged.
    if (rigidbody.model_definition_is_constant) {
      for (auto& [_, point] : rigidbody.model_definition) {
        problem.SetParameterBlockConstant(point.data());
      }
    }
    // Set this rigidbody's world pose constant if flagged.
    if (rigidbody.world_pose_is_constant) {
      utils::SetPoseConstantInProblem(problem, rigidbody.T_world_rigidbody);
    }
  }
  // Add gravity vector.
  problem.AddParameterBlock(gravity_.data(), gravity_.size());
  num_parameters_added += gravity_.size();
  if (!gravity_enabled_) {
    problem.SetParameterBlockConstant(gravity_.data());
  }
  return num_parameters_added;
}

void WorldModel::EnableGravityEstimation(bool enable) {
  gravity_enabled_;
}

absl::flat_hash_map<int, Landmark>& WorldModel::landmarks() {
  return landmark_id_to_landmark_;
}

const absl::flat_hash_map<int, Landmark>& WorldModel::landmarks() const {
  return landmark_id_to_landmark_;
}

absl::flat_hash_map<int, RigidBody>& WorldModel::rigidbodies() {
  return rigidbody_id_to_rigidbody_;
}

const absl::flat_hash_map<int, RigidBody>& WorldModel::rigidbodies() const {
  return rigidbody_id_to_rigidbody_;
}

Eigen::Vector3d& WorldModel::gravity() {
  return gravity_;
}

const Eigen::Vector3d& WorldModel::gravity() const {
  return gravity_;
}

void WorldModel::SetGravity(const Eigen::Vector3d& gravity) {
  gravity_ = gravity;
}

const Eigen::Vector3d& WorldModel::GetGravity() const {
  return gravity_;
}

int WorldModel::NumberOfLandmarks() const {
  return landmark_id_to_landmark_.size();
}

int WorldModel::NumberOfRigidBodies() const {
  return rigidbody_id_to_rigidbody_.size();
}

void WorldModel::ClearLandmarks() {
  landmark_id_to_landmark_.clear();
}

void WorldModel::ClearRigidBodies() {
  rigidbody_id_to_rigidbody_.clear();
}

void WorldModel::Clear() {
  ClearLandmarks();
  ClearRigidBodies();
}



} // namespace calico
