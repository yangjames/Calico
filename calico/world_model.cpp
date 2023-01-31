#include "calico/world_model.h"

#include "absl/status/status.h"
#include "absl/strings/string_view.h"


namespace calico {

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
