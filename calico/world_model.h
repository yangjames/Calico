#ifndef CALICO_WORLD_MODEL_H_
#define CALICO_WORLD_MODEL_H_

#include "calico/typedefs.h"

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "Eigen/Dense"


namespace calico {

// `Landmark` object which represents a unique 3d point resolved in the world
// frame. For optimization problems, set `world_point_is_constant` to true
// if it is a free parameter that needs to be estimated, false otherwise.
struct Landmark {
  Eigen::Vector3d point;
  int id;
  bool point_is_constant;
};

// `Rigidbody` object which represents a constellation of points on a rigid
// body. `model_definition` contains 3d points that make up the rigid body
// resolved in the rigid body frame. For example, this could be planar points
// on a fiducial target in a calibration problem relative to some origin defined
// on the target. `T_world_rigidbody` is the rigid transform TO the world frame
// FROM the rigidbody frame. Both the model definition and its world pose can be
// set as either constant or as free parameters in an optimization problem via
// the flags `world_pose_is_constant` and `model_definition_is_constant`.
struct RigidBody {
  // Maps feature id to its 3d point definition resolved in the rigid body frame
  absl::flat_hash_map<int, Eigen::Vector3d> model_definition;
  Pose3 T_world_rigidbody;
  int id;
  bool world_pose_is_constant;
  bool model_definition_is_constant;
};

// `WorldModel` manages all `Landmark` and `RigidBody` objects in an
// optimization problem.
class WorldModel {
 public:
  WorldModel() = default;
  ~WorldModel() = default;

  // Add a `Landmark` object to the world model. Returns an InvalidArgument status
  // code if the landmark id is not unique.
  absl::Status AddLandmark(const Landmark& landmark);

  // Add a `RigidBody` object to the world model. Returns an InvalidArgument
  // status code if the rigid body id is not unique.
  absl::Status AddRigidBody(const RigidBody& rigidbody);

  // Add internal parameters to a ceres problem. Any internal parameters set to
  // constant are marked as such in the problem.
  //absl::Status AddParametersToProblem(ceres::Problem& problem);

  // Accessor for landmarks.
  absl::flat_hash_map<int, Landmark>& landmarks();
  const absl::flat_hash_map<int, Landmark>& landmarks() const;

  // Accessor for rigid bodies.
  absl::flat_hash_map<int, RigidBody>& rigidbodies();
  const absl::flat_hash_map<int, RigidBody>& rigidbodies() const;

  // Get the number of landmarks currently in this world model.
  int NumberOfLandmarks() const;

  // Get the number of rigid bodies currently in this world model.
  int NumberOfRigidBodies() const;

  // Remove all `Landmark` objects from the world model.
  void ClearLandmarks();

  // Remove all `RigidBody` objects from the world model.
  void ClearRigidBodies();

  // Remove all objects from the world model.
  void Clear();

 private:
  absl::flat_hash_map<int, RigidBody> rigidbody_id_to_rigidbody_;
  absl::flat_hash_map<int, Landmark> landmark_id_to_landmark_;

};

} // namespace calico

#endif // CALICO_WORLD_MODEL_H_
