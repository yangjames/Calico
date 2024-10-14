#ifndef CALICO_WORLD_MODEL_H_
#define CALICO_WORLD_MODEL_H_

#include <memory>

#include "calico/typedefs.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "ceres/problem.h"
#include "Eigen/Dense"


namespace calico {

/// Default frame id for a landmark (i.e. World frame).
constexpr int kLandmarkFrameId = -1;

/// `Landmark` object which represents a unique 3d point resolved in the world
/// frame \f$\mathbf{t}^w_{wx_i}\f$. Landmark `id` must be unique. For
/// optimization problems, set `world_point_is_constant` to true if it is a
/// free parameter that needs to be estimated, false otherwise.
struct Landmark {
  /// \brief This landmark's location in the world frame. \f$\mathbf{t}^w_{wx}\f$
  Eigen::Vector3d point;
  /// \brief Unique id for this landmark.
  int id;
  /// \brief Flag for keeping this landmark's point constant or as a free
  /// parameter.
  bool point_is_constant;
};

/// `Rigidbody` object which represents a constellation of points on a rigid body.
/// \n\n
/// `model_definition` contains 3d points that make up the rigid body
/// resolved in the rigid body frame \f$\mathbf{t}^b_{bx_i}\f$. For example, this
/// could be planar points on a fiducial target in a calibration problem relative
/// to some origin defined on the target.\n\n
/// `T_world_rigidbody` is the rigid transform TO the world FROM the rigidbody
/// frame such that the model defintion points can be resolved in the world frame
/// by the following:
/// \f[
///   \mathbf{T}^w_b = \left[\begin{matrix}
///      \mathbf{R}^w_b & \mathbf{t}^w_{wb}\\
///      \mathbf{0}&1
///   \end{matrix}\right]\\
///   \mathbf{t}^w_{wx_i} = \mathbf{t}^w_{wb} + \mathbf{R}^w_b\mathbf{t}^b_{bx_i}
/// \f]
/// \n\n
/// Both the model definition and its world pose can be
/// set as either constant or as free parameters in an optimization problem via
/// the flags `world_pose_is_constant` and `model_definition_is_constant`.\n\n
/// TODO(yangjames): Replace std::unordered_map with absl::flat_hash_map.
struct RigidBody {
  /// \brief Maps feature id to its 3d point definition resolved in the rigid body
  /// frame, \f$\mathbf{t}^b_{bx_i}\f$. Each feature needs a unique integer id.
  /// The numerical value of the id does not matter.
  std::unordered_map<int, Eigen::Vector3d> model_definition;
  /// \brief FROM world TO rigidbody transform, \f$\mathbf{T}^w_b\f$
  Pose3d T_world_rigidbody;
  /// \brief Id of this rigid body. When adding rigid bodies to the
  /// BatchOptimizer, id's for all rigid bodies must be unique.
  int id;
  /// \brief Flag for keeping this rigid body's world-from-rigidbody pose
  /// constant or as a free parameter.
  bool world_pose_is_constant;
  /// \brief Flag for keeping the model definition constant or as free
  /// parameters.
  bool model_definition_is_constant;
};

/// `WorldModel` manages all `Landmark` and `RigidBody` objects in an
/// optimization problem.
class WorldModel {
 public:
  /// Default magnitude of gravity vector. Gravity resolved in the standard
  /// inertial frame is defined as
  /// \f$\mathbf{g}^w=\left[0, 0, -9.80665\right]^T\f$
  static constexpr double kGravityDefaultZ = -9.80665;

  WorldModel();
  ~WorldModel();

  /// Add a `Landmark` object to the world model. Returns an InvalidArgument status
  /// code if the landmark id is not unique.
  absl::Status AddLandmark(Landmark* landmark, bool take_ownership = true);

  /// Add a `RigidBody` object to the world model. Returns an InvalidArgument
  /// status code if the rigid body id is not unique.
  absl::Status AddRigidBody(RigidBody* rigidbody, bool take_ownership = true);

  /// Add internal parameters to a ceres problem. Any internal parameters set to
  /// constant are marked as such in the problem. Returns the total number of
  /// parameters added to the problem.
  int AddParametersToProblem(ceres::Problem& problem);

  /// Accessor for landmarks.
  const absl::flat_hash_map<int, std::unique_ptr<Landmark>>& landmarks() const;
  absl::flat_hash_map<int, std::unique_ptr<Landmark>>& landmarks();

  /// Accessor for rigid bodies.
  const absl::flat_hash_map<int, std::unique_ptr<RigidBody>>& rigidbodies() const;
  absl::flat_hash_map<int, std::unique_ptr<RigidBody>>& rigidbodies();

  /// Enable flag for gravity estimation.
  void EnableGravityEstimation(bool enable);

  /// Accessor for gravity vector.
  const Eigen::Vector3d& gravity() const;
  Eigen::Vector3d& gravity();

  /// Setter for gravity for easy pybind integration.
  void SetGravity(const Eigen::Vector3d& gravity);

  /// Getter for gravity for easy pybind integration.
  const Eigen::Vector3d& GetGravity() const;


  /// Get the number of landmarks currently in this world model.
  int NumberOfLandmarks() const;

  /// Get the number of rigid bodies currently in this world model.
  int NumberOfRigidBodies() const;

  /// Remove all `Landmark` objects from the world model.
  void ClearLandmarks();

  /// Remove all `RigidBody` objects from the world model.
  void ClearRigidBodies();

  /// Remove all objects from the world model.
  void Clear();

 private:
  absl::flat_hash_map<int, bool> rigidbody_id_to_own_rigidbody_;
  absl::flat_hash_map<int, bool> landmark_id_to_own_landmark_;
  absl::flat_hash_map<int, std::unique_ptr<RigidBody>> rigidbody_id_to_rigidbody_;
  absl::flat_hash_map<int, std::unique_ptr<Landmark>> landmark_id_to_landmark_;

  Eigen::Vector3d gravity_;
  bool gravity_enabled_;
};

} // namespace calico

#endif // CALICO_WORLD_MODEL_H_
