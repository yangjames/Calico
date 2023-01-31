#ifndef CALICO_OPTIMIZATION_UTILS_H_
#define CALICO_OPTIMIZATION_UTILS_H_

#include "calico/typedefs.h"
#include "ceres/ceres.h"

namespace calico::utils {

// Convenience function for adding a Pose3 type to a ceres problem with
// correct parameterization and manifold specification.
  void AddPoseToProblem(ceres::Problem& problem, Pose3& pose) {
  problem.AddParameterBlock(pose.translation().data(),
                            pose.translation().size());
  ceres::Manifold* quat_manifold =  new ceres::EigenQuaternionManifold;
  problem.AddParameterBlock(pose.rotation().coeffs().data(),
                            pose.rotation().coeffs().size(), quat_manifold);
}

void SetPoseConstantInProblem(ceres::Problem& problem, Pose3& pose) {
  problem.SetParameterBlockConstant(pose.translation().data());
  problem.SetParameterBlockConstant(pose.rotation().coeffs().data());
}

} // namespace calico

#endif // CALICO_OPTIMIZATION_UTILS_H_
