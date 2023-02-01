#ifndef CALICO_OPTIMIZATION_UTILS_H_
#define CALICO_OPTIMIZATION_UTILS_H_

#include "calico/typedefs.h"
#include "ceres/ceres.h"

namespace calico::utils {

// Convenience function for adding a Pose3 type to a ceres problem with
// correct parameterization and manifold specification.
int AddPoseToProblem(ceres::Problem& problem, Pose3& pose) {
  int num_parameters_added = 0;
  problem.AddParameterBlock(pose.translation().data(),
                            pose.translation().size());
  num_parameters_added += pose.translation().size();
  ceres::Manifold* quat_manifold =  new ceres::EigenQuaternionManifold;
  problem.AddParameterBlock(pose.rotation().coeffs().data(),
                            pose.rotation().coeffs().size(), quat_manifold);
  num_parameters_added += pose.rotation().coeffs().size();
  return num_parameters_added;
}

void SetPoseConstantInProblem(ceres::Problem& problem, Pose3& pose) {
  problem.SetParameterBlockConstant(pose.translation().data());
  problem.SetParameterBlockConstant(pose.rotation().coeffs().data());
}

} // namespace calico

#endif // CALICO_OPTIMIZATION_UTILS_H_
