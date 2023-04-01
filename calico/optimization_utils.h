#ifndef CALICO_OPTIMIZATION_UTILS_H_
#define CALICO_OPTIMIZATION_UTILS_H_

#include "calico/typedefs.h"
#include "ceres/ceres.h"


namespace calico::utils {

enum class LossFunctionType : int {
  kNone,
  kHuber,
  kCauchy,
};

inline ceres::LossFunction* CreateLossFunction(
    LossFunctionType loss, double scale) {
  switch (loss) {
    case LossFunctionType::kNone: {
      return nullptr;
    }
    case LossFunctionType::kHuber: {
      return new ceres::HuberLoss(scale);
    }
    case LossFunctionType::kCauchy: {
      return new ceres::CauchyLoss(scale);
    }
    default: {
      return nullptr;
    }
  }
}

// Convenience function for adding a Pose3d type to a ceres problem with
// correct parameterization and manifold specification.
inline int AddPoseToProblem(ceres::Problem& problem, Pose3d& pose) {
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

inline void SetPoseConstantInProblem(ceres::Problem& problem, Pose3d& pose) {
  problem.SetParameterBlockConstant(pose.translation().data());
  problem.SetParameterBlockConstant(pose.rotation().coeffs().data());
}

} // namespace calico

#endif // CALICO_OPTIMIZATION_UTILS_H_
