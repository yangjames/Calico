#ifndef CALICO_TEST_UTILS_H_
#define CALICO_TEST_UTILS_H_

#include "absl/container/flat_hash_map.h"
#include "calico/typedefs.h"
#include "Eigen/Dense"


namespace calico {

class DefaultSyntheticTest {
 public:
  DefaultSyntheticTest() {
    // Create position and orientation setpoints through which we will smoothly
    // interpolate.
    const Eigen::Quaterniond q_world_sensorrig0(
        Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
    const Eigen::Vector3d t_world_sensorrig0 =
      Eigen::Vector3d(0.0, 0.0, 1.0);
    std::vector<double> angle_displacements =
        {0.0, kAngleAmplitude, 0.0, -kAngleAmplitude, 0.0};
    std::vector<double> position_displacements =
      {0.0, kPosAmplitude, 0.0, -kPosAmplitude, 0.0};

    // Excitation per axis.
    std::vector<double> interpolation_times(kNumSamplesPerSegment);
    const double dt_interpolation =
        1.0 / static_cast<double>(kNumSamplesPerSegment);
    const double dt_actual = dt_interpolation * kSegmentDuration;
    for (int i = 0; i < kNumSamplesPerSegment; ++i) {
      const double temp = dt_interpolation * i;
      interpolation_times[i] = (std::sin(temp * M_PI - M_PI_2) + 1.0) / 2.0;
    }
    double current_time = 0.0;
    for (const Eigen::Vector3d& axis : std::vector{
             Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(),
                 Eigen::Vector3d::UnitZ()}) {
      // Angular excitation.
      for (int i = 1; i < angle_displacements.size(); ++i) {
        const double& theta0 = angle_displacements[i - 1];
        const double& theta1 = angle_displacements[i];
        const double dtheta = theta1 - theta0;
        for (const double& interp_time : interpolation_times) {
          const double theta =    dtheta * interp_time + theta0;
          const Eigen::Quaterniond q_sensorrig0_sensorrig(
              Eigen::AngleAxisd(theta, axis));
          const Eigen::Quaterniond q_world_sensorrig =
              q_world_sensorrig0 * q_sensorrig0_sensorrig;
          trajectory_world_sensorrig_[current_time] =
              Pose3(q_world_sensorrig, t_world_sensorrig0);
          current_time += dt_actual;
        }
      }
      // Linear excitation.
      for (int i = 1; i < position_displacements.size(); ++i) {
        const double& pos0 = position_displacements[i - 1];
        const double& pos1 = position_displacements[i];
        const double dpos = pos1 - pos0;
        for (const double& interp_time : interpolation_times) {
          const double pos = dpos * interp_time + pos0;
          trajectory_world_sensorrig_[current_time] =
              Pose3(q_world_sensorrig0, axis * pos + t_world_sensorrig0);
          current_time += dt_actual;
        }
      }
    }
    for (const auto& [stamp, _] : trajectory_world_sensorrig_) {
      trajectory_key_values_.push_back(stamp);
    }
    std::sort(trajectory_key_values_.begin(), trajectory_key_values_.end());
  }

  const absl::flat_hash_map<double, Pose3>& TrajectoryAsMap() const {
    return trajectory_world_sensorrig_;
  }
  const std::vector<double>& TrajectoryMapKeys() const {
    return trajectory_key_values_;
  }

 private:
  absl::flat_hash_map<double, Pose3> trajectory_world_sensorrig_;
  std::vector<double> trajectory_key_values_;

  static constexpr double kDeg2Rad = M_PI / 180.0;
  // Sensor rig trajectory specs.
  static constexpr int kNumSamplesPerSegment = 10;
  static constexpr double kPosAmplitude = 0.5;
  static constexpr double kAngleAmplitude = 30 * kDeg2Rad;
  static constexpr double kSegmentDuration = 1.0;

};
} // namespace

#endif // CALICO_TEST_UTILS_H_
