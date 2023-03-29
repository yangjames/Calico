#ifndef CALICO_CHART_DETECTORS_APRILGRID_DETECTOR_H_
#define CALICO_CHART_DETECTORS_APRILGRID_DETECTOR_H_


#include <memory>
#include <unordered_map>

#include "apriltags/TagDetector.h"
#include "calico/world_model.h"
#include "opencv2/opencv.hpp"


namespace calico::chart_detectors {

struct AprilGridConfig {
  int tagCols;  // Number of apriltags per column.
  int tagRows;  // Number of apriltags per row.
  double tagSize;  // Size of each apriltag edge in meters.
  double tagSpacing;  // Ratio of space between tags to their size.
                      // For example:
                      //   space between tags=0.5m, tagSize=2m -> tagSpacing=0.25
};

class AprilGridDetector {
 public:

  // Users have two options for constructing a detector. First is passing a
  // configuration filename directly to the constructor. Second is by passing
  // a hand-made configuration. The former invokes the latter.
  AprilGridDetector(const std::string& config_filename);
  AprilGridDetector(const AprilGridConfig& config);

  // Run detection on an image. Returns a map containing each corner id and their
  // corresponding pixel locations.
  std::unordered_map<int, Eigen::Vector2d> Detect(const cv::Mat& image) const;

  // Getter for this detector's rigid body definition.
  RigidBody GetRigidBodyDefinition() const;


 private:
  AprilGridConfig config_;
  std::unordered_map<int, Eigen::Vector3d> model_definition_;
  std::unique_ptr<AprilTags::TagDetector> detector_;

  // Convenience function for setting up the apriltag detector.
  void SetupDetector();
};
} // namespace calico::chart_detectors


#endif // CALICO_CHART_DETECTORS_APRILGRID_DETECTOR_H_
