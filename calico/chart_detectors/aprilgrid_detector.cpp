#include "calico/chart_detectors/aprilgrid_detector.h"

#include "apriltags/Tag16h5.h"
#include "apriltags/Tag25h7.h"
#include "apriltags/Tag25h9.h"
#include "apriltags/Tag36h9.h"
#include "apriltags/Tag36h11.h"
#include "yaml-cpp/yaml.h"


namespace calico::chart_detectors {

AprilGridDetector::AprilGridDetector(const std::string& config_filename) {
  YAML::Node config_yaml = YAML::LoadFile(config_filename);
  config_ = AprilGridConfig {
      .tagCols = config_yaml["tagCols"].as<int>(),
      .tagRows = config_yaml["tagRows"].as<int>(),
      .tagSize = config_yaml["tagSize"].as<double>(),
      .tagSpacing = config_yaml["tagSpacing"].as<double>(),
  };
  SetupDetector();
}
AprilGridDetector::AprilGridDetector(const AprilGridConfig& config) :
  config_(config) {
  SetupDetector();
}

void AprilGridDetector::SetupDetector() {
  // None of these parameters really need to change. Kalibr's aprilgrid generator
  // doesn't explicitly support other tag families and requires a 2-bit black
  // border.
  const AprilTags::TagCodes tag_codes = AprilTags::tagCodes36h11;
  detector_ = std::unique_ptr<AprilTags::TagDetector>(
      new AprilTags::TagDetector(tag_codes));
  const double& tag_width = config_.tagSize;
  const double tag_width_with_spacing = tag_width * (1.0 + config_.tagSpacing);
  for (int row = 0; row < config_.tagRows; ++row) {
    for (int col = 0; col < config_.tagCols; ++col) {
      const double tag_origin_x = tag_width_with_spacing * col;
      const double tag_origin_y = tag_width_with_spacing * row;
      const int tag_number = row * config_.tagCols + col;
      for (int k = 0; k < 4; ++k) {
        const double corner_x = tag_origin_x + tag_width * (k == 1 || k == 2);
        const double corner_y = tag_origin_y + tag_width * (k == 2 || k == 3);
        const int feature_id = tag_number * 4 + k;
        model_definition_[feature_id] = Eigen::Vector3d(corner_x, corner_y, 0.0);
      }
    }
  }
}

std::unordered_map<int, Eigen::Vector2d>
AprilGridDetector::Detect(const cv::Mat& image) const {
  const auto detected_tags = detector_->extractTags(image);
  std::unordered_map<int, Eigen::Vector2d> detections;
  for (const auto detected_tag : detected_tags) {
    const int tag_id = detected_tag.id;
    for (int i = 0; i < 4; ++i) {
      const int feature_id = 4 * tag_id + i;
      const auto point = detected_tag.p[i];
      detections.insert(
          {feature_id, Eigen::Vector2d(point.first, point.second)});
    }
  }
  return detections;
}

RigidBody AprilGridDetector::GetRigidBodyDefinition() const {
  const RigidBody chart {
    .model_definition = model_definition_,
    .id = 0,
  };
  return chart;
}

} // namespace calico::chart_detectors
