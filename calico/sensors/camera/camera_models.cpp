#include "calico/sensors/camera/camera_models.h"

namespace calico::sensors::camera {

absl::StatusOr<std::unique_ptr<CameraModel>> CameraModel::Create(
    const CameraIntrinsicsModel camera_model) {
  switch (camera_model) {
    case CameraIntrinsicsModel::kOpenCv5: {
      return std::make_unique<OpenCv5Model>();
    }
    default: {
      return absl::UnimplementedError(
          absl::StrCat("Camera model with ID ", camera_model));
    }
  }
}

} // namespace calico::sensors::camera
