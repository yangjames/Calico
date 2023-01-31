#include "calico/sensors/camera_models.h"

namespace calico::sensors {

std::unique_ptr<CameraModel> CameraModel::Create(
    CameraIntrinsicsModel camera_model) {
  switch (camera_model) {
    case CameraIntrinsicsModel::kOpenCv5: {
      return std::make_unique<OpenCv5Model>();
    }
    default: {
      return std::move(std::unique_ptr<CameraModel>{});
    }
  }
}

} // namespace calico::sensors
