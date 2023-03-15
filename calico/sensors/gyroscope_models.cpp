#include "calico/sensors/gyroscope_models.h"

namespace calico::sensors {

std::unique_ptr<GyroscopeModel> GyroscopeModel::Create(
    GyroscopeIntrinsicsModel gyroscope_model) {
  switch (gyroscope_model) {
    case GyroscopeIntrinsicsModel::kScaleOnly: {
      return std::make_unique<ScaleOnlyModel>();
    }
    case GyroscopeIntrinsicsModel::kScaleAndBias: {
      return std::make_unique<ScaleAndBiasModel>();
    }
    default: {
      return std::move(std::unique_ptr<GyroscopeModel>{});
    }
  }
}

} // namespace calico::sensors
