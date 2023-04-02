#include "calico/sensors/gyroscope_models.h"

namespace calico::sensors {

std::unique_ptr<GyroscopeModel> GyroscopeModel::Create(
    GyroscopeIntrinsicsModel gyroscope_model) {
  switch (gyroscope_model) {
    case GyroscopeIntrinsicsModel::kGyroscopeScaleOnly: {
      return std::make_unique<GyroscopeScaleOnlyModel>();
    }
    case GyroscopeIntrinsicsModel::kGyroscopeScaleAndBias: {
      return std::make_unique<GyroscopeScaleAndBiasModel>();
    }
    case GyroscopeIntrinsicsModel::kGyroscopeVectorNav: {
      return std::make_unique<GyroscopeVectorNavModel>();
    }
    default: {
      return std::move(std::unique_ptr<GyroscopeModel>{});
    }
  }
}

} // namespace calico::sensors
