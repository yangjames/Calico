#include "calico/sensors/accelerometer_models.h"

namespace calico::sensors {

std::unique_ptr<AccelerometerModel> AccelerometerModel::Create(
    AccelerometerIntrinsicsModel accelerometer_model) {
  switch (accelerometer_model) {
    case AccelerometerIntrinsicsModel::kAccelerometerScaleOnly: {
      return std::make_unique<AccelerometerScaleOnlyModel>();
    }
    case AccelerometerIntrinsicsModel::kAccelerometerScaleAndBias: {
      return std::make_unique<AccelerometerScaleAndBiasModel>();
    }
    default: {
      return std::move(std::unique_ptr<AccelerometerModel>{});
    }
  }
}

} // namespace calico::sensors
