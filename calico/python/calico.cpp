#include "calico/sensors/accelerometer.h"
#include "calico/sensors/camera.h"
#include "calico/sensors/gyroscope.h"
#include "calico/typedefs.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"


namespace calico::sensors {

PYBIND11_MODULE(calico, m) {
  m.doc() = "Calico";
  namespace py = pybind11;

  // absl Status
  py::enum_<absl::StatusCode>(m, "StatusCode")
    .value("kOk", absl::StatusCode::kOk)
    .value("kInvalidArgument", absl::StatusCode::kInvalidArgument);

  py::class_<absl::Status>(m, "Status")
    .def(py::init<>())
    .def("ok", &absl::Status::ok)
    .def("code", &absl::Status::code)
    .def("message", [](absl::Status& self) {
      return std::string(self.message());
    });

  // Typedefs.
  py::class_<Pose3d>(m, "Pose3d")
    .def(py::init<>())
    .def(py::init<Pose3d const &>())
    .def_property("rotation", &Pose3d::GetRotation, &Pose3d::SetRotation)
    .def_property("translation",
                  &Pose3d::GetTranslation, &Pose3d::SetTranslation);

  // Accelerometer class.
  py::enum_<AccelerometerIntrinsicsModel>(m, "AccelerometerIntrinsicsModel")
    .value("kNone", AccelerometerIntrinsicsModel::kNone)
    .value("kAccelerometerScaleOnly",
           AccelerometerIntrinsicsModel::kAccelerometerScaleOnly)
    .value("kAccelerometerScaleAndBias",
           AccelerometerIntrinsicsModel::kAccelerometerScaleAndBias);

  py::class_<AccelerometerObservationId>(m, "AccelerometerObservationId")
    .def(py::init<>())
    .def(py::init<AccelerometerObservationId const &>())
    .def_readwrite("stamp", &AccelerometerObservationId::stamp)
    .def_readwrite("sequence", &AccelerometerObservationId::sequence);

  py::class_<AccelerometerMeasurement>(m, "AccelerometerMeasurement")
    .def(py::init<>())
    .def_readwrite("measurement", &AccelerometerMeasurement::measurement)
    .def_readwrite("id", &AccelerometerMeasurement::id);

  py::class_<Accelerometer>(m, "Accelerometer")
    .def(py::init<>())
    .def("SetName", &Accelerometer::SetName)
    .def("GetName", &Accelerometer::GetName)
    .def("SetExtrinsics", &Accelerometer::SetExtrinsics)
    .def("GetExtrinsics", &Accelerometer::GetExtrinsics)
    .def("SetIntrinsics", &Accelerometer::SetIntrinsics)
    .def("GetIntrinsics", &Accelerometer::GetIntrinsics)
    .def("SetLatency", &Accelerometer::SetLatency)
    .def("GetLatency", &Accelerometer::GetLatency)
    .def("EnableExtrinsicsEstimation", &Accelerometer::EnableExtrinsicsEstimation)
    .def("EnableIntrinsicsEstimation", &Accelerometer::EnableExtrinsicsEstimation)
    .def("EnableLatencyEstimation", &Accelerometer::EnableLatencyEstimation)
    .def("SetModel", &Accelerometer::SetModel)
    .def("GetModel", &Accelerometer::GetModel)
    .def("AddMeasurement", &Accelerometer::AddMeasurement)
    .def("AddMeasurements", &Accelerometer::AddMeasurements);

  // Gyroscope class.
  py::enum_<GyroscopeIntrinsicsModel>(m, "GyroscopeIntrinsicsModel")
    .value("kNone", GyroscopeIntrinsicsModel::kNone)
    .value("kGyroscopeScaleOnly",
           GyroscopeIntrinsicsModel::kGyroscopeScaleOnly)
    .value("kGyroscopeScaleAndBias",
           GyroscopeIntrinsicsModel::kGyroscopeScaleAndBias);

  py::class_<GyroscopeObservationId>(m, "GyroscopeObservationId")
    .def(py::init<>())
    .def(py::init<GyroscopeObservationId const &>())
    .def_readwrite("stamp", &GyroscopeObservationId::stamp)
    .def_readwrite("sequence", &GyroscopeObservationId::sequence);

  py::class_<GyroscopeMeasurement>(m, "GyroscopeMeasurement")
    .def(py::init<>())
    .def_readwrite("measurement", &GyroscopeMeasurement::measurement)
    .def_readwrite("id", &GyroscopeMeasurement::id);

  py::class_<Gyroscope>(m, "Gyroscope")
    .def(py::init<>())
    .def("SetName", &Gyroscope::SetName)
    .def("GetName", &Gyroscope::GetName)
    .def("SetExtrinsics", &Gyroscope::SetExtrinsics)
    .def("GetExtrinsics", &Gyroscope::GetExtrinsics)
    .def("SetIntrinsics", &Gyroscope::SetIntrinsics)
    .def("GetIntrinsics", &Gyroscope::GetIntrinsics)
    .def("SetLatency", &Gyroscope::SetLatency)
    .def("GetLatency", &Gyroscope::GetLatency)
    .def("EnableExtrinsicsEstimation", &Gyroscope::EnableExtrinsicsEstimation)
    .def("EnableIntrinsicsEstimation", &Gyroscope::EnableExtrinsicsEstimation)
    .def("EnableLatencyEstimation", &Gyroscope::EnableLatencyEstimation)
    .def("SetModel", &Gyroscope::SetModel)
    .def("GetModel", &Gyroscope::GetModel)
    .def("AddMeasurement", &Gyroscope::AddMeasurement)
    .def("AddMeasurements", &Gyroscope::AddMeasurements);


  // Camera class.
  py::enum_<CameraIntrinsicsModel>(m, "CameraIntrinsicsModel")
    .value("kNone", CameraIntrinsicsModel::kNone)
    .value("kOpenCv5", CameraIntrinsicsModel::kOpenCv5);

  py::class_<CameraObservationId>(m, "CameraObservationId")
    .def(py::init<>())
    .def(py::init<CameraObservationId const &>())
    .def_readwrite("stamp", &CameraObservationId::stamp)
    .def_readwrite("image_id", &CameraObservationId::image_id)
    .def_readwrite("model_id", &CameraObservationId::model_id)
    .def_readwrite("feature_id", &CameraObservationId::feature_id);

  py::class_<CameraMeasurement>(m, "CameraMeasurement")
    .def(py::init<>())
    .def_readwrite("pixel", &CameraMeasurement::pixel)
    .def_readwrite("id", &CameraMeasurement::id);

  py::class_<Camera>(m, "Camera")
    .def(py::init<>())
    .def("SetName", &Camera::SetName)
    .def("GetName", &Camera::GetName)
    .def("SetExtrinsics", &Camera::SetExtrinsics)
    .def("GetExtrinsics", &Camera::GetExtrinsics)
    .def("SetIntrinsics", &Camera::SetIntrinsics)
    .def("GetIntrinsics", &Camera::GetIntrinsics)
    .def("SetLatency", &Camera::SetLatency)
    .def("GetLatency", &Camera::GetLatency)
    .def("EnableExtrinsicsEstimation", &Camera::EnableExtrinsicsEstimation)
    .def("EnableIntrinsicsEstimation", &Camera::EnableExtrinsicsEstimation)
    .def("EnableLatencyEstimation", &Camera::EnableLatencyEstimation)
    .def("SetModel", &Camera::SetModel)
    .def("GetModel", &Camera::GetModel)
    .def("AddMeasurement", &Camera::AddMeasurement)
    .def("AddMeasurements", &Camera::AddMeasurements);

  // Trajectory class.
  py::class_<Trajectory>(m, "Trajectory")
    .def(py::init<>())
    .def("AddPoses", py::overload_cast<
         const std::unordered_map<double, Pose3d>&>(&Trajectory::AddPoses));

  // World model class.
  py::class_<Landmark>(m, "Landmark")
    .def(py::init<>())
    .def_readwrite("point", &Landmark::point)
    .def_readwrite("id", &Landmark::id)
    .def_readwrite("point_is_constant", &Landmark::point_is_constant);

  py::class_<RigidBody>(m, "RigidBody")
    .def(py::init<>());

  py::class_<WorldModel>(m, "WorldModel")
    .def(py::init<>())
    ;
  
}

} // namespace calico::senosrs
