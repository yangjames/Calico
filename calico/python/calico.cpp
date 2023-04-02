#include "calico/batch_optimizer.h"
#include "calico/chart_detectors/aprilgrid_detector.h"
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


PYBIND11_MODULE(_calico, m) {
  m.doc() = "Calico";
  namespace py = pybind11;
  using namespace calico;
  using namespace calico::chart_detectors;
  using namespace calico::sensors;
  using namespace calico::utils;

  // absl::Status
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

  // Loss function types.
  py::enum_<LossFunctionType>(m, "LossFunctionType")
    .value("kNone", LossFunctionType::kNone)
    .value("kHuber", LossFunctionType::kHuber)
    .value("kCauchy", LossFunctionType::kCauchy);

  // Base sensor class.
  py::class_<Sensor, std::shared_ptr<Sensor>>(m, "Sensor");

  // Accelerometer class.
  py::enum_<AccelerometerIntrinsicsModel>(m, "AccelerometerIntrinsicsModel")
    .value("kNone", AccelerometerIntrinsicsModel::kNone)
    .value("kAccelerometerScaleOnly",
           AccelerometerIntrinsicsModel::kAccelerometerScaleOnly)
    .value("kAccelerometerScaleAndBias",
           AccelerometerIntrinsicsModel::kAccelerometerScaleAndBias)
    .value("kAccelerometerVectorNav",
           AccelerometerIntrinsicsModel::kAccelerometerVectorNav);

  py::class_<AccelerometerObservationId>(m, "AccelerometerObservationId")
    .def(py::init<>())
    .def(py::init<AccelerometerObservationId const &>())
    .def_readwrite("stamp", &AccelerometerObservationId::stamp)
    .def_readwrite("sequence", &AccelerometerObservationId::sequence);

  py::class_<AccelerometerMeasurement>(m, "AccelerometerMeasurement")
    .def(py::init<>())
    .def_readwrite("measurement", &AccelerometerMeasurement::measurement)
    .def_readwrite("id", &AccelerometerMeasurement::id);

  py::class_<Accelerometer, std::shared_ptr<Accelerometer>, Sensor>
      (m, "Accelerometer")
    .def(py::init<>())
    .def("SetName", &Accelerometer::SetName)
    .def("GetName", &Accelerometer::GetName)
    .def("SetExtrinsics", &Accelerometer::SetExtrinsics)
    .def("GetExtrinsics", &Accelerometer::GetExtrinsics)
    .def("SetIntrinsics",
         [](Accelerometer& self, const Eigen::VectorXd& intrinsics) {
           const auto status = self.SetIntrinsics(intrinsics);
           if (!status.ok()) {
             throw std::runtime_error(
                 std::string("Error: ") + std::string(status.message()));
           }
         })
    .def("GetIntrinsics", &Accelerometer::GetIntrinsics)
    .def("SetLatency", &Accelerometer::SetLatency)
    .def("GetLatency", &Accelerometer::GetLatency)
    .def("EnableExtrinsicsEstimation", &Accelerometer::EnableExtrinsicsEstimation)
    .def("EnableIntrinsicsEstimation", &Accelerometer::EnableIntrinsicsEstimation)
    .def("EnableLatencyEstimation", &Accelerometer::EnableLatencyEstimation)
    .def("SetModel", &Accelerometer::SetModel)
    .def("GetModel", &Accelerometer::GetModel)
    .def("SetLossFunction", &Accelerometer::SetLossFunction)
    .def("AddMeasurement", &Accelerometer::AddMeasurement)
    .def("AddMeasurements", &Accelerometer::AddMeasurements);

  // Gyroscope class.
  py::enum_<GyroscopeIntrinsicsModel>(m, "GyroscopeIntrinsicsModel")
    .value("kNone", GyroscopeIntrinsicsModel::kNone)
    .value("kGyroscopeScaleOnly",
           GyroscopeIntrinsicsModel::kGyroscopeScaleOnly)
    .value("kGyroscopeScaleAndBias",
           GyroscopeIntrinsicsModel::kGyroscopeScaleAndBias)
    .value("kGyroscopeVectorNav",
           GyroscopeIntrinsicsModel::kGyroscopeVectorNav);

  py::class_<GyroscopeObservationId>(m, "GyroscopeObservationId")
    .def(py::init<>())
    .def(py::init<GyroscopeObservationId const &>())
    .def_readwrite("stamp", &GyroscopeObservationId::stamp)
    .def_readwrite("sequence", &GyroscopeObservationId::sequence);

  py::class_<GyroscopeMeasurement>(m, "GyroscopeMeasurement")
    .def(py::init<>())
    .def_readwrite("measurement", &GyroscopeMeasurement::measurement)
    .def_readwrite("id", &GyroscopeMeasurement::id);

  py::class_<Gyroscope, std::shared_ptr<Gyroscope>, Sensor>(m, "Gyroscope")
    .def(py::init<>())
    .def("SetName", &Gyroscope::SetName)
    .def("GetName", &Gyroscope::GetName)
    .def("SetExtrinsics", &Gyroscope::SetExtrinsics)
    .def("GetExtrinsics", &Gyroscope::GetExtrinsics)
    .def("SetIntrinsics",
         [](Gyroscope& self, const Eigen::VectorXd& intrinsics) {
           const auto status = self.SetIntrinsics(intrinsics);
           if (!status.ok()) {
             throw std::runtime_error(
                 std::string("Error: ") + std::string(status.message()));
           }
         })
    .def("GetIntrinsics", &Gyroscope::GetIntrinsics)
    .def("SetLatency", &Gyroscope::SetLatency)
    .def("GetLatency", &Gyroscope::GetLatency)
    .def("EnableExtrinsicsEstimation", &Gyroscope::EnableExtrinsicsEstimation)
    .def("EnableIntrinsicsEstimation", &Gyroscope::EnableIntrinsicsEstimation)
    .def("EnableLatencyEstimation", &Gyroscope::EnableLatencyEstimation)
    .def("SetModel", &Gyroscope::SetModel)
    .def("GetModel", &Gyroscope::GetModel)
    .def("SetLossFunction", &Gyroscope::SetLossFunction)
    .def("AddMeasurement", &Gyroscope::AddMeasurement)
    .def("AddMeasurements", &Gyroscope::AddMeasurements);


  // Camera class.
  py::enum_<CameraIntrinsicsModel>(m, "CameraIntrinsicsModel")
    .value("kNone", CameraIntrinsicsModel::kNone)
    .value("kOpenCv5", CameraIntrinsicsModel::kOpenCv5)
    .value("kKannalaBrandt", CameraIntrinsicsModel::kKannalaBrandt);

  py::class_<CameraObservationId>(m, "CameraObservationId")
    .def(py::init<>())
    .def(py::init<CameraObservationId const &>())
    .def("__hash__", absl::Hash<CameraObservationId>())
    .def("__str__",
         [](const CameraObservationId& self) {
           return "stamp: " + std::to_string(self.stamp) + ", image_id: " +
             std::to_string(self.image_id) + ", model_id: " +
             std::to_string(self.model_id) + ", feature_id: " +
             std::to_string(self.feature_id);
         })
    .def_readwrite("stamp", &CameraObservationId::stamp)
    .def_readwrite("image_id", &CameraObservationId::image_id)
    .def_readwrite("model_id", &CameraObservationId::model_id)
    .def_readwrite("feature_id", &CameraObservationId::feature_id);

  py::class_<CameraMeasurement>(m, "CameraMeasurement")
    .def(py::init<>())
    .def_readwrite("pixel", &CameraMeasurement::pixel)
    .def_readwrite("id", &CameraMeasurement::id);

  py::class_<Camera, std::shared_ptr<Camera>, Sensor>(m, "Camera")
    .def(py::init<>())
    .def("SetName", &Camera::SetName)
    .def("GetName", &Camera::GetName)
    .def("SetExtrinsics", &Camera::SetExtrinsics)
    .def("GetExtrinsics", &Camera::GetExtrinsics)
    .def("SetIntrinsics",
         [](Camera& self, const Eigen::VectorXd& intrinsics) {
           const auto status = self.SetIntrinsics(intrinsics);
           if (!status.ok()) {
             throw std::runtime_error(
                 std::string("Error: ") + std::string(status.message()));
           }
         })
    .def("GetIntrinsics", &Camera::GetIntrinsics)
    .def("SetLatency", &Camera::SetLatency)
    .def("GetLatency", &Camera::GetLatency)
    .def("EnableExtrinsicsEstimation", &Camera::EnableExtrinsicsEstimation)
    .def("EnableIntrinsicsEstimation", &Camera::EnableIntrinsicsEstimation)
    .def("EnableLatencyEstimation", &Camera::EnableLatencyEstimation)
    .def("SetModel", &Camera::SetModel)
    .def("GetModel", &Camera::GetModel)
    .def("SetLossFunction", &Camera::SetLossFunction)
    .def("AddMeasurement", &Camera::AddMeasurement)
    .def("AddMeasurements", &Camera::AddMeasurements)
    .def("GetMeasurementResidualPairs",
         [](const Camera& self) {
           const auto pairs = self.GetMeasurementResidualPairs();
           if (!pairs.status().ok()) {
             throw std::runtime_error(
                 std::string("Error: ") + std::string(pairs.status().message()));
           }
           return pairs.value();
         })
    .def("GetMeasurementIdToMeasurement",
         [](Camera& self) {
           std::unordered_map<CameraObservationId, CameraMeasurement,
                              absl::Hash<CameraObservationId>>
               id_to_measurement;
           for (const auto [id, measurement] :
                  self.GetMeasurementIdToMeasurement()) {
             id_to_measurement[id] = measurement;
           }
           return id_to_measurement;
         })
    .def("MarkOutlierById",
         [](Camera& self, CameraObservationId id) {
           const auto status = self.MarkOutlierById(id);
           if (!status.ok()) {
             throw std::runtime_error(
                 std::string("Error: ") + std::string(status.message()));
           }
         })
    .def("MarkOutliersById",
         [](Camera& self, const std::vector<CameraObservationId>& ids) {
           const auto status = self.MarkOutliersById(ids);
           if (!status.ok()) {
             throw std::runtime_error(
                 std::string("Error: ") + std::string(status.message()));
           }
         });

  // Trajectory class.
  py::class_<Trajectory, std::shared_ptr<Trajectory>>(m, "Trajectory")
    .def(py::init<>())
    .def("FitSpline",
         [](Trajectory& self, const std::unordered_map<double, Pose3d>& poses,
            double knot_frequency, int spline_order) {
           absl::flat_hash_map<double, Pose3d> poses_absl;
           for (const auto& [key, value] : poses) {
             poses_absl[key] = value;
           }
           const auto status = self.FitSpline(poses_absl, knot_frequency,
                                              spline_order);
           if (!status.ok()) {
             throw std::runtime_error(
                 std::string("Error: ") + std::string(status.message()));
           }
         },
         py::arg("poses"),
         py::arg("knot_frequency") = Trajectory::kDefaultKnotFrequency,
         py::arg("spline_order") = Trajectory::kDefaultSplineOrder);

  // World model class.
  py::class_<Landmark>(m, "Landmark")
    .def(py::init<>())
    .def_readwrite("point", &Landmark::point)
    .def_readwrite("id", &Landmark::id)
    .def_readwrite("point_is_constant", &Landmark::point_is_constant);

  py::class_<RigidBody>(m, "RigidBody")
    .def(py::init<>())
    .def_readwrite("model_definition", &RigidBody::model_definition)
    .def_readwrite("T_world_rigidbody", &RigidBody::T_world_rigidbody)
    .def_readwrite("id", &RigidBody::id)
    .def_readwrite("world_pose_is_constant", &RigidBody::world_pose_is_constant)
    .def_readwrite("model_definition_is_constant",
                   &RigidBody::model_definition_is_constant);

  py::class_<WorldModel, std::shared_ptr<WorldModel>>(m, "WorldModel")
    .def(py::init<>())
    .def("AddLandmark", &WorldModel::AddLandmark)
    .def("AddRigidBody",
         [](WorldModel& self, const RigidBody& rigidbody) {
           const auto status = self.AddRigidBody(rigidbody);
           if (!status.ok()) {
             throw std::runtime_error(
                 std::string("Error: ") + std::string(status.message()));
           }
         });

  // ceres::Summary
  py::class_<ceres::Solver::Summary>(m, "Summary")
    .def("BriefReport", &ceres::Solver::Summary::BriefReport)
    .def("FullReport", &ceres::Solver::Summary::FullReport)
    .def("IsSolutionUsable", &ceres::Solver::Summary::IsSolutionUsable)
    .def_readonly("initial_cost", &ceres::Solver::Summary::initial_cost)
    .def_readonly("final_cost", &ceres::Solver::Summary::final_cost)
    .def_readonly("num_residual_blocks",
                  &ceres::Solver::Summary::num_residual_blocks)
    .def_readonly("num_residuals",
                  &ceres::Solver::Summary::num_residuals)
    .def_readonly("num_parameter_blocks",
                  &ceres::Solver::Summary::num_parameter_blocks)
    .def_readonly("num_parameters",
                  &ceres::Solver::Summary::num_parameters)
    .def_readonly("num_parameter_blocks_reduced",
                  &ceres::Solver::Summary::num_parameter_blocks_reduced)
    .def_readonly("num_parameters_reduced",
                  &ceres::Solver::Summary::num_parameters_reduced)
    .def_readonly("num_effective_parameters_reduced",
                  &ceres::Solver::Summary::num_effective_parameters_reduced)
    .def_readonly("num_residual_blocks_reduced",
                  &ceres::Solver::Summary::num_residual_blocks_reduced)
    .def_readonly("num_residuals_reduced",
                  &ceres::Solver::Summary::num_residuals_reduced);

  // ceres::Solver::Options
  py::class_<ceres::Solver::Options>(m, "SolverOptions")
    .def_readwrite("minimizer_type", &ceres::Solver::Options::minimizer_type)
    .def_readwrite("max_num_iterations",
                   &ceres::Solver::Options::max_num_iterations)
    .def_readwrite("num_threads", &ceres::Solver::Options::num_threads)
    .def_readwrite("function_tolerance",
                   &ceres::Solver::Options::function_tolerance)
    .def_readwrite("gradient_tolerance",
                   &ceres::Solver::Options::gradient_tolerance)
    .def_readwrite("parameter_tolerance",
                   &ceres::Solver::Options::parameter_tolerance)
    .def_readwrite("linear_solver_type",
                   &ceres::Solver::Options::linear_solver_type)
    .def_readwrite("preconditioner_type",
                   &ceres::Solver::Options::preconditioner_type)
    .def_readwrite("minimizer_progress_to_stdout",
                   &ceres::Solver::Options::minimizer_progress_to_stdout);

  // Getter for default solver options.
  m.def("DefaultSolverOptions", &DefaultSolverOptions);

  // BatchOptimizer class.
  py::class_<BatchOptimizer>(m, "BatchOptimizer")
    .def(py::init())
    .def("AddSensor",
         [](BatchOptimizer& self, std::shared_ptr<Sensor> sensor) {
           self.AddSensor(sensor.get(), /*take_ownership=*/false);
         })

    .def("AddTrajectory",
         [](BatchOptimizer& self, std::shared_ptr<Trajectory> trajectory) {
           self.AddTrajectory(trajectory.get(), /*take_ownership=*/false);
         })
    .def("AddWorldModel",
         [](BatchOptimizer& self, std::shared_ptr<WorldModel> world_model) {
           self.AddWorldModel(world_model.get(), /*take_ownership=*/false);
         })
    .def("Optimize",
         [](BatchOptimizer& self, const ceres::Solver::Options& options) {
           const auto summary = self.Optimize(options);
           if (!summary.ok()) {
             throw std::runtime_error(
                 std::string("Error: ") +
                 std::string(summary.status().message()));
           }
           return summary.value();
         },
         py::arg("options") = DefaultSolverOptions());

  // AprilGridDetector class.
  py::class_<AprilGridDetector>(m, "AprilGridDetector")
    .def(py::init<std::string>())
    .def("Detect",
         [](AprilGridDetector& self, const py::array_t<uint8_t>& img) {
           py::buffer_info buf = img.request();
           cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC1,
                       static_cast<void*>(buf.ptr));
           return self.Detect(mat);
         })
    .def("GetRigidBodyDefinition", &AprilGridDetector::GetRigidBodyDefinition);
}
