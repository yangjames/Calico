import calico

import numpy as np
import unittest
import copy

class TestCalicoPythonBindings(unittest.TestCase):
    def test_Pose3d(self):
        expected_rot = [-0.774982, -0.1549964, -0.2324946, 0.5668556]
        expected_pos = [1.5, 2.3, 6.8]
        temp_pose = calico.Pose3d()
        temp_pose.rotation = expected_rot
        temp_pose.translation = expected_pos
        np.testing.assert_allclose(temp_pose.rotation, np.array(expected_rot), 1e-7)
        np.testing.assert_equal(temp_pose.translation, np.array(expected_pos))

    def test_accelerometer(self):
        accelerometer = calico.Accelerometer()
        # Set/get name.
        test_name = 'test'
        accelerometer.SetName(test_name)
        self.assertEqual(test_name, accelerometer.GetName())
        # Set/get model.
        test_model = (
            calico.AccelerometerIntrinsicsModel.kAccelerometerScaleOnly)
        self.assertTrue(accelerometer.SetModel(test_model).ok())
        self.assertEqual(test_model, accelerometer.GetModel())
        # Set/get intrinsics.
        test_intrinsics = [1]
        self.assertTrue(accelerometer.SetIntrinsics(test_intrinsics).ok())
        np.testing.assert_equal(test_intrinsics, accelerometer.GetIntrinsics())
        # Set/get extrinsics.
        test_extrinsics = calico.Pose3d()
        test_extrinsics.rotation = [-0.774982, -0.1549964, -0.2324946, 0.5668556]
        test_extrinsics.translation = [1.5, 2.3, 6.8]
        accelerometer.SetExtrinsics(test_extrinsics)
        actual_extrinsics = accelerometer.GetExtrinsics()
        np.testing.assert_allclose(test_extrinsics.rotation,
                                   actual_extrinsics.rotation, 1e-7)
        np.testing.assert_equal(test_extrinsics.translation, actual_extrinsics.translation)
        # Set/get latency.
        test_latency = 0.02
        self.assertTrue(accelerometer.SetLatency(test_latency).ok())
        self.assertEqual(test_latency, accelerometer.GetLatency())
        # Add measurements.
        measurement = calico.AccelerometerMeasurement()
        measurement.id.stamp = 0
        measurement.id.sequence = 0
        self.assertTrue(accelerometer.AddMeasurement(measurement).ok())
        measurements = []
        for i in range(3):
            new_measurement = calico.AccelerometerMeasurement()
            new_measurement.id.stamp = i + 1
            new_measurement.id.sequence = i + 1
            measurements.append(new_measurement)
        self.assertTrue(accelerometer.AddMeasurements(measurements).ok())
        
    def test_gyroscope(self):
        gyroscope = calico.Gyroscope()
        # Set/get name.
        test_name = 'test'
        gyroscope.SetName(test_name)
        self.assertEqual(test_name, gyroscope.GetName())
        # Set/get model.
        test_model = (
            calico.GyroscopeIntrinsicsModel.kGyroscopeScaleOnly)
        self.assertTrue(gyroscope.SetModel(test_model).ok())
        self.assertEqual(test_model, gyroscope.GetModel())
        # Set/get intrinsics.
        test_intrinsics = [1]
        self.assertTrue(gyroscope.SetIntrinsics(test_intrinsics).ok())
        np.testing.assert_equal(test_intrinsics, gyroscope.GetIntrinsics())
        # Set/get extrinsics.
        test_extrinsics = calico.Pose3d()
        test_extrinsics.rotation = [-0.774982, -0.1549964, -0.2324946, 0.5668556]
        test_extrinsics.translation = [1.5, 2.3, 6.8]
        gyroscope.SetExtrinsics(test_extrinsics)
        actual_extrinsics = gyroscope.GetExtrinsics()
        np.testing.assert_allclose(test_extrinsics.rotation,
                                   actual_extrinsics.rotation, 1e-7)
        np.testing.assert_equal(test_extrinsics.translation, actual_extrinsics.translation)
        # Set/get latency.
        test_latency = 0.02
        self.assertTrue(gyroscope.SetLatency(test_latency).ok())
        self.assertEqual(test_latency, gyroscope.GetLatency())
        # Add measurements.
        measurement = calico.GyroscopeMeasurement()
        measurement.id.stamp = 0
        measurement.id.sequence = 0
        self.assertTrue(gyroscope.AddMeasurement(measurement).ok())
        measurements = []
        for i in range(3):
            new_measurement = calico.GyroscopeMeasurement()
            new_measurement.id.stamp = i + 1
            new_measurement.id.sequence = i + 1
            measurements.append(new_measurement)
        self.assertTrue(gyroscope.AddMeasurements(measurements).ok())

    def test_camera(self):
        camera = calico.Camera()
        # Set/get name.
        test_name = 'test'
        camera.SetName(test_name)
        self.assertEqual(test_name, camera.GetName())
        # Set/get model.
        test_model = (
            calico.CameraIntrinsicsModel.kOpenCv5)
        self.assertTrue(camera.SetModel(test_model).ok())
        self.assertEqual(test_model, camera.GetModel())
        # Set/get intrinsics.
        test_intrinsics = [1, 2, 3, 4, 5, 6, 7, 8]
        self.assertTrue(camera.SetIntrinsics(test_intrinsics).ok())
        np.testing.assert_equal(test_intrinsics, camera.GetIntrinsics())
        # Set/get extrinsics.
        test_extrinsics = calico.Pose3d()
        test_extrinsics.rotation = [-0.774982, -0.1549964, -0.2324946, 0.5668556]
        test_extrinsics.translation = [1.5, 2.3, 6.8]
        camera.SetExtrinsics(test_extrinsics)
        actual_extrinsics = camera.GetExtrinsics()
        np.testing.assert_allclose(test_extrinsics.rotation,
                                   actual_extrinsics.rotation, 1e-7)
        np.testing.assert_equal(test_extrinsics.translation, actual_extrinsics.translation)
        # Set/get latency.
        test_latency = 0.02
        self.assertTrue(camera.SetLatency(test_latency).ok())
        self.assertEqual(test_latency, camera.GetLatency())
        # Add measurements.
        measurement = calico.CameraMeasurement()
        measurement.id.stamp = 0
        measurement.id.image_id = 0
        self.assertTrue(camera.AddMeasurement(measurement).ok())
        measurements = []
        for i in range(3):
            new_measurement = calico.CameraMeasurement()
            new_measurement.id.stamp = i + 1
            new_measurement.id.image_id = i + 1
            measurements.append(new_measurement)
        self.assertTrue(camera.AddMeasurements(measurements).ok())

    def test_trajectory(self):
        trajectory = calico.Trajectory()
        poses = {0.0:calico.Pose3d(), 1.0:calico.Pose3d()}
        self.assertTrue(trajectory.AddPoses(poses).ok())
        
        
if __name__ == '__main__':
    unittest.main()
