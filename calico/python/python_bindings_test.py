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

        
if __name__ == '__main__':
    unittest.main()
