#!/usr/bin/env python3
import calico

import numpy as np
from scipy.spatial.transform import Rotation as R
import unittest


class TestCalicoPythonUtils(unittest.TestCase):

    def test_DetectionsToCameraMeasurements(self):
        test_detections = {}
        for i in range(600):
            test_detections[i] = np.array([float(i), float(i)])
        test_stamp = 1.0
        test_seq = 32
        measurements = calico.DetectionsToCameraMeasurements(
            test_detections, test_stamp, test_seq)
        self.assertEqual(len(measurements), len(test_detections))
        for measurement in measurements:
            np.testing.assert_equal(test_detections[measurement.id.feature_id],
                                    measurement.pixel)
            self.assertEqual(measurement.id.stamp, test_stamp)
            self.assertEqual(measurement.id.image_id, test_seq)

    def test_InitializePinholeAndPoses(self):
        # Ground-truth intrinsics.
        true_fx = 400
        true_fy = 410
        true_s = 10
        true_cx = 100
        true_cy = 250
        true_intrinsics = [true_fx, true_fy, true_s, true_cx, true_cy]
        true_K = np.array([[true_fx, true_s, true_cx],
                           [0, true_fy, true_cy],
                           [0, 0, 1.0]])
        # Create sample sensor poses.
        R_camera_world = [
            R.from_rotvec([np.pi, np.pi/3.0, 0.0]).as_matrix(),
            R.from_rotvec([np.pi, -np.pi/3, 0]).as_matrix(),
            R.from_rotvec([np.pi, np.pi/12, 0]).as_matrix(),
            R.from_rotvec([np.pi + np.pi/12, 0, 0]).as_matrix(),
            R.from_rotvec([np.pi, np.pi/6, np.pi/12]).as_matrix(),
        ]
        t_camera_world = [
            np.array([0.5, 0.5, 1]),
            np.array([0.6, 0.6, 1.25]),
            np.array([0.5, 0.5, 0.75]),
            np.array([0.4, 0.4, 1.1]),
            np.array([0.5, 0.6, 0.9]),
        ]
        # Create planar points resolved in world frame homogeneous coordinates.
        world_points = np.zeros((4, 121), dtype=float)
        idx = 0
        for x in range(11):
            for y in range(11):
                world_points[:, idx] = [0.1 * x, 0.1 * y, 0.0, 1.0]
                idx += 1
        model_definition = {}
        for i in range(world_points.shape[1]):
            model_definition[i] = np.array(world_points[:3, i].flatten())
        # Generate sample detections by projecting planar points through our
        # pinhole camera using sample poses.
        detections = []
        for i, (R_camera_world_i, t_camera_world_i) in\
            enumerate(zip(R_camera_world, t_camera_world)):
            camera_points = np.dot(
                np.hstack((R_camera_world_i, t_camera_world_i.reshape((3,1)))), world_points)
            projections = np.dot(true_K, camera_points)
            proj_x = projections[0, :] / projections[2, :]
            proj_y = projections[1, :] / projections[2, :]
            detection = {}
            for j, (px, py) in enumerate(zip(proj_x, proj_y)):
                detection[j] = np.array([px, py])
            detections.append(detection)
        # Estimate everything and compare.
        actual_intrinsics, actual_R_world_camera, actual_t_world_camera =\
            calico.InitializePinholeAndPoses(detections, model_definition)
        np.testing.assert_almost_equal(
            true_intrinsics, actual_intrinsics, decimal=3)
        for actual_R, actual_t, expected_R, expected_t in \
            zip(actual_R_world_camera, actual_t_world_camera, R_camera_world,
                t_camera_world):
            r_actual = -R.from_matrix(actual_R).as_rotvec()
            r_expected = R.from_matrix(expected_R).as_rotvec()
            t_actual = -np.dot(actual_R.T, actual_t)
            t_expected = expected_t
            np.testing.assert_almost_equal(r_actual, r_expected, decimal=6)
            np.testing.assert_almost_equal(t_actual, t_expected, decimal=5)
        
if __name__ == '__main__':
    unittest.main()
