"""@package Python utils
Utility functions for calico python bindings.
"""
import calico

import cv2
import numpy as np
from typing import Dict, List, Tuple, Union
import yaml


def ComputeRmseHeatmapAndFeatureCount(
  measurement_residual_pairs: List[Tuple[calico.CameraMeasurement, np.ndarray]],
  image_width: int, image_height: int, num_rows:int = 8, num_cols:int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """ Compute the RMSE heatmap with specified resolution.

  Args:
    measurement_residual_pairs:
      List of camera measurements paired with their residuals.
    image_width:
      Width of the original image.
    image_height:
      Height of the original image.
    num_rows:
      Number of rows we want to divide the image into.
    num_cols:
      Number of columns we want to divide the image into.

  Returns:
    A tuple containing:
      1. An rmse heatmap image with dimensions image_width x image_height.
      2. A binned version of the RMSE heatmap as a num_rows x num_cols array.
      3. A num_rows x num_cols array representing the number of features detected
         in a particular region of the image space.
  """
  local_count = np.zeros((num_rows, num_cols))
  local_rmse = np.zeros((num_rows, num_cols))
  for measurement, residual in measurement_residual_pairs:
    local_col = int(np.floor((measurement.pixel[0] / image_width) * num_cols))
    local_row = int(np.floor((measurement.pixel[1] / image_height) * num_rows))
    local_col = max(min(local_col, num_cols - 1), 0)
    local_row = max(min(local_row, num_rows - 1), 0)
    local_count[local_row, local_col] += 1
    local_rmse[local_row, local_col] += np.sum(residual**2)
  rmse_heatmap = np.sqrt(local_rmse / local_count)
  rmse_heatmap_image = cv2.resize(
      rmse_heatmap, dsize=(image_width, image_height),
      interpolation=cv2.INTER_NEAREST)
  return rmse_heatmap_image, rmse_heatmap, local_count

def DrawDetections(
    img: np.ndarray,
    detections: Dict[int, np.ndarray]
) -> np.ndarray:
  """Small helper function for drawing detections onto the original image.

  Args:
    img: Original grayscale image.
    detections: Dictionary mapping feature id to its pixel location.

  Returns:
    Color image with detections drawn.
  """
  img_color = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
  for feature_id, corner in detections.items():
    corner_id = feature_id % 4
    tag_id = feature_id // 4
    color = (
      255 * (corner_id == 2),
      255 * (corner_id == 1 or corner_id == 3),
      255 * (corner_id == 0 or corner_id == 3),
    )
    cv2.putText(img_color, str(tag_id) + '.' + str(corner_id),
                tuple(corner.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1)
    cv2.circle(img_color, tuple(corner.astype(int)), 3, color, 3)
  return img_color


def DetectionsToCameraMeasurements(
    detections: Dict[int, np.ndarray],
    stamp: float,
    seq: int,
) -> List[calico.CameraMeasurement]:
  """Convenience function for converting a calibration chart detection into
  camera measurement types.
  """
  measurements = []
  for feature_id, point in detections.items():
    measurement = calico.CameraMeasurement()
    measurement.id.stamp = stamp
    measurement.id.image_id = seq
    measurement.id.model_id = 0  # always 0 for Aprilgrid since multiple
                                 # charts are not supported.
    measurement.id.feature_id = feature_id
    measurement.pixel = point
    measurements.append(measurement)
  return measurements


def InitializePinholeAndPoses(
    all_detections: List[Dict[int, np.ndarray]],
    model_definition: Dict[int, np.ndarray]
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
  """Implements Zhang's pinhole estimation algorithm.
  https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf
  For convenience, we also return the pose of the camera w.r.t. the calibration
  chart.

  Args:
    measurements:
      List of calibration chart detections where each list element represents
      detections for one image frame.
    model_definition:
      Dictionary mapping feature ID of a calibration chart to its metric
      coordinate resolved in the chart's frame.

  Returns:
    intrinsics:
      Pinhole parameters as a 5-vector in the order [fx, fy, s, cx, cy] such that
      K = [fx  s cx]
          [ 0 fy cy]
          [ 0  0  1]
    R_chart_camera:
      List of 3x3 
  """
  V = np.zeros((2*len(all_detections), 6))
  H_camera_chart = []
  pixels = []
  model_points = []
  for i, detections in enumerate(all_detections):
    # Estimate homography.
    n_detections = len(detections)
    pixels_i = np.zeros((n_detections, 2))
    model_points_i = np.zeros((n_detections, 2))
    for j, (feature_id, pixel) in enumerate(detections.items()):
      pixels_i[j, :] = pixel
      model_points_i[j, :] = model_definition[feature_id][:2]
    H_camera_chart_i, _ = cv2.findHomography(model_points_i, pixels_i)
    H_camera_chart.append(H_camera_chart_i)
    pixels.append(pixels_i)
    model_points.append(model_points_i)
    # Populate linear problem.
    h11, h12, h13, h21, h22, h23, h31, h32, h33 = H_camera_chart_i.flatten()
    v11 = np.array([h11 ** 2, 2.0 * h11 * h21, h21 ** 2, 2.0 * h11 * h31,
                    2.0 * h21 * h31, h31 ** 2])
    v12 = np.array([h11*h12, h11*h22 + h12*h21, h21*h22, h11*h32 + h12*h31,
                    h21 * h32 + h22 * h31, h31 * h32])
    v22 = np.array([h12 ** 2, 2.0 * h12 * h22, h22 ** 2, 2.0 * h12 * h32,
                    2.0 * h22 * h32, h32 **2 ])
    V[2 * i, :] = v12
    V[2 * i + 1, :] = v11 - v22
  # Solve for intrinsics.
  _, _, Vt = np.linalg.svd(np.dot(V.T, V))
  b = Vt[-1,:].flatten()
  c1 = b[0] * b[2] * b[5] - b[1] ** 2 * b[5] - b[0] * b[4] ** 2 + \
    2.0 * b[1] * b[3] * b[4] - b[2] * b[3] ** 2
  c2 = b[0] * b[2] - b[1]**2
  c2 *= np.sign(c2)
  alpha = np.sqrt(c1 / (c2 * b[0]))
  beta = np.sqrt(c1 / c2 ** 2 * b[0])
  gamma = -np.sqrt(c1 / (c2 ** 2 * b[0])) * b[1]
  u0 = (b[1] * b[4] - b[2] * b[3]) / c2
  v0 = (b[1] * b[3] - b[0] * b[4]) / c2
  intrinsics = [alpha, beta, gamma, u0, v0]
  # Solve for camera poses.
  R_chart_camera = []
  t_chart_camera = []
  K_inv = np.array([[1.0/alpha, -gamma/(alpha*beta), (v0*gamma - u0*beta)/(alpha*beta)],
                    [0, 1.0/beta, -v0/beta],
                    [0, 0, 1]])
  for (pixels_i, model_points_i, H_camera_chart_i) in zip(pixels, model_points, H_camera_chart):
    Rt = np.dot(K_inv, H_camera_chart_i)
    scale = (np.linalg.norm(Rt[:, 0]) + \
             np.linalg.norm(Rt[:, 1])) * 0.5
    R_camera_chart = np.zeros((3, 3))
    R_camera_chart[:, 0] = Rt[:, 0] / scale
    R_camera_chart[:, 1] = Rt[:, 1] / scale
    R_camera_chart[:, 2] = np.cross(R_camera_chart[:, 0], R_camera_chart[:, 1])
    U, _, Vt = np.linalg.svd(R_camera_chart)
    R_camera_chart = np.dot(U, Vt)
    t_camera_chart = Rt[:, 2] / scale
    R_chart_camera.append(R_camera_chart.T)
    t_chart_camera.append(-np.dot(R_camera_chart.T, t_camera_chart))
  return intrinsics, R_chart_camera, t_chart_camera
