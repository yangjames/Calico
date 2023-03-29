#!/usr/bin/env python3

import os

import calico
import cv_bridge
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
import rosbag

    
if __name__ == '__main__':
  visualize = False
  # Load in rosbag and chart definition.
  bag_path = '/home/james/Downloads/multicamera-calibration-data/cam_april.bag'
  bag = rosbag.Bag(bag_path, 'r')
  chart_definition_path = (
    '/home/james/Downloads/multicamera-calibration-data/april_6x6.yaml')
  detector = calico.AprilGridDetector(chart_definition_path)

  # Construct data containers.
  topic_left = '/cam0/image_raw'
  topic_right = '/cam1/image_raw'
  camera_left = calico.Camera()
  camera_left.SetName(topic_left)
  camera_left.SetModel(calico.CameraIntrinsicsModel.kOpenCv5)
  camera_right = calico.Camera()
  camera_right.SetName(topic_right)
  camera_right.SetModel(calico.CameraIntrinsicsModel.kOpenCv5)

  # Extract aprilgrid detections from images.
  all_detections = {
    topic_left: [],
    topic_right: []
  }
  bridge = cv_bridge.CvBridge()
  total_expected_messages = \
    bag.get_message_count(topic_left) + bag.get_message_count(topic_right)
  msg_count = 0
  for topic, msg, stamp in bag.read_messages(topics = [topic_left, topic_right]):
    msg_count += 1
    print(f'\r{(100.0 * msg_count) / total_expected_messages:.1f}%:'
          f' Processed {msg_count} of {total_expected_messages} images.', end='')
    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding = 'passthrough')
    detections = detector.Detect(cv_img)
    if len(detections) < 28:
      # Show the picture without any detections.
      if visualize:
        cv2.imshow(topic, cv_img)
        cv2.waitKey(1)
      continue
    all_detections[topic].append((msg.header.stamp.to_sec(), detections))
    measurements = calico.DetectionsToCameraMeasurements(
      detections, msg.header.stamp.to_sec(), msg.header.seq)
    if topic == topic_left:
      camera_left.AddMeasurements(measurements)
    elif topic == topic_right:
      camera_right.AddMeasurements(measurements)
    # Visualize if flagged.
    if visualize:
      im = calico.DrawDetections(cv_img, detections)
      cv2.imshow(topic, im)
      cv2.waitKey(1)
  bag.close()
  print()

  # Initialize our world model.
  print('Initializing world model.')
  chart_rigid_body = detector.GetRigidBodyDefinition()
  chart_rigid_body.world_pose_is_constant = True
  chart_rigid_body.model_definition_is_constant = True
  world_model = calico.WorldModel()
  world_model.AddRigidBody(chart_rigid_body)

  # Intialize left/right camera intrinsics and left camera-to-chart poses.
  print('Initializing intrinsics for cameras.')
  left_detections = [detections[1] for detections in all_detections[topic_left]]
  intrinsics_left, R_chart_left, t_chart_left = \
    calico.InitializePinholeAndPoses(
      left_detections, chart_rigid_body.model_definition)
  right_detections = [
    detections[1] for detections in all_detections[topic_right]]
  intrinsics_right, _, _ = calico.InitializePinholeAndPoses(
    right_detections, chart_rigid_body.model_definition)
  print(intrinsics_left)
  print(intrinsics_right)
  f = np.mean([
    intrinsics_left[0], intrinsics_left[1],
    intrinsics_right[0], intrinsics_right[1]])
  cx = np.mean([intrinsics_left[3], intrinsics_right[3]])
  cy = np.mean([intrinsics_left[4], intrinsics_right[4]])
  initial_intrinsics = [f, cx, cy, 0, 0, 0, 0, 0]
  camera_left.SetIntrinsics(initial_intrinsics)
  camera_right.SetIntrinsics(initial_intrinsics)

  # Initialize sensor rig trajectory.
  print('Initializing sensor trajectory.')
  stamps = [detections[0] for detections in all_detections[topic_left]]
  poses_chart_sensorrig = {}
  for stamp, R_chart_left_i, t_chart_left_i in\
      zip(stamps, R_chart_left, t_chart_left):
    pose_chart_sensorrig = calico.Pose3d()
    pose_chart_sensorrig.rotation = \
      R.from_matrix(R_chart_left_i).as_quat()[[3, 0, 1, 2]]
    pose_chart_sensorrig.translation = t_chart_left_i
    poses_chart_sensorrig[stamp] = pose_chart_sensorrig
  trajectory_chart_sensorrig = calico.Trajectory()
  trajectory_chart_sensorrig.AddPoses(poses_chart_sensorrig)

  # Run optimization.
  print('Running optimization.')
  print('Intial intrinsics: ')
  print(f'{camera_left.GetName()}: {camera_left.GetIntrinsics()}')
  print(f'{camera_right.GetName()}: {camera_right.GetIntrinsics()}')
  print('Initial extrinsics: ')
  print(f'{camera_right.GetName()}: q - {camera_right.GetExtrinsics().rotation}, t - {camera_right.GetExtrinsics().translation}')
  camera_left.EnableIntrinsicsEstimation(True)
  camera_left.EnableExtrinsicsEstimation(False)
  camera_left.EnableLatencyEstimation(False)
  camera_right.EnableIntrinsicsEstimation(True)
  camera_right.EnableExtrinsicsEstimation(True)
  camera_right.EnableLatencyEstimation(False)

  optimizer = calico.BatchOptimizer()
  optimizer.AddSensor(camera_left)
  optimizer.AddSensor(camera_right)
  optimizer.AddWorldModel(world_model)
  optimizer.AddTrajectory(trajectory_chart_sensorrig)

  options = calico.DefaultSolverOptions()
  options.num_threads = 4
  summary = optimizer.Optimize(options)

  print('Final intrinsics: ')
  print(f'{camera_left.GetName()}: {camera_left.GetIntrinsics()}')
  print(f'{camera_right.GetName()}: {camera_right.GetIntrinsics()}')
  print('Final extrinsics: ')
  print(f'{camera_right.GetName()}: q - {camera_right.GetExtrinsics().rotation}, t - {camera_right.GetExtrinsics().translation}')
  print(summary.FullReport())
