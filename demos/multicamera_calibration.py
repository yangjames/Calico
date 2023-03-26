#!/usr/bin/env python3

import os
import yaml

import apriltag
import calico
import cv_bridge
import cv2
import numpy as np
import rosbag

import matplotlib.pyplot as plt

from typing import Dict

def DrawDetections(
    img: np.ndarray,
    detections: Dict[int, np.ndarray]
) -> np.ndarray:
  img_color = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
  for feature_id, corner in detections.items():
    corner_id = feature_id % 4
    tag_id = feature_id // 4
    color = (
      255 * (corner_id == 2 or corner_id == 3),
      255 * (corner_id == 1 or corner_id == 3),
      255 * (corner_id == 0),
    )
    cv2.putText(img_color, str(tag_id) + '.' + str(corner_id),
                tuple(corner.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1)
    # color = (255, 0, 255)
    # cv2.putText(img_color, str(feature_id),
    #             tuple(corner.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             color, 1)
    cv2.circle(img_color, tuple(corner.astype(int)), 3, color, 3)
  return img_color


if __name__ == '__main__':
  bag_path = '/home/james/Downloads/multicamera-calibration-data/cam_april.bag'
  bag = rosbag.Bag(bag_path, 'r')
  bridge = cv_bridge.CvBridge()

  chart_definition_path = (
    '/home/james/Downloads/multicamera-calibration-data/april_6x6.yaml')
  detector = calico.AprilGridDetector(chart_definition_path)

  chart_rigid_body = detector.GetRigidBodyDefinition()

  topic_left = '/cam0/image_raw'
  topic_right = '/cam1/image_raw'
  for topic, msg, stamp in bag.read_messages(topics = [topic_left, topic_right]):
    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding = 'passthrough')
    detections = detector.Detect(cv_img)
    if len(detections) == 0:
      continue
    im = DrawDetections(cv_img, detections)
    cv2.imshow(topic, im)
    cv2.waitKey(0)
  bag.close()

