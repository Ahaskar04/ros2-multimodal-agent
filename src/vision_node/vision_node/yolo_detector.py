#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class YOLODetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        # Initialize YOLO model (YOLOv8n - nano, fastest)
        self.get_logger().info('Loading YOLO model...')
        self.model = YOLO('yolov8n.pt')  # Downloads automatically on first run
        self.get_logger().info('YOLO model loaded!')

        # CV Bridge for converting ROS images to OpenCV
        self.bridge = CvBridge()

        # Subscribers and Publishers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',  # Standard Gazebo camera topic
            self.image_callback,
            10
        )

        self.detection_pub = self.create_publisher(
            String,
            '/vision/detections',
            10
        )

        self.get_logger().info('YOLO Detector node started!')

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Run YOLO detection
            results = self.model(cv_image, verbose=False)

            # Process detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get detection info
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self.model.names[cls]

                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1

                    # Only publish if confidence > 0.5
                    if conf > 0.5:
                        detection_msg = f"{label},{conf:.2f},{center_x:.1f},{center_y:.1f},{width:.1f},{height:.1f}"
                        self.detection_pub.publish(String(data=detection_msg))
                        self.get_logger().info(f'Detected: {label} ({conf:.2f}) at ({center_x:.1f}, {center_y:.1f})')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = YOLODetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
