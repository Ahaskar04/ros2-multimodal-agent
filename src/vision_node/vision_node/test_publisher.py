#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import urllib.request

class TestImagePublisher(Node):
    def __init__(self):
        super().__init__('test_image_publisher')

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)

        # Load a real test image with detectable objects
        self.load_test_image()

        # Publish at 1 Hz
        self.timer = self.create_timer(1.0, self.publish_image)
        self.get_logger().info('Test image publisher started!')

    def load_test_image(self):
        # Try to download a sample image from the internet
        try:
            url = 'https://ultralytics.com/images/bus.jpg'
            self.get_logger().info(f'Downloading test image from {url}...')

            req = urllib.request.urlopen(url)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            self.test_image = cv2.imdecode(arr, -1)

            self.get_logger().info('Test image downloaded successfully!')
        except Exception as e:
            self.get_logger().warn(f'Could not download image: {e}. Using generated image.')
            # Fallback to a simple image
            self.test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128

    def publish_image(self):
        msg = self.bridge.cv2_to_imgmsg(self.test_image, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        self.publisher.publish(msg)
        self.get_logger().info('Published test image')

def main(args=None):
    rclpy.init(args=args)
    node = TestImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
