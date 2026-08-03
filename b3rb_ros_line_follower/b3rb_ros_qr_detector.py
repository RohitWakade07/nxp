# Copyright 2024-2026 NXP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
import json
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np

class QRDetector(Node):
    """
    ROS 2 Node that processes raw camera images to scan for QR codes.
    It publishes the detected QR code payload on the `/qr_detection` topic.
    """
    def __init__(self):
        super().__init__('qr_detector')

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        # Publisher for QR code detection results.
        self.publisher_qr = self.create_publisher(
            String,
            '/qr_detection',
            10)

        self.get_logger().info("QR Detector Node started. Waiting for images...")

    def camera_image_callback(self, message):
        """Processes incoming camera frames to detect QR codes."""
        # Convert compressed image message to OpenCV format
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return

        payload = self.detect_qr_code(image)

        if payload is not None:
            msg = String()
            msg.data = json.dumps(payload)
            self.publisher_qr.publish(msg)
            self.get_logger().info(f"Found QR Data: {payload['data']}")

            # Draw bounding box and text
            bbox = payload['bbox']
            if len(bbox) > 0 and len(bbox[0]) == 4:
                pts = np.array(bbox[0], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(image, [pts], True, (0, 255, 0), 3)
                
                # Put text near the top-left corner
                pt1 = (int(pts[0][0][0]), int(pts[0][0][1]))
                cv2.putText(image, payload['data'], (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # cv2.imshow("QR Code Detection", image)
        cv2.waitKey(1)

    def detect_qr_code(self, image):
        """
        Detect and decode QR code in the image.
        """
        try:
            try:
                from pyzbar.pyzbar import decode
            except ImportError:
                decode = None
            if decode is not None:
                decoded_objects = decode(image)
                if decoded_objects:
                    obj = decoded_objects[0]
                    data = obj.data.decode('utf-8')
                    rect = obj.rect
                    bbox_list = [[[rect.left, rect.top], 
                                  [rect.left + rect.width, rect.top], 
                                  [rect.left + rect.width, rect.top + rect.height], 
                                  [rect.left, rect.top + rect.height]]]
                    return {
                        "detected": True,
                        "data": data,
                        "bbox": bbox_list
                    }
            else:
                detector = cv2.QRCodeDetector()
                data, bbox, straight_qrcode = detector.detectAndDecode(image)
                if bbox is not None and data != "":
                    bbox_list = bbox.tolist() if hasattr(bbox, 'tolist') else bbox
                    return {
                        "detected": True,
                        "data": data,
                        "bbox": bbox_list
                    }
        except Exception as e:
            self.get_logger().debug(f"QR Detection failed: {e}")

        return None

def main(args=None):
    rclpy.init(args=args)
    node = QRDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()