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
import os
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
import time

from synapse_msgs.msg import ServerCommunication

# ROS 2 package path resolver
from ament_index_python.packages import get_package_share_directory

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

CLASSES = {
    0: "A",
    1: "B",
    2: "C",
    3: "Left",
    4: "Right",
    5: "Straight",
    6: "X",
    7: "Y",
    8: "Z"
}

# Color Map (BGR) for different classes
CLASS_COLORS = {
    "A": (0, 0, 255),          # Red
    "B": (0, 255, 0),          # Green
    "C": (255, 0, 0),          # Blue
    "Left": (0, 255, 255),     # Yellow
    "Right": (255, 255, 0),    # Cyan
    "Straight": (255, 0, 255), # Magenta
    "X": (0, 165, 255),        # Orange
    "Y": (128, 0, 128),        # Purple
    "Z": (0, 100, 0)           # Dark Green
}

class ObjectRecognizer(Node):
    """
    ROS 2 Node that processes raw camera images to recognize traffic sign boards.
    It publishes the detected sign type/labels on the `/sign_board_detection` topic.
    """
    def __init__(self):
        super().__init__('object_recognizer')

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        # Publisher for sign board detection results.
        self.publisher_sign = self.create_publisher(
            String,
            '/sign_board_detection',
            10)

        # Subscription for Server Communication
        self.subscription_server = self.create_subscription(
            ServerCommunication,
            '/ServerCommunication',
            self.server_communication_callback,
            10)

        self.model = None
        self.received_first_image = False
        
        self.mission_received_time = 0.0
        self.sign_paused_due_to_mission = False

        if YOLO is not None:
            try:
                # Dynamically locate the package share directory
                package_share_directory = get_package_share_directory('b3rb_ros_line_follower')
                model_path = os.path.join(package_share_directory, 'weights', 'best.pt')
                
                if os.path.exists(model_path):
                    self.model = YOLO(model_path)
                    self.get_logger().info(f"Successfully loaded YOLOv8 model from {model_path}")
                else:
                    self.get_logger().error(f"Model file not found at expected path: {model_path}")
                    self.get_logger().warn("Ensure the 'weights' folder is properly installed via CMakeLists.txt or setup.py!")
            except Exception as e:
                self.get_logger().error(f"Failed to locate or load YOLOv8 model: {e}")
        else:
            self.get_logger().warn("ultralytics is not installed. Object Recognizer cannot run model inference.")

        self.get_logger().info("Object Recognizer Node started. Waiting for images...")

    def server_communication_callback(self, message):
        # Buggy receives commands on dest=1 with ack=0
        if message.dest == 1 and message.ack == 0:
            self.mission_received_time = time.time()
            self.sign_paused_due_to_mission = True
            self.get_logger().info(
                f"Mission received: '{message.msg}'. Pausing sign detection for 10 seconds."
            )

    def camera_image_callback(self, message):
        """Processes incoming camera frames to classify traffic signs."""
        if not self.received_first_image:
            self.received_first_image = True
            self.get_logger().info("Received first camera image frame! Subscription active.")

        # Convert compressed image message to OpenCV format
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return

        current_time = time.time()
        if self.sign_paused_due_to_mission:
            if current_time - self.mission_received_time < 10.0:
                # ── PAUSED: skip inference, show countdown overlay ──
                orig_h, orig_w = image.shape[:2]
                panel_width = 250
                side_panel = np.ones((orig_h, panel_width, 3), dtype=np.uint8) * 255
                cv2.putText(side_panel, "Detection Paused", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(side_panel,
                            f"Resuming in {10.0 - (current_time - self.mission_received_time):.1f}s",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                combined_img = np.hstack((image, side_panel))
                cv2.imshow("Sign Board Detection", combined_img)
                cv2.waitKey(1)
                return
            else:
                self.sign_paused_due_to_mission = False  # resume
                self.get_logger().info("10 seconds pause complete. Resuming sign detection.")

        payload = self.classify_sign(image)

        # Create a side panel (white background)
        orig_h, orig_w = image.shape[:2]
        panel_width = 250
        side_panel = np.ones((orig_h, panel_width, 3), dtype=np.uint8) * 255
        
        # Add title to side panel
        cv2.putText(side_panel, "Detected Signs:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_offset = 60

        if payload is not None and payload.get("detected", False):
            sign_config = self.get_sign_key_value_pairs(payload.get("detections", []))
            payload["sign_config"] = sign_config

            msg = String()
            msg.data = json.dumps(payload)
            self.publisher_sign.publish(msg)
            
            num_dets = len(payload.get("detections", []))
            self.get_logger().info(f"Detected {num_dets} signs (inference_time={payload.get('inference_time_ms', 0.0):.2f}ms)")
            print(sign_config)

            # Draw bounding boxes and labels for all detections
            for det in payload.get("detections", []):
                box = det.get("box", [0, 0, 0, 0])
                x1, y1, x2, y2 = [int(v) for v in box]
                class_name = det['class']
                confidence = det['confidence']
                label = f"{class_name} ({confidence:.2f})"
                
                color = CLASS_COLORS.get(class_name, (255, 0, 255))
                
                # Draw box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Write to side panel (black text on white background)
                cv2.putText(side_panel, f"- {class_name}: {confidence:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                y_offset += 20

        # Combine original image and side panelata 
        combined_img = np.hstack((image, side_panel))

        cv2.imshow("Sign Board Detection", combined_img)
        cv2.waitKey(1)

    def classify_sign(self, image):
        """
        Classify traffic sign boards using YOLOv8 model.
        """
        if self.model is None:
            return None

        try:
            import time
            start_time = time.time()
            
            # Predict (lowered conf from 0.40 to 0.25 to detect harder-to-see arrows)
            results = self.model.predict(image, conf=0.25, iou=0.45, verbose=False)
            
            inference_time_ms = (time.time() - start_time) * 1000.0
            self.get_logger().info(f"Model Inference Time: {inference_time_ms:.2f} ms")

            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                
                detections = []
                for i in range(len(boxes)):
                    box = boxes[i]
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    xyxy = box.xyxy[0].tolist() # [xmin, ymin, xmax, ymax]
                    
                    xmin, ymin, xmax, ymax = xyxy
                    w = xmax - xmin
                    h = ymax - ymin
                    cx = xmin + w / 2.0
                    cy = ymin + h / 2.0
                    
                    detections.append({
                        "class": CLASSES.get(class_id, f"Unknown_{class_id}"),
                        "confidence": confidence,
                        "cx": cx,
                        "cy": cy,
                        "w": w,
                        "h": h,
                        "box": [float(xmin), float(ymin), float(xmax), float(ymax)]
                    })

                return {
                    "detected": True,
                    "detections": detections,
                    "inference_time_ms": inference_time_ms
                }

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")

        return None

    def get_sign_key_value_pairs(self, detections):
        """Detect letters/arrows in a frame and return {letter: direction} pairs."""
        letters = []
        arrows = []
        for det in detections:
            if det["class"] in ["A", "B", "C", "X", "Y", "Z"]:
                letters.append(det)
            elif det["class"] in ["Left", "Right", "Straight", "Up"]:
                arrows.append(det)

        pairings = {}
        for letter in letters:
            best_arrow = None
            best_dist = float('inf')
            for arrow in arrows:
                dist = (letter["cx"] - arrow["cx"])**2 + (letter["cy"] - arrow["cy"])**2
                if dist < best_dist:
                    best_dist = dist
                    best_arrow = arrow["class"]
            
            if best_arrow:
                pairings[letter["class"]] = best_arrow

        direction_map = {"Left": "left", "Right": "right", "Straight": "straight", "Up": "up"}
        sign_config = {
            letter: direction_map.get(arrow_name, arrow_name.lower())
            for letter, arrow_name in pairings.items()
        }
        return sign_config

from rclpy.executors import ExternalShutdownException

def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()