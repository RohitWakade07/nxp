# Copyright 2024-2026 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc.
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
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
import numpy as np
import cv2
import math
from synapse_msgs.msg import EdgeVectors

QOS_PROFILE_DEFAULT = 10
PI = math.pi

RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)

# OPTIMIZATION: Set view height percentage for edge vector processing.
VECTOR_IMAGE_HEIGHT_PERCENTAGE = 0.40
# OPTIMIZATION: Maximum full image width percentage (100% horizontal FOV).
VECTOR_IMAGE_WIDTH_PERCENTAGE = 100.0 
# OPTIMIZATION: Increased minimum magnitude to ignore small black specks/noise.
VECTOR_MAGNITUDE_MINIMUM = 4.0 

class EdgeVectorsPublisher(Node):
    """
    ROS 2 Node that processes raw camera images to detect the lane edges (left/right bounds).
    Optimized for NXP Gazebo Track: White road, black borders, and intersections.
    """
    def __init__(self):
        super().__init__('edge_vectors_publisher')

        self.subscription_camera_raw = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_raw_image_callback,
            QOS_PROFILE_DEFAULT)

        self.publisher_edge_vectors = self.create_publisher(
            EdgeVectors,
            '/edge_vectors',
            QOS_PROFILE_DEFAULT)

        self.publisher_thresh_image = self.create_publisher(
            CompressedImage,
            "/debug_images/thresh_image",
            QOS_PROFILE_DEFAULT)

        self.publisher_vector_image = self.create_publisher(
            CompressedImage,
            "/debug_images/vector_image",
            QOS_PROFILE_DEFAULT)

        self.image_height = 0
        self.image_width = 0
        self.lower_image_height = 0
        self.upper_image_height = 0

        self.show_perception_window = True
        if self.show_perception_window:
            cv2.namedWindow("Perception Window - Edge Vectors", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Perception Window - Edge Vectors", 1920, 1080)
            cv2.startWindowThread()

    def publish_debug_image(self, publisher, image):
        """Helper function to publish OpenCV debug images to ROS topics."""
        message = CompressedImage()
        _, encoded_data = cv2.imencode('.jpg', image)
        message.format = "jpeg"
        message.data = encoded_data.tobytes()
        publisher.publish(message)

    def get_vector_angle_in_radians(self, vector):
        """Calculates the slope angle of a vector in radians."""
        if ((vector[0][0] - vector[1][0]) == 0):  # Prevent division by zero
            theta = PI / 2
        else:
            slope = (vector[1][1] - vector[0][1]) / (vector[0][0] - vector[1][0])
            theta = math.atan(slope)
        return theta

    def compute_vectors_from_image(self, image, thresh):
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        cv2.drawContours(image, contours, -1, (255, 0, 0), 1)  # Draw all contours in blue
        vectors = []
        
        for i in range(len(contours)):
            coordinates = contours[i][:, 0, :]
            min_y_value = np.min(coordinates[:, 1])
            max_y_value = np.max(coordinates[:, 1])

            min_y_coords = np.array(coordinates[coordinates[:, 1] == min_y_value])
            max_y_coords = np.array(coordinates[coordinates[:, 1] == max_y_value])

            min_y_coord = min_y_coords[np.argmin(min_y_coords[:, 0])].copy()
            max_y_coord = max_y_coords[np.argmax(max_y_coords[:, 0])].copy()

            magnitude = np.linalg.norm(min_y_coord - max_y_coord)
            
            if (magnitude > VECTOR_MAGNITUDE_MINIMUM):
                rover_point = [self.image_width / 2, self.lower_image_height]
                middle_point = (min_y_coord + max_y_coord) / 2
                distance = np.linalg.norm(middle_point - rover_point)

                angle = self.get_vector_angle_in_radians([min_y_coord, max_y_coord])
                if angle > 0:
                    min_y_coord[0] = np.max(min_y_coords[:, 0])
                else:
                    max_y_coord[0] = np.max(max_y_coords[:, 0])

                vectors.append([list(min_y_coord), list(max_y_coord), distance])

            # --- New Behavior: Corner Detection ---
            # Kept original logic intact; this appends an extra, tighter vector for corners
            epsilon = 0.01 * cv2.arcLength(contours[i], False)
            approx = cv2.approxPolyDP(contours[i], epsilon, False)[:, 0, :]

            if len(approx) >= 3:
                rover_point_np = np.array([self.image_width / 2, self.lower_image_height])
                best_seg, best_dist = None, float('inf')
                for j in range(len(approx) - 1):
                    p1, p2 = approx[j], approx[j + 1]
                    mid = (p1 + p2) / 2.0
                    dist = np.linalg.norm(mid - rover_point_np)
                    if dist < best_dist:
                        best_dist = dist
                        best_seg = (p1, p2)
                
                if best_seg is not None:
                    p1, p2 = best_seg
                    if p1[1] <= p2[1]:
                        c_min_y, c_max_y = p1.copy(), p2.copy()
                    else:
                        c_min_y, c_max_y = p2.copy(), p1.copy()
                    
                    c_magnitude = np.linalg.norm(c_min_y - c_max_y)
                    if c_magnitude > VECTOR_MAGNITUDE_MINIMUM:
                        c_middle = (c_min_y + c_max_y) / 2.0
                        c_dist = np.linalg.norm(c_middle - rover_point_np)
                        vectors.append([list(c_min_y), list(c_max_y), c_dist])

        return vectors, image

    def process_image_for_edge_vectors(self, image):
        """
        Applies preprocessing (Blur, Grayscale, Thresholding, Morphology) and extracts lane vectors.
        """
        self.image_height, self.image_width, _ = image.shape
        self.lower_image_height = int(self.image_height * VECTOR_IMAGE_HEIGHT_PERCENTAGE)
        self.upper_image_height = int(self.image_height - self.lower_image_height)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Mask out highly saturated colors (objects) to prevent them from being detected as dark lane lines
        color_mask = (hsv[:, :, 1] > 50).astype(np.uint8) * 255
        
        # Dilate the color mask to cover the grey/black bases of the traffic cones,
        # which sit just below the orange parts. This ensures the base isn't seen as a lane.
        kernel = np.ones((50, 50), np.uint8)
        color_mask = cv2.dilate(color_mask, kernel, iterations=1)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray[color_mask > 0] = 255 # Make colored pixels and their dilated regions white
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # FIX 3: Increased threshold from 35 to 85. 
        # The extreme edges of the camera view are anti-aliased (grey). 35 was turning them white.
        threshold_black = 85 
        thresh = cv2.threshold(blurred, threshold_black, 255, cv2.THRESH_BINARY_INV)[1]

        kernel = np.ones((8, 8), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        thresh_cropped = thresh[self.image_height - self.lower_image_height:]
        image_cropped = image[self.image_height - self.lower_image_height:].copy()

        vectors, debug_img = self.compute_vectors_from_image(image_cropped, thresh_cropped)
        vectors = sorted(vectors, key=lambda x: x[2])

        half_width = self.image_width / 2
        vectors_left = [v for v in vectors if ((v[0][0] + v[1][0]) / 2) < half_width]
        vectors_right = [v for v in vectors if ((v[0][0] + v[1][0]) / 2) >= half_width]

        final_vectors = []
        for side_vectors in [vectors_left, vectors_right]:
            if len(side_vectors) > 0:
                best_vector = side_vectors[0]
                
                cv2.line(debug_img, (int(best_vector[0][0]), int(best_vector[0][1])), (int(best_vector[1][0]), int(best_vector[1][1])), GREEN_COLOR, 3)
                
                # Transform coordinates back to original image space
                best_vector[0][1] += self.upper_image_height
                best_vector[1][1] += self.upper_image_height
                final_vectors.append(best_vector[:2])

        self.publish_debug_image(self.publisher_thresh_image, thresh_cropped)
        self.publish_debug_image(self.publisher_vector_image, debug_img)

        if getattr(self, 'show_perception_window', False):
            full_debug_img = image.copy()
            full_debug_img[self.upper_image_height:] = debug_img
            cv2.line(full_debug_img, (0, self.upper_image_height), (self.image_width, self.upper_image_height), (0, 255, 255), 3)
            cv2.imshow("Perception Window - Edge Vectors", full_debug_img)
            cv2.waitKey(1)

        return final_vectors

    def camera_raw_image_callback(self, message):
        """Processes incoming raw Image camera frames."""
        try:
            channels = 3
            if '8' in message.encoding and len(message.data) == message.height * message.width * 4:
                channels = 4
            else:
                channels = len(message.data) // (message.height * message.width)

            image = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.width, channels)
            if channels == 4:
                if 'rgb' in message.encoding.lower():
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            elif channels == 3:
                if 'rgb' in message.encoding.lower():
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if np.mean(image) < 2.0:
                self.get_logger().warn("Camera image is almost entirely black! Gazebo rendering might be lagging.", throttle_duration_sec=3.0)

            self.process_and_publish(image)
        except Exception as e:
            self.get_logger().error(f"Failed to process raw image: {e}")

    def camera_compressed_image_callback(self, message):
        """Processes incoming CompressedImage camera frames."""
        try:
            np_arr = np.frombuffer(message.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is not None:
                self.process_and_publish(image)
        except Exception as e:
            self.get_logger().error(f"Failed to process compressed image: {e}")

    def process_and_publish(self, image):
        """Processes camera image and publishes EdgeVectors."""
        vectors = self.process_image_for_edge_vectors(image)

        vectors_message = EdgeVectors()
        vectors_message.image_height = image.shape[0]
        vectors_message.image_width = image.shape[1]
        vectors_message.vector_count = 0

        if len(vectors) > 0:
            vectors_message.vector_1[0].x = float(vectors[0][0][0])
            vectors_message.vector_1[0].y = float(vectors[0][0][1])
            vectors_message.vector_1[1].x = float(vectors[0][1][0])
            vectors_message.vector_1[1].y = float(vectors[0][1][1])
            vectors_message.vector_count += 1

        if len(vectors) > 1:
            vectors_message.vector_2[0].x = float(vectors[1][0][0])
            vectors_message.vector_2[0].y = float(vectors[1][0][1])
            vectors_message.vector_2[1].x = float(vectors[1][1][0])
            vectors_message.vector_2[1].y = float(vectors[1][1][1])
            vectors_message.vector_count += 1

        self.publisher_edge_vectors.publish(vectors_message)

def main(args=None):
    rclpy.init(args=args)
    node = EdgeVectorsPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()