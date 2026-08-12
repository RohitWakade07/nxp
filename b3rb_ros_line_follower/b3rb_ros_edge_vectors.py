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
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
import math
from synapse_msgs.msg import EdgeVectors

QOS_PROFILE_DEFAULT = 10
PI = math.pi

RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)

# HINT: You can adjust what percentage of the image from the bottom is analyzed.
# Lower portions are closer to the buggy, while upper portions see further ahead.
VECTOR_IMAGE_HEIGHT_PERCENTAGE = 0.35
VECTOR_MAGNITUDE_MINIMUM = 2.25

# Left/right edge tracking: a candidate vector is only accepted as "the same
# line as last frame" if its near-point x has moved less than this fraction
# of the image width since the previous frame. Prevents a shadow/sign-post/
# wrong-branch contour that merely happens to land on the left half of the
# image from being mistaken for the real left edge (e.g. after drifting off
# track), since a real lane line can't jump across the frame between frames.
MAX_EDGE_JUMP_RATIO = 0.18
# Consecutive frames a side is allowed to go undetected before we give up
# the lock and allow a fresh re-acquisition via the half-width fallback
# (e.g. after a sharp turn genuinely loses the line for a while).
EDGE_LOST_FRAME_LIMIT = 6

# HSV thresholds for the black lane boundary stripes. Value (brightness) is
# the discriminating channel for black-on-white lane markings; Saturation is
# capped low since black/gray/white are all low-saturation, and this keeps
# saturated track-side colors (grass, signs) from being mistaken for lane
# lines the way a flat grayscale cutoff can under uneven lighting/shadows.
HSV_BLACK_LOWER = np.array([0, 0, 0])
HSV_BLACK_UPPER = np.array([160, 60, 5])

class EdgeVectorsPublisher(Node):
    """
    ROS 2 Node that processes raw camera images to detect the lane edges (left/right bounds).
    It publishes the detected boundaries as synapse_msgs/EdgeVectors.
    """
    def __init__(self):
        super().__init__('edge_vectors_publisher')

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            QOS_PROFILE_DEFAULT)

        # Publisher for edge vectors.
        self.publisher_edge_vectors = self.create_publisher(
            EdgeVectors,
            '/edge_vectors',
            QOS_PROFILE_DEFAULT)

        # Publisher for thresh image (for debugging thresholding/segmentation).
        self.publisher_thresh_image = self.create_publisher(
            CompressedImage,
            "/debug_images/thresh_image",
            QOS_PROFILE_DEFAULT)

        # Publisher for vector image (for debugging vector drawing).
        self.publisher_vector_image = self.create_publisher(
            CompressedImage,
            "/debug_images/vector_image",
            QOS_PROFILE_DEFAULT)

        self.image_height = 0
        self.image_width = 0
        self.lower_image_height = 0
        self.upper_image_height = 0

        # Left/right edge continuity tracking (see select_left_right_vectors).
        self.prev_left_x = None
        self.prev_right_x = None
        self.left_lost_frames = 0
        self.right_lost_frames = 0

        self.show_perception_window = True
        if self.show_perception_window:
            cv2.namedWindow("Perception Window - Edge Vectors", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Perception Window - Edge Vectors", 800, 600)
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
        """
        Analyzes the binary threshold image and extracts left and right lane edge vectors.
        This uses basic contour finding and coordinates calculations.
        
        You can optimize this algorithm or implement alternative methods here.
        """
        # Find contours around black edge stripes detected in the binary threshold image.
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        vectors = []
        for i in range(len(contours)):
            coordinates = contours[i][:, 0, :]

            # Get coordinates representing the boundaries of the contour
            min_y_value = np.min(coordinates[:, 1])
            max_y_value = np.max(coordinates[:, 1])

            min_y_coords = np.array(coordinates[coordinates[:, 1] == min_y_value])
            max_y_coords = np.array(coordinates[coordinates[:, 1] == max_y_value])

            min_y_coord = min_y_coords[0]
            max_y_coord = max_y_coords[0]

            # Calculate contour vector magnitude
            magnitude = np.linalg.norm(min_y_coord - max_y_coord)
            if (magnitude > VECTOR_MAGNITUDE_MINIMUM):
                # Calculate distance from the camera center at the bottom of the crop
                rover_point = [self.image_width / 2, self.lower_image_height]
                middle_point = (min_y_coord + max_y_coord) / 2
                distance = np.linalg.norm(middle_point - rover_point)

                # Correct point coordinates based on vector slope angle
                angle = self.get_vector_angle_in_radians([min_y_coord, max_y_coord])
                if angle > 0:
                    min_y_coord[0] = np.max(min_y_coords[:, 0])
                else:
                    max_y_coord[0] = np.max(max_y_coords[:, 0])

                # Store vectors with distance metadata for sorting
                vectors.append([list(min_y_coord), list(max_y_coord), distance])

            # Draw all detected raw vectors in blue on the debug image
            cv2.line(image, tuple(min_y_coord), tuple(max_y_coord), BLUE_COLOR, 2)

        return vectors, image

    def select_left_right_vectors(self, vectors):
        """
        Assigns candidate vectors to the left/right lane boundary using
        continuity with the previous frame's tracked position, instead of a
        stateless "which half of the image is it in" split. A purely
        positional split can't distinguish a real edge from a wrong-side
        detection (shadow, sign post, wrong branch at a junction) -- both
        look identical to a single frame with no memory. Tracking the
        near-point x (closest to the buggy, least perspective-distorted)
        frame-to-frame means a candidate must be roughly where the edge was
        a moment ago to be accepted, so a stray detection on the "left" half
        after drifting off track won't get mistaken for the tracked line.

        `vectors` is the magnitude-filtered, distance-sorted list from
        compute_vectors_from_image: each entry is [min_y_coord, max_y_coord,
        distance], where max_y_coord is the point nearest the buggy.

        Returns (left_vector, right_vector), either of which may be None.
        """
        half_width = self.image_width / 2.0
        max_jump = self.image_width * MAX_EDGE_JUMP_RATIO

        remaining = list(vectors)

        def take_closest_match(prev_x):
            """Pops and returns the candidate whose near-point x is closest
            to prev_x, if within max_jump; otherwise leaves `remaining`
            untouched and returns None."""
            best_index, best_dist = None, max_jump
            for i, v in enumerate(remaining):
                dist = abs(v[1][0] - prev_x)
                if dist <= best_dist:
                    best_dist = dist
                    best_index = i
            return remaining.pop(best_index) if best_index is not None else None

        # 1. Try to keep tracking whichever edges we already had a lock on.
        left_vector = take_closest_match(self.prev_left_x) if self.prev_left_x is not None else None
        right_vector = take_closest_match(self.prev_right_x) if self.prev_right_x is not None else None

        # 2. Update loss counters; forget a stale lock after it's been gone
        # too long so a real re-acquisition becomes possible again.
        if left_vector is None:
            self.left_lost_frames += 1
            if self.left_lost_frames > EDGE_LOST_FRAME_LIMIT:
                self.prev_left_x = None
        else:
            self.left_lost_frames = 0

        if right_vector is None:
            self.right_lost_frames += 1
            if self.right_lost_frames > EDGE_LOST_FRAME_LIMIT:
                self.prev_right_x = None
        else:
            self.right_lost_frames = 0

        # 3. Bootstrap / re-acquire via the simple half-width split, but only
        # for a side with no active lock (first frame, or lock just given up).
        if left_vector is None and self.prev_left_x is None:
            candidates = [v for v in remaining if v[1][0] < half_width]
            if candidates:
                left_vector = candidates[0]
                remaining.remove(left_vector)

        if right_vector is None and self.prev_right_x is None:
            candidates = [v for v in remaining if v[1][0] >= half_width]
            if candidates:
                right_vector = candidates[0]
                remaining.remove(right_vector)

        if left_vector is not None:
            self.prev_left_x = left_vector[1][0]
        if right_vector is not None:
            self.prev_right_x = right_vector[1][0]

        return left_vector, right_vector

    def process_image_for_edge_vectors(self, image):
        """
        Applies basic preprocessing (Grayscale + Thresholding) and extracts lane vectors.
        
        OPTIMIZATION HINTS:
        - Inverse Perspective Mapping (IPM) / Perspective Warp: Transforming the image into a
          "birds-eye view" before processing makes steering math linear and much easier to calculate.
        - Regions of Interest (ROI): Make sure to filter out the sky, horizon, or the buggy's own chassis
          to avoid spurious noise.
        - Sliding Windows / Polynomial Fitting: Instead of simple lines, you could fit a quadratic curve
          (y = Ax^2 + Bx + C) to handle bends, curves, and intersections more smoothly.
        """
        self.image_height, self.image_width, _ = image.shape
        self.lower_image_height = int(self.image_height * VECTOR_IMAGE_HEIGHT_PERCENTAGE)
        self.upper_image_height = int(self.image_height - self.lower_image_height)

        # 1. Convert to HSV and threshold on Value (black stripes are low-V
        # regardless of hue) instead of a flat grayscale cutoff, so shadows
        # and lighting variation across the track don't shift the threshold.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(hsv, HSV_BLACK_LOWER, HSV_BLACK_UPPER)

        # 3. Crop the image to focus on the lower section close to the buggy
        thresh_cropped = thresh[self.image_height - self.lower_image_height:]
        image_cropped = image[self.image_height - self.lower_image_height:].copy()

        # 4. Compute vectors from the binary image contours
        vectors, debug_img = self.compute_vectors_from_image(image_cropped, thresh_cropped)

        # 5. Sort vectors based on distance from buggy (we prioritize vectors closer to us)
        vectors = sorted(vectors, key=lambda x: x[2])

        # 6. Assign left/right using continuity with the previous frame
        # rather than a stateless half-width split (see
        # select_left_right_vectors for why that matters).
        left_vector, right_vector = self.select_left_right_vectors(vectors)

        final_vectors = []
        for best_vector in (left_vector, right_vector):
            if best_vector is not None:
                # Draw the selected key lane vector in green on the debug image
                cv2.line(debug_img, tuple(best_vector[0]), tuple(best_vector[1]), GREEN_COLOR, 2)

                # Transform coordinates back to the original uncropped image space
                best_vector[0][1] += self.upper_image_height
                best_vector[1][1] += self.upper_image_height
                final_vectors.append(best_vector[:2])

        # Publish visual debugging images (viewable in tools like Foxglove)
        self.publish_debug_image(self.publisher_thresh_image, thresh_cropped)
        self.publish_debug_image(self.publisher_vector_image, debug_img)

        if getattr(self, 'show_perception_window', False):
            full_debug_img = image.copy()
            full_debug_img[self.upper_image_height:] = debug_img

            # Draw CONE DETECTED text if orange pixels exceed threshold in lower half
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([15, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
            roi = cv2.bitwise_or(mask1, mask2)[image.shape[0] // 2:, :]
            if cv2.countNonZero(roi) > 500:
                cv2.putText(full_debug_img, "CONE DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)

            cv2.imshow("Perception Window - Edge Vectors", full_debug_img)
            cv2.waitKey(1)

        return final_vectors

    def camera_image_callback(self, message):
        """Processes incoming camera frames and publishes detected EdgeVectors."""
        # Convert compressed image message to OpenCV format
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        vectors = self.process_image_for_edge_vectors(image)

        # Construct and publish the ROS 2 EdgeVectors message
        vectors_message = EdgeVectors()
        vectors_message.image_height = image.shape[0]
        vectors_message.image_width = image.shape[1]
        vectors_message.vector_count = 0

        # Vector 1 (usually representing Left boundary)
        if len(vectors) > 0:
            vectors_message.vector_1[0].x = float(vectors[0][0][0])
            vectors_message.vector_1[0].y = float(vectors[0][0][1])
            vectors_message.vector_1[1].x = float(vectors[0][1][0])
            vectors_message.vector_1[1].y = float(vectors[0][1][1])
            vectors_message.vector_count += 1

        # Vector 2 (usually representing Right boundary)
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