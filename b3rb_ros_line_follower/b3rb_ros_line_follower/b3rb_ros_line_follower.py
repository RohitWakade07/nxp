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
import time
import json

import math
import numpy as np
import cv2
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, ServerCommunication
from nav_msgs.msg import Odometry
import subprocess

from enum import Enum

SPEED_MIN = 0.0
SPEED_MAX = 1.0
TURN_MIN = -1.0
TURN_MAX = 1.0
QOS_PROFILE_DEFAULT = 10

class NodeStatus(Enum):
    SUCCESS = 1
    FAILURE = 2
    RUNNING = 3

class Behavior:
    def tick(self, node):
        raise NotImplementedError

class Selector(Behavior):
    def __init__(self, children):
        self.children = children
    def tick(self, node):
        for child in self.children:
            status = child.tick(node)
            if status != NodeStatus.FAILURE:
                return status
        return NodeStatus.FAILURE

class Sequence(Behavior):
    def __init__(self, children):
        self.children = children
    def tick(self, node):
        for child in self.children:
            status = child.tick(node)
            if status != NodeStatus.SUCCESS:
                return status
        return NodeStatus.SUCCESS

class Condition(Behavior):
    def __init__(self, condition_func):
        self.condition_func = condition_func
    def tick(self, node):
        if self.condition_func(node):
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE

class HandleCommunication(Behavior):
    def tick(self, node):
        if node.comm_state == "QR_DRIVE_DELAY":
            current_time = node.get_clock().now().nanoseconds / 1e9
            if current_time - node.qr_drive_start_time >= 2.0:
                node.comm_state = "SEND_QR"
                node.rover_move_manual_mode(0.0, 0.0)
                node.get_logger().info("2 seconds passed, stopping and sending QR.")
            return NodeStatus.FAILURE
            
        elif node.comm_state == "SEND_QR":
            node.send_server_msg(node.last_qr_data, is_ack=False)
            node.comm_state = "WAITING_ACK"
            node.get_logger().info(f"Sent QR {node.last_qr_data}, waiting for ACK")
            node.rover_move_manual_mode(0.0, 0.0)
            return NodeStatus.RUNNING
            
        elif node.comm_state == "WAITING_ACK":
            node.rover_move_manual_mode(0.0, 0.0)
            if node.received_ack:
                node.comm_state = "WAITING_TARGET"
                node.received_ack = False
                node.get_logger().info("Received ACK, waiting for target")
            return NodeStatus.RUNNING
            
        elif node.comm_state == "WAITING_TARGET":
            node.rover_move_manual_mode(0.0, 0.0)
            if node.received_target is not None:
                node.send_server_msg("", is_ack=True, req_uid=node.last_uid)
                node.current_target = node.received_target
                node.received_target = None
                node.comm_state = "IDLE"
                node.resume_from_stop_time = node.get_clock().now().nanoseconds / 1e9
                node.get_logger().info(f"Received target {node.current_target}, resuming navigation")
                return NodeStatus.SUCCESS
            return NodeStatus.RUNNING
            
        elif node.comm_state == "PARKED":
            node.rover_move_manual_mode(0.0, 0.0)
            if not node.sent_parked:
                node.send_server_msg("PARKED", is_ack=False)
                node.sent_parked = True
                node.get_logger().info("Sent PARKED, waiting for OK")
            
            if node.received_target == "OK":
                node.get_logger().info("Parking Confirmed OK!")
                node.comm_state = "DONE"
            elif node.received_target == "INVALID":
                node.get_logger().error("Parking INVALID!")
                node.comm_state = "DONE"
            return NodeStatus.RUNNING
            
        elif node.comm_state == "DONE":
            node.rover_move_manual_mode(0.0, 0.0)
            return NodeStatus.SUCCESS

        return NodeStatus.FAILURE

class MissionStateHandler(Behavior):
    def tick(self, node):
        if getattr(node, 'mission_state', 'INIT') == "INIT":
            if getattr(node, 'received_target', None) is not None and node.comm_state == "IDLE":
                node.current_target = node.received_target
                node.received_target = None
                node.get_logger().info(f"INIT: Received target {node.current_target} from control panel, starting!")
                
            if getattr(node, 'current_target', None) is None:
                node.rover_move_manual_mode(0.0, 0.0)
                return NodeStatus.RUNNING
            else:
                node.mission_state = "DRIVE_TO_SOURCE"
                return NodeStatus.FAILURE

        elif node.mission_state == "DRIVE_TO_SOURCE":
            if getattr(node, 'last_qr_data', None) is not None and node.comm_state == "IDLE":
                node.mission_state = "DRIVE_30M_POST_QR"
                node.distance_since_qr = 0.0
            return NodeStatus.FAILURE

        elif node.mission_state == "DRIVE_30M_POST_QR":
            if getattr(node, 'distance_since_qr', 0.0) >= getattr(node, 'qr_stop_distance', 30.0):
                node.mission_state = "STOPPED_AT_SOURCE"
                node.qr_handshake_started = False
            return NodeStatus.FAILURE

        elif node.mission_state == "STOPPED_AT_SOURCE":
            node.rover_move_manual_mode(0.0, 0.0)
            if not getattr(node, 'qr_handshake_started', False):
                node.comm_state = "SEND_QR"
                node.qr_handshake_started = True
            
            if node.comm_state == "IDLE":
                node.mission_state = "DRIVE_TO_DESTINATION"
                node.qr_handshake_started = False
                node.last_qr_data = None  # Clear so we wait for a NEW QR at destination
            return NodeStatus.RUNNING

        elif node.mission_state == "DRIVE_TO_DESTINATION":
            if getattr(node, 'last_qr_data', None) is not None and node.comm_state == "IDLE":
                node.mission_count += 1
                if node.mission_count >= getattr(node, 'total_missions', 3):
                    node.mission_state = "PARKED_TRANSITION"
                else:
                    node.mission_state = "DRIVE_TO_SOURCE"
                node.last_qr_data = None  # Clear for next mission's source
            return NodeStatus.FAILURE

        elif node.mission_state == "PARKED_TRANSITION":
            node.comm_state = "PARKED"
            node.mission_state = "DONE"
            return NodeStatus.RUNNING

        elif node.mission_state == "DONE":
            node.rover_move_manual_mode(0.0, 0.0)
            return NodeStatus.RUNNING

        return NodeStatus.FAILURE

class OvertakeBehavior(Behavior):
    def tick(self, node):
        if not hasattr(node, 'overtake_state'):
            node.overtake_state = 'IDLE'
            node.overtake_start_time = 0.0
            node.overtake_direction = 1.0

        if node.overtake_state != 'IDLE':
            return self._execute_overtake(node)

        if not hasattr(node, 'latest_scan') or node.latest_scan is None:
            return NodeStatus.FAILURE

        scan = node.latest_scan
        angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)
        ranges = np.array(scan.ranges)
        min_len = min(len(angles), len(ranges))
        angles = angles[:min_len]
        ranges = ranges[:min_len]

        valid_idx = (ranges > scan.range_min) & (ranges < 4.0) & ~np.isinf(ranges)
        valid_ranges = ranges[valid_idx]
        valid_angles = angles[valid_idx]

        obs_x = valid_ranges * np.cos(valid_angles)
        obs_y = valid_ranges * np.sin(valid_angles)

        # Detect obstacle directly in front (x < 1.0m, y < 0.4m)
        front_mask = (obs_x > 0.1) & (obs_x < 1.0) & (np.abs(obs_y) <= 0.4)
        if np.any(front_mask):
            obs_width = np.max(obs_y[front_mask]) - np.min(obs_y[front_mask])
            if obs_width > 0.25: # Slightly reduced threshold for faster reaction
                node.overtake_state = 'SWERVE_OUT'
                node.overtake_start_time = node.get_clock().now().nanoseconds / 1e9
                mean_y = np.mean(obs_y[front_mask])
                
                # If obstacle is on the right (mean_y < 0), swerve left (dir = 1.0)
                node.overtake_direction = 1.0 if mean_y < 0 else -1.0
                node.get_logger().info(f"Overtake: Obstacle detected! SWERVE_OUT dir={node.overtake_direction}")
                return self._execute_overtake(node)

        return NodeStatus.FAILURE

    def _execute_overtake(self, node):
        current_time = node.get_clock().now().nanoseconds / 1e9
        elapsed = current_time - node.overtake_start_time
        
        speed = 0.4
        steer_mag = 0.8

        if node.overtake_state == 'SWERVE_OUT':
            # Phase 1: Swerve out of the lane
            if elapsed < 0.8: # Shortened slightly to prevent going entirely off-track
                node.rover_move_manual_mode(speed, node.overtake_direction * steer_mag)
                return NodeStatus.RUNNING
            else:
                node.overtake_state = 'STRAIGHT'
                node.overtake_start_time = current_time
                node.get_logger().info("Overtake: STRAIGHT (Checking side clearance...)")
                return NodeStatus.RUNNING

        elif node.overtake_state == 'STRAIGHT':
            # Phase 2: Drive parallel and scan the side
            node.rover_move_manual_mode(speed, 0.0)
            
            scan = node.latest_scan
            ranges = np.array(scan.ranges)
            angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)[:len(ranges)]
            
            # Define "Side Vision" cone. 
            # If we swerved Left (direction 1.0), look Right (negative angles -1.5 to -0.3 rad)
            if node.overtake_direction == 1.0:
                side_mask = (angles > -1.5) & (angles < -0.3)
            else:
                side_mask = (angles > 0.3) & (angles < 1.5)
                
            side_ranges = ranges[side_mask]
            
            # Check if there are any points dangerously close (under 0.8 meters) on our side
            side_clear = True
            if len(side_ranges) > 0:
                min_side_dist = np.min(side_ranges)
                if min_side_dist < 0.8:
                    side_clear = False

            # We must drive straight for at least 0.5s to clear our own back wheels, 
            # AND the side must be clear of the obstacle.
            if side_clear and elapsed > 0.5:
                node.overtake_state = 'SWERVE_IN'
                node.overtake_start_time = current_time
                node.get_logger().info("Overtake: Obstacle cleared! SWERVE_IN")
                
            # Failsafe: If we are stuck driving straight for 3 seconds, force return to lane
            if elapsed > 3.0:
                node.overtake_state = 'SWERVE_IN'
                node.overtake_start_time = current_time
                node.get_logger().warn("Overtake: Clearance timeout. Forcing SWERVE_IN")

            return NodeStatus.RUNNING
        
        elif node.overtake_state == 'SWERVE_IN':
            # Phase 3: Return to lane
            if elapsed < 0.8:
                node.rover_move_manual_mode(speed, -node.overtake_direction * steer_mag)
                return NodeStatus.RUNNING
            else:
                node.overtake_state = 'IDLE'
                node.get_logger().info("Overtake: DONE")
                return NodeStatus.SUCCESS

        return NodeStatus.RUNNING

class NavigationAction(Behavior):
    def tick(self, node):
        if node.vectors is None:
            node.rover_move_manual_mode(0.0, 0.0)
            return NodeStatus.RUNNING
            
        vectors = node.vectors
        current_time = node.get_clock().now().nanoseconds / 1e9

        # Fallback to stored last valid vectors if resuming from a stop and current vectors are empty
        if (current_time - getattr(node, 'resume_from_stop_time', 0.0) < 3.0) and vectors.vector_count == 0 and getattr(node, 'last_valid_vectors', None) is not None:
            vectors = node.last_valid_vectors
            node.get_logger().info("Using stored last valid vectors post-stop to prevent collision", throttle_duration_sec=1.0)

        half_width = vectors.image_width / 2.0
        target_speed = 0.5
        desired_dir = None
        raw_turn = 0.0

        if vectors.vector_count == 0:
            node.blind_frame_count += 1
            # Immediately slow down to 0.2 when no lane is detected
            target_speed = 0.2
            target = getattr(node, 'current_target', None) or getattr(node, 'target_patient', None)
            if target and hasattr(node, 'sign_config_map') and target in node.sign_config_map:
                desired_dir = node.sign_config_map[target]
                node.get_logger().info(f"Intersection: Target '{target}' mapped to direction '{desired_dir}'", throttle_duration_sec=1.0)
            elif hasattr(node, 'last_detected_sign'):
                desired_dir = node.last_detected_sign

            if desired_dir in ["left", "Left", "right", "Right"]:
                # Kinematic turn calculation (Ackermann)
                wheelbase = 0.3
                turn_radius = 0.6  # Estimated center-lane turning radius for 90-degree intersection
                delta = math.atan(wheelbase / turn_radius)
                max_steer_angle = 0.5  # Physical steering limit in radians
                kinematic_turn = delta / max_steer_angle
                
                if desired_dir in ["left", "Left"]:
                    raw_turn = kinematic_turn
                else:
                    raw_turn = -kinematic_turn
            elif desired_dir in ["straight", "Straight", "up", "Up"]:
                # --- Wide intersection fix ---
                # Force a hard zero, override any residual smoothed_turn/lattice bias,
                # and crawl even slower until lines reappear. Don't let the planner
                # steer based on stale LiDAR data in an open crossing.
                raw_turn = 0.0
                target_speed = 0.2  # minimum speed of 0.2
                node.smoothed_turn = 0.0  # snap-reset instead of blending — kills residual drift immediately
                node.use_lattice_planner_override = True  # NEW flag, see below
            else:
                # Safety-net fallback: we don't have a resolved direction, so hold the current smoothed turn steady
                # rather than guessing a new direction, as this is now a rare edge case.
                raw_turn = node.smoothed_turn
                node.get_logger().warn("Intersection: No resolved direction available, holding current smoothed turn.", throttle_duration_sec=1.0)
        elif vectors.vector_count == 1:
            node.blind_frame_count = 0  # reset when lines reappear
            deviation = vectors.vector_1[1].x - vectors.vector_1[0].x
            error = deviation / half_width
            derivative = error - node.last_error
            raw_turn = (error * 1.2) + (derivative * 0.8)
            target_speed = max(0.3, 0.7 - abs(error) * 0.4)
            node.last_error = error

            # Track error history
            node.error_history.append(error)
            if len(node.error_history) > 10:
                node.error_history.pop(0)
        elif vectors.vector_count == 2:
            node.blind_frame_count = 0  # reset when lines reappear
            # Pure Pursuit: Look-ahead point (top of the lane lines)
            top_x_left = vectors.vector_1[0].x
            top_x_right = vectors.vector_2[0].x
            lane_center = (top_x_left + top_x_right) / 2.0
            deviation = half_width - lane_center
            error = deviation / half_width
            derivative = error - node.last_error
            raw_turn = (error * 1.5) + (derivative * 1.0)
            # Slow down slightly on curves to maintain smoothness at higher speeds
            target_speed = max(0.3, 0.7 - abs(error) * 0.5)
            node.last_error = error

            # Track error history
            node.error_history.append(error)
            if len(node.error_history) > 10:
                node.error_history.pop(0)

        # Higher alpha to reduce phase lag (which causes oscillations)
        alpha = 0.6
        if desired_dir not in ["straight", "Straight", "up", "Up"] or vectors.vector_count != 0:
            node.smoothed_turn = alpha * raw_turn + (1.0 - alpha) * node.smoothed_turn
            node.use_lattice_planner_override = False
        
        # Intersection approach slow down
        if getattr(node, 'sign_approach_start_time', 0.0) > 0.0:
            if current_time - node.sign_approach_start_time > 1.0:
                target_speed = min(target_speed, 0.2)
                
            # Reset if we are clear of the intersection (no signs for a while and have lines)
            if current_time - getattr(node, 'last_sign_detection_time', 0.0) > 4.0 and vectors.vector_count > 0:
                node.sign_approach_start_time = 0.0
                node.last_detected_sign = None

        # Smooth the speed transition
        if not hasattr(node, 'smoothed_speed'):
            node.smoothed_speed = 0.5
        node.smoothed_speed = 0.7 * node.smoothed_speed + 0.3 * target_speed

        node.get_logger().info(f"Lane Tracking Error: {node.last_error:.2f} (midpoint deviation)", throttle_duration_sec=0.5)

        if getattr(node, 'use_lattice_planner_override', False):
            # Bypass lattice planner entirely during wide straight-through crossings —
            # its obstacle avoidance can misfire on open cross-junction LiDAR returns.
            planner_speed, planner_turn = node.smoothed_speed, 0.0
        else:
            planner_speed, planner_turn = node._run_lattice_planner(node.smoothed_turn, node.smoothed_speed)
            
        node.rover_move_manual_mode(planner_speed, planner_turn)
        return NodeStatus.RUNNING

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')

        self.subscription_vectors = self.create_subscription(EdgeVectors, '/edge_vectors', self.edge_vectors_callback, QOS_PROFILE_DEFAULT)
        self.subscription_lidar = self.create_subscription(LaserScan, '/scan', self.lidar_callback, QOS_PROFILE_DEFAULT)
        self.subscription_server = self.create_subscription(ServerCommunication, '/ServerCommunication', self.server_communication_callback, QOS_PROFILE_DEFAULT)
        self.subscription_qr = self.create_subscription(String, '/qr_detection', self.qr_detection_callback, QOS_PROFILE_DEFAULT)
        self.subscription_signs = self.create_subscription(String, '/sign_board_detection', self.sign_board_callback, QOS_PROFILE_DEFAULT)
        self.subscription_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, QOS_PROFILE_DEFAULT)

        self.publisher_joy = self.create_publisher(Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)
        self.publisher_server = self.create_publisher(ServerCommunication, '/ServerCommunication', QOS_PROFILE_DEFAULT)

        self.target_speed = 0.0
        self.target_turn = 0.0
        
        self.latest_scan = None
        self.vectors = None
        self.smoothed_turn = 0.0
        self.last_error = 0.0
        
        # NEW: rolling buffer of recent errors
        self.error_history = []
        # NEW: consecutive frames with 0 vectors
        self.blind_frame_count = 0
        # NEW: persists the predicted turn while blind
        self.blind_turn_estimate = 0.0
        # NEW: smooths out speed drops when blind
        self.smoothed_speed = 0.5
        
        # Stored vectors for resuming post-stop
        self.last_valid_vectors = None
        self.resume_from_stop_time = 0.0
        
        self.show_lattice_window = True
        if self.show_lattice_window:
            cv2.namedWindow("Lattice Planner", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Lattice Planner", 400, 400)

        # Mission state variables
        self.last_detected_sign = None
        self.sign_approach_start_time = 0.0
        self.last_sign_detection_time = 0.0
        self.sign_config_map = {}
        self.target_patient = None
        self.comm_state = "IDLE"
        self.last_qr_data = None
        self.last_processed_qr = None
        self.qr_drive_start_time = 0.0
        
        # New MissionStateHandler state variables
        self.mission_state = "INIT"
        self.mission_count = 0
        self.total_missions = 3
        self.distance_since_qr = 0.0
        self.last_odom_pos = None
        self.qr_stop_distance = 30.0
        self.qr_handshake_started = False

        # Server Comm Tracking
        self.uid_counter = 0
        self.last_sent_uid = 0
        self.last_uid = 0
        self.received_ack = False
        self.received_target = None
        self.current_target = None
        self.sent_parked = False

        # Behavior Tree Root
        self.bt_root = Selector([
            Sequence([
                Condition(lambda n: n.comm_state != "IDLE"),
                HandleCommunication()
            ]),
            MissionStateHandler(),
            OvertakeBehavior(),
            NavigationAction()
        ])

        self.control_timer = self.create_timer(0.05, self.tick_bt_and_publish)
        self.get_logger().info("Line Follower initialized. Behavior Tree Active.")

    def tick_bt_and_publish(self):
        self.bt_root.tick(self)
        
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        is_overtaking = 1.0 if getattr(self, 'overtake_state', 'IDLE') != 'IDLE' else 0.0
        msg.axes = [0.0, self.target_speed, is_overtaking, self.target_turn]
        self.publisher_joy.publish(msg)

    def rover_move_manual_mode(self, speed, turn):
        self.target_speed = float(max(min(speed, SPEED_MAX), -SPEED_MAX))
        self.target_turn = float(max(min(turn, TURN_MAX), -TURN_MAX))

    def edge_vectors_callback(self, message):
        self.vectors = message
        if message.vector_count > 0:
            self.last_valid_vectors = message

    def _run_lattice_planner(self, desired_steer, desired_speed=0.8):
        if not hasattr(self, 'latest_scan') or self.latest_scan is None:
            return desired_speed, desired_steer

        scan = self.latest_scan
        angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)
        ranges = np.array(scan.ranges)
        min_len = min(len(angles), len(ranges))
        angles = angles[:min_len]
        ranges = ranges[:min_len]

        valid_idx = (ranges > scan.range_min) & (ranges < 4.0) & ~np.isinf(ranges)
        valid_ranges = ranges[valid_idx]
        valid_angles = angles[valid_idx]

        obs_x = valid_ranges * np.cos(valid_angles)
        obs_y = valid_ranges * np.sin(valid_angles)

        # Filter obstacles to ignore track walls 
        # Filter obstacles to ignore track walls 
        track_mask = (obs_x > 0.0) & (obs_x < 2.0) & (np.abs(obs_y) <= 0.40)
        obs_x = obs_x[track_mask]
        obs_y = obs_y[track_mask]
        obstacles = np.column_stack((obs_x, obs_y))

        # --- Planner Parameters ---
        dt = 0.20
        steps = 10 # 2.0 seconds forward look-ahead (10 * 0.20s)
        wheelbase = 0.3
        LAT_ACCEL_MAX = 1.2
        num_candidates = 25 # Increased candidate density
        
        candidates = np.linspace(-1.0, 1.0, num_candidates)
        max_steer_angle = 0.5 # Physical steering limit in radians
        
        vis_img = None
        if getattr(self, 'show_lattice_window', False):
            vis_img = np.ones((500, 500, 3), dtype=np.uint8) * 255
            for ox, oy in zip(obs_x, obs_y):
                px, py = int(250 - oy * 100), int(500 - ox * 100)
                if 0 <= px < 500 and 0 <= py < 500:
                    cv2.circle(vis_img, (px, py), 3, (0, 0, 0), -1)

        min_cost = float('inf')
        best_steer = desired_steer
        best_path_points = None
        best_obs_dist = float('inf')
        best_speed = desired_speed

        # --- Evaluate Paths ---
        for cand_steer in candidates:
            delta = cand_steer * max_steer_angle
            
            # 1. Exact Kinematic Path Generation
            if abs(delta) < 1e-4:
                # Straight line
                x = np.linspace(0, desired_speed * dt * steps, steps)
                y = np.zeros(steps)
                v_step = desired_speed
            else:
                # Ackermann arc
                R = wheelbase / math.tan(delta)
                v_curve_limit = math.sqrt(LAT_ACCEL_MAX * abs(R))
                v_step = max(0.2, min(desired_speed, v_curve_limit))
                
                d_theta = (v_step / R) * dt
                theta = np.cumsum(np.full(steps, d_theta))
                x = R * np.sin(theta)
                y = R * (1 - np.cos(theta))

            path_points = np.column_stack((x, y))
            min_path_dist = float('inf')
            collision = False

            # 2. Vectorized Collision Checking
            if len(obstacles) > 0:
                # Calculate distance from every path point to every obstacle using broadcasting
                diff = path_points[:, np.newaxis, :] - obstacles[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                min_dists_per_step = np.min(dists, axis=1)
                min_path_dist = np.min(min_dists_per_step)

                if min_path_dist < 0.15: # Hard collision boundary (15cm)
                    continue

            # 3. Cost Evaluation
            cost = 0.0
            
            # Soft Obstacle Penalty (Reduced buffer for closer passing)
            if min_path_dist < 0.6:
                cost += 20.0 * math.exp(-4.0 * (min_path_dist - 0.15))

            # Tracking Penalty (deviation from ideal lane keeping)
            steer_diff = abs(cand_steer - desired_steer)
            cost += 10.0 * steer_diff + 25.0 * (steer_diff ** 2)

            # Smoothness Penalty (deviation from current actuator state)
            cost += 5.0 * abs(cand_steer - self.target_turn)

            # Directional Bias (Overtake logic preference)
            if len(obstacles) > 0:
                closest_obs_idx = np.argmin(np.linalg.norm(obstacles, axis=1))
                closest_obs_y = obstacles[closest_obs_idx, 1]
                if closest_obs_y > 0 and cand_steer > 0:
                    cost += 8.0 * cand_steer
                elif closest_obs_y < 0 and cand_steer < 0:
                    cost += 8.0 * abs(cand_steer)

            # 4. Selection
            if cost < min_cost:
                min_cost = cost
                best_steer = cand_steer
                best_path_points = path_points
                best_obs_dist = min_path_dist
                best_speed = v_step

        # --- Fallback & Speed Adjustment ---
        if min_cost == float('inf'):
            self.get_logger().warn("Lattice Planner: All paths blocked! Reactive turn.", throttle_duration_sec=1.0)
            if len(obstacles) > 0:
                closest_obs_idx = np.argmin(np.linalg.norm(obstacles, axis=1))
                if obstacles[closest_obs_idx, 1] > 0:
                    best_steer = -1.0 # Obstacle on left, turn right
                else:
                    best_steer = 1.0  # Obstacle on right, turn left
            return max(0.2, desired_speed * 0.4), best_steer

        # Dynamic proximity slowdown
        final_speed = best_speed
        if best_obs_dist < 0.35:
            final_speed = max(0.3, best_speed * (best_obs_dist / 0.35))

        # --- Visualization ---
        if vis_img is not None and best_path_points is not None:
            for i in range(len(best_path_points) - 1):
                p1, p2 = best_path_points[i], best_path_points[i + 1]
                cv2.line(vis_img, (int(250 - p1[1] * 100), int(500 - p1[0] * 100)),
                          (int(250 - p2[1] * 100), int(500 - p2[0] * 100)), (0, 200, 0), 2)
            # cv2.imshow("Lattice Planner", vis_img)
            cv2.waitKey(1)

        return final_speed, best_steer

    def lidar_callback(self, message):
        self.latest_scan = message

    def odom_callback(self, msg):
        current_pos = msg.pose.pose.position
        if getattr(self, 'last_odom_pos', None) is not None and getattr(self, 'mission_state', '') == "DRIVE_30M_POST_QR":
            dx = current_pos.x - self.last_odom_pos.x
            dy = current_pos.y - self.last_odom_pos.y
            self.distance_since_qr += math.sqrt(dx*dx + dy*dy)
        self.last_odom_pos = current_pos

    def server_communication_callback(self, message):
        if message.dest == 1:
            if message.ack == 1 and message.uid == self.last_sent_uid:
                self.received_ack = True
            elif message.ack == 0:
                self.received_target = message.msg
                self.last_uid = message.uid

    def send_server_msg(self, text_msg, is_ack=False, req_uid=None):
        server_msg = ServerCommunication()
        server_msg.src = 1
        server_msg.dest = 2
        
        if is_ack:
            server_msg.uid = req_uid
            server_msg.ack = 1
        else:
            self.uid_counter = (self.uid_counter + 1) % 256
            self.last_sent_uid = self.uid_counter
            server_msg.uid = self.uid_counter
            server_msg.ack = 0
            
        server_msg.msg = text_msg
        self.publisher_server.publish(server_msg)

    def qr_detection_callback(self, message):
        qr_data = message.data
        if self.comm_state == "IDLE" and qr_data != self.last_processed_qr:
            # --- Fake hospital / fake patient filter ---
            target_to_loc = {
                'A': 'PATIENT_1',
                'B': 'PATIENT_2',
                'C': 'PATIENT_3',
                'X': 'HOSPITAL_1',
                'Y': 'HOSPITAL_2',
                'Z': 'HOSPITAL_3'
            }
            current_target = getattr(self, 'current_target', None)
            
            if current_target in target_to_loc:
                expected_loc = target_to_loc[current_target]
                # Check if the expected location string is within the JSON qr_data
                if f"{{LOC: {expected_loc}}}" not in qr_data:
                    self.get_logger().info(f"Ignoring fake QR: expected '{expected_loc}' for target '{current_target}', but got '{qr_data}'")
                    self.last_processed_qr = qr_data
                    return

            self.last_qr_data = qr_data
            self.last_processed_qr = qr_data
            self.qr_drive_start_time = self.get_clock().now().nanoseconds / 1e9
            self.comm_state = "QR_DRIVE_DELAY"
            self.get_logger().info(f"QR Detected: {qr_data}. Driving for 2 more seconds.")
            
        # Optional: logic to detect parking spot
        if "PARKING" in qr_data and self.comm_state == "IDLE":
            self.comm_state = "PARKED"

    def sign_board_callback(self, message):
        if message.data:
            try:
                payload = json.loads(message.data)
                current_time = self.get_clock().now().nanoseconds / 1e9
                if payload.get("detected") and len(payload.get("detections", [])) > 0:
                    sign_class = payload["detections"][0]["class"]
                    if sign_class != getattr(self, 'last_detected_sign', None) or getattr(self, 'sign_approach_start_time', 0.0) == 0.0:
                        self.last_detected_sign = sign_class
                        self.sign_approach_start_time = current_time
                        self.get_logger().info(f"Stored detected sign: {self.last_detected_sign}, started approach timer")
                    self.last_sign_detection_time = current_time

                sign_config = payload.get("sign_config")
                if sign_config:
                    self.sign_config_map.update(sign_config)
                    self.get_logger().info(f"Stored sign_config map: {self.sign_config_map}")
            except Exception as e:
                self.get_logger().error(f"Failed to parse sign board data: {e}")

    def _run_follow_the_gap(self, desired_steer, desired_speed=0.8):
        """
        Purely reactive obstacle avoidance using the F1TENTH Follow the Gap algorithm.
        """
        # 1. Check for valid LiDAR data
        if not hasattr(self, 'latest_scan') or self.latest_scan is None:
            return desired_speed, desired_steer

        scan = self.latest_scan
        ranges = np.array(scan.ranges)
        angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)[:len(ranges)]

        # 2. Limit Field of View to front 140 degrees (-70 to +70) to ignore side track walls
        fov = math.radians(70)
        fov_mask = (angles > -fov) & (angles < fov)
        ranges = ranges[fov_mask]
        angles = angles[fov_mask]

        # Clean data: Replace NaNs/Infs and cap max distance
        max_range = 2.5
        ranges = np.nan_to_num(ranges, posinf=max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, max_range)

        # 3. Vision Priority: If track is clear straight ahead, let the camera drive
        front_mask = (angles > -0.3) & (angles < 0.3)
        if len(ranges[front_mask]) > 0 and np.min(ranges[front_mask]) > 1.2:
            # We also clear the visualization window so it doesn't freeze on old data
            if getattr(self, 'show_lattice_window', False):
                # cv2.imshow("Lattice Planner", np.ones((500, 500, 3), dtype=np.uint8) * 255)
                cv2.waitKey(1)
            return desired_speed, desired_steer

        # 4. Find the closest obstacle
        closest_idx = np.argmin(ranges)
        closest_dist = max(ranges[closest_idx], 0.1) # Avoid division by zero

        # 5. Draw Safety Bubble (Mask out the obstacle + half your rover width)
        bubble_radius = 0.45  # Meters. Adjust based on your rover's physical width
        
        # Calculate how many array indices this radius covers at this specific distance
        angle_span = math.atan2(bubble_radius, closest_dist)
        idx_span = int(angle_span / scan.angle_increment)

        start_idx = max(0, closest_idx - idx_span)
        end_idx = min(len(ranges), closest_idx + idx_span + 1)
        
        # Zero out the points inside the bubble so the rover treats them as a wall
        proc_ranges = np.copy(ranges)
        proc_ranges[start_idx:end_idx] = 0.0

        # 6. Find all Gaps
        gap_threshold = 1.0  # Meters. Points further than this are considered "free space"
        gaps = []
        current_gap = []
        
        for i, r in enumerate(proc_ranges):
            if r > gap_threshold:
                current_gap.append(i)
            else:
                if len(current_gap) > 0:
                    gaps.append(current_gap)
                    current_gap = []
        if len(current_gap) > 0:
            gaps.append(current_gap)

        # Filter gaps to make sure the vehicle can physically fit (e.g. at least 15 points wide)
        valid_gaps = [g for g in gaps if len(g) >= 15]
        if not valid_gaps:
            valid_gaps = gaps  # Fallback to any gaps if none are wide enough

        if not valid_gaps:
            # Fallback: No gaps found! Turn hard away from the closest point
            self.get_logger().warn("FTG: All paths blocked! Emergency avoid.", throttle_duration_sec=1.0)
            avoid_steer = -1.0 if angles[closest_idx] > 0 else 1.0
            return 0.2, avoid_steer

        # 7. Find the Best Point across ALL valid gaps
        # Instead of blindly picking the largest gap, we choose the gap and point
        # that aligns closest to the lane tracker's desired steering.
        desired_angle = desired_steer * 0.5 
        best_point_idx = None
        min_angle_diff = float('inf')
        
        for gap in valid_gaps:
            for idx in gap:
                angle_diff = abs(angles[idx] - desired_angle)
                if angle_diff < min_angle_diff:
                    min_angle_diff = angle_diff
                    best_point_idx = idx

        if best_point_idx is None:
            largest_gap = max(valid_gaps, key=len)
            best_point_idx = largest_gap[len(largest_gap) // 2] 

        target_angle = angles[best_point_idx]

        # 8. Convert target angle back to a steering command (-1.0 to 1.0)
        best_steer = max(-1.0, min(1.0, target_angle / 0.5))

        # Dynamically slow down on sharp avoidance maneuvers
        final_speed = max(0.3, desired_speed - abs(best_steer) * 0.4)

        # --- Debug Visualization ---
        # Kept the window name "Lattice Planner" so you don't need to change OpenCV setups
        if getattr(self, 'show_lattice_window', False):
            vis_img = np.ones((500, 500, 3), dtype=np.uint8) * 255
            
            # Draw standard LiDAR points (Gray)
            for r, a in zip(ranges, angles):
                px = int(250 - (r * math.sin(a)) * 100)
                py = int(500 - (r * math.cos(a)) * 100)
                if 0 <= px < 500 and 0 <= py < 500:
                    cv2.circle(vis_img, (px, py), 2, (200, 200, 200), -1)
                    
            # Draw the safety bubble (Red)
            for i in range(start_idx, end_idx):
                r, a = ranges[i], angles[i]
                px = int(250 - (r * math.sin(a)) * 100)
                py = int(500 - (r * math.cos(a)) * 100)
                if 0 <= px < 500 and 0 <= py < 500:
                    cv2.circle(vis_img, (px, py), 4, (0, 0, 255), -1)
                    
            # Draw the chosen steering path (Green)
            target_px = int(250 - (2.0 * math.sin(target_angle)) * 100)
            target_py = int(500 - (2.0 * math.cos(target_angle)) * 100)
            cv2.line(vis_img, (250, 500), (target_px, target_py), (0, 255, 0), 3)
            
            # cv2.imshow("Lattice Planner", vis_img)
            cv2.waitKey(1)

        self.get_logger().info(f"FTG: Obstacle Avoided. Steer {best_steer:.2f}", throttle_duration_sec=0.5)

        return final_speed, best_steer

def main(args=None):
    try:
        subprocess.Popen(['kazam'])
        print("Started Kazam screen recorder.")
    except Exception as e:
        pass

    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()