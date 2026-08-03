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
import subprocess

from enum import Enum

SPEED_MIN = 0.0
SPEED_MAX = 1.0
TURN_MIN = -1.0
TURN_MAX = 1.0
QOS_PROFILE_DEFAULT = 10

# --- SPEED & ACCELERATION CONFIGURATION PARAMETERS ---
TARGET_CRUISE_SPEED = 0.5          # Desired cruising speed on straight paths
MIN_CORNER_SPEED = 0.2             # Minimum speed floor on sharp curves / slow maneuvers
MAX_ACCEL_LIMIT = 2.0              # Maximum acceleration rate limit (units/sec^2)
SPEED_CURVE_SLOWDOWN_GAIN = 0.5    # Speed reduction factor on curves based on steering error

# --- LATTICE PLANNER CONFIGURATION PARAMETERS ---
LATTICE_LOOKAHEAD_DT = 0.20        # Time step per simulation step (seconds)
LATTICE_STEPS = 10                 # Total steps ahead (10 steps * 0.20s = 2.0s lookahead)
LATTICE_WHEELBASE = 0.3            # Vehicle wheelbase (meters)
LATTICE_LAT_ACCEL_MAX = 1.2        # Max lateral acceleration limit (m/s^2) for cornering speed calculation
LATTICE_NUM_CANDIDATES = 25        # Number of steering arc trajectories sampled between [-1.0, 1.0]
LATTICE_MAX_STEER_ANGLE = 0.62     # Maximum physical steering angle (radians)
LATTICE_COLLISION_RADIUS = 0.15    # Hard collision distance threshold (meters)
LATTICE_TRACK_X_MAX = 2.0          # Forward distance filter limit for relevant obstacles (meters)
LATTICE_TRACK_Y_MAX = 0.40         # Lateral width filter limit for track obstacles (meters)

# --- COST FUNCTION PARAMETERS ("OBSTACLE HUGGING" & TRAJECTORY TUNING) ---
COST_OBSTACLE_BUFFER_DIST = 0.7    # Distance threshold (m) below which obstacle proximity cost is applied
COST_OBSTACLE_WEIGHT = 5.0        # Weight multiplier for obstacle proximity penalty
COST_OBSTACLE_DECAY = 6.0          # Exponential decay factor (higher value = passes closer/hugs obstacle)
COST_TRACKING_LINEAR_WEIGHT = 10.0 # Weight for linear deviation from ideal line-following steering
COST_TRACKING_QUAD_WEIGHT = 15.0   # Weight for quadratic deviation from ideal line-following steering
COST_SMOOTHNESS_WEIGHT = 3.0       # Weight for steering angle change from current steering state
COST_DIRECTIONAL_BIAS_WEIGHT = 8.0 # Penalty weight for steering towards the side of the closest obstacle

# --- REPULSIVE FORCE PARAMETERS ---
REPULSIVE_FORCE_GAIN = 0.0        # Tuning gain for obstacle repulsion force
REPULSIVE_LOOKAHEAD_DIST = 0     # Maximum distance to consider obstacles for repulsion

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
        half_width = vectors.image_width / 2.0
        speed = 0.5  # Normal speed
        raw_turn = 0.0

        if vectors.vector_count == 0:
            speed = 0.2

            # --- OBSTACLE AVOIDANCE WHEN NO LANE IS DETECTED ---
            # Check if LiDAR sees an obstacle in the forward zone.
            # If yes: treat lane as straight and use LiDAR repulsion to steer away.
            lidar_obstacle_avoidance_active = False
            if node.latest_scan is not None:
                _scan = node.latest_scan
                _angles = np.arange(_scan.angle_min, _scan.angle_max, _scan.angle_increment)
                _ranges = np.array(_scan.ranges)
                _min_len = min(len(_angles), len(_ranges))
                _angles = _angles[:_min_len]
                _ranges = _ranges[:_min_len]

                # Only look at a narrow forward cone (±40°) for real obstacles
                _fwd_mask = (_angles > -0.70) & (_angles < 0.70)
                _fwd_ranges = _ranges[_fwd_mask]
                _fwd_angles = _angles[_fwd_mask]

                # Clean: ignore invalid readings
                _valid = (_fwd_ranges > _scan.range_min) & (_fwd_ranges < 4.0) & ~np.isinf(_fwd_ranges)
                _fwd_ranges = _fwd_ranges[_valid]
                _fwd_angles = _fwd_angles[_valid]

                if len(_fwd_ranges) > 0:
                    _min_fwd_dist = float(np.min(_fwd_ranges))
                    # Obstacle within 1.2 m triggers avoidance
                    if _min_fwd_dist < 1.2:
                        lidar_obstacle_avoidance_active = True
                        # Compute a summed repulsive steer from all close front points
                        _obs_x = _fwd_ranges * np.cos(_fwd_angles)
                        _obs_y = _fwd_ranges * np.sin(_fwd_angles)
                        _avoid_steer = 0.0
                        for ox, oy in zip(_obs_x, _obs_y):
                            _dist = max(0.1, math.sqrt(ox**2 + oy**2))
                            _force = 0.15 / (_dist ** 2)
                            if oy > 0:
                                _avoid_steer -= _force  # obstacle left → steer right
                            else:
                                _avoid_steer += _force  # obstacle right → steer left
                        _avoid_steer = max(-1.0, min(1.0, _avoid_steer))
                        raw_turn = _avoid_steer
                        speed = 0.25
                        node.get_logger().info(
                            f"No lane + obstacle at {_min_fwd_dist:.2f}m: LiDAR avoidance steer={_avoid_steer:.2f}",
                            throttle_duration_sec=0.5)

            if not lidar_obstacle_avoidance_active:
                target = getattr(node, 'current_target', None) or getattr(node, 'target_patient', None)
                desired_dir = None
                if target and hasattr(node, 'sign_config_map') and target in node.sign_config_map:
                    desired_dir = node.sign_config_map[target]
                    node.get_logger().info(f"Intersection: Target '{target}' mapped to direction '{desired_dir}'", throttle_duration_sec=1.0)
                elif hasattr(node, 'last_detected_sign'):
                    desired_dir = node.last_detected_sign

                if desired_dir in ["left", "Left"]:
                    raw_turn = 0.6
                elif desired_dir in ["right", "Right"]:
                    raw_turn = -0.6
                elif desired_dir in ["straight", "Straight", "up", "Up"]:
                    raw_turn = 0.0
                else:
                    if node.last_error > 0.2:
                        raw_turn = 0.6  # line is to the left, steer left to find it
                    elif node.last_error < -0.2:
                        raw_turn = -0.6 # line is to the right, steer right to find it
                    else:
                        raw_turn = -0.5
        elif vectors.vector_count == 1:
            deviation = vectors.vector_1[1].x - vectors.vector_1[0].x
            error = deviation / half_width
            derivative = error - node.last_error
            raw_turn = (error * 1.2) + (derivative * 0.8)
            speed = max(MIN_CORNER_SPEED, TARGET_CRUISE_SPEED - abs(error) * SPEED_CURVE_SLOWDOWN_GAIN)
            node.last_error = error
        elif vectors.vector_count == 2:
            # Pure Pursuit: Look-ahead point (top of the lane lines)
            top_x_left = vectors.vector_1[0].x
            top_x_right = vectors.vector_2[0].x
            lane_center = (top_x_left + top_x_right) / 2.0
            deviation = half_width - lane_center
            error = deviation / half_width
            derivative = error - node.last_error
            raw_turn = (error * 1.5) + (derivative * 1.0)
            # Slow down slightly on curves to maintain smoothness at higher speeds
            speed = max(MIN_CORNER_SPEED, TARGET_CRUISE_SPEED - abs(error) * SPEED_CURVE_SLOWDOWN_GAIN)
            node.last_error = error

        # Compute repulsive force from nearby obstacles
        repulsive_steer_offset = 0.0
        if node.latest_scan is not None:
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

            # Filter to relevant track region
            track_mask = (obs_x > 0.0) & (obs_x < LATTICE_TRACK_X_MAX) & (np.abs(obs_y) <= LATTICE_TRACK_Y_MAX)
            obs_x = obs_x[track_mask]
            obs_y = obs_y[track_mask]

            if len(obs_x) > 0:
                for ox, oy in zip(obs_x, obs_y):
                    dist = math.sqrt(ox**2 + oy**2)
                    if dist > 0.1 and dist < REPULSIVE_LOOKAHEAD_DIST:
                        # Repulsive force points away from the obstacle
                        # oy > 0 means obstacle on left -> steer right (negative turn)
                        # oy < 0 means obstacle on right -> steer left (positive turn)
                        force = REPULSIVE_FORCE_GAIN / (dist ** 2)
                        if oy > 0:
                            repulsive_steer_offset -= force
                        else:
                            repulsive_steer_offset += force
                
                # Cap the maximum contribution of repulsive steer offset
                repulsive_steer_offset = max(-0.5, min(0.5, repulsive_steer_offset))

        # Higher alpha to reduce phase lag (which causes oscillations)
        alpha = 0.6
        node.smoothed_turn = alpha * raw_turn + (1.0 - alpha) * node.smoothed_turn

        # Add the repulsive force offset to the smoothed turn command
        target_turn_command = max(-1.0, min(1.0, node.smoothed_turn + repulsive_steer_offset))

        node.get_logger().info(f"Lane Tracking Error: {node.last_error:.2f} (midpoint deviation), Repulsive Offset: {repulsive_steer_offset:.2f}", throttle_duration_sec=0.5)

        planner_speed, planner_turn = node._run_lattice_planner(target_turn_command, speed)
            
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

        self.publisher_joy = self.create_publisher(Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)
        self.publisher_server = self.create_publisher(ServerCommunication, '/ServerCommunication', QOS_PROFILE_DEFAULT)

        self.target_speed = 0.0
        self.target_turn = 0.0
        
        self.latest_scan = None
        self.vectors = None
        self.smoothed_turn = 0.0
        self.last_error = 0.0
        
        self.show_lattice_window = True
        if self.show_lattice_window:
            cv2.namedWindow("Lattice Planner", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Lattice Planner", 400, 400)

        # Mission state variables
        self.last_detected_sign = None
        self.sign_config_map = {}
        self.target_patient = None
        self.comm_state = "IDLE"
        self.last_qr_data = None
        self.last_processed_qr = None
        self.qr_drive_start_time = 0.0
        
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
            # OvertakeBehavior(),  # Disabled S-curve overtake behavior for now
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
        # Smoothly limit linear acceleration and deceleration
        dt = 0.05
        max_speed_change = MAX_ACCEL_LIMIT * dt
        speed_diff = speed - self.target_speed
        if abs(speed_diff) > max_speed_change:
            speed = self.target_speed + np.sign(speed_diff) * max_speed_change

        self.target_speed = float(max(min(speed, SPEED_MAX), -SPEED_MAX))
        self.target_turn = float(max(min(turn, TURN_MAX), -TURN_MAX))

    def edge_vectors_callback(self, message):
        self.vectors = message

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
        # Filter obstacles to ignore track walls 
        track_mask = (obs_x > 0.0) & (obs_x < LATTICE_TRACK_X_MAX) & (np.abs(obs_y) <= LATTICE_TRACK_Y_MAX)
        obs_x = obs_x[track_mask]
        obs_y = obs_y[track_mask]
        obstacles = np.column_stack((obs_x, obs_y))

        # --- Planner Setup ---
        dt = LATTICE_LOOKAHEAD_DT
        steps = LATTICE_STEPS
        wheelbase = LATTICE_WHEELBASE
        LAT_ACCEL_MAX = LATTICE_LAT_ACCEL_MAX
        num_candidates = LATTICE_NUM_CANDIDATES
        
        candidates = np.linspace(-1.0, 1.0, num_candidates)
        max_steer_angle = LATTICE_MAX_STEER_ANGLE
        
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

                if min_path_dist < LATTICE_COLLISION_RADIUS: # Hard collision boundary
                    continue

            # 3. Cost Evaluation
            cost = 0.0
            
            # Soft Obstacle Penalty (Allows obstacle hugging / close passing based on parameters)
            if min_path_dist < COST_OBSTACLE_BUFFER_DIST:
                cost += COST_OBSTACLE_WEIGHT * math.exp(-COST_OBSTACLE_DECAY * (min_path_dist - LATTICE_COLLISION_RADIUS))

            # Tracking Penalty (deviation from ideal lane keeping)
            steer_diff = abs(cand_steer - desired_steer)
            cost += COST_TRACKING_LINEAR_WEIGHT * steer_diff + COST_TRACKING_QUAD_WEIGHT * (steer_diff ** 2)

            # Smoothness Penalty (deviation from current actuator state)
            cost += COST_SMOOTHNESS_WEIGHT * abs(cand_steer - self.target_turn)

            # Directional Bias (Overtake logic preference)
            if len(obstacles) > 0:
                closest_obs_idx = np.argmin(np.linalg.norm(obstacles, axis=1))
                closest_obs_y = obstacles[closest_obs_idx, 1]
                if closest_obs_y > 0 and cand_steer > 0:
                    cost += COST_DIRECTIONAL_BIAS_WEIGHT * cand_steer
                elif closest_obs_y < 0 and cand_steer < 0:
                    cost += COST_DIRECTIONAL_BIAS_WEIGHT * abs(cand_steer)

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
            cv2.imshow("Lattice Planner", vis_img)
            cv2.waitKey(1)

        return final_speed, best_steer

    def lidar_callback(self, message):
        self.latest_scan = message

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
            self.last_qr_data = qr_data
            self.last_processed_qr = qr_data
            self.qr_drive_start_time = self.get_clock().now().nanoseconds / 1e9
            self.comm_state = "QR_DRIVE_DELAY"
            self.get_logger().info(f"QR Detected: {qr_data}. Driving for 2 more seconds.")
            
        # Optional: logic to detect parking spot
        if qr_data == "PARKING" and self.comm_state == "IDLE":
            self.comm_state = "PARKED"

    def sign_board_callback(self, message):
        if message.data:
            try:
                payload = json.loads(message.data)
                if payload.get("detected") and len(payload.get("detections", [])) > 0:
                    self.last_detected_sign = payload["detections"][0]["class"]
                    self.get_logger().info(f"Stored detected sign: {self.last_detected_sign}")

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
                cv2.imshow("Lattice Planner", np.ones((500, 500, 3), dtype=np.uint8) * 255)
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
            
            cv2.imshow("Lattice Planner", vis_img)
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