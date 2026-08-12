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
from sensor_msgs.msg import Joy, LaserScan, CompressedImage
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
TARGET_CRUISE_SPEED = 1.0          # nominal speed on straight/normal lane tracking
MIN_CORNER_SPEED = 0.2             # floor speed, never goes slower than this while actively tracking a lane
MAX_ACCEL_LIMIT = 5.0              # caps how fast speed can ramp up or down per tick (units/sec²)
MAX_STEER_RATE = 4.0               # caps how fast steering can change per tick
SPEED_CURVE_SLOWDOWN_GAIN = 0.5    # how much speed is shaved off per unit of steering error

# --- LATTICE PLANNER CONFIGURATION PARAMETERS ---
LATTICE_LOOKAHEAD_DT = 0.20        # Time step per simulation step (seconds)
LATTICE_STEPS = 10                 # Total steps ahead (10 steps * 0.20s = 2.0s lookahead)
LATTICE_WHEELBASE = 0.3            # Vehicle wheelbase (meters)
LATTICE_LAT_ACCEL_MAX = 1.4        # Max lateral acceleration limit (m/s^2) for cornering speed calculation
LATTICE_NUM_CANDIDATES = 25        # Number of steering arc trajectories sampled between [-1.0, 1.0]
LATTICE_MAX_STEER_ANGLE = 0.62     # Maximum physical steering angle (radians)
LATTICE_COLLISION_RADIUS = 0.15    # Hard collision distance threshold (meters)
LATTICE_TRACK_X_MAX = 1.5          # Forward distance filter limit for relevant obstacles (meters)
LATTICE_TRACK_Y_MAX = 0.15         # Lateral width filter limit for track obstacles (meters)

# --- COST FUNCTION PARAMETERS ("OBSTACLE HUGGING" & TRAJECTORY TUNING) ---
COST_OBSTACLE_BUFFER_DIST = 0.5    # Distance threshold (m) below which obstacle proximity cost is applied
COST_OBSTACLE_WEIGHT = 30.0        # Weight multiplier for obstacle proximity penalty
COST_OBSTACLE_DECAY = 15.0          # Exponential decay factor (higher value = passes closer/hugs obstacle)
COST_TRACKING_LINEAR_WEIGHT = 150.0 # Weight for linear deviation from ideal line-following steering (was 10.0)
COST_TRACKING_QUAD_WEIGHT = 400.0  # Weight for quadratic deviation from ideal line-following steering (was 25.0)
COST_SMOOTHNESS_WEIGHT = 250.0       # Weight for steering angle change from current steering state
COST_DIRECTIONAL_BIAS_WEIGHT = 8.0 # Penalty weight for steering towards the side of the closest obstacle

# --- REPULSIVE FORCE PARAMETERS ---
REPULSIVE_FORCE_GAIN = 0.0        # Tuning gain for obstacle repulsion force
REPULSIVE_LOOKAHEAD_DIST = 0     # Maximum distance to consider obstacles for repulsion

# --- MISSION CONSTANTS ---
SIGN_MAP = {
    'A': 'PATIENT_1', 'B': 'PATIENT_2', 'C': 'PATIENT_3',
    'X': 'HOSPITAL_1', 'Y': 'HOSPITAL_2', 'Z': 'HOSPITAL_3',
    'PATIENT_1': 'A', 'PATIENT_2': 'B', 'PATIENT_3': 'C',
    'HOSPITAL_1': 'X', 'HOSPITAL_2': 'Y', 'HOSPITAL_3': 'Z',
}
PATIENT_LIFETIME_SEC = 60.0
ZONE_TIMEOUT_SEC = 30.0
MAX_COMM_RETRIES = 5

# --- AUTO PARKING BEHAVIOR CONSTANTS ---
PARKING_TRIGGER_HOSPITAL_COUNT = 3
PARKING_APPROACH_SPEED = 0.1          # Slow crawl speed inside the bay
PARKING_TURN_IN_SPEED  = 0.12         # Speed while turning into the bay
PARKING_CORRIDOR_X_MAX = 1.5
PARKING_CORRIDOR_Y_MAX = 0.6
PARKING_STOP_FRONT_DIST = 0.35        # Stop when a wall/end-of-bay is this close ahead
PARKING_CENTERING_GAIN = 2.5
PARKING_MAX_STEER = 0.8
PARKING_TIMEOUT_SEC = 30.0            # Increased timeout for 6s approach + turn + crawl
PARKING_SIDE_DETECT_DIST = 1.5       # Max lateral LiDAR dist to detect the cone-row opening (m)
PARKING_SIDE_ANGLE_L = 1.0           # Left side scan window start (rad, ~57°)
PARKING_SIDE_ANGLE_R = 1.57          # Left side scan window end  (rad, ~90°)
PARKING_TURN_IN_STEER = 0.85         # Full left Ackermann steer while turning into bay
PARKING_APPROACH_SECS = 9.0         # Drive straight for 6 seconds to align and find cones
PARKING_TURN_IN_SECS  = 2.0          # How long to hold the left-turn into the bay
PARKING_STRAIGHTEN_SECS = 1.5        # After turning, straighten up inside bay

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
        current_time = node.get_clock().now().nanoseconds / 1e9
        
        if node.comm_state == "QR_DRIVE_DELAY":
            if current_time - node.qr_drive_start_time >= 12.0:
                node.is_hospital_zone = False
                node.comm_state = "SEND_QR"
                node.comm_msg_to_send = node.last_qr_data
                node.rover_move_manual_mode(0.0, 0.0)
                
                if node.vectors and node.vectors.vector_count > 0:
                    half_width = node.vectors.image_width / 2.0
                    if node.vectors.vector_count == 2:
                        lane_center = (node.vectors.vector_1[0].x + node.vectors.vector_2[0].x) / 2.0
                        node.last_error = (half_width - lane_center) / half_width
                    elif node.vectors.vector_count == 1:
                        deviation = node.vectors.vector_1[1].x - node.vectors.vector_1[0].x
                        node.last_error = deviation / half_width
                
                node.get_logger().info("12 seconds passed, stopping to send QR.")
            return NodeStatus.FAILURE
            
        elif node.comm_state == "SEND_QR":
            node.send_server_msg(node.comm_msg_to_send, is_ack=False)
            node.comm_state = "WAITING_ACK"
            node.comm_last_send_time = current_time
            node.get_logger().info(f"Sent {node.comm_msg_to_send}, waiting for ACK")
            node.rover_move_manual_mode(0.0, 0.0)
            return NodeStatus.RUNNING
            
        elif node.comm_state == "WAITING_ACK":
            node.rover_move_manual_mode(0.0, 0.0)
            if node.received_ack:
                node.comm_state = "WAITING_TARGET"
                node.received_ack = False
                node.comm_retry_count = 0
                node.comm_last_send_time = current_time
                node.get_logger().info("Received ACK, waiting for target")
            elif current_time - node.comm_last_send_time > 1.0:
                node.comm_retry_count += 1
                node.comm_state = "SEND_QR"
                node.get_logger().warn(f"Comm Retry {node.comm_retry_count}/5")
            return NodeStatus.RUNNING
            
        elif node.comm_state == "WAITING_TARGET":
            node.rover_move_manual_mode(0.0, 0.0)
            if node.received_target is not None:
                if node.last_uid not in node.uid_to_mission_state:
                    # ACK the server; keep received_target for MissionController to consume
                    node.send_server_msg("", is_ack=True, req_uid=node.last_uid)
                    node.uid_to_mission_state.add(node.last_uid)
                    node.comm_state = "IDLE"  # MissionController reads received_target on next tick
                    node.comm_retry_count = 0
                    node.get_logger().info(f"Received target '{node.received_target}' from server, ACK sent")
                    return NodeStatus.SUCCESS
                else:
                    # Duplicate UID — ACK again and discard
                    node.send_server_msg("", is_ack=True, req_uid=node.last_uid)
                    node.received_target = None
                    node.get_logger().warn(f"Duplicate UID {node.last_uid} ignored")
            elif current_time - node.comm_last_send_time > 5.0:
                node.comm_retry_count += 1
                node.comm_state = "SEND_QR"
                node.get_logger().warn("WAITING_TARGET timeout, retrying send")
            return NodeStatus.RUNNING
            
        elif node.comm_state == "PARKED":
            node.rover_move_manual_mode(0.0, 0.0)
            if not node.sent_parked:
                node.send_server_msg("PARKED", is_ack=False)
                node.sent_parked = True
                node.comm_last_send_time = current_time
                node.get_logger().info("Sent PARKED, waiting for OK")
            
            if node.received_target == "OK":
                node.get_logger().info("Parking Confirmed OK!")
                node.comm_state = "DONE"
            elif node.received_target == "INVALID":
                node.get_logger().error("Parking INVALID!")
                node.comm_state = "DONE"
            elif current_time - node.comm_last_send_time > 1.0:
                 node.sent_parked = False
                 node.comm_retry_count += 1
            return NodeStatus.RUNNING
            
        elif node.comm_state == "DONE":
            node.rover_move_manual_mode(0.0, 0.0)
            return NodeStatus.SUCCESS

        return NodeStatus.FAILURE
            

class WatchdogParallel(Behavior):
    def __init__(self, children):
        self.children = children
    def tick(self, node):
        for child in self.children:
            child.tick(node)
        return NodeStatus.SUCCESS  # Must return SUCCESS so Sequence continues to navigation

class TimingWatchdog(Behavior):
    def tick(self, node):
        if node.mission_state in ["SEEKING_PATIENT", "IN_PATIENT_ZONE", "AWAITING_ASSIGNMENT"]:
            current_time = node.get_clock().now().nanoseconds / 1e9
            if node.patient_spawn_time > 0 and (current_time - node.patient_spawn_time) > PATIENT_LIFETIME_SEC:
                node.get_logger().error(f"TimingWatchdog: Patient {node.current_patient_target} expired! Penalty.")
                node.penalties.append({"type": "time_expired", "target": node.current_patient_target, "time": current_time})
                node.patient_spawn_time = current_time  # Reset for now so it doesn't spam
        return NodeStatus.SUCCESS

class ZoneExitWatchdog(Behavior):
    def tick(self, node):
        if node.in_zone is not None:
            current_time = node.get_clock().now().nanoseconds / 1e9
            # Fake zone exit based on time if we haven't seen a QR in 12 seconds
            if current_time - node.zone_entry_time > 12.0:
                old_zone = node.in_zone
                node.in_zone = None
                node.get_logger().info(f"ZoneExitWatchdog: Exited zone {old_zone}")
                
                # Check for penalties
                if node.mission_state == "AWAITING_ASSIGNMENT":
                    node.get_logger().error("Penalty: Exited PATIENT zone before assignment received!")
                    node.penalties.append({"type": "early_exit", "zone": old_zone, "time": current_time})
        return NodeStatus.SUCCESS

class ServerDisconnectWatchdog(Behavior):
    def tick(self, node):
        if node.comm_retry_count >= MAX_COMM_RETRIES:
            if node.mission_state != "ABORT_DISCONNECTED":
                node.get_logger().error("ServerDisconnectWatchdog: Comm retries exceeded. Aborting mission!")
                node.mission_state = "ABORT_DISCONNECTED"
        return NodeStatus.SUCCESS

class ParkingBehavior(Behavior):
    """
    Multi-phase Ackermann parking maneuver.

    The parking bay is on the LEFT side of the road.  The sequence is:

      Phase 0 – APPROACH_SLOW
        The buggy is still on the main road.  We use the LEFT-side LiDAR
        sector (angles ~57°–90° from forward) to detect when a clear gap
        (no cone closer than PARKING_SIDE_DETECT_DIST) appears, meaning we
        have reached the mouth of the parking bay.  While searching we crawl
        slowly forward (vision-based lane centering still running via the
        EdgeVectors topic that was already published).

      Phase 1 – TURN_IN
        Apply maximum left Ackermann steering for PARKING_TURN_IN_SECS.
        The rover describes a tight left arc off the road and into the bay.

      Phase 2 – STRAIGHTEN
        Release steering to 0 for PARKING_STRAIGHTEN_SECS so that the rover
        is aligned with the bay corridor.

      Phase 3 – CRAWL_IN
        Drive straight ahead at PARKING_APPROACH_SPEED.  Use the LEFT/RIGHT
        LiDAR corridor measurements to stay centred between the cone rows
        (P-controller). Stop when the front LiDAR detects the end-of-bay wall
        closer than PARKING_STOP_FRONT_DIST, or on timeout.
    """

    PHASE_APPROACH_SLOW = 0
    PHASE_TURN_IN       = 1
    PHASE_STRAIGHTEN    = 2
    PHASE_CRAWL_IN      = 3

    def __init__(self):
        self.reset()

    def reset(self):
        self.phase      = self.PHASE_APPROACH_SLOW
        self.phase_start = None
        self.start_time = None

    def tick(self, node):
        current_time = node.get_clock().now().nanoseconds / 1e9

        # Global timer starts when tick() is first called
        if self.start_time is None:
            self.start_time = current_time
            self.phase_start = current_time
            node.get_logger().info("ParkingBehavior: starting — Phase 0 APPROACH_SLOW")

        elapsed_total = current_time - self.start_time
        elapsed_phase = current_time - (self.phase_start or current_time)

        # --- Global failsafe timeout ---
        if elapsed_total > PARKING_TIMEOUT_SEC:
            node.rover_move_manual_mode(0.0, 0.0)
            node.get_logger().warn("ParkingBehavior: Timeout — stopping as failsafe.")
            self.reset()
            return NodeStatus.SUCCESS

        # --- LiDAR preprocessing (shared across phases) ---
        scan = node.latest_scan
        obs_x = obs_y = None
        if scan is not None:
            angles  = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)
            ranges_ = np.array(scan.ranges)
            n       = min(len(angles), len(ranges_))
            angles, ranges_ = angles[:n], ranges_[:n]
            valid   = (ranges_ > scan.range_min) & (ranges_ < 6.0) & ~np.isinf(ranges_)
            r_v, a_v = ranges_[valid], angles[valid]
            obs_x = r_v * np.cos(a_v)
            obs_y = r_v * np.sin(a_v)

        # ────────────────────────────────────────────────────────────────────
        # Phase 0 — APPROACH_SLOW
        #   Drive straight for 2 seconds (PARKING_APPROACH_SECS) to find/align with cones properly.
        # ────────────────────────────────────────────────────────────────────
        if self.phase == self.PHASE_APPROACH_SLOW:
            if elapsed_phase < PARKING_APPROACH_SECS:
                node.rover_move_manual_mode(PARKING_TURN_IN_SPEED, 0.0)
                node.get_logger().info(
                    f"ParkingBehavior Ph0 APPROACH_SLOW: driving straight to position t={elapsed_phase:.1f}s/{PARKING_APPROACH_SECS:.1f}s",
                    throttle_duration_sec=0.5)
            else:
                self.phase = self.PHASE_TURN_IN
                self.phase_start = current_time
                node.get_logger().info("ParkingBehavior: Phase 0 complete — starting Phase 1 TURN_IN")
            return NodeStatus.RUNNING

        # ────────────────────────────────────────────────────────────────────
        # Phase 1 — TURN_IN
        #   Apply full left Ackermann steering to swing into the bay.
        # ────────────────────────────────────────────────────────────────────
        elif self.phase == self.PHASE_TURN_IN:
            if elapsed_phase < PARKING_TURN_IN_SECS:
                node.rover_move_manual_mode(PARKING_TURN_IN_SPEED, PARKING_TURN_IN_STEER)
                node.get_logger().info(
                    f"ParkingBehavior Ph1 TURN_IN: steer={PARKING_TURN_IN_STEER:.2f} t={elapsed_phase:.1f}s",
                    throttle_duration_sec=0.5)
            else:
                self.phase = self.PHASE_STRAIGHTEN
                self.phase_start = current_time
                node.get_logger().info("ParkingBehavior: Phase 2 STRAIGHTEN")
            return NodeStatus.RUNNING

        # ────────────────────────────────────────────────────────────────────
        # Phase 2 — STRAIGHTEN
        #   Release steering so the rover lines up with the bay axis.
        # ────────────────────────────────────────────────────────────────────
        elif self.phase == self.PHASE_STRAIGHTEN:
            if elapsed_phase < PARKING_STRAIGHTEN_SECS:
                node.rover_move_manual_mode(PARKING_APPROACH_SPEED, 0.0)
                node.get_logger().info(
                    f"ParkingBehavior Ph2 STRAIGHTEN: t={elapsed_phase:.1f}s",
                    throttle_duration_sec=0.5)
            else:
                self.phase = self.PHASE_CRAWL_IN
                self.phase_start = current_time
                node.get_logger().info("ParkingBehavior: Phase 3 CRAWL_IN")
            return NodeStatus.RUNNING

        # ────────────────────────────────────────────────────────────────────
        # Phase 3 — CRAWL_IN
        #   Slowly drive straight ahead into the bay, centred between cones.
        #   Stop when the end-of-bay wall is close ahead.
        # ────────────────────────────────────────────────────────────────────
        elif self.phase == self.PHASE_CRAWL_IN:
            # End-of-bay detection: obstacle dead ahead
            if obs_x is not None:
                front_mask = (
                    (obs_x > 0.05) &
                    (obs_x < PARKING_STOP_FRONT_DIST) &
                    (np.abs(obs_y) < 0.35)
                )
                if np.any(front_mask):
                    node.rover_move_manual_mode(0.0, 0.0)
                    node.get_logger().info("ParkingBehavior: Front obstacle — PARKED!")
                    self.reset()
                    return NodeStatus.SUCCESS

            # Corridor centering using LiDAR
            steer = 0.0
            if obs_x is not None:
                corridor_mask = (
                    (obs_x > 0.1) &
                    (obs_x < PARKING_CORRIDOR_X_MAX) &
                    (np.abs(obs_y) < PARKING_CORRIDOR_Y_MAX)
                )
                cy = obs_y[corridor_mask]
                if len(cy) > 0:
                    left_pts  = cy[cy > 0]
                    right_pts = cy[cy < 0]
                    left_dist  = np.min(left_pts)          if len(left_pts)  > 0 else PARKING_CORRIDOR_Y_MAX
                    right_dist = np.min(np.abs(right_pts)) if len(right_pts) > 0 else PARKING_CORRIDOR_Y_MAX
                    lateral_error = right_dist - left_dist
                    steer = max(-PARKING_MAX_STEER,
                                min(PARKING_MAX_STEER, lateral_error * PARKING_CENTERING_GAIN))

            # Optionally override with vision-vector steering if cones are visible
            vectors = node.vectors
            if vectors is not None and vectors.vector_count > 0:
                half_width = vectors.image_width / 2.0
                raw_turn = 0.0
                if vectors.vector_count == 2:
                    mid_x_bottom = (vectors.vector_1[0].x + vectors.vector_2[0].x) / 2.0
                    mid_x_top    = (vectors.vector_1[1].x + vectors.vector_2[1].x) / 2.0
                    dx_mid = mid_x_top - mid_x_bottom
                    dy_mid = ((vectors.vector_1[1].y + vectors.vector_2[1].y) / 2.0 -
                              (vectors.vector_1[0].y + vectors.vector_2[0].y) / 2.0)
                    if dy_mid != 0:
                        raw_turn = math.atan2(dx_mid, -dy_mid)
                elif vectors.vector_count == 1:
                    dx = vectors.vector_1[1].x - vectors.vector_1[0].x
                    dy = vectors.vector_1[1].y - vectors.vector_1[0].y
                    if dy != 0:
                        angle      = math.atan2(dx, -dy)
                        offset_x   = vectors.vector_1[0].x - half_width
                        raw_turn   = angle + (offset_x / half_width) * 0.5
                vision_steer = max(-PARKING_MAX_STEER, min(PARKING_MAX_STEER, raw_turn * 2.0))
                # Blend: 60% vision, 40% LiDAR centering
                steer = 0.6 * vision_steer + 0.4 * steer

            node.get_logger().info(
                f"ParkingBehavior Ph3 CRAWL_IN: steer={steer:.2f} t={elapsed_phase:.1f}s",
                throttle_duration_sec=0.5)
            node.rover_move_manual_mode(PARKING_APPROACH_SPEED, steer)
            return NodeStatus.RUNNING

        # Fallback
        node.rover_move_manual_mode(0.0, 0.0)
        return NodeStatus.RUNNING

class MissionController(Behavior):
    def __init__(self):
        self.patient_cycle = PatientCycleBehavior()
        self.parking_behavior = ParkingBehavior()
        
    def tick(self, node):
        if node.mission_state == "ENTER_CITY":
            initial_patient = node.received_target if node.received_target else 'A'
            node.received_target = None
            node.current_patient_target = SIGN_MAP.get(initial_patient, initial_patient)
            node.current_target = node.current_patient_target
            node.patient_spawn_time = node.get_clock().now().nanoseconds / 1e9
            node.mission_state = "SEEKING_PATIENT"
            node.get_logger().info(f"Mission Start: Default target '{initial_patient}' ({node.current_patient_target}). Navigating immediately.")
            return NodeStatus.RUNNING
            
        elif node.mission_state == "AWAITING_INITIAL_ASSIGNMENT":
            # Fallback if state is manually set
            node.rover_move_manual_mode(0.0, 0.0)
            if node.received_target:
                initial_patient = node.received_target
                node.received_target = None
                node.current_patient_target = SIGN_MAP.get(initial_patient, initial_patient)
                node.current_target = node.current_patient_target
                node.patient_spawn_time = node.get_clock().now().nanoseconds / 1e9
                node.mission_state = "SEEKING_PATIENT"
                node.get_logger().info(f"Initial target '{initial_patient}' -> navigating to {node.current_patient_target}")
            return NodeStatus.RUNNING
            
        elif node.mission_state in [
            "SEEKING_PATIENT", "IN_PATIENT_ZONE",
            "SEEKING_HOSPITAL", "IN_HOSPITAL_ZONE"
        ]:
            return self.patient_cycle.tick(node)
            
        elif node.mission_state == "EXITING":
            node.get_logger().info("All patients served. Heading to parking zone...", throttle_duration_sec=5.0)
            node.mission_state = "SEEKING_PARK"
            
        elif node.mission_state == "SEEKING_PARK":
            # Take full motor control — stop letting NavigationAction drive straight.
            # Slow down and look for the bay entrance with LiDAR; then
            # hand off to ParkingBehavior once the gap is aligned.
            if node.in_zone == "PARKING":
                # QR zone triggered — enter bay immediately
                node.mission_state = "PARKING_MANEUVER"
                node.get_logger().info("Parking zone QR reached. Handing off to ParkingBehavior.")
                return NodeStatus.RUNNING
            # While still seeking: ParkingBehavior Phase 0 handles the slow approach
            # so we can reuse its gap-detection logic. Drive slowly forward.
            status = self.parking_behavior.tick(node)
            if status == NodeStatus.SUCCESS:
                # Gap found and we entered — but wait for QR or just begin maneuver
                node.mission_state = "PARKING_MANEUVER"
                node.get_logger().info("Gap found by ParkingBehavior. Starting maneuver.")
            return NodeStatus.RUNNING

        elif node.mission_state == "PARKING_MANEUVER":
            status = self.parking_behavior.tick(node)
            if status == NodeStatus.SUCCESS:
                node.mission_state = "PARKED"
                node.get_logger().info("Parking maneuver complete. Mission finished.")
            return NodeStatus.RUNNING
                
        elif node.mission_state == "PARKED":
            node.rover_move_manual_mode(0.0, 0.0)
            return NodeStatus.RUNNING
                
        elif node.mission_state == "MISSION_DONE":
            node.rover_move_manual_mode(0.0, 0.0)
            return NodeStatus.RUNNING
            
        elif node.mission_state == "ABORT_DISCONNECTED":
            node.rover_move_manual_mode(0.0, 0.0)
            return NodeStatus.RUNNING
            
        return NodeStatus.FAILURE

class PatientCycleBehavior(Behavior):
    """Purely server-directed mission cycle.
    The buggy navigates, detects zones via QR, stops, and waits for send_target.py
    to send the next assignment. No automatic sends to the server.
    """
    def tick(self, node):
        if node.mission_state == "SEEKING_PATIENT":
            # Tell NavigationAction which building to head to
            node.current_target = node.current_patient_target
            # Arrived when zone flips to PATIENT and the QR matches our target
            if node.in_zone == "PATIENT" and node.last_qr_read_data:
                if node.current_patient_target in node.last_qr_read_data:
                    current_time = node.get_clock().now().nanoseconds / 1e9
                    if not hasattr(node, 'zone_stop_start_time') or node.zone_stop_start_time is None:
                        node.zone_stop_start_time = current_time
                        node.get_logger().info(f"Target QR detected ({node.last_qr_read_data})! Driving for 10 seconds before stopping beside board...")
                    elif current_time - node.zone_stop_start_time >= 10.0:
                        node.mission_state = "IN_PATIENT_ZONE"
                        node.zone_stop_start_time = None
                        node.send_server_msg(node.last_qr_read_data, is_ack=False)
                        node.get_logger().info(
                            f"\033[1;32m[SERVER PUBLISH] Arrived at {node.current_patient_target} (QR={node.last_qr_read_data}) after 10s drive delay. Sent 1 report to server. Waiting for hospital assignment...\033[0m"
                        )
                else:
                    node.get_logger().info(
                        f"Wrong patient zone QR: '{node.last_qr_read_data}' "
                        f"(wanted {node.current_patient_target}). Continuing."
                    )

        elif node.mission_state == "IN_PATIENT_ZONE":
            # Stay stopped and wait for the server to send a hospital letter (X/Y/Z)
            node.rover_move_manual_mode(0.0, 0.0)
            if node.received_target:
                raw = node.received_target
                node.received_target = None
                # Map letter -> canonical target (e.g. X -> HOSPITAL_1)
                hospital = SIGN_MAP.get(raw, raw)
                node.current_hospital_target = hospital
                node.current_target = hospital
                node.in_zone = None
                node.last_qr_read_data = None
                node.last_processed_qr = None
                node.mission_state = "SEEKING_HOSPITAL"
                node.get_logger().info(f"Hospital assigned by server: '{raw}' -> {hospital}. Navigating.")
            return NodeStatus.RUNNING

        elif node.mission_state == "SEEKING_HOSPITAL":
            node.current_target = node.current_hospital_target
            # Arrived when zone flips to HOSPITAL and QR matches
            if node.in_zone == "HOSPITAL" and node.last_qr_read_data:
                if node.current_hospital_target in node.last_qr_read_data:
                    current_time = node.get_clock().now().nanoseconds / 1e9
                    if not hasattr(node, 'zone_stop_start_time') or node.zone_stop_start_time is None:
                        node.zone_stop_start_time = current_time
                        node.get_logger().info(f"Target QR detected ({node.last_qr_read_data})! Driving for 7 seconds before stopping beside hospital...")
                    elif current_time - node.zone_stop_start_time >= 7.0:
                        node.mission_state = "IN_HOSPITAL_ZONE"
                        node.patients_done += 1
                        node.zone_stop_start_time = None
                        node.received_ack = False
                        node.send_server_msg(node.last_qr_read_data, is_ack=False)
                        node.get_logger().info(
                            f"\033[1;32m[SERVER PUBLISH] Arrived at {node.current_hospital_target} (QR={node.last_qr_read_data}) after 7s drive delay. Sent 1 report to server. Patients done: {node.patients_done}. Waiting for server ACK...\033[0m"
                        )
                else:
                    node.get_logger().info(
                        f"Wrong hospital zone QR: '{node.last_qr_read_data}' "
                        f"(wanted {node.current_hospital_target}). Continuing."
                    )

        elif node.mission_state == "IN_HOSPITAL_ZONE":
            node.rover_move_manual_mode(0.0, 0.0)
            
            # If 3rd hospital delivery completed, stay stopped and wait specifically for 'PARKING' command from server
            if node.patients_done >= PARKING_TRIGGER_HOSPITAL_COUNT:
                has_parking_target = node.received_target and node.received_target in ("DONE", "PARKING")
                
                if has_parking_target:
                    node.received_ack = False
                    node.received_target = None
                    node.hospital_resume_straight = False
                    node.mission_state = "EXITING"
                    node.get_logger().info(
                        f"{node.patients_done} hospitals completed and explicit 'PARKING' server message received. Heading to parking."
                    )
                    return NodeStatus.SUCCESS
                else:
                    node.get_logger().info(
                        f"At hospital {node.patients_done}/{PARKING_TRIGGER_HOSPITAL_COUNT}. Waiting for 'PARKING' command from server...",
                        throttle_duration_sec=2.0
                    )
                    return NodeStatus.RUNNING
                
            if node.received_target:
                raw = node.received_target
                node.received_target = None
                node.hospital_resume_time = node.get_clock().now().nanoseconds / 1e9
                node.hospital_resume_straight = True

                if raw in ("DONE", "PARKING"):
                    node.hospital_resume_straight = False
                    node.mission_state = "EXITING"
                    node.get_logger().info(f"Server sent '{raw}'. Immediately heading to park.")
                else:
                    next_patient = SIGN_MAP.get(raw, raw)
                    node.current_patient_target = next_patient
                    node.current_target = next_patient
                    node.in_zone = None
                    node.last_qr_read_data = None
                    node.last_processed_qr = None
                    node.patient_spawn_time = node.get_clock().now().nanoseconds / 1e9
                    node.mission_state = "SEEKING_PATIENT"
                    node.get_logger().info(f"Next patient from server: '{raw}' -> {next_patient}. Navigating.")
            return NodeStatus.RUNNING

        return NodeStatus.FAILURE

class ZoneGuard(Behavior):
    def __init__(self, expected_zone):
        self.expected_zone = expected_zone
    def tick(self, node):
        if node.in_zone == self.expected_zone:
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE

class LargeObstacleAvoidanceBehavior(Behavior):
    def tick(self, node):
        if not hasattr(node, 'overtake_state'):
            node.overtake_state = 'IDLE'
            node.overtake_wp_x = 0.0
            node.overtake_wp_y = 0.0
            node.overtake_robot_x = 0.0
            node.overtake_robot_y = 0.0
            node.overtake_robot_yaw = 0.0

        if node.overtake_state == 'IDLE':
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

            # Detect obstacle directly in front (up to 0.65m)
            front_mask = (obs_x > 0.1) & (obs_x < 0.65) & (np.abs(obs_y) <= 0.4)
            if np.any(front_mask):
                obs_width = np.max(obs_y[front_mask]) - np.min(obs_y[front_mask])
                if obs_width > 0.4: # > 50% lane width (assuming 0.8m lane)
                    node.overtake_state = 'OVERTAKING'
                    mean_y = np.mean(obs_y[front_mask])
                    max_x = np.max(obs_x[front_mask])
                    
                    # Reset local odometry
                    node.overtake_robot_x = 0.0
                    node.overtake_robot_y = 0.0
                    node.overtake_robot_yaw = 0.0
                    
                    # Waypoint is 1.0m past the obstacle
                    node.overtake_wp_x = max_x + 1.0
                    
                    # Offset waypoint significantly to avoid clipping the obstacle
                    if mean_y < 0:
                        node.overtake_wp_y = 0.8
                    else:
                        node.overtake_wp_y = -0.8
                        
                    node.last_overtake_time = node.get_clock().now().nanoseconds / 1e9
                    node.get_logger().info(f"Large Obstacle! Width={obs_width:.2f}. Waypoint set to ({node.overtake_wp_x:.2f}, {node.overtake_wp_y:.2f})")
                    return self._execute_overtake(node)
            return NodeStatus.FAILURE
            
        else:
            return self._execute_overtake(node)

    def _execute_overtake(self, node):
        current_time = node.get_clock().now().nanoseconds / 1e9
        if not hasattr(node, 'last_overtake_time'):
            node.last_overtake_time = current_time
            
        dt = current_time - node.last_overtake_time
        node.last_overtake_time = current_time
        
        speed = 0.4
        L = 0.3 # wheelbase
        
        # Current steering angle in radians (approx based on target_turn)
        steer_rad = getattr(node, 'target_turn', 0.0) * 0.62
        
        # Update robot pose (Kinematic bicycle model)
        node.overtake_robot_x += speed * math.cos(node.overtake_robot_yaw) * dt
        node.overtake_robot_y += speed * math.sin(node.overtake_robot_yaw) * dt
        node.overtake_robot_yaw += (speed / L) * math.tan(steer_rad) * dt
        
        # Calculate waypoint in robot's local frame
        dx = node.overtake_wp_x - node.overtake_robot_x
        dy = node.overtake_wp_y - node.overtake_robot_y
        
        local_wp_x = dx * math.cos(-node.overtake_robot_yaw) - dy * math.sin(-node.overtake_robot_yaw)
        local_wp_y = dx * math.sin(-node.overtake_robot_yaw) + dy * math.cos(-node.overtake_robot_yaw)
        
        if local_wp_x < 0.2:
            node.overtake_state = 'IDLE'
            node.get_logger().info("Overtake: Waypoint reached. Returning to lane.")
            return NodeStatus.SUCCESS
            
        # Pure pursuit steering to the local waypoint
        L_d_sq = local_wp_x**2 + local_wp_y**2
        if L_d_sq < 0.01:
            steer_mag = 0.0
        else:
            curvature = 2.0 * local_wp_y / L_d_sq
            target_steer_rad = math.atan(L * curvature)
            steer_mag = max(-1.0, min(1.0, target_steer_rad / 0.62))
        
        node.rover_move_manual_mode(speed, steer_mag)
        return NodeStatus.RUNNING

class NavigationAction(Behavior):
    def tick(self, node):
        current_time = node.get_clock().now().nanoseconds / 1e9
        resume_time = getattr(node, 'hospital_resume_time', 0.0)
        vectors = node.vectors
        if getattr(node, 'hospital_resume_straight', False):
            if resume_time > 0.0 and (current_time - resume_time) < 4.0:
                if vectors is None or vectors.vector_count == 0:
                    # Construct virtual straight vectors so that the Lattice Planner and obstacle avoidance stay active
                    from synapse_msgs.msg import EdgeVectors
                    virtual_vectors = EdgeVectors()
                    virtual_vectors.image_width = 640
                    virtual_vectors.image_height = 480
                    virtual_vectors.vector_count = 2
                    
                    # Left vector
                    virtual_vectors.vector_1[0].x = 160.0
                    virtual_vectors.vector_1[0].y = 0.0
                    virtual_vectors.vector_1[1].x = 160.0
                    virtual_vectors.vector_1[1].y = 480.0
                    
                    # Right vector
                    virtual_vectors.vector_2[0].x = 480.0
                    virtual_vectors.vector_2[0].y = 0.0
                    virtual_vectors.vector_2[1].x = 480.0
                    virtual_vectors.vector_2[1].y = 480.0
                    
                    vectors = virtual_vectors
                    node.get_logger().info("Hospital resume: using virtual straight vectors", throttle_duration_sec=0.5)
            else:
                node.hospital_resume_straight = False
                node.hospital_lane_memory_active = False

        if vectors is None:
            node.rover_move_manual_mode(0.0, 0.0)
            return NodeStatus.RUNNING

        # Filter vectors if near obstacle/cones to avoid trusting short contours (e.g. from cones)
        near_obstacle = False
        if getattr(node, 'cone_detected', False):
            near_obstacle = True
        elif hasattr(node, 'latest_scan') and node.latest_scan is not None:
            scan = node.latest_scan
            angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)
            ranges = np.array(scan.ranges)
            min_len = min(len(angles), len(ranges))
            angles = angles[:min_len]
            ranges = ranges[:min_len]
            valid_idx = (ranges > scan.range_min) & (ranges < 4.0) & ~np.isinf(ranges)
            if np.any(valid_idx):
                obs_x = ranges[valid_idx] * np.cos(angles[valid_idx])
                obs_y = ranges[valid_idx] * np.sin(angles[valid_idx])
                front_mask = (obs_x > 0.1) & (obs_x < 1.2) & (np.abs(obs_y) <= 0.4)
                if np.any(front_mask):
                    near_obstacle = True

        if near_obstacle:
            def is_proper_vector(pt_start, pt_end):
                dx = pt_end.x - pt_start.x
                dy = pt_end.y - pt_start.y
                mag = math.hypot(dx, dy)
                return mag > 80.0 and abs(dy) > 60.0

            valid_count = 0
            if vectors.vector_count == 2:
                v1_valid = is_proper_vector(vectors.vector_1[0], vectors.vector_1[1])
                v2_valid = is_proper_vector(vectors.vector_2[0], vectors.vector_2[1])
                if v1_valid and v2_valid:
                    valid_count = 2
                elif v1_valid:
                    valid_count = 1
                elif v2_valid:
                    valid_count = 1
                    from copy import copy
                    vectors = copy(vectors)
                    vectors.vector_1 = vectors.vector_2
                else:
                    valid_count = 0
            elif vectors.vector_count == 1:
                if is_proper_vector(vectors.vector_1[0], vectors.vector_1[1]):
                    valid_count = 1
                else:
                    valid_count = 0

            if valid_count < vectors.vector_count:
                from copy import copy
                vectors = copy(vectors)
                vectors.vector_count = valid_count
                node.get_logger().info(f"Near obstacle/cone: ignoring short contours, treating vector_count as {valid_count}", throttle_duration_sec=1.0)

        half_width = vectors.image_width / 2.0
        speed = 0.5  # Normal speed
        raw_turn = 0.0

        if getattr(node, 'hospital_lane_memory_active', False):
            current_time = node.get_clock().now().nanoseconds / 1e9
            resume_time = getattr(node, 'hospital_resume_time', 0.0)
            # Irrespective of confidence, use memory mode for the first 3 seconds after resuming
            if resume_time > 0.0 and (current_time - resume_time) >= 3.0:
                if vectors.vector_count == 2:
                    dx1 = vectors.vector_1[1].x - vectors.vector_1[0].x
                    dy1 = vectors.vector_1[1].y - vectors.vector_1[0].y
                    mag1 = math.hypot(dx1, dy1)
                    
                    dx2 = vectors.vector_2[1].x - vectors.vector_2[0].x
                    dy2 = vectors.vector_2[1].y - vectors.vector_2[0].y
                    mag2 = math.hypot(dx2, dy2)
                    
                    crop_height = vectors.image_height * 0.40
                    confidence1 = (mag1 / crop_height) * 100.0 if crop_height > 0 else 0
                    confidence2 = (mag2 / crop_height) * 100.0 if crop_height > 0 else 0
                    
                    if confidence1 > 70.0 and confidence2 > 70.0:
                        node.hospital_lane_memory_active = False
                        node.get_logger().info(f"Both lanes > 70% confidence (L:{confidence1:.1f}%, R:{confidence2:.1f}%). Exiting memory mode.")

        in_hospital_memory = getattr(node, 'hospital_lane_memory_active', False)

        if vectors.vector_count > 0:
            import copy
            node.last_valid_vectors = copy.deepcopy(vectors)
        elif in_hospital_memory and hasattr(node, 'last_valid_vectors'):
            vectors = node.last_valid_vectors
            node.get_logger().info("Using last stored vector in hospital zone", throttle_duration_sec=1.0)

        if vectors.vector_count > 0:
            if getattr(node, 'in_intersection', False):
                node.intersection_exit_ticks = getattr(node, 'intersection_exit_ticks', 0) + 1
                if node.intersection_exit_ticks > 20:
                    node.in_intersection = False
                    node.sign_detected = False
                    node.last_detected_sign = None
                    node.last_turn_clear_time = node.get_clock().now().nanoseconds / 1e9
                    node.get_logger().info("Intersection cleared. Resetting sign detection and initiating 3s cooldown.")

        if vectors.vector_count == 0:
            node.in_intersection = True
            node.intersection_exit_ticks = 0
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
                target_key = SIGN_MAP.get(target, target)
                node.get_logger().info(f"DEBUG Intersection: target={target} (key={target_key}), sign_config_map={getattr(node, 'sign_config_map', {})}")
                if target_key and hasattr(node, 'sign_config_map') and target_key in node.sign_config_map:
                    desired_dir = node.sign_config_map[target_key]
                    node.get_logger().info(f"Intersection: Target '{target}' mapped to direction '{desired_dir}'")
                elif hasattr(node, 'last_detected_sign') and node.last_detected_sign is not None:
                    desired_dir = node.last_detected_sign
                    node.get_logger().info(f"Intersection: Fallback to last_detected_sign '{desired_dir}'")

                if desired_dir in ["left", "Left"]:
                    raw_turn = 0.6
                elif desired_dir in ["right", "Right"]:
                    raw_turn = -0.6
                elif desired_dir in ["straight", "Straight", "up", "Up"]:
                    if getattr(node, 'in_zone', None) is None and not getattr(node, 'hospital_lane_memory_active', False):
                        # Steer 3 degrees counter-clockwise (left) (~0.08 normalized turn) to find the path at a cross section
                        raw_turn = 0.08
                        node.get_logger().info("Intersection (Straight): Steering 3 degrees left to find lane", throttle_duration_sec=0.5)
                    else:
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
            raw_turn = (error * 0.6) + (derivative * 1.5)
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
            raw_turn = (error * 1.2) + (derivative * 1.8)
            speed = max(MIN_CORNER_SPEED, TARGET_CRUISE_SPEED - abs(error) * SPEED_CURVE_SLOWDOWN_GAIN)
            node.last_error = error

        if getattr(node, 'cone_detected', False):
            speed = 0.7
            node.get_logger().info("Cone detected! Slowing down for slalom.", throttle_duration_sec=0.5)

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

        if vectors.vector_count == 0 and not in_hospital_memory:
            # Bypass lattice planner at intersections so it doesn't try to avoid the corner walls and ruin the turn
            planner_speed, planner_turn = speed, target_turn_command
        else:
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
        self.subscription_camera = self.create_subscription(CompressedImage, '/camera/image_raw/compressed', self.camera_callback, QOS_PROFILE_DEFAULT)

        self.cone_detected = False
        self.cone_detected_time = 0.0

        self.publisher_joy = self.create_publisher(Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)
        self.publisher_server = self.create_publisher(ServerCommunication, '/ServerCommunication', QOS_PROFILE_DEFAULT)
        self.publisher_parking_guide = self.create_publisher(String, '/parking_guide_data', QOS_PROFILE_DEFAULT)

        self.target_speed = 0.0
        self.target_turn = 0.0
        
        self.latest_scan = None
        self.vectors = None
        self.smoothed_turn = 0.0
        self.last_error = 0.0
        
        self.show_lattice_window = True  # Disabled: imshow at 20Hz kills RTF significantly

        # Mission state variables
        self.last_detected_sign = None
        self.sign_detected = False
        self.in_intersection = False
        self.intersection_exit_ticks = 0
        self.is_hospital_zone = False
        self.last_turn_clear_time = 0.0
        self.hospital_lane_memory_active = False
        self.hospital_resume_time = 0.0
        self.hospital_resume_straight = False
        
        # Debounce tracking for sign config
        self.current_sign_payload = None
        self.current_sign_payload_count = 0
        self.payload_locked = False
        
        self.sign_config_map = {}
        
        # Blackboard
        self.mission_state = "ENTER_CITY"
        self.patients_done = 0
        self.current_patient_target = None
        self.current_hospital_target = None
        self.in_zone = None
        self.zone_entry_time = 0.0
        self.patient_spawn_time = 0.0
        self.penalties = []
        self.bonus_awarded = False
        
        self.comm_retry_count = 0
        self.uid_to_mission_state = set()
        self.comm_msg_to_send = ""
        self.comm_last_send_time = 0.0
        self.last_qr_read_data = None

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
        # HandleCommunication is NOT in the Selector — all server messages come from
        # send_target.py manually. The buggy only listens via server_communication_callback.
        self.bt_root = Sequence([
            WatchdogParallel([
                TimingWatchdog(),
                ZoneExitWatchdog(),
                ServerDisconnectWatchdog()
            ]),
            Selector([
                MissionController(),
                # LargeObstacleAvoidanceBehavior(),
                NavigationAction()
            ])
        ])

        self.control_timer = self.create_timer(0.02, self.tick_bt_and_publish)  # 50Hz
        self.get_logger().info("Line Follower initialized. Behavior Tree Active.")

    def tick_bt_and_publish(self):
        self.bt_root.tick(self)
        
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        is_overtaking = 1.0 if getattr(self, 'overtake_state', 'IDLE') != 'IDLE' else 0.0
        msg.axes = [0.0, self.target_speed, is_overtaking, self.target_turn]
        self.publisher_joy.publish(msg)

        # Publish parking guide data so edge_vectors can draw Ackermann arc projections
        guide_msg = String()
        guide_msg.data = json.dumps({
            'mission_state': self.mission_state,
            'steer': float(self.target_turn),
            'speed': float(self.target_speed),
        })
        self.publisher_parking_guide.publish(guide_msg)

    def rover_move_manual_mode(self, speed, turn):
        # Smoothly limit linear acceleration and deceleration
        dt = 0.05
        accel = getattr(self, 'current_accel_limit', MAX_ACCEL_LIMIT)
        max_speed_change = accel * dt
        speed_diff = speed - self.target_speed
        if abs(speed_diff) > max_speed_change:
            speed = self.target_speed + np.sign(speed_diff) * max_speed_change

        # Rate-limit steering
        if getattr(self, '_emergency_steer', False):
            self._emergency_steer = False
        else:
            max_turn_change = MAX_STEER_RATE * dt
            turn_diff = turn - self.target_turn
            if abs(turn_diff) > max_turn_change:
                turn = self.target_turn + np.sign(turn_diff) * max_turn_change

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
        
        # Check if there is a large obstacle using full LiDAR forward range (up to 4.0m)
        has_large_obstacle = False
        has_obstacle = False
        dynamic_buffer_dist = 0.65 # Default for small obstacles
        obs_center_y = 0.0
        
        if len(obs_x) > 0:
            # Detect obstacles in the forward lane using max range
            front_mask = (obs_x > 0.1) & (np.abs(obs_y) <= 0.4) 
            if np.any(front_mask):
                has_obstacle = True
                obs_center_y = float(np.mean(obs_y[front_mask]))
                obs_width = np.max(obs_y[front_mask]) - np.min(obs_y[front_mask])
                if obs_width > 0.3:
                    has_large_obstacle = True
                    dynamic_buffer_dist = 2.0

        # 2D Grid of Polynomial Parameters
        # c2_norm controls initial steering (-1.0 to 1.0)
        # c3_norm controls curve change over distance (-1.0 to 1.0)
        c2_candidates = np.linspace(-1.0, 1.0, 25)
        if has_obstacle:
            if obs_center_y > 0.0:
                c2_candidates = np.linspace(-1.0, 0.0, 13)
            else:
                c2_candidates = np.linspace(0.0, 1.0, 13)
                
        if has_large_obstacle:
            c3_candidates = np.linspace(-1.0, 1.0, 25)
        else:
            c3_candidates = np.array([0.0]) # Only simple arcs for normal driving
        
        max_steer_angle = LATTICE_MAX_STEER_ANGLE
        # c2_max corresponds to max steering angle: kappa = tan(delta)/L = 2*c2
        c2_max = math.tan(max_steer_angle) / (2.0 * wheelbase)
        c3_max = 0.5 # Tuning parameter for how aggressively the S-curve bends back
        
        vis_img = None
        if getattr(self, 'show_lattice_window', False):
            vis_img = np.ones((500, 500, 3), dtype=np.uint8) * 255
            for ox, oy in zip(obs_x, obs_y):
                px, py = int(250 - oy * 100), int(500 - ox * 100)
                if 0 <= px < 500 and 0 <= py < 500:
                    cv2.circle(vis_img, (px, py), 3, (0, 0, 0), -1)

        min_cost = float('inf')
        best_steer = desired_steer
        
        in_hospital_memory = getattr(self, 'hospital_lane_memory_active', False)
        obstacle_w = COST_OBSTACLE_WEIGHT
        tracking_lin_w = COST_TRACKING_LINEAR_WEIGHT
        tracking_quad_w = COST_TRACKING_QUAD_WEIGHT

        best_path_points = None
        best_obs_dist = float('inf')
        best_speed = desired_speed

        # --- Evaluate Paths ---
        for c2_norm in c2_candidates:
            for c3_norm in c3_candidates:
                c2 = c2_norm * c2_max
                c3 = c3_norm * c3_max
                cand_steer = c2_norm # Maps directly to immediate steering command
                
                delta = cand_steer * max_steer_angle
                
                # 1. Cubic Polynomial Path Generation (y = c3*x^3 + c2*x^2)
                if abs(delta) < 1e-4:
                    v_step = desired_speed
                else:
                    R = wheelbase / math.tan(delta)
                    v_curve_limit = math.sqrt(LAT_ACCEL_MAX * abs(R))
                    v_step = max(0.2, min(desired_speed, v_curve_limit))
                
                max_x = v_step * dt * steps
                x = np.linspace(0, max_x, steps)
                y = c3 * (x ** 3) + c2 * (x ** 2)
                path_points = np.column_stack((x, y))

                min_path_dist = float('inf')
                collision = False

                # 2. Vectorized Collision Checking
                if len(obstacles) > 0:
                    diff = path_points[:, np.newaxis, :] - obstacles[np.newaxis, :, :]
                    dists = np.linalg.norm(diff, axis=2)
                    min_dists_per_step = np.min(dists, axis=1)
                    min_path_dist = np.min(min_dists_per_step)

                    if min_path_dist < LATTICE_COLLISION_RADIUS: # Hard collision boundary
                        if vis_img is not None:
                            for i in range(len(path_points) - 1):
                                p1, p2 = path_points[i], path_points[i + 1]
                                cv2.line(vis_img, (int(250 - p1[1] * 100), int(500 - p1[0] * 100)),
                                          (int(250 - p2[1] * 100), int(500 - p2[0] * 100)), (0, 0, 200), 1)
                        continue

                if vis_img is not None:
                    for i in range(len(path_points) - 1):
                        p1, p2 = path_points[i], path_points[i + 1]
                        cv2.line(vis_img, (int(250 - p1[1] * 100), int(500 - p1[0] * 100)),
                                  (int(250 - p2[1] * 100), int(500 - p2[0] * 100)), (0, 200, 200), 1)

                # 3. Cost Evaluation
                cost = 0.0
                
                # Soft Obstacle Penalty
                if min_path_dist <= dynamic_buffer_dist:
                    calc_dist = max(0.20, min_path_dist)
                    multiplier = (dynamic_buffer_dist - calc_dist) / dynamic_buffer_dist
                    cost += obstacle_w * (multiplier * 1000.0)

                # Tracking Penalty
                steer_diff = abs(cand_steer - desired_steer)
                current_lin_w = tracking_lin_w
                current_quad_w = tracking_quad_w
                
                if min_path_dist <= dynamic_buffer_dist:
                    calc_dist_track = max(0.20, min_path_dist)
                    relaxation_factor = max(0.2, (calc_dist_track / dynamic_buffer_dist))
                    current_lin_w *= relaxation_factor
                    current_quad_w *= relaxation_factor
                    
                cost += current_lin_w * steer_diff + current_quad_w * (steer_diff ** 2)

                # Smoothness Penalty
                cost += COST_SMOOTHNESS_WEIGHT * abs(cand_steer - self.target_turn)

                # Polynomial Complexity Penalty (prefers simple arcs/lines if obstacles are clear)
                cost += abs(c3_norm) * 10.0

                # Directional Bias
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
        current_obs_dist = np.min(np.linalg.norm(obstacles, axis=1)) if len(obstacles) > 0 else float('inf')

        if not hasattr(self, 'lattice_is_locked'):
            self.lattice_is_locked = False
            self.locked_steer = 0.0
            self.locked_speed = 0.0

        if self.lattice_is_locked:
            if current_obs_dist > 0.75:
                self.lattice_is_locked = False
            else:
                best_steer = self.locked_steer
                best_speed = self.locked_speed
                min_cost = 0.0
                best_obs_dist = 0.6

        if not self.lattice_is_locked and current_obs_dist <= 0.65 and min_cost != float('inf'):
            self.lattice_is_locked = True
            self.locked_steer = best_steer
            self.locked_speed = best_speed
            best_obs_dist = 0.6

        if min_cost == float('inf'):
            self.get_logger().warn("Lattice Planner: All paths blocked! Reactive turn.", throttle_duration_sec=1.0)
            if len(obstacles) > 0:
                closest_obs_idx = np.argmin(np.linalg.norm(obstacles, axis=1))
                if obstacles[closest_obs_idx, 1] > 0:
                    best_steer = -1.0 # Obstacle on left, turn right
                else:
                    best_steer = 1.0  # Obstacle on right, turn left
            self._emergency_steer = True
            return max(0.2, desired_speed * 0.4), best_steer

        # Dynamic proximity slowdown
        final_speed = best_speed
        self.current_accel_limit = MAX_ACCEL_LIMIT
        if best_obs_dist < 0.5:
            final_speed = max(0.3, best_speed * (best_obs_dist / 0.5))
            self.current_accel_limit = 0.5  # Decrease acceleration near obstacle

        # --- Visualization ---
        if vis_img is not None and best_path_points is not None:
            for i in range(len(best_path_points) - 1):
                p1, p2 = best_path_points[i], best_path_points[i + 1]
                cv2.line(vis_img, (int(250 - p1[1] * 100), int(500 - p1[0] * 100)),
                          (int(250 - p2[1] * 100), int(500 - p2[0] * 100)), (0, 200, 0), 2)
            cv2.imshow("Lattice Planner", vis_img)
            cv2.waitKey(1)

        return final_speed, best_steer

    def camera_callback(self, message):
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red/Dark Orange color bounds (for simulated traffic cones)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([15, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        mask = cv2.bitwise_or(mask1, mask2)

        h, w = mask.shape
        roi = mask[h // 2 :, :]  # only lower half of image
        orange_pixels = cv2.countNonZero(roi)

        current_time = self.get_clock().now().nanoseconds / 1e9
        if orange_pixels > 500:
            self.cone_detected = True
            self.cone_detected_time = current_time
        else:
            if current_time - getattr(self, 'cone_detected_time', 0.0) > 0.5:
                self.cone_detected = False

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
        try:
            payload = json.loads(message.data)
            qr_data = payload.get("data", message.data)
        except json.JSONDecodeError:
            qr_data = message.data
            
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Only accept QR data if it matches our assigned mission target
        current_target = getattr(self, 'current_target', None)
        target_key = SIGN_MAP.get(current_target, current_target) if current_target else None
        
        # Match either canonical target (e.g. HOSPITAL_1) or target key (e.g. X) in QR payload
        is_assigned_target = False
        if current_target and (current_target in qr_data or (target_key and target_key in qr_data)):
            is_assigned_target = True
        elif "PARKING" in qr_data and self.mission_state == "EXITING":
            is_assigned_target = True

        if is_assigned_target and qr_data != self.last_processed_qr:
            self.last_qr_data = qr_data
            self.last_processed_qr = qr_data
            self.last_qr_read_data = qr_data
            self.zone_entry_time = current_time
            
            if "hospital" in qr_data.lower() or "X" in qr_data or "Y" in qr_data or "Z" in qr_data:
                self.in_zone = "HOSPITAL"
                self.is_hospital_zone = True
                self.hospital_lane_memory_active = True
                self.get_logger().info(f"Entered ASSIGNED HOSPITAL zone: {qr_data}")
            elif "A" in qr_data or "B" in qr_data or "C" in qr_data:
                self.in_zone = "PATIENT"
                self.get_logger().info(f"Entered ASSIGNED PATIENT zone: {qr_data}")
            elif "PARKING" in qr_data:
                self.in_zone = "PARKING"
                self.get_logger().info(f"Entered PARKING zone: {qr_data}")
            # PatientCycleBehavior manages comm_state; do NOT override it here

    def sign_board_callback(self, message):
        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self.last_turn_clear_time < 3.0:
            return
        if message.data:
            try:
                payload = json.loads(message.data)
                if payload.get("detected") and len(payload.get("detections", [])) > 0:
                    # Only store direction arrows, not letters (A, B, C, X, Y, Z)
                    DIRECTION_CLASSES = {"Left", "Right", "Straight", "Up"}
                    for det in payload.get("detections", []):
                        if det["class"] in DIRECTION_CLASSES and det.get("confidence", 1.0) >= 0.65:
                            self.last_detected_sign = det["class"]
                            self.get_logger().info(f"Stored detected sign direction: {self.last_detected_sign} (conf: {det.get('confidence', 1.0):.2f})")

                sign_config = payload.get("sign_config")
                if sign_config:
                    if self.current_sign_payload == sign_config:
                        if not self.payload_locked:
                            self.current_sign_payload_count += 1
                            if self.current_sign_payload_count >= 10:
                                self.payload_locked = True
                                self.sign_config_map.update(sign_config)
                                self.get_logger().info(f"Locked in sign_config map after 10 consistent reads: {self.sign_config_map}")
                    else:
                        self.current_sign_payload = sign_config
                        self.current_sign_payload_count = 1
                        self.payload_locked = False
                        self.get_logger().info(f"Sign variation detected, reset count for: {sign_config}")
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