import re

with open('b3rb_ros_line_follower.py', 'r') as f:
    content = f.read()

# 1. Add Constants
constants_code = """
SIGN_MAP = {
    'A': 'PATIENT_1',   'B': 'PATIENT_2',   'C': 'PATIENT_3',
    'X': 'HOSPITAL_1',  'Y': 'HOSPITAL_2',  'Z': 'HOSPITAL_3',
    'PATIENT_1': 'A',   'PATIENT_2': 'B',   'PATIENT_3': 'C',
    'HOSPITAL_1': 'X',  'HOSPITAL_2': 'Y',  'HOSPITAL_3': 'Z',
}
PATIENT_LIFETIME_SEC = 60.0
MAX_COMM_RETRIES = 5
"""
content = re.sub(r'(REPULSIVE_LOOKAHEAD_DIST = 0.*?\n)', r'\1' + constants_code, content)

# 2. Add Parallel Behavior
parallel_code = """
class Parallel(Behavior):
    def __init__(self, children):
        self.children = children
    def tick(self, node):
        for child in self.children:
            child.tick(node)
        return NodeStatus.SUCCESS
"""
content = re.sub(r'(class Condition\(Behavior\):.*?return NodeStatus\.FAILURE\n)', r'\1' + parallel_code, content, flags=re.DOTALL)

# 3. Replace HandleCommunication and OvertakeBehavior with the new ones
behaviors_code = """
class HandleCommunication(Behavior):
    def tick(self, node):
        current_time = node.get_clock().now().nanoseconds / 1e9
        
        if node.comm_state == "IDLE":
            return NodeStatus.FAILURE
            
        elif node.comm_state == "QR_DRIVE_DELAY":
            if current_time - node.qr_drive_start_time >= 12.0:
                node.is_hospital_zone = False
                node.comm_state = "SEND_QR"
                node.comm_msg_to_send = node.last_qr_data
                node.rover_move_manual_mode(0.0, 0.0)
                node.comm_last_send_time = current_time
                node.comm_retry_count = 0
            return NodeStatus.RUNNING

        elif node.comm_state == "SEND_QR":
            node.send_server_msg(node.comm_msg_to_send, is_ack=False)
            node.comm_state = "WAITING_ACK"
            node.comm_last_send_time = current_time
            node.rover_move_manual_mode(0.0, 0.0)
            return NodeStatus.RUNNING

        elif node.comm_state == "WAITING_ACK":
            node.rover_move_manual_mode(0.0, 0.0)
            if node.received_ack:
                node.comm_state = "WAITING_TARGET"
                node.received_ack = False
                node.comm_last_send_time = current_time
                node.comm_retry_count = 0
            elif current_time - node.comm_last_send_time > 1.0:
                node.comm_retry_count += 1
                node.comm_state = "SEND_QR"
            return NodeStatus.RUNNING
            
        elif node.comm_state == "WAITING_TARGET":
            node.rover_move_manual_mode(0.0, 0.0)
            if node.received_target is not None:
                if node.last_uid not in getattr(node, 'uid_to_mission_state', set()):
                    node.send_server_msg("", is_ack=True, req_uid=node.last_uid)
                    if not hasattr(node, 'uid_to_mission_state'):
                        node.uid_to_mission_state = set()
                    node.uid_to_mission_state.add(node.last_uid)
                    node.comm_state = "IDLE"
                else:
                    node.send_server_msg("", is_ack=True, req_uid=node.last_uid)
                    node.received_target = None
            elif current_time - node.comm_last_send_time > 5.0:
                node.comm_retry_count += 1
                node.comm_state = "SEND_QR"
            return NodeStatus.RUNNING
            
        elif node.comm_state == "PARKED":
            node.rover_move_manual_mode(0.0, 0.0)
            if not getattr(node, 'sent_parked', False):
                node.send_server_msg("PARKED", is_ack=False)
                node.sent_parked = True
            
            if node.received_target == "OK":
                node.comm_state = "DONE"
            return NodeStatus.RUNNING
            
        elif node.comm_state == "DONE":
            node.rover_move_manual_mode(0.0, 0.0)
            return NodeStatus.SUCCESS

        return NodeStatus.FAILURE

class PatientCycleBehavior(Behavior):
    def tick(self, node):
        if node.mission_state == "SEEKING_PATIENT":
            node.current_target = getattr(node, 'current_patient_target', None)
            if getattr(node, 'in_zone', None) == "PATIENT" and getattr(node, 'last_qr_read_data', None):
                if node.current_patient_target and node.current_patient_target in node.last_qr_read_data:
                    current_time = node.get_clock().now().nanoseconds / 1e9
                    if not hasattr(node, 'zone_stop_start_time') or node.zone_stop_start_time is None:
                        node.zone_stop_start_time = current_time
                    elif current_time - node.zone_stop_start_time >= 5.0:
                        node.mission_state = "IN_PATIENT_ZONE"
                        node.zone_stop_start_time = None
                        node.comm_msg_to_send = node.last_qr_read_data
                        node.comm_state = "SEND_QR"
            return NodeStatus.FAILURE
            
        elif node.mission_state == "IN_PATIENT_ZONE":
            node.rover_move_manual_mode(0.0, 0.0)
            if node.received_target:
                raw = node.received_target
                node.received_target = None
                hospital = SIGN_MAP.get(raw, raw)
                node.current_hospital_target = hospital
                node.current_target = hospital
                node.in_zone = None
                node.last_qr_read_data = None
                node.last_processed_qr = None
                node.mission_state = "SEEKING_HOSPITAL"
            return NodeStatus.RUNNING
            
        elif node.mission_state == "SEEKING_HOSPITAL":
            node.current_target = getattr(node, 'current_hospital_target', None)
            if getattr(node, 'in_zone', None) == "HOSPITAL" and getattr(node, 'last_qr_read_data', None):
                if getattr(node, 'current_hospital_target', '') and getattr(node, 'current_hospital_target', '') in node.last_qr_read_data:
                    current_time = node.get_clock().now().nanoseconds / 1e9
                    if not hasattr(node, 'zone_stop_start_time') or node.zone_stop_start_time is None:
                        node.zone_stop_start_time = current_time
                    elif current_time - node.zone_stop_start_time >= 7.0:
                        node.mission_state = "IN_HOSPITAL_ZONE"
                        node.patients_done = getattr(node, 'patients_done', 0) + 1
                        node.zone_stop_start_time = None
                        node.comm_msg_to_send = node.last_qr_read_data
                        node.comm_state = "SEND_QR"
            return NodeStatus.FAILURE
            
        elif node.mission_state == "IN_HOSPITAL_ZONE":
            node.rover_move_manual_mode(0.0, 0.0)
            if node.received_target:
                raw = node.received_target
                node.received_target = None
                if raw in ("DONE", "PARKING"):
                    node.mission_state = "EXITING"
                else:
                    next_patient = SIGN_MAP.get(raw, raw)
                    node.current_patient_target = next_patient
                    node.current_target = next_patient
                    node.in_zone = None
                    node.last_qr_read_data = None
                    node.last_processed_qr = None
                    node.patient_spawn_time = node.get_clock().now().nanoseconds / 1e9
                    node.mission_state = "SEEKING_PATIENT"
            return NodeStatus.RUNNING

        elif node.mission_state == "EXITING":
            if getattr(node, 'in_zone', None) == "PARKING":
                node.rover_move_manual_mode(0.0, 0.0)
                node.mission_state = "SEEKING_PARK"
            return NodeStatus.FAILURE
            
        elif node.mission_state == "SEEKING_PARK":
            node.comm_state = "PARKED"
            node.mission_state = "PARKED"
            return NodeStatus.RUNNING
            
        elif node.mission_state == "PARKED":
            return NodeStatus.RUNNING
            
        return NodeStatus.FAILURE

class TimingWatchdog(Behavior):
    def tick(self, node):
        if getattr(node, 'mission_state', None) in ["SEEKING_PATIENT","IN_PATIENT_ZONE","AWAITING_INITIAL_ASSIGNMENT"]:
            current_time = node.get_clock().now().nanoseconds / 1e9
            if getattr(node, 'patient_spawn_time', 0.0) > 0 and (current_time - getattr(node, 'patient_spawn_time', 0.0)) > PATIENT_LIFETIME_SEC:
                node.get_logger().error(f"Patient {getattr(node, 'current_patient_target', None)} expired! Penalty.")
                if not hasattr(node, 'penalties'): node.penalties = []
                node.penalties.append({"type": "time_expired", "target": getattr(node, 'current_patient_target', None)})
                node.patient_spawn_time = current_time
        return NodeStatus.SUCCESS

class ZoneExitWatchdog(Behavior):
    def tick(self, node):
        if getattr(node, 'in_zone', None) is not None:
            current_time = node.get_clock().now().nanoseconds / 1e9
            if current_time - getattr(node, 'zone_entry_time', 0.0) > 12.0:
                node.in_zone = None
                if getattr(node, 'mission_state', None) == "AWAITING_INITIAL_ASSIGNMENT":
                    if not hasattr(node, 'penalties'): node.penalties = []
                    node.penalties.append({"type": "early_exit"})
        return NodeStatus.SUCCESS

class ServerDisconnectWatchdog(Behavior):
    def tick(self, node):
        if getattr(node, 'comm_retry_count', 0) >= MAX_COMM_RETRIES:
            node.mission_state = "ABORT_DISCONNECTED"
        return NodeStatus.SUCCESS
"""
content = re.sub(r'class HandleCommunication\(Behavior\):.*?return NodeStatus\.RUNNING\n\n', behaviors_code + '\n', content, flags=re.DOTALL)

# 4. Modify NavigationAction to use SIGN_MAP and hospital lane memory
nav_action_replace = """
            if not lidar_obstacle_avoidance_active:
                target = getattr(node, 'current_target', None) or getattr(node, 'target_patient', None)
                target_key = SIGN_MAP.get(target, target)

                desired_dir = None
                if target_key and hasattr(node, 'sign_config_map') and target_key in node.sign_config_map:
                    desired_dir = node.sign_config_map[target_key]
                elif hasattr(node, 'last_detected_sign') and getattr(node, 'last_detected_sign', None) is not None:
                    desired_dir = node.last_detected_sign
"""
content = re.sub(r'            if not lidar_obstacle_avoidance_active:\n                target =.*?\n                desired_dir = None\n                if target and hasattr\(node, \'sign_config_map\'\) and target in node\.sign_config_map:\n                    desired_dir = node\.sign_config_map\[target\]\n                    node\.get_logger\(\)\.info\(f"Intersection: Target \'{target}\' mapped to direction \'{desired_dir}\'", throttle_duration_sec=1\.0\)\n                elif hasattr\(node, \'last_detected_sign\'\):\n                    desired_dir = node\.last_detected_sign\n', nav_action_replace, content, flags=re.DOTALL)

nav_action_memory = """
        else:
            if getattr(node, 'in_intersection', False):
                node.intersection_exit_ticks = getattr(node, 'intersection_exit_ticks', 0) + 1
                if node.intersection_exit_ticks > 20:
                    node.in_intersection = False
                    node.sign_detected = False
                    node.last_detected_sign = None
                    node.last_turn_clear_time = node.get_clock().now().nanoseconds / 1e9
                    node.get_logger().info("Intersection cleared. Resetting sign detection and initiating 6s cooldown.")

        in_hospital_memory = getattr(node, 'hospital_lane_memory_active', False)

        if vectors.vector_count == 1:
            deviation = vectors.vector_1[1].x - vectors.vector_1[0].x
            error = deviation / half_width
            derivative = error - node.last_error
            raw_turn = (error * 1.2) + (derivative * 0.8)
            speed = max(MIN_CORNER_SPEED, TARGET_CRUISE_SPEED - abs(error) * SPEED_CURVE_SLOWDOWN_GAIN)
            node.last_error = error
        elif vectors.vector_count == 2:
            if in_hospital_memory:
                mag1 = math.sqrt((vectors.vector_1[1].x - vectors.vector_1[0].x)**2 + (vectors.vector_1[1].y - vectors.vector_1[0].y)**2)
                mag2 = math.sqrt((vectors.vector_2[1].x - vectors.vector_2[0].x)**2 + (vectors.vector_2[1].y - vectors.vector_2[0].y)**2)
                crop_height = vectors.image_height
                conf1 = (mag1 / crop_height) * 100.0 if crop_height > 0 else 0
                conf2 = (mag2 / crop_height) * 100.0 if crop_height > 0 else 0
                if conf1 > 70.0 and conf2 > 70.0:
                    node.hospital_lane_memory_active = False
                    in_hospital_memory = False

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

        if in_hospital_memory:
            error = node.last_error
            derivative = 0.0
            raw_turn = error * 1.5
            speed = 0.4
"""
content = re.sub(r'        elif vectors.vector_count == 1:.*?node\.last_error = error\n', nav_action_memory, content, flags=re.DOTALL)

# Add node.in_intersection flag inside vector_count == 0 block
content = re.sub(r'(        if vectors\.vector_count == 0:\n            speed = 0\.2\n)', r'\1            node.in_intersection = True\n            node.intersection_exit_ticks = 0\n', content)

# 5. BT Root and State init in LineFollower.__init__
init_vars = """
        # Mission state variables
        self.mission_state = "AWAITING_INITIAL_ASSIGNMENT"
        self.last_detected_sign = None
        self.sign_config_map = {}
        self.target_patient = None
        self.comm_state = "IDLE"
        self.last_qr_data = None
        self.last_processed_qr = None
        self.last_qr_read_data = None
        self.zone_entry_time = 0.0
        self.in_zone = None
        self.is_hospital_zone = False
        self.hospital_lane_memory_active = False
        
        # Server Comm Tracking
        self.uid_counter = 0
        self.last_sent_uid = 0
        self.last_uid = 0
        self.received_ack = False
        self.received_target = None
        self.current_target = None
        self.sent_parked = False
        self.comm_retry_count = 0
        self.comm_last_send_time = 0.0
        self.uid_to_mission_state = set()
        
        self.last_turn_clear_time = 0.0
"""
content = re.sub(r'        # Mission state variables.*?self\.sent_parked = False\n', init_vars, content, flags=re.DOTALL)

bt_root = """
        # Behavior Tree Root
        self.bt_root = Sequence([
            Parallel([
                TimingWatchdog(),
                ZoneExitWatchdog(),
                ServerDisconnectWatchdog()
            ]),
            Selector([
                Sequence([
                    Condition(lambda n: n.comm_state != "IDLE"),
                    HandleCommunication()
                ]),
                PatientCycleBehavior(),
                NavigationAction()
            ])
        ])
"""
content = re.sub(r'        # Behavior Tree Root.*?\]\)\n', bt_root, content, flags=re.DOTALL)

# 6. sign_board_callback
sign_board_code = """
    def sign_board_callback(self, message):
        current_time = self.get_clock().now().nanoseconds / 1e9

        if current_time - getattr(self, 'last_turn_clear_time', 0.0) < 6.0:
            return
            
        if message.data:
            try:
                payload = json.loads(message.data)
                if payload.get("detected") and len(payload.get("detections", [])) > 0:
                    DIRECTION_CLASSES = {"Left", "Right", "Straight", "Up"}
                    for det in payload.get("detections", []):
                        if det["class"] in DIRECTION_CLASSES and det.get("confidence", 1.0) >= 0.65:
                            self.last_detected_sign = det["class"]
                            self.get_logger().info(f"Stored detected sign direction: {self.last_detected_sign} (conf: {det.get('confidence', 1.0):.2f})")

                sign_config = payload.get("sign_config")
                if sign_config:
                    if getattr(self, 'current_sign_payload', None) == sign_config:
                        if not getattr(self, 'payload_locked', False):
                            self.current_sign_payload_count = getattr(self, 'current_sign_payload_count', 0) + 1
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
"""
content = re.sub(r'    def sign_board_callback\(self, message\):.*?self\.get_logger\(\)\.error\(f"Failed to parse sign board data: \{e\}"\)\n', sign_board_code, content, flags=re.DOTALL)

# 7. qr_detection_callback
qr_callback_code = """
    def qr_detection_callback(self, message):
        try:
            payload  = json.loads(message.data)
            qr_data  = payload.get("data", message.data)
        except json.JSONDecodeError:
            qr_data = message.data

        current_time = self.get_clock().now().nanoseconds / 1e9

        current_target = getattr(self, 'current_target', None)
        target_key = SIGN_MAP.get(current_target, current_target) if current_target else None

        is_assigned_target = False
        if current_target and (current_target in qr_data or (target_key and target_key in qr_data)):
            is_assigned_target = True
        elif "PARKING" in qr_data and getattr(self, 'mission_state', None) == "EXITING":
            is_assigned_target = True

        if is_assigned_target and qr_data != getattr(self, 'last_processed_qr', None):
            self.last_qr_data = qr_data
            self.last_processed_qr = qr_data
            self.last_qr_read_data = qr_data
            self.zone_entry_time = current_time

            if "hospital" in qr_data.lower() or "X" in qr_data or "Y" in qr_data or "Z" in qr_data:
                self.in_zone = "HOSPITAL"
                self.is_hospital_zone = True
                self.hospital_lane_memory_active = True
                self.comm_state = "QR_DRIVE_DELAY"
                self.qr_drive_start_time = current_time
                self.get_logger().info(f"Entered ASSIGNED HOSPITAL zone: {qr_data}")
            elif "A" in qr_data or "B" in qr_data or "C" in qr_data:
                self.in_zone = "PATIENT"
                self.comm_state = "QR_DRIVE_DELAY"
                self.qr_drive_start_time = current_time
                self.get_logger().info(f"Entered ASSIGNED PATIENT zone: {qr_data}")
            elif "PARKING" in qr_data:
                self.in_zone = "PARKING"
                self.comm_state = "PARKED"
                self.get_logger().info(f"Entered PARKING zone: {qr_data}")
"""
content = re.sub(r'    def qr_detection_callback\(self, message\):.*?self\.comm_state = "PARKED"\n', qr_callback_code, content, flags=re.DOTALL)

with open('b3rb_ros_line_follower.py', 'w') as f:
    f.write(content)
