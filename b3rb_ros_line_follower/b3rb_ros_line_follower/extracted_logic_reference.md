# 🧠 Extracted Logic: Signboard Detection, QR Detection & Mission Planning

> **Source files:**
> - [`b3rb_ros_line_follower.py`](file:///home/chaitanya/cognipilot/cranium/src/b3rb_ros_line_follower/b3rb_ros_line_follower/b3rb_ros_line_follower/b3rb_ros_line_follower.py)
> - [`b3rb_ros_object_recog.py`](file:///home/chaitanya/cognipilot/cranium/src/b3rb_ros_line_follower/b3rb_ros_line_follower/b3rb_ros_line_follower/b3rb_ros_object_recog.py)
> - [`b3rb_ros_qr_detector.py`](file:///home/chaitanya/cognipilot/cranium/src/b3rb_ros_line_follower/b3rb_ros_line_follower/b3rb_ros_line_follower/b3rb_ros_qr_detector.py)

---

## 📐 Architecture Overview

```
Camera (compressed) ──► QRDetector ──────────────────► /qr_detection ─────────────┐
Camera (compressed) ──► ObjectRecognizer (YOLOv8) ──► /sign_board_detection ──────┤
LiDAR ──────────────────────────────────────────────► /scan                       │
/ServerCommunication ◄──────────────────────────────────────────────────────────┐ │
                                                                                 │ │
LineFollower ──────────────────────────────────────────────────────────────────►┘ │
  ├── BT Root (10 Hz tick)                                                         │
  │   ├── WatchdogParallel: TimingWatchdog, ZoneExitWatchdog, ServerDisconnectWD   │
  │   └── Selector: MissionController ──► PatientCycleBehavior                    │
  │                 NavigationAction (lattice planner, lane follow)                │
  ├── sign_board_callback ◄──────────────────────────────────────────────────────┘
  └── qr_detection_callback ◄────────────────────────────────────────────────────┘
```

---

## 1️⃣ SIGNBOARD DETECTION SYSTEM (`b3rb_ros_object_recog.py`)

### 1.1 Class Map
```python
CLASSES = {
    0: "A", 1: "B", 2: "C",       # Patient zone letters
    3: "Left", 4: "Right", 5: "Straight",  # Direction arrows
    6: "X", 7: "Y", 8: "Z"        # Hospital zone letters
}
```

### 1.2 YOLOv8 Model Loading
```python
package_share_directory = get_package_share_directory('b3rb_ros_line_follower')
model_path = os.path.join(package_share_directory, 'weights', 'best.pt')
self.model = YOLO(model_path)
```
> **Setup.py** installs weights via: `('share/' + package_name + '/weights', glob('weights/*'))`

### 1.3 ⏸ 10-Second Detection Pause After Mission Assignment

This is the **key sign-board suppression logic**. When the server sends a new mission command, sign detection is paused for **10 seconds** to prevent false positives interfering with navigation.

```python
# In ObjectRecognizer.__init__:
self.mission_received_time = 0.0
self.sign_paused_due_to_mission = False

# In server_communication_callback:
def server_communication_callback(self, message):
    # Buggy receives commands on dest=1 with ack=0
    if message.dest == 1 and message.ack == 0:
        import time
        self.mission_received_time = time.time()
        self.sign_paused_due_to_mission = True
        self.get_logger().info(
            f"Mission received: '{message.msg}'. Pausing sign detection for 10 seconds."
        )

# In camera_image_callback:
def camera_image_callback(self, message):
    import time
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
            return  # ← EXIT: skip inference
        else:
            self.sign_paused_due_to_mission = False  # resume
            self.get_logger().info("10 seconds pause complete. Resuming sign detection.")
```

### 1.4 YOLO Inference & Confidence Score Check

```python
def classify_sign(self, image):
    results = self.model.predict(image, conf=0.25, iou=0.45, verbose=False)
    # conf=0.25 → model-level minimum. Per-detection filtering at 0.65 happens in sign_board_callback

    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        detections = []
        for i in range(len(boxes)):
            box = boxes[i]
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())   # ← CONFIDENCE SCORE
            xyxy = box.xyxy[0].tolist()

            xmin, ymin, xmax, ymax = xyxy
            w = xmax - xmin; h = ymax - ymin
            cx = xmin + w / 2.0; cy = ymin + h / 2.0

            detections.append({
                "class": CLASSES.get(class_id, f"Unknown_{class_id}"),
                "confidence": confidence,    # ← included in payload
                "cx": cx, "cy": cy,
                "w": w, "h": h,
                "box": [float(xmin), float(ymin), float(xmax), float(ymax)]
            })

        return {
            "detected": True,
            "detections": detections,
            "inference_time_ms": inference_time_ms
        }
    return None
```

### 1.5 Letter–Arrow Pairing Logic (`get_sign_key_value_pairs`)

Pairs a building letter (A/B/C/X/Y/Z) with its closest direction arrow (Left/Right/Straight):

```python
def get_sign_key_value_pairs(self, detections):
    letters = [d for d in detections if d["class"] in ["A","B","C","X","Y","Z"]]
    arrows  = [d for d in detections if d["class"] in ["Left","Right","Straight","Up"]]

    pairings = {}
    for letter in letters:
        best_arrow = None
        best_dist  = float('inf')
        for arrow in arrows:
            dist = (letter["cx"] - arrow["cx"])**2 + (letter["cy"] - arrow["cy"])**2
            if dist < best_dist:
                best_dist = dist
                best_arrow = arrow["class"]
        if best_arrow:
            pairings[letter["class"]] = best_arrow

    direction_map = {"Left":"left","Right":"right","Straight":"straight","Up":"up"}
    sign_config = {letter: direction_map.get(arrow, arrow.lower())
                   for letter, arrow in pairings.items()}
    return sign_config  # e.g. {"A": "left", "X": "right"}
```

**Published payload structure:**
```json
{
  "detected": true,
  "detections": [
    {"class": "Left", "confidence": 0.87, "cx": 120.5, "cy": 200.3, "w": 40, "h": 35, "box": [...]},
    {"class": "A", "confidence": 0.92, "cx": 160.1, "cy": 200.5, "w": 30, "h": 30, "box": [...]}
  ],
  "inference_time_ms": 45.2,
  "sign_config": {"A": "left"}
}
```

---

## 2️⃣ SIGNBOARD CALLBACK IN LINE_FOLLOWER (`b3rb_ros_line_follower.py`)

### 2.1 State Variables (in `__init__`)
```python
self.last_detected_sign      = None   # most recently confirmed direction arrow class
self.sign_detected           = False  # flag: intersection sign has been acted on
self.in_intersection         = False  # flag: currently inside an intersection
self.intersection_exit_ticks = 0      # count of ticks since leaving intersection
self.last_turn_clear_time    = 0.0    # timestamp after intersection cleared

# sign_config debounce/locking mechanism
self.current_sign_payload       = None   # last seen sign_config dict
self.current_sign_payload_count = 0      # how many consecutive identical reads
self.payload_locked             = False  # True once 10 consistent reads confirmed
self.sign_config_map            = {}     # LOCKED map: {building: direction}
```

### 2.2 `sign_board_callback` — Full Logic

```python
def sign_board_callback(self, message):
    current_time = self.get_clock().now().nanoseconds / 1e9

    # ── GATE 1: 6-second cooldown after intersection clearance ──
    if current_time - self.last_turn_clear_time < 6.0:
        return   # ignore signs too soon after completing a turn

    if message.data:
        try:
            payload = json.loads(message.data)

            if payload.get("detected") and len(payload.get("detections", [])) > 0:
                # ── Only accept DIRECTION arrows, NOT building letters ──
                DIRECTION_CLASSES = {"Left", "Right", "Straight", "Up"}
                for det in payload.get("detections", []):
                    # ── CONFIDENCE GATE: minimum 0.65 confidence ──
                    if det["class"] in DIRECTION_CLASSES and det.get("confidence", 1.0) >= 0.65:
                        self.last_detected_sign = det["class"]
                        self.get_logger().info(
                            f"Stored detected sign direction: {self.last_detected_sign} "
                            f"(conf: {det.get('confidence', 1.0):.2f})"
                        )

            # ── SIGN CONFIG DEBOUNCE / LOCKING ──
            sign_config = payload.get("sign_config")
            if sign_config:
                if self.current_sign_payload == sign_config:
                    if not self.payload_locked:
                        self.current_sign_payload_count += 1
                        if self.current_sign_payload_count >= 10:   # ← 10 consistent reads
                            self.payload_locked = True
                            self.sign_config_map.update(sign_config)
                            self.get_logger().info(
                                f"Locked in sign_config map after 10 consistent reads: {self.sign_config_map}"
                            )
                else:
                    # Config changed → reset debounce
                    self.current_sign_payload       = sign_config
                    self.current_sign_payload_count = 1
                    self.payload_locked             = False
                    self.get_logger().info(f"Sign variation detected, reset count for: {sign_config}")

        except Exception as e:
            self.get_logger().error(f"Failed to parse sign board data: {e}")
```

### 2.3 How `sign_config_map` is used at intersections (in `NavigationAction.tick`)

```python
elif vectors.vector_count == 0:     # ← INTERSECTION DETECTED (no lane lines)
    node.in_intersection = True
    node.intersection_exit_ticks = 0
    speed = 0.2

    target     = getattr(node, 'current_target', None)
    target_key = SIGN_MAP.get(target, target)   # e.g. "HOSPITAL_1" → "X"

    desired_dir = None
    if target_key and hasattr(node, 'sign_config_map') and target_key in node.sign_config_map:
        desired_dir = node.sign_config_map[target_key]   # ← use locked sign map
    elif hasattr(node, 'last_detected_sign') and node.last_detected_sign is not None:
        desired_dir = node.last_detected_sign             # ← fallback to last arrow seen

    if desired_dir in ["left", "Left"]:
        raw_turn = 0.6
    elif desired_dir in ["right", "Right"]:
        raw_turn = -0.6
    elif desired_dir in ["straight", "Straight", "up", "Up"]:
        raw_turn = 0.0
    else:
        # Emergency fallback: use last PD error to guess direction
        if node.last_error > 0.2:
            raw_turn = 0.6
        elif node.last_error < -0.2:
            raw_turn = -0.6
        else:
            raw_turn = -0.5
```

### 2.4 Intersection exit & sign reset

```python
if getattr(node, 'in_intersection', False):
    node.intersection_exit_ticks = getattr(node, 'intersection_exit_ticks', 0) + 1
    if node.intersection_exit_ticks > 20:           # ← 20 ticks @ 10Hz = 2 seconds
        node.in_intersection         = False
        node.sign_detected           = False
        node.last_detected_sign      = None          # ← reset sign
        node.last_turn_clear_time    = node.get_clock().now().nanoseconds / 1e9
        # This starts the 6-second cooldown gate in sign_board_callback
        node.get_logger().info("Intersection cleared. Resetting sign detection and initiating 6s cooldown.")
```

---

## 3️⃣ QR CODE DETECTION (`b3rb_ros_qr_detector.py`)

### 3.1 Full QR Detector Node

```python
class QRDetector(Node):
    def __init__(self):
        super().__init__('qr_detector')

        # Subscribe to COMPRESSED camera images
        self.subscription_camera = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed',
            self.camera_image_callback, 10)

        # Publish decoded QR payload as JSON string
        self.publisher_qr = self.create_publisher(String, '/qr_detection', 10)

    def camera_image_callback(self, message):
        np_arr = np.frombuffer(message.data, np.uint8)
        image  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return

        payload = self.detect_qr_code(image)

        if payload is not None:
            msg = String()
            msg.data = json.dumps(payload)
            self.publisher_qr.publish(msg)
            self.get_logger().info(f"Found QR Data: {payload['data']}")

            # Visual debug overlay
            bbox = payload['bbox']
            if len(bbox) > 0 and len(bbox[0]) == 4:
                pts = np.array(bbox[0], np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [pts], True, (0, 255, 0), 3)
                pt1 = (int(pts[0][0][0]), int(pts[0][0][1]))
                cv2.putText(image, payload['data'], (pt1[0], pt1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.waitKey(1)

    def detect_qr_code(self, image):
        try:
            # PRIMARY: use pyzbar (more robust)
            try:
                from pyzbar.pyzbar import decode
            except ImportError:
                decode = None

            if decode is not None:
                decoded_objects = decode(image)
                if decoded_objects:
                    obj  = decoded_objects[0]
                    data = obj.data.decode('utf-8')
                    rect = obj.rect
                    bbox_list = [[[rect.left, rect.top],
                                  [rect.left + rect.width, rect.top],
                                  [rect.left + rect.width, rect.top + rect.height],
                                  [rect.left, rect.top + rect.height]]]
                    return {"detected": True, "data": data, "bbox": bbox_list}

            else:
                # FALLBACK: OpenCV built-in QR detector
                detector = cv2.QRCodeDetector()
                data, bbox, straight_qrcode = detector.detectAndDecode(image)
                if bbox is not None and data != "":
                    bbox_list = bbox.tolist() if hasattr(bbox, 'tolist') else bbox
                    return {"detected": True, "data": data, "bbox": bbox_list}

        except Exception as e:
            self.get_logger().debug(f"QR Detection failed: {e}")

        return None
```

**Published QR payload structure:**
```json
{
  "detected": true,
  "data": "PATIENT_1",
  "bbox": [[[x1,y1], [x2,y1], [x2,y2], [x1,y2]]]
}
```

---

## 4️⃣ QR DETECTION CALLBACK IN LINE_FOLLOWER

### 4.1 State Variables (in `__init__`)
```python
self.last_qr_data      = None    # raw QR string from most recent valid scan
self.last_qr_read_data = None    # same; used by PatientCycleBehavior
self.last_processed_qr = None    # dedup guard: do not process same QR twice
self.zone_entry_time   = 0.0     # timestamp when zone was entered
self.in_zone           = None    # "PATIENT" | "HOSPITAL" | "PARKING" | None
self.is_hospital_zone  = False   # True when inside hospital QR zone
self.hospital_lane_memory_active = False  # switches NavigationAction to memory mode
```

### 4.2 `qr_detection_callback` — Full Logic

```python
def qr_detection_callback(self, message):
    try:
        payload  = json.loads(message.data)
        qr_data  = payload.get("data", message.data)
    except json.JSONDecodeError:
        qr_data = message.data

    current_time = self.get_clock().now().nanoseconds / 1e9

    # ── ASSIGNMENT GATE: only accept QR that matches current mission target ──
    current_target = getattr(self, 'current_target', None)
    target_key     = SIGN_MAP.get(current_target, current_target) if current_target else None

    is_assigned_target = False
    if current_target and (current_target in qr_data or
                           (target_key and target_key in qr_data)):
        is_assigned_target = True
    elif "PARKING" in qr_data and self.mission_state == "EXITING":
        is_assigned_target = True

    # ── DEDUP GATE: ignore if we already processed this exact QR string ──
    if is_assigned_target and qr_data != self.last_processed_qr:
        self.last_qr_data      = qr_data
        self.last_processed_qr = qr_data
        self.last_qr_read_data = qr_data
        self.zone_entry_time   = current_time

        # ── Zone Classification ──
        if "hospital" in qr_data.lower() or "X" in qr_data or "Y" in qr_data or "Z" in qr_data:
            self.in_zone                    = "HOSPITAL"
            self.is_hospital_zone           = True
            self.hospital_lane_memory_active = True    # ← enables memory lane-follow mode
            self.get_logger().info(f"Entered ASSIGNED HOSPITAL zone: {qr_data}")

        elif "A" in qr_data or "B" in qr_data or "C" in qr_data:
            self.in_zone = "PATIENT"
            self.get_logger().info(f"Entered ASSIGNED PATIENT zone: {qr_data}")

        elif "PARKING" in qr_data:
            self.in_zone = "PARKING"
            self.get_logger().info(f"Entered PARKING zone: {qr_data}")
```

---

## 5️⃣ MISSION PLANNING SYSTEM

### 5.1 Mission State Machine (all states)

```
ENTER_CITY
  └─► AWAITING_INITIAL_ASSIGNMENT  (wait for server → received_target)
        └─► SEEKING_PATIENT         (drive to patient building)
              └─► IN_PATIENT_ZONE   (stopped, wait for hospital assignment)
                    └─► SEEKING_HOSPITAL (drive to hospital)
                          └─► IN_HOSPITAL_ZONE (stopped, wait for next patient or DONE)
                                └─► [loop back to SEEKING_PATIENT]
                                └─► EXITING
                                      └─► SEEKING_PARK
                                            └─► PARKED
                                                  └─► MISSION_DONE

ABORT_DISCONNECTED  (any state → when comm_retry_count >= 5)
```

### 5.2 SIGN_MAP (bidirectional letter↔building lookup)

```python
SIGN_MAP = {
    'A': 'PATIENT_1',   'B': 'PATIENT_2',   'C': 'PATIENT_3',
    'X': 'HOSPITAL_1',  'Y': 'HOSPITAL_2',  'Z': 'HOSPITAL_3',
    'PATIENT_1': 'A',   'PATIENT_2': 'B',   'PATIENT_3': 'C',
    'HOSPITAL_1': 'X',  'HOSPITAL_2': 'Y',  'HOSPITAL_3': 'Z',
}
```

### 5.3 Constants

```python
PATIENT_LIFETIME_SEC = 60.0    # max seconds to reach patient before penalty
ZONE_TIMEOUT_SEC     = 30.0    # max seconds in zone (currently unused in code, reference)
MAX_COMM_RETRIES     = 5       # triggers ABORT_DISCONNECTED
```

### 5.4 `PatientCycleBehavior.tick` — Core Mission Logic

```python
# ── SEEKING_PATIENT ──
if node.mission_state == "SEEKING_PATIENT":
    node.current_target = node.current_patient_target
    if node.in_zone == "PATIENT" and node.last_qr_read_data:
        if node.current_patient_target in node.last_qr_read_data:
            current_time = node.get_clock().now().nanoseconds / 1e9
            if not hasattr(node, 'zone_stop_start_time') or node.zone_stop_start_time is None:
                node.zone_stop_start_time = current_time   # start 5s timer
            elif current_time - node.zone_stop_start_time >= 5.0:   # ← 5-second drive delay
                node.mission_state    = "IN_PATIENT_ZONE"
                node.zone_stop_start_time = None
                node.send_server_msg(node.last_qr_read_data, is_ack=False)   # ← single server report
        else:
            pass  # wrong zone, keep driving

# ── IN_PATIENT_ZONE ──
elif node.mission_state == "IN_PATIENT_ZONE":
    node.rover_move_manual_mode(0.0, 0.0)     # ← STOP
    if node.received_target:
        raw      = node.received_target
        node.received_target = None
        hospital = SIGN_MAP.get(raw, raw)      # e.g. "X" → "HOSPITAL_1"
        node.current_hospital_target  = hospital
        node.current_target           = hospital
        node.in_zone                  = None   # clear zone
        node.last_qr_read_data        = None
        node.last_processed_qr        = None
        node.mission_state            = "SEEKING_HOSPITAL"

# ── SEEKING_HOSPITAL ──
elif node.mission_state == "SEEKING_HOSPITAL":
    node.current_target = node.current_hospital_target
    if node.in_zone == "HOSPITAL" and node.last_qr_read_data:
        if node.current_hospital_target in node.last_qr_read_data:
            current_time = node.get_clock().now().nanoseconds / 1e9
            if not hasattr(node, 'zone_stop_start_time') or node.zone_stop_start_time is None:
                node.zone_stop_start_time = current_time  # start 7s timer
            elif current_time - node.zone_stop_start_time >= 7.0:   # ← 7-second drive delay
                node.mission_state    = "IN_HOSPITAL_ZONE"
                node.patients_done   += 1
                node.zone_stop_start_time = None
                node.send_server_msg(node.last_qr_read_data, is_ack=False)

# ── IN_HOSPITAL_ZONE ──
elif node.mission_state == "IN_HOSPITAL_ZONE":
    node.rover_move_manual_mode(0.0, 0.0)    # ← STOP
    if node.received_target:
        raw = node.received_target
        node.received_target = None
        if raw in ("DONE", "PARKING"):
            node.mission_state = "EXITING"
        else:
            next_patient              = SIGN_MAP.get(raw, raw)
            node.current_patient_target = next_patient
            node.current_target         = next_patient
            node.in_zone                = None
            node.last_qr_read_data      = None
            node.last_processed_qr      = None
            node.patient_spawn_time     = node.get_clock().now().nanoseconds / 1e9
            node.mission_state          = "SEEKING_PATIENT"
```

### 5.5 `HandleCommunication` — Server Protocol State Machine

The comm system uses a 5-state sub-machine inside the BT:

| State | Action | Transition |
|---|---|---|
| `IDLE` | Nothing | → `QR_DRIVE_DELAY` when QR zone entered |
| `QR_DRIVE_DELAY` | Keep driving for 12 sec after QR scan | → `SEND_QR` after 12s |
| `SEND_QR` | Send QR data to server | → `WAITING_ACK` |
| `WAITING_ACK` | Wait for ACK (timeout 1s, retry) | → `WAITING_TARGET` on ACK received |
| `WAITING_TARGET` | Wait for hospital/patient assignment (timeout 5s) | → `IDLE` on target received |
| `PARKED` | Send "PARKED", wait for "OK" | → `DONE` |
| `DONE` | Stop rover permanently | — |

```python
# QR_DRIVE_DELAY → SEND_QR transition (12-second drive):
if node.comm_state == "QR_DRIVE_DELAY":
    if current_time - node.qr_drive_start_time >= 12.0:
        node.is_hospital_zone   = False
        node.comm_state         = "SEND_QR"
        node.comm_msg_to_send   = node.last_qr_data
        node.rover_move_manual_mode(0.0, 0.0)  # stop before sending

# WAITING_ACK retry after 1 second:
elif node.comm_state == "WAITING_ACK":
    if node.received_ack:
        node.comm_state    = "WAITING_TARGET"
        node.received_ack  = False
    elif current_time - node.comm_last_send_time > 1.0:
        node.comm_retry_count += 1
        node.comm_state = "SEND_QR"   # retry

# WAITING_TARGET retry after 5 seconds:
elif node.comm_state == "WAITING_TARGET":
    if node.received_target is not None:
        if node.last_uid not in node.uid_to_mission_state:
            node.send_server_msg("", is_ack=True, req_uid=node.last_uid)  # ACK server
            node.uid_to_mission_state.add(node.last_uid)
            node.comm_state = "IDLE"
    elif current_time - node.comm_last_send_time > 5.0:
        node.comm_retry_count += 1
        node.comm_state = "SEND_QR"   # timeout, retry
```

### 5.6 Watchdogs

```python
class TimingWatchdog(Behavior):
    """Tracks 60-second patient timer; logs penalty if exceeded."""
    def tick(self, node):
        if node.mission_state in ["SEEKING_PATIENT","IN_PATIENT_ZONE","AWAITING_ASSIGNMENT"]:
            current_time = node.get_clock().now().nanoseconds / 1e9
            if node.patient_spawn_time > 0 and (current_time - node.patient_spawn_time) > 60.0:
                node.get_logger().error(f"Patient {node.current_patient_target} expired! Penalty.")
                node.penalties.append({...})
                node.patient_spawn_time = current_time  # reset to prevent spam
        return NodeStatus.SUCCESS

class ZoneExitWatchdog(Behavior):
    """If in a zone for > 12 seconds without another QR, force-exit the zone."""
    def tick(self, node):
        if node.in_zone is not None:
            current_time = node.get_clock().now().nanoseconds / 1e9
            if current_time - node.zone_entry_time > 12.0:
                old_zone    = node.in_zone
                node.in_zone = None
                if node.mission_state == "AWAITING_ASSIGNMENT":
                    node.penalties.append({...})  # penalty for early exit
        return NodeStatus.SUCCESS

class ServerDisconnectWatchdog(Behavior):
    """After 5 comm retries, abort mission."""
    def tick(self, node):
        if node.comm_retry_count >= MAX_COMM_RETRIES:   # MAX_COMM_RETRIES = 5
            node.mission_state = "ABORT_DISCONNECTED"
        return NodeStatus.SUCCESS
```

---

## 6️⃣ SERVER COMMUNICATION PROTOCOL

### 6.1 Message Format (ServerCommunication msg)

| Field | Type | Meaning |
|---|---|---|
| `src` | int | 1 = Buggy, 2 = Server |
| `dest` | int | 1 = Buggy, 2 = Server |
| `uid` | int | 0–255 rolling counter |
| `ack` | int | 1 = ACK, 0 = Data |
| `msg` | str | Payload (QR data / target letter) |

### 6.2 `send_server_msg`

```python
def send_server_msg(self, text_msg, is_ack=False, req_uid=None):
    server_msg = ServerCommunication()
    server_msg.src  = 1
    server_msg.dest = 2

    if is_ack:
        server_msg.uid = req_uid     # echo back server's UID
        server_msg.ack = 1
    else:
        self.uid_counter       = (self.uid_counter + 1) % 256
        self.last_sent_uid     = self.uid_counter
        server_msg.uid         = self.uid_counter
        server_msg.ack         = 0

    server_msg.msg = text_msg
    self.publisher_server.publish(server_msg)
```

### 6.3 `server_communication_callback`

```python
def server_communication_callback(self, message):
    if message.dest == 1:
        if message.ack == 1 and message.uid == self.last_sent_uid:
            self.received_ack = True           # ACK for our own message
        elif message.ack == 0:
            self.received_target = message.msg  # New target or instruction
            self.last_uid        = message.uid  # store UID for dedup
```

### 6.4 Duplicate UID Guard

```python
self.uid_to_mission_state = set()   # set of UIDs already processed

# In WAITING_TARGET handler:
if node.last_uid not in node.uid_to_mission_state:
    # ACK the server and consume the target
    node.send_server_msg("", is_ack=True, req_uid=node.last_uid)
    node.uid_to_mission_state.add(node.last_uid)
    node.comm_state = "IDLE"
else:
    # Duplicate — ACK again but discard target
    node.send_server_msg("", is_ack=True, req_uid=node.last_uid)
    node.received_target = None
```

---

## 7️⃣ HOSPITAL LANE MEMORY MODE

When the buggy enters a hospital zone (detected via QR), lane lines may be obstructed. The system activates **memory mode**:

```python
# In qr_detection_callback (triggered by hospital QR):
self.hospital_lane_memory_active = True

# In NavigationAction.tick:
in_hospital_memory = getattr(node, 'hospital_lane_memory_active', False)

if in_hospital_memory:
    error      = node.last_error       # use last known error
    derivative = 0.0
    raw_turn   = error * 1.5
    speed      = 0.4                   # reduced speed

# Memory exits when BOTH lane lines are detected with > 70% confidence:
if vectors.vector_count == 2:
    confidence1 = (mag1 / crop_height) * 100.0
    confidence2 = (mag2 / crop_height) * 100.0
    if confidence1 > 70.0 and confidence2 > 70.0:
        node.hospital_lane_memory_active = False
```

---

## 8️⃣ FILES TO UPDATE FOR INTEGRATION

| File | What to Update |
|---|---|
| [`b3rb_ros_object_recog.py`](file:///home/chaitanya/cognipilot/cranium/src/b3rb_ros_line_follower/b3rb_ros_line_follower/b3rb_ros_line_follower/b3rb_ros_object_recog.py) | Confidence threshold at YOLO call (currently `conf=0.25`); the per-detection gate is `0.65` in line_follower |
| [`b3rb_ros_qr_detector.py`](file:///home/chaitanya/cognipilot/cranium/src/b3rb_ros_line_follower/b3rb_ros_line_follower/b3rb_ros_line_follower/b3rb_ros_qr_detector.py) | Add zone classification hint to published payload if needed; install `pyzbar` for primary decoder |
| [`b3rb_ros_line_follower.py`](file:///home/chaitanya/cognipilot/cranium/src/b3rb_ros_line_follower/b3rb_ros_line_follower/b3rb_ros_line_follower/b3rb_ros_line_follower.py) | All callbacks, BT behaviors, and mission state machine are here |
| [`setup.py`](file:///home/chaitanya/cognipilot/cranium/src/b3rb_ros_line_follower/b3rb_ros_line_follower/setup.py) | Ensure `weights/best.pt` is listed in data_files glob |
| `package.xml` | Add `python3-pyzbar` if using pyzbar for QR |
| `weights/best.pt` | YOLOv8 trained model file must exist at this path |

---

## 9️⃣ CRITICAL TIMING PARAMETERS (ALL IN ONE PLACE)

| Parameter | Value | Location | Purpose |
|---|---|---|---|
| `conf=0.25` | YOLO model | `classify_sign()` | Minimum detection threshold |
| `confidence >= 0.65` | Per-detection | `sign_board_callback()` | Direction arrow acceptance gate |
| `>= 10` consistent reads | Debounce count | `sign_board_callback()` | Lock sign_config map |
| `6.0` seconds | Cooldown | `sign_board_callback()` | Post-intersection sign ignore window |
| `> 20` ticks (=2s @ 10Hz) | Intersection exit | `NavigationAction.tick()` | Clear intersection flag |
| `10.0` seconds | Pause | `ObjectRecognizer` | Sign detection pause after mission received |
| `5.0` seconds | Drive delay | `PatientCycleBehavior` | Drive time at patient zone before stopping |
| `7.0` seconds | Drive delay | `PatientCycleBehavior` | Drive time at hospital zone before stopping |
| `12.0` seconds | Drive delay | `HandleCommunication` | Drive time before sending QR to server |
| `12.0` seconds | Zone exit | `ZoneExitWatchdog` | Force-exit zone if no new QR |
| `60.0` seconds | Patient timer | `TimingWatchdog` | Patient pickup deadline |
| `1.0` second | ACK timeout | `HandleCommunication` | Retry send if no ACK |
| `5.0` seconds | Target timeout | `HandleCommunication` | Retry send if no target |
| `5` retries | Max retries | `ServerDisconnectWatchdog` | Abort on repeated comm failures |
| `70%` confidence | Lane confidence | `NavigationAction.tick()` | Exit hospital memory mode |

---

## 🔟 MASTER INTEGRATION PROMPT

```
You are integrating three cooperating ROS 2 nodes for a competition buggy:
  1. ObjectRecognizer (b3rb_ros_object_recog.py) — YOLOv8 sign detection
  2. QRDetector (b3rb_ros_qr_detector.py) — QR code zone detection
  3. LineFollower (b3rb_ros_line_follower.py) — Mission brain + navigation

Apply the following logic in its entirety:

────────────────────────────────────────────
SIGN DETECTION (ObjectRecognizer):
────────────────────────────────────────────
• Load YOLOv8 model from: get_package_share_directory('b3rb_ros_line_follower')/weights/best.pt
• Classes: {0:A,1:B,2:C,3:Left,4:Right,5:Straight,6:X,7:Y,8:Z}
• Run YOLO inference with conf=0.25, iou=0.45 on every compressed camera frame.
• Subscribe to /ServerCommunication; on receiving any dest=1,ack=0 message:
    - Set self.sign_paused_due_to_mission = True and record self.mission_received_time = time.time()
• In camera_image_callback: if sign_paused_due_to_mission and elapsed < 10.0:
    - Skip inference entirely, show "Detection Paused / Resuming in Xs" overlay on cv2.imshow.
    - After 10.0s, set sign_paused_due_to_mission = False and resume inference.
• After inference: pair letters (A/B/C/X/Y/Z) with their nearest arrow (Left/Right/Straight/Up)
  using Euclidean distance of bounding box centers.
• Publish to /sign_board_detection as JSON:
    {"detected": true, "detections": [{class, confidence, cx, cy, w, h, box}...],
     "inference_time_ms": float, "sign_config": {"A":"left","X":"right",...}}

────────────────────────────────────────────
QR DETECTION (QRDetector):
────────────────────────────────────────────
• Subscribe to /camera/image_raw/compressed.
• Decode QR using pyzbar (primary). If pyzbar not installed, fall back to cv2.QRCodeDetector.
• On successful decode, publish to /qr_detection as JSON:
    {"detected": true, "data": "<qr_string>", "bbox": [[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]]}
• Draw polyline bounding box and text overlay on image (for debug display).

────────────────────────────────────────────
LINE FOLLOWER — SIGN BOARD CALLBACK:
────────────────────────────────────────────
• Subscribe to /sign_board_detection.
• Gate 1: if (current_time - self.last_turn_clear_time) < 6.0: return (ignore during cooldown).
• Gate 2: for each detection, only accept if class in {"Left","Right","Straight","Up"}
          AND confidence >= 0.65.
  → Store as self.last_detected_sign.
• Debounce sign_config: if same sign_config dict received >= 10 consecutive times,
  lock it into self.sign_config_map (used at intersections).
  Reset counter if sign_config changes.

────────────────────────────────────────────
LINE FOLLOWER — QR CALLBACK:
────────────────────────────────────────────
• Subscribe to /qr_detection.
• Gate 1 (assignment): only process QR if it matches self.current_target or target_key from SIGN_MAP,
  OR if "PARKING" in qr_data and mission_state == "EXITING".
• Gate 2 (dedup): if qr_data == self.last_processed_qr: skip.
• On valid new QR:
    - Set self.last_qr_data, self.last_processed_qr, self.last_qr_read_data = qr_data
    - Set self.zone_entry_time = current_time
    - If hospital QR (contains "hospital" or X/Y/Z):
        self.in_zone = "HOSPITAL"; self.hospital_lane_memory_active = True
    - If patient QR (contains A/B/C):
        self.in_zone = "PATIENT"
    - If "PARKING":
        self.in_zone = "PARKING"

────────────────────────────────────────────
LINE FOLLOWER — MISSION STATE MACHINE:
────────────────────────────────────────────
States (in order): ENTER_CITY → AWAITING_INITIAL_ASSIGNMENT → SEEKING_PATIENT →
IN_PATIENT_ZONE → SEEKING_HOSPITAL → IN_HOSPITAL_ZONE → [loop] → EXITING →
SEEKING_PARK → PARKED → MISSION_DONE | ABORT_DISCONNECTED

SEEKING_PATIENT:
  - If in_zone=="PATIENT" and current_patient_target in last_qr_read_data:
    - Start zone_stop_start_time timer.
    - After 5.0s: transition to IN_PATIENT_ZONE, send QR data to server.
  - Else: continue driving.

IN_PATIENT_ZONE:
  - Stop rover (speed=0, turn=0).
  - Wait for received_target from server.
  - On receive: map letter→HOSPITAL_N, clear zone+qr state, go to SEEKING_HOSPITAL.

SEEKING_HOSPITAL:
  - If in_zone=="HOSPITAL" and current_hospital_target in last_qr_read_data:
    - Start zone_stop_start_time timer.
    - After 7.0s: transition to IN_HOSPITAL_ZONE, increment patients_done, send QR to server.

IN_HOSPITAL_ZONE:
  - Stop rover.
  - Wait for received_target.
  - If "DONE"/"PARKING": go to EXITING.
  - Else: map letter→PATIENT_N, reset zone+qr, increment patient_spawn_time, go to SEEKING_PATIENT.

EXITING → SEEKING_PARK:
  - If in_zone == "PARKING": stop, go to PARKED.

PARKED:
  - Send "PARKED" to server. Wait for "OK".
  - On "OK": go to DONE.

────────────────────────────────────────────
WATCHDOGS (run every BT tick at 10Hz):
────────────────────────────────────────────
• TimingWatchdog: if patient timer exceeds 60s, log penalty.
• ZoneExitWatchdog: if in_zone is set for > 12s without new QR, force in_zone = None.
• ServerDisconnectWatchdog: if comm_retry_count >= 5, set mission_state = ABORT_DISCONNECTED.

────────────────────────────────────────────
INTERSECTION NAVIGATION:
────────────────────────────────────────────
• Intersections = vector_count == 0 (no lane lines visible).
• At intersection:
    1. Lookup current_target in sign_config_map (locked map).
    2. Fallback to last_detected_sign if not found.
    3. Steer: left→turn=+0.6, right→turn=-0.6, straight→turn=0.0.
    4. After > 20 ticks (2s), clear in_intersection, reset last_detected_sign,
       set last_turn_clear_time (starts 6s sign cooldown).

────────────────────────────────────────────
HOSPITAL LANE MEMORY:
────────────────────────────────────────────
• Activated by hospital QR: hospital_lane_memory_active = True.
• While active: use last_error for steering, speed=0.4.
• Deactivate when BOTH lane vectors detected with magnitude/crop_height > 70%.

────────────────────────────────────────────
FILES TO UPDATE:
────────────────────────────────────────────
1. setup.py → ensure `('share/b3rb_ros_line_follower/weights', glob('weights/*'))` is in data_files.
2. package.xml → add python3-pyzbar dependency.
3. weights/best.pt → place trained YOLOv8 model here.
4. All three Python files → apply logic as described above.
```
