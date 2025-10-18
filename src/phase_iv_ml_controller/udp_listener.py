import socket
import json
import time
import math
import threading
from collections import deque
from queue import Queue, Empty
from pynput.keyboard import Controller, Key
import sys
import os
# Add shared_utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared_utils'))
import network_utils
from zeroconf import ServiceInfo, Zeroconf
import joblib
import pandas as pd
import numpy as np
from scipy import stats
from scipy.fft import fft

# --- Global State ---
keyboard = Controller()
is_walking = False
last_step_time = 0
walking_thread = None
stop_walking_event = threading.Event()
# The core state for our character's direction
facing_direction = "right"
# A variable to store the phone's current orientation
last_known_orientation = {"x": 0, "y": 0, "z": 0, "w": 1}

# --- NEW: Walk Fuel System ---
walk_fuel_seconds = 0.0
last_frame_time = time.time()

# --- NEW: Attack debouncing to prevent rapid-fire attacks ---
last_attack_time = 0
ATTACK_COOLDOWN_SEC = 0.3  # Minimum time between attacks


# --- NEW: The core mathematical helper function ---
def rotate_vector_by_quaternion(vector, quat):
    """Rotates a 3D vector by a quaternion using standard quaternion rotation formula."""
    q_vec = [quat["x"], quat["y"], quat["z"]]
    q_scalar = quat["w"]

    # Standard formula for vector rotation by quaternion
    a = [
        2 * (q_vec[1] * vector[2] - q_vec[2] * vector[1]),
        2 * (q_vec[2] * vector[0] - q_vec[0] * vector[2]),
        2 * (q_vec[0] * vector[1] - q_vec[1] * vector[0]),
    ]

    b = [q_scalar * a[0], q_scalar * a[1], q_scalar * a[2]]

    c = [
        q_vec[1] * a[2] - q_vec[2] * a[1],
        q_vec[2] * a[0] - q_vec[0] * a[2],
        q_vec[0] * a[1] - q_vec[1] * a[0],
    ]

    rotated_vector = [
        vector[0] + b[0] + c[0],
        vector[1] + b[1] + c[1],
        vector[2] + b[2] + c[2],
    ]

    return rotated_vector


# --- Configuration Loading ---
def load_config():
    """Load configuration from config.json file"""
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("ERROR: config.json not found! Please create the configuration file.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in config.json: {e}")
        exit(1)


def get_key(key_string):
    """Converts a string from config to a pynput Key object if needed."""
    if key_string.startswith("Key."):
        key_name = key_string.split(".")[-1]
        if hasattr(Key, key_name):
            return getattr(Key, key_name)
    return key_string


# Load configuration at startup
config = load_config()

# Auto-detect and update IP address
print("🔍 Auto-detecting IP address...")
network_utils.update_config_ip()
# Reload config to get the updated IP
config = load_config()

# Extract configuration values
LISTEN_IP = config["network"]["listen_ip"]
LISTEN_PORT = config["network"]["listen_port"]
FUEL_ADDED_PER_STEP = config["thresholds"]["fuel_added_per_step_sec"]
MAX_FUEL_CAPACITY = config["thresholds"]["max_fuel_sec"]
PUNCH_THRESHOLD = config["thresholds"]["punch_threshold_xy_accel"]
JUMP_THRESHOLD = config["thresholds"]["jump_threshold_z_accel"]
TURN_THRESHOLD = config["thresholds"]["turn_threshold_degrees"]
# NEW: Hardcoded stability threshold for pitch/roll stability check
STABILITY_THRESHOLD_DEGREES = 50.0
# NEW: Z-axis stability factor for attack detection (prevents attack during jumps)
ATTACK_Z_STABILITY_FACTOR = 0.7  # Z-axis must be < punch_threshold * this factor

# Load key mappings from config
KEY_MAP = {
    "left": get_key(config["keyboard_mappings"]["left"]),
    "right": get_key(config["keyboard_mappings"]["right"]),
    "jump": get_key(config["keyboard_mappings"]["jump"]),
    "attack": get_key(config["keyboard_mappings"]["attack"]),
}


# --- ML Model Loading ---
def load_ml_models():
    """Load trained ML models for gesture recognition."""
    try:
        model = joblib.load("models/gesture_classifier.pkl")
        scaler = joblib.load("models/feature_scaler.pkl")
        feature_names = joblib.load("models/feature_names.pkl")
        print("✓ ML models loaded successfully")
        return model, scaler, feature_names
    except FileNotFoundError as e:
        print(f"⚠️  ML model files not found: {e}")
        print("   Running in threshold-based mode (legacy)")
        return None, None, None


def extract_window_features(window_df):
    """Extract comprehensive features from a time window of sensor data.

    This is the same function used during training.
    """
    features = {}

    # Separate by sensor type
    accel = window_df[window_df["sensor"] == "linear_acceleration"]
    gyro = window_df[window_df["sensor"] == "gyroscope"]
    rot = window_df[window_df["sensor"] == "rotation_vector"]

    # ========== ACCELERATION FEATURES ==========
    if len(accel) > 0:
        for axis in ["accel_x", "accel_y", "accel_z"]:
            values = accel[axis].dropna()

            if len(values) > 0:
                features[f"{axis}_mean"] = values.mean()
                features[f"{axis}_std"] = values.std()
                features[f"{axis}_max"] = values.max()
                features[f"{axis}_min"] = values.min()
                features[f"{axis}_range"] = values.max() - values.min()
                features[f"{axis}_median"] = values.median()
                features[f"{axis}_skew"] = stats.skew(values)
                features[f"{axis}_kurtosis"] = stats.kurtosis(values)

                threshold = values.mean() + 2 * values.std()
                features[f"{axis}_peak_count"] = len(values[values > threshold])

                if len(values) > 2:
                    fft_vals = np.abs(fft(values))[: len(values) // 2]
                    if len(fft_vals) > 0:
                        features[f"{axis}_fft_max"] = fft_vals.max()
                        features[f"{axis}_dominant_freq"] = fft_vals.argmax()
                        features[f"{axis}_fft_mean"] = fft_vals.mean()

    # ========== GYROSCOPE FEATURES ==========
    if len(gyro) > 0:
        for axis in ["gyro_x", "gyro_y", "gyro_z"]:
            values = gyro[axis].dropna()

            if len(values) > 0:
                features[f"{axis}_mean"] = values.mean()
                features[f"{axis}_std"] = values.std()
                features[f"{axis}_max_abs"] = values.abs().max()
                features[f"{axis}_range"] = values.max() - values.min()
                features[f"{axis}_skew"] = stats.skew(values)
                features[f"{axis}_kurtosis"] = stats.kurtosis(values)
                features[f"{axis}_rms"] = np.sqrt(np.mean(values**2))

                if len(values) > 2:
                    fft_vals = np.abs(fft(values))[: len(values) // 2]
                    if len(fft_vals) > 0:
                        features[f"{axis}_fft_max"] = fft_vals.max()

    # ========== ROTATION FEATURES ==========
    if len(rot) > 0:
        for axis in ["rot_x", "rot_y", "rot_z", "rot_w"]:
            values = rot[axis].dropna()

            if len(values) > 0:
                features[f"{axis}_mean"] = values.mean()
                features[f"{axis}_std"] = values.std()
                features[f"{axis}_range"] = values.max() - values.min()

    # ========== CROSS-SENSOR FEATURES ==========
    if len(accel) > 0:
        accel_mag = np.sqrt(
            accel["accel_x"].fillna(0) ** 2
            + accel["accel_y"].fillna(0) ** 2
            + accel["accel_z"].fillna(0) ** 2
        )
        features["accel_magnitude_mean"] = accel_mag.mean()
        features["accel_magnitude_max"] = accel_mag.max()
        features["accel_magnitude_std"] = accel_mag.std()

    if len(gyro) > 0:
        gyro_mag = np.sqrt(
            gyro["gyro_x"].fillna(0) ** 2
            + gyro["gyro_y"].fillna(0) ** 2
            + gyro["gyro_z"].fillna(0) ** 2
        )
        features["gyro_magnitude_mean"] = gyro_mag.mean()
        features["gyro_magnitude_max"] = gyro_mag.max()
        features["gyro_magnitude_std"] = gyro_mag.std()

    return features


# Load ML models
ml_model, ml_scaler, ml_feature_names = load_ml_models()
ML_ENABLED = ml_model is not None

# Multi-threaded architecture: Collector → Predictor → Actor
# Reduced window size for low-latency prediction (0.3s micro-windows)
WINDOW_SIZE_SEC = 0.3  # Micro-window for fast gesture detection
WINDOW_SIZE_SAMPLES = int(WINDOW_SIZE_SEC * 50)  # ~15 samples at 50Hz
ML_CONFIDENCE_THRESHOLD = 0.7
CONFIDENCE_GATING_COUNT = 5  # Require N consecutive predictions for state change

# Thread-safe queues for communication
sensor_queue = Queue(maxsize=1000)  # Collector → Predictor
action_queue = Queue(maxsize=100)   # Predictor → Actor


# --- Walker Thread ---
def walker_thread_func():
    global is_walking
    is_walking = True

    # Press left or right arrow based on facing direction
    key_to_press = KEY_MAP["right"] if facing_direction == "right" else KEY_MAP["left"]
    keyboard.press(key_to_press)

    stop_walking_event.wait()

    keyboard.release(key_to_press)
    is_walking = False


# --- Helper Functions ---
def quaternion_to_roll(qx, qy, qz, qw):
    """Convert quaternion to roll angle in degrees."""
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    return math.degrees(roll)


def quaternion_to_euler(q):
    """Converts a quaternion dict into yaw, pitch, roll in degrees."""
    x, y, z, w = q["x"], q["y"], q["z"], q["w"]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.degrees(math.copysign(math.pi / 2, sinp))
    else:
        pitch = math.degrees(math.asin(sinp))

    # Yaw (z-axis rotation) - This is our Azimuth
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))

    return yaw, pitch, roll


# ========================================================================
# MULTI-THREADED ARCHITECTURE: Collector → Predictor → Actor
# ========================================================================

def collector_thread(sock, sensor_queue, stop_event):
    """Thread 1: Collect sensor data from UDP and push to queue.
    
    This thread runs at network speed and does NO processing.
    It simply reads UDP packets and dumps them into the sensor queue.
    """
    print("[COLLECTOR] Thread started")
    
    while not stop_event.is_set():
        try:
            # Non-blocking receive with timeout
            sock.settimeout(0.1)
            data, addr = sock.recvfrom(2048)
            
            try:
                parsed_json = json.loads(data.decode())
                current_time = time.time()
                
                sensor_type = parsed_json.get("sensor")
                sensor_reading = {
                    "timestamp": current_time,
                    "sensor": sensor_type,
                }
                
                # Add sensor-specific values
                if sensor_type == "linear_acceleration":
                    vals = parsed_json.get("values", {})
                    sensor_reading.update({
                        "accel_x": vals.get("x", 0),
                        "accel_y": vals.get("y", 0),
                        "accel_z": vals.get("z", 0),
                    })
                elif sensor_type == "gyroscope":
                    vals = parsed_json.get("values", {})
                    sensor_reading.update({
                        "gyro_x": vals.get("x", 0),
                        "gyro_y": vals.get("y", 0),
                        "gyro_z": vals.get("z", 0),
                    })
                elif sensor_type == "rotation_vector":
                    vals = parsed_json.get("values", {})
                    sensor_reading.update({
                        "rot_x": vals.get("x", 0),
                        "rot_y": vals.get("y", 0),
                        "rot_z": vals.get("z", 0),
                        "rot_w": vals.get("w", 1),
                    })
                elif sensor_type == "step_detector":
                    sensor_reading["step"] = True
                
                # Push to queue (non-blocking with timeout)
                try:
                    sensor_queue.put(sensor_reading, timeout=0.01)
                except:
                    pass  # Queue full, drop packet
                    
            except json.JSONDecodeError:
                pass  # Invalid JSON, skip
                
        except socket.timeout:
            continue  # No data, continue
        except Exception as e:
            if not stop_event.is_set():
                print(f"[COLLECTOR] Error: {e}")
            break
    
    print("[COLLECTOR] Thread stopped")


def predictor_thread(ml_model, ml_scaler, ml_feature_names, sensor_queue, action_queue, stop_event):
    """Thread 2: Run ML inference on sensor data.
    
    This thread runs in a tight loop, maintaining a micro-window buffer,
    extracting features, and running predictions as fast as possible.
    """
    print("[PREDICTOR] Thread started")
    
    # Micro-window buffer (0.3 seconds at 50Hz = ~15 samples)
    sensor_buffer = deque(maxlen=WINDOW_SIZE_SAMPLES)
    
    while not stop_event.is_set():
        try:
            # Pull data from sensor queue (non-blocking)
            try:
                sensor_reading = sensor_queue.get(timeout=0.01)
                sensor_buffer.append(sensor_reading)
            except Empty:
                # No new data, still try to predict if buffer has enough
                pass
            
            # Run prediction if we have enough data
            if len(sensor_buffer) >= int(WINDOW_SIZE_SAMPLES * 0.8):  # 80% full
                try:
                    # Convert buffer to DataFrame
                    buffer_df = pd.DataFrame(list(sensor_buffer))
                    
                    # Extract features
                    features = extract_window_features(buffer_df)
                    
                    # Create feature vector matching training format
                    feature_vector = pd.DataFrame([features])
                    feature_vector = feature_vector.reindex(
                        columns=ml_feature_names, fill_value=0
                    )
                    
                    # Scale features
                    features_scaled = ml_scaler.transform(feature_vector)
                    
                    # Predict gesture
                    prediction = ml_model.predict(features_scaled)[0]
                    confidence = ml_model.predict_proba(features_scaled).max()
                    
                    # Push prediction to action queue if confident
                    if confidence >= ML_CONFIDENCE_THRESHOLD:
                        action = {
                            "gesture": prediction,
                            "confidence": confidence,
                            "timestamp": time.time()
                        }
                        try:
                            action_queue.put(action, timeout=0.01)
                        except:
                            pass  # Queue full, drop prediction
                    
                except Exception as e:
                    # Silently handle prediction errors
                    pass
                    
        except Exception as e:
            if not stop_event.is_set():
                print(f"[PREDICTOR] Error: {e}")
    
    print("[PREDICTOR] Thread stopped")


def actor_thread(action_queue, sensor_queue, stop_event):
    """Thread 3: Execute keyboard actions based on predictions.
    
    This thread listens to the action queue and executes keyboard commands.
    It implements confidence gating (requires N consecutive predictions)
    and manages walking state.
    """
    print("[ACTOR] Thread started")
    
    global is_walking, walking_thread, facing_direction, walk_fuel_seconds
    
    # State management
    prediction_history = deque(maxlen=CONFIDENCE_GATING_COUNT)
    last_frame_time = time.time()
    
    while not stop_event.is_set():
        try:
            # Update walk fuel
            current_time = time.time()
            delta_time = current_time - last_frame_time
            last_frame_time = current_time
            
            walk_fuel_seconds = max(0.0, walk_fuel_seconds - delta_time)
            
            # Check for step detector events (for walk fuel)
            try:
                while True:
                    sensor_reading = sensor_queue.get_nowait()
                    if sensor_reading.get("sensor") == "step_detector":
                        walk_fuel_seconds = min(
                            walk_fuel_seconds + FUEL_ADDED_PER_STEP,
                            MAX_FUEL_CAPACITY
                        )
            except Empty:
                pass
            
            # Manage walking state based on fuel
            if walk_fuel_seconds > 0 and not is_walking and walking_thread is None:
                stop_walking_event.clear()
                walking_thread = threading.Thread(target=walker_thread_func)
                walking_thread.start()
            elif walk_fuel_seconds <= 0 and is_walking:
                stop_walking_event.set()
                if walking_thread is not None:
                    walking_thread.join()
                    walking_thread = None
            
            # Process action from action queue
            try:
                action = action_queue.get(timeout=0.1)
                gesture = action["gesture"]
                confidence = action["confidence"]
                
                # Add to prediction history for confidence gating
                prediction_history.append(gesture)
                
                # Check if we have N consecutive predictions of the same gesture
                if len(prediction_history) == CONFIDENCE_GATING_COUNT:
                    # Check if all recent predictions agree
                    if len(set(prediction_history)) == 1:
                        # Execute the gesture
                        if gesture == "jump":
                            print(f"\n[ML] JUMP ({confidence:.2f})")
                            keyboard.press(KEY_MAP["jump"])
                            time.sleep(0.1)
                            keyboard.release(KEY_MAP["jump"])
                        elif gesture == "punch":
                            print(f"\n[ML] ATTACK ({confidence:.2f})")
                            keyboard.press(KEY_MAP["attack"])
                            time.sleep(0.1)
                            keyboard.release(KEY_MAP["attack"])
                        elif gesture == "turn":
                            print(f"\n[ML] TURN ({confidence:.2f})")
                            facing_direction = (
                                "left" if facing_direction == "right" else "right"
                            )
                        # walk is handled by step detector/fuel system
                        # noise is ignored
                        
            except Empty:
                continue  # No action, continue
                
        except Exception as e:
            if not stop_event.is_set():
                print(f"[ACTOR] Error: {e}")
    
    print("[ACTOR] Thread stopped")


# --- Main Listener Logic ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LISTEN_IP, LISTEN_PORT))

# --- Network Service Discovery Setup ---
print("🔗 Registering service for automatic discovery...")
zeroconf = Zeroconf()
service_type = "_silksong._udp.local."
service_name = f"SilksongController.{service_type}"

# Convert IP address to bytes for ServiceInfo
ip_bytes = socket.inet_aton(LISTEN_IP)

service_info = ServiceInfo(
    service_type,
    service_name,
    addresses=[ip_bytes],
    port=LISTEN_PORT,
    properties={"version": "1.0"},
)

zeroconf.register_service(service_info)
print(f"✓ Service registered as '{service_name}'")

ml_mode = "ML-POWERED" if ML_ENABLED else "THRESHOLD-BASED"
print(f"--- Silksong Controller v2.0 ({ml_mode}) ---")
print(f"Listening on {LISTEN_IP}:{LISTEN_PORT}")
if ML_ENABLED:
    print(f"✓ Machine Learning Model Active")
    print(f"  Architecture: Collector → Predictor → Actor")
    print(f"  Window Size: {WINDOW_SIZE_SEC}s (micro-window)")
    print(f"  Confidence Threshold: {ML_CONFIDENCE_THRESHOLD:.0%}")
    print(f"  Confidence Gating: {CONFIDENCE_GATING_COUNT} consecutive predictions")
print("Official Hollow Knight/Silksong key mappings:")
print(
    f"  Movement: {config['keyboard_mappings']['left']}/{config['keyboard_mappings']['right']} (direction-based)"
)
print(
    f"  Jump: {config['keyboard_mappings']['jump']} | Attack: {config['keyboard_mappings']['attack']}"
)
print("---------------------------------------")

# ========================================================================
# MAIN: Start Multi-Threaded Architecture
# ========================================================================

if ML_ENABLED:
    # Create stop event for graceful shutdown
    stop_event = threading.Event()
    
    # Start the three threads
    collector = threading.Thread(
        target=collector_thread,
        args=(sock, sensor_queue, stop_event),
        name="Collector"
    )
    
    predictor = threading.Thread(
        target=predictor_thread,
        args=(ml_model, ml_scaler, ml_feature_names, sensor_queue, action_queue, stop_event),
        name="Predictor"
    )
    
    actor = threading.Thread(
        target=actor_thread,
        args=(action_queue, sensor_queue, stop_event),
        name="Actor"
    )
    
    # Start all threads
    collector.start()
    predictor.start()
    actor.start()
    
    print("\n✓ All threads started successfully")
    print("  Collector: UDP → Queue")
    print("  Predictor: Queue → ML → Actions")
    print("  Actor: Actions → Keyboard")
    print("\nPress Ctrl+C to stop...\n")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        stop_event.set()
        
        # Wait for threads to finish
        print("  Stopping collector...")
        collector.join(timeout=2)
        print("  Stopping predictor...")
        predictor.join(timeout=2)
        print("  Stopping actor...")
        actor.join(timeout=2)
        
        # Clean up walking thread if still running
        if is_walking and walking_thread is not None:
            stop_walking_event.set()
            walking_thread.join()
    finally:
        print("🔗 Unregistering service...")
        zeroconf.unregister_service(service_info)
        zeroconf.close()
        sock.close()
        print("Controller stopped.")

else:
    print("\n⚠️  ML models not available, cannot run multi-threaded controller")
    print("    Please train models first using CS156_Silksong_Watch.ipynb")
    zeroconf.unregister_service(service_info)
    zeroconf.close()
    sock.close()
