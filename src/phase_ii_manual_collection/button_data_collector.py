#!/usr/bin/env python3
"""
Button Data Collector - Python Backend
Receives UDP label events from Android button grid app and saves labeled sensor data.

Usage:
    python button_data_collector.py [--skip-noise]

Options:
    --skip-noise    Skip baseline noise capture (for testing)

Requirements:
    - Watch app streaming sensor data on port 12345
    - Button app sending label events on port 12345
    - All devices on same WiFi network

IMPORTANT: Data Format
    Each UDP packet contains ONE sensor reading (not all three sensors).
    Watch app sends separate packets for:
    - linear_acceleration (accel_x, accel_y, accel_z)
    - gyroscope (gyro_x, gyro_y, gyro_z)
    - rotation_vector (rot_x, rot_y, rot_z, rot_w)

    The CSV files store each packet as a row with the sensor type.
    This is CORRECT behavior - each row has non-zero values only for its sensor type.

    To verify data quality, use: python src/inspect_csv_data.py <csv_file>
"""

import json
import socket
import threading
import time
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
import csv


class ButtonDataCollector:
    def __init__(self, udp_port=12345, output_dir="data/button_collected", skip_noise=False):
        self.udp_port = udp_port
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skip_noise = skip_noise

        # Buffer for sensor data (keep last 30 seconds at 50Hz = 1500 samples)
        self.sensor_buffer = deque(maxlen=1500)

        # Currently recording action (None means NOISE mode - default state)
        self.active_recording = None

        # Noise capture
        self.noise_buffer = []
        self.baseline_noise_captured = skip_noise  # Skip if flag set
        self.baseline_noise_duration = 30  # seconds
        self.noise_start_time = time.time()  # Initialize immediately

        # Lock for thread safety
        self.lock = threading.Lock()

        # Statistics
        self.action_counts = {
            'walk': 0, 'idle': 0, 'punch': 0,
            'jump': 0, 'turn_left': 0, 'turn_right': 0,
            'noise': 0  # Track noise segments
        }

        # Session info
        self.session_start = datetime.now()
        self.total_recordings = 0

        # Connection tracking
        self.watch_connected = False
        self.phone_connected = False
        self.last_watch_data = 0
        self.last_phone_data = 0
        self.sensor_data_count = 0
        self.label_event_count = 0

    def start(self):
        """Start the UDP listener"""
        print(f"🎯 Button Data Collector Started")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        print(f"🌐 Listening on UDP port {self.udp_port}")
        print(f"⏰ Session start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n� Waiting for connections...")
        print(f"   📱 Watch app: Waiting...")
        print(f"   📲 Phone app: Waiting...")

        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', self.udp_port))
        sock.settimeout(0.5)  # Non-blocking with timeout

        ready_to_start = False

        try:
            # Wait for both connections before starting
            while not ready_to_start:
                try:
                    data, addr = sock.recvfrom(4096)
                    message = data.decode('utf-8')

                    # Parse JSON message
                    try:
                        msg = json.loads(message)

                        # Check message type
                        if msg.get('sensor'):
                            # Sensor data from watch
                            if not self.watch_connected:
                                self.watch_connected = True
                                self.last_watch_data = time.time()
                                print(f"   ✅ Watch connected from {addr[0]}")
                            self.sensor_data_count += 1

                        elif msg.get('type') == 'label_event':
                            # Label event from phone
                            if not self.phone_connected:
                                self.phone_connected = True
                                self.last_phone_data = time.time()
                                print(f"   ✅ Phone connected from {addr[0]}")
                            self.label_event_count += 1

                        # Check if both are ready
                        if self.watch_connected and self.phone_connected and not ready_to_start:
                            print(f"\n🎉 Both devices connected!")
                            print(f"   📊 Sensor packets: {self.sensor_data_count}")
                            print(f"   📱 Label events: {self.label_event_count}")
                            print(f"\n{'='*60}")
                            print(f"✨ READY TO START DATA COLLECTION ✨")
                            print(f"{'='*60}")
                            print(f"\n👉 Press ENTER to begin collecting data...")

                            # Wait for user input
                            input()

                            print(f"\n🚀 Collection started!")

                            if not self.skip_noise:
                                print(f" DEFAULT STATE: NOISE MODE (all data labeled as noise unless button pressed)")
                                print(f"📊 Capturing {self.baseline_noise_duration}s baseline noise...\n")
                            else:
                                print(f"⚡ SKIP NOISE MODE: Noise capture disabled for testing")
                                print(f"✋ Ready for button presses immediately!\n")

                            ready_to_start = True
                            self.noise_start_time = time.time()

                    except json.JSONDecodeError:
                        pass

                except socket.timeout:
                    # Check connection health
                    current_time = time.time()
                    if self.watch_connected and current_time - self.last_watch_data > 5:
                        print(f"   ⚠️  Watch connection lost")
                        self.watch_connected = False
                    if self.phone_connected and current_time - self.last_phone_data > 5:
                        print(f"   ⚠️  Phone connection lost")
                        self.phone_connected = False
                    continue

            # Main collection loop
            while True:
                try:
                    data, addr = sock.recvfrom(4096)
                    message = data.decode('utf-8')

                    # Parse JSON message
                    try:
                        msg = json.loads(message)
                        self.handle_message(msg, addr)
                    except json.JSONDecodeError:
                        pass

                except socket.timeout:
                    continue

        except KeyboardInterrupt:
            print("\n\n🛑 Stopping data collector...")
            if not self.skip_noise:
                self.segment_and_save_noise()
            self.print_statistics()
        finally:
            sock.close()

    def handle_message(self, msg, addr):
        """Handle incoming UDP message"""
        msg_type = msg.get('type')

        if msg_type == 'label_event':
            self.handle_label_event(msg, addr)
        elif msg.get('sensor'):
            # Sensor data - parse and buffer
            self.last_watch_data = time.time()

            # Parse sensor values (handle both formats)
            # Note: Each packet contains ONE sensor type, not all three!
            # We store each sensor reading separately with its timestamp
            values = msg.get('values', {})
            sensor_type = msg.get('sensor', 'unknown')
            timestamp = msg.get('timestamp_ns', msg.get('timestamp', time.time() * 1e9))

            # Initialize all sensor values to 0 (only the relevant sensor will have data)
            accel_x = accel_y = accel_z = 0.0
            gyro_x = gyro_y = gyro_z = 0.0
            rot_x = rot_y = rot_z = 0.0
            rot_w = 1.0

            # Extract values based on sensor type
            if sensor_type == 'linear_acceleration':
                accel_x = values.get('x', msg.get('accel_x', 0.0))
                accel_y = values.get('y', msg.get('accel_y', 0.0))
                accel_z = values.get('z', msg.get('accel_z', 0.0))
            elif sensor_type == 'gyroscope':
                gyro_x = values.get('x', msg.get('gyro_x', 0.0))
                gyro_y = values.get('y', msg.get('gyro_y', 0.0))
                gyro_z = values.get('z', msg.get('gyro_z', 0.0))
            elif sensor_type == 'rotation_vector':
                rot_x = values.get('x', msg.get('rot_x', 0.0))
                rot_y = values.get('y', msg.get('rot_y', 0.0))
                rot_z = values.get('z', msg.get('rot_z', 0.0))
                rot_w = values.get('w', msg.get('rot_w', 1.0))
            else:
                # Fallback to flat format (old watch app)
                accel_x = msg.get('accel_x', 0.0)
                accel_y = msg.get('accel_y', 0.0)
                accel_z = msg.get('accel_z', 0.0)
                gyro_x = msg.get('gyro_x', 0.0)
                gyro_y = msg.get('gyro_y', 0.0)
                gyro_z = msg.get('gyro_z', 0.0)
                rot_x = msg.get('rot_x', 0.0)
                rot_y = msg.get('rot_y', 0.0)
                rot_z = msg.get('rot_z', 0.0)
                rot_w = msg.get('rot_w', 1.0)

            sensor_entry = {
                'timestamp': timestamp,
                'sensor': sensor_type,
                'accel_x': accel_x,
                'accel_y': accel_y,
                'accel_z': accel_z,
                'gyro_x': gyro_x,
                'gyro_y': gyro_y,
                'gyro_z': gyro_z,
                'rot_x': rot_x,
                'rot_y': rot_y,
                'rot_z': rot_z,
                'rot_w': rot_w
            }

            # Add to main buffer
            with self.lock:
                self.sensor_buffer.append(sensor_entry)

            # Capture baseline noise (first 30 seconds after noise_start_time)
            if not self.baseline_noise_captured and self.noise_start_time:
                elapsed = time.time() - self.noise_start_time
                if elapsed <= self.baseline_noise_duration:
                    # Still in baseline capture window
                    self.noise_buffer.append(sensor_entry)

                    # Print progress every 5 seconds
                    if int(elapsed) % 5 == 0 and len(self.noise_buffer) % 250 == 0:
                        print(f"   📊 Baseline: {int(elapsed)}s / {self.baseline_noise_duration}s ({len(self.noise_buffer)} samples)")
                else:
                    # Baseline complete - save noise data immediately
                    self.baseline_noise_captured = True
                    samples = len(self.noise_buffer)
                    print(f"✅ Baseline noise captured ({samples} samples)")

                    # Save baseline noise to file immediately
                    if samples > 0:
                        baseline_file = self.output_dir / f"baseline_noise_{int(time.time())}.csv"
                        with open(baseline_file, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                'timestamp', 'sensor',
                                'accel_x', 'accel_y', 'accel_z',
                                'gyro_x', 'gyro_y', 'gyro_z',
                                'rot_x', 'rot_y', 'rot_z', 'rot_w'
                            ])
                            for entry in self.noise_buffer:
                                writer.writerow([
                                    entry['timestamp'], entry['sensor'],
                                    entry['accel_x'], entry['accel_y'], entry['accel_z'],
                                    entry['gyro_x'], entry['gyro_y'], entry['gyro_z'],
                                    entry['rot_x'], entry['rot_y'], entry['rot_z'], entry['rot_w']
                                ])
                        print(f"   💾 Baseline saved to {baseline_file.name}")

                    print("✋ Ready for button presses...\n")

            # If no button pressed (default NOISE state), continue collecting noise
            elif self.active_recording is None and not self.skip_noise:
                self.noise_buffer.append(sensor_entry)

    def handle_label_event(self, msg, addr):
        """Handle label start/end events from button app"""
        action = msg.get('action')
        event = msg.get('event')
        timestamp = msg.get('timestamp_ms')

        # Update phone connection status
        self.last_phone_data = time.time()

        # Handle test ping
        if event == 'ping':
            print(f"📱 Test ping received from {addr[0]}")
            return

        if event == 'start':
            with self.lock:
                if self.active_recording:
                    print(f"⚠️  Already recording {self.active_recording['action']}, ignoring new start")
                    return

                self.active_recording = {
                    'action': action,
                    'start_time': timestamp,
                    'start_index': len(self.sensor_buffer)
                }

                print(f"🔴 Recording {action.upper()} (from {addr[0]})")

        elif event == 'end':
            with self.lock:
                if not self.active_recording:
                    print(f"⚠️  No active recording to end")
                    return

                if self.active_recording['action'] != action:
                    print(f"⚠️  Mismatch: recording {self.active_recording['action']} but got end for {action}")
                    return

                # Calculate duration
                duration_ms = timestamp - self.active_recording['start_time']
                duration_sec = duration_ms / 1000.0

                # Save the recording
                filename = self.save_recording(
                    action=action,
                    start_time=self.active_recording['start_time'],
                    end_time=timestamp,
                    count=msg.get('count', 0)
                )

                # Update statistics
                self.action_counts[action] = self.action_counts.get(action, 0) + 1
                self.total_recordings += 1

                count = msg.get('count', 0)
                print(f"✅ Saved {action.upper()} ({duration_sec:.2f}s) → {filename} [Count: {count}]")
                self.print_progress()

                self.active_recording = None

    def save_recording(self, action, start_time, end_time, count):
        """Save recording to CSV file"""
        # Create filename
        filename = f"{action}_{start_time}_to_{end_time}.csv"
        filepath = self.output_dir / filename

        # Extract sensor data from buffer within time window
        with self.lock:
            # Convert millisecond timestamps to nanoseconds for comparison
            start_ns = start_time * 1_000_000
            end_ns = end_time * 1_000_000

            # Filter sensor data within time window
            recording_data = [
                entry for entry in self.sensor_buffer
                if start_ns <= entry['timestamp'] <= end_ns
            ]

        # Save to CSV with corrected column order (matches old format)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'accel_x', 'accel_y', 'accel_z',
                'gyro_x', 'gyro_y', 'gyro_z',
                'rot_w', 'rot_x', 'rot_y', 'rot_z',
                'sensor', 'timestamp'
            ])

            # Write sensor data with correct order
            for entry in recording_data:
                writer.writerow([
                    entry['accel_x'], entry['accel_y'], entry['accel_z'],
                    entry['gyro_x'], entry['gyro_y'], entry['gyro_z'],
                    entry['rot_w'], entry['rot_x'], entry['rot_y'], entry['rot_z'],
                    entry['sensor'],
                    entry['timestamp']
                ])

        print(f"   💾 Saved {len(recording_data)} sensor samples to {filename}")

        return filename

    def segment_and_save_noise(self):
        """Segment noise buffer into fixed-size chunks and save exactly 30 per classifier"""
        import random

        if not self.noise_buffer:
            print("⚠️  No noise data collected")
            return

        print(f"\n🔇 Processing noise data ({len(self.noise_buffer)} samples)...")

        # All samples should now be properly structured
        valid_noise = self.noise_buffer

        print(f"✅ Validated {len(valid_noise)} noise samples")

        # Segment for locomotion classifier (5-second chunks)
        locomotion_segments = self._segment_noise(valid_noise, duration_sec=5.0, samples_per_sec=50)

        # Segment for action classifier (1-second chunks)
        action_segments = self._segment_noise(valid_noise, duration_sec=1.0, samples_per_sec=50)

        # Randomly select exactly 30 samples per classifier
        if len(locomotion_segments) >= 30:
            selected_locomotion = random.sample(locomotion_segments, 30)
            print(f"📦 Selected 30 locomotion noise segments (5s each) from {len(locomotion_segments)} available")
        else:
            selected_locomotion = locomotion_segments
            print(f"⚠️  Only {len(locomotion_segments)} locomotion segments available (need 30)")

        if len(action_segments) >= 30:
            selected_action = random.sample(action_segments, 30)
            print(f"📦 Selected 30 action noise segments (1s each) from {len(action_segments)} available")
        else:
            selected_action = action_segments
            print(f"⚠️  Only {len(action_segments)} action segments available (need 30)")

        # Save locomotion noise segments
        for i, segment in enumerate(selected_locomotion, 1):
            filename = f"noise_locomotion_seg_{i:03d}.csv"
            self._save_noise_segment(filename, segment)

        # Save action noise segments
        for i, segment in enumerate(selected_action, 1):
            filename = f"noise_action_seg_{i:03d}.csv"
            self._save_noise_segment(filename, segment)

        self.action_counts['noise'] = len(selected_locomotion) + len(selected_action)
        print(f"✅ Saved {self.action_counts['noise']} noise segments")

    def _segment_noise(self, noise_data, duration_sec, samples_per_sec):
        """Segment noise data into fixed-duration chunks"""
        segment_size = int(duration_sec * samples_per_sec)
        segments = []

        for i in range(0, len(noise_data) - segment_size + 1, segment_size):
            segment = noise_data[i:i + segment_size]
            if len(segment) == segment_size:
                segments.append(segment)

        return segments

    def _save_noise_segment(self, filename, segment):
        """Save a noise segment to CSV with corrected column order"""
        filepath = self.output_dir / filename

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'accel_x', 'accel_y', 'accel_z',
                'gyro_x', 'gyro_y', 'gyro_z',
                'rot_w', 'rot_x', 'rot_y', 'rot_z',
                'sensor', 'timestamp'
            ])

            # Write actual sensor data with correct order
            for entry in segment:
                writer.writerow([
                    entry['accel_x'], entry['accel_y'], entry['accel_z'],
                    entry['gyro_x'], entry['gyro_y'], entry['gyro_z'],
                    entry['rot_w'], entry['rot_x'], entry['rot_y'], entry['rot_z'],
                    entry['sensor'],
                    entry['timestamp']
                ])

    def print_progress(self):
        """Print current collection progress"""
        print(f"\n📊 Progress: {self.total_recordings} total recordings")

        # Print counts in grid format (with defaults for missing keys)
        print(f"   WALK: {self.action_counts.get('walk', 0):2d}  IDLE: {self.action_counts.get('idle', 0):2d}")
        print(f"  PUNCH: {self.action_counts.get('punch', 0):2d}  JUMP: {self.action_counts.get('jump', 0):2d}")
        print(f"  TURN_L: {self.action_counts.get('turn_left', 0):2d}  TURN_R: {self.action_counts.get('turn_right', 0):2d}")
        print(f"  NOISE: {self.action_counts.get('noise', 0):2d}\n")

    def print_statistics(self):
        """Print final session statistics"""
        duration = datetime.now() - self.session_start

        print("\n" + "="*50)
        print("📈 SESSION STATISTICS")
        print("="*50)
        print(f"Duration: {duration}")
        print(f"Total recordings: {self.total_recordings}")
        print(f"\nAction breakdown:")
        for action, count in sorted(self.action_counts.items()):
            print(f"  {action.upper():12s}: {count:3d} samples")

        print(f"\n💾 Files saved to: {self.output_dir.absolute()}")
        print("="*50)


def main():
    """Main entry point"""
    # Parse command line arguments
    skip_noise = '--skip-noise' in sys.argv

    collector = ButtonDataCollector(skip_noise=skip_noise)

    print("""
╔═══════════════════════════════════════════════════════════╗
║         Button Data Collection - Python Backend          ║
╚═══════════════════════════════════════════════════════════╝

Instructions:
1. Make sure Pixel Watch app is running and streaming
2. Launch Android button grid app on phone
3. Configure phone app with this computer's IP address
4. Press and hold buttons while performing gestures
5. Watch this console for confirmation messages

Options:
  --skip-noise    Skip baseline noise capture (for testing)

Press Ctrl+C to stop and see statistics.
""")

    if skip_noise:
        print("⚡ SKIP NOISE MODE ENABLED - Noise capture will be skipped\n")

    try:
        collector.start()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
