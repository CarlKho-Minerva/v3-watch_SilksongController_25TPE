#!/usr/bin/env python3
"""
Data Collection Dashboard - Real-time Verification Interface

Provides a live dashboard showing:
- Connection status for Watch and Phone apps
- Real-time sensor data values
- Data rate metrics
- Recent samples
- Buffer status
- Recording status

This helps verify that data is actually coming in from both devices.

Usage:
    python data_collection_dashboard.py [--skip-noise]
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
import os

# ANSI Colors for terminal
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


class DataCollectionDashboard:
    def __init__(self, udp_port=12345, output_dir="data/button_collected", skip_noise=False):
        self.udp_port = udp_port
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skip_noise = skip_noise

        # Buffer for sensor data (keep last 30 seconds at 50Hz = 1500 samples)
        self.sensor_buffer = deque(maxlen=1500)

        # Currently recording action
        self.active_recording = None

        # Noise capture
        self.noise_buffer = []
        self.baseline_noise_captured = skip_noise
        self.baseline_noise_duration = 30
        self.noise_start_time = None

        # Lock for thread safety
        self.lock = threading.Lock()

        # Statistics
        self.action_counts = {
            'walk': 0, 'idle': 0, 'punch': 0,
            'jump': 0, 'turn_left': 0, 'turn_right': 0,
            'noise': 0
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

        # Data rate tracking
        self.sensor_rate_window = deque(maxlen=50)  # Last 50 sensor packets
        self.label_rate_window = deque(maxlen=10)   # Last 10 label events

        # Latest sensor values (for display)
        self.latest_accel = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.latest_gyro = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.latest_rotation = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
        self.latest_sensor_type = "unknown"
        self.latest_timestamp = 0

        # Dashboard control
        self.running = True
        self.collection_started = False
        self.dashboard_thread = None

    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def format_value(self, value, width=7):
        """Format a float value with fixed width"""
        return f"{value:>{width}.3f}"

    def get_data_rate(self, window):
        """Calculate data rate from timestamp window"""
        if len(window) < 2:
            return 0.0
        time_span = window[-1] - window[0]
        if time_span > 0:
            return len(window) / time_span
        return 0.0

    def draw_dashboard(self):
        """Draw the dashboard interface"""
        self.clear_screen()

        # Header
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}         DATA COLLECTION DASHBOARD - Real-time Verification{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")

        # Session Info
        session_duration = datetime.now() - self.session_start
        print(f"{Colors.BOLD}Session:{Colors.RESET} {self.session_start.strftime('%Y-%m-%d %H:%M:%S')} "
              f"({session_duration})")
        print(f"{Colors.BOLD}Output:{Colors.RESET} {self.output_dir.absolute()}\n")

        # Connection Status
        print(f"{Colors.BOLD}{Colors.YELLOW}CONNECTION STATUS{Colors.RESET}")
        print(f"{Colors.YELLOW}{'─'*80}{Colors.RESET}")

        watch_status = f"{Colors.GREEN}✓ CONNECTED{Colors.RESET}" if self.watch_connected else f"{Colors.RED}✗ DISCONNECTED{Colors.RESET}"
        phone_status = f"{Colors.GREEN}✓ CONNECTED{Colors.RESET}" if self.phone_connected else f"{Colors.RED}✗ DISCONNECTED{Colors.RESET}"

        sensor_rate = self.get_data_rate(self.sensor_rate_window)
        label_rate = self.get_data_rate(self.label_rate_window)

        print(f"  📱 Watch App:  {watch_status}  "
              f"| Packets: {self.sensor_data_count:>6}  "
              f"| Rate: {sensor_rate:>5.1f} Hz")
        print(f"  📲 Phone App:  {phone_status}  "
              f"| Events:  {self.label_event_count:>6}  "
              f"| Rate: {label_rate:>5.2f} Hz")

        # Collection Status
        print(f"\n{Colors.BOLD}{Colors.CYAN}COLLECTION STATUS{Colors.RESET}")
        print(f"{Colors.CYAN}{'─'*80}{Colors.RESET}")

        if not self.collection_started:
            print(f"  {Colors.YELLOW}⏸  WAITING TO START{Colors.RESET}")
            print(f"  {Colors.DIM}(Press ENTER when both devices are connected){Colors.RESET}")
        elif not self.baseline_noise_captured and not self.skip_noise:
            elapsed = time.time() - self.noise_start_time if self.noise_start_time else 0
            remaining = max(0, self.baseline_noise_duration - elapsed)
            progress = min(100, (elapsed / self.baseline_noise_duration) * 100)
            bar_width = 40
            filled = int(bar_width * progress / 100)
            bar = '█' * filled + '░' * (bar_width - filled)
            print(f"  📊 Baseline Noise: [{bar}] {progress:.0f}%")
            print(f"  ⏱  Time: {elapsed:.1f}s / {self.baseline_noise_duration}s "
                  f"(Remaining: {remaining:.1f}s)")
            print(f"  📦 Samples: {len(self.noise_buffer)}")
        elif self.active_recording:
            action = self.active_recording['action']
            duration_ms = time.time() * 1000 - self.active_recording['start_time']
            duration_sec = duration_ms / 1000.0
            print(f"  {Colors.RED}🔴 RECORDING: {action.upper()}{Colors.RESET}")
            print(f"  ⏱  Duration: {duration_sec:.2f}s")
            print(f"  📦 Buffer size: {len(self.sensor_buffer)}")
        else:
            print(f"  {Colors.GREEN}✅ READY{Colors.RESET} - Waiting for button press")
            print(f"  📦 Buffer size: {len(self.sensor_buffer)}")

        # Latest Sensor Data
        print(f"\n{Colors.BOLD}{Colors.GREEN}LATEST SENSOR DATA{Colors.RESET}")
        print(f"{Colors.GREEN}{'─'*80}{Colors.RESET}")

        # Show if we have recent data
        data_age = time.time() - self.last_watch_data if self.last_watch_data else 999
        if data_age < 1.0:
            freshness = f"{Colors.GREEN}FRESH{Colors.RESET}"
        elif data_age < 3.0:
            freshness = f"{Colors.YELLOW}STALE{Colors.RESET}"
        else:
            freshness = f"{Colors.RED}NO DATA{Colors.RESET}"

        print(f"  Sensor: {self.latest_sensor_type:<20} | Freshness: {freshness} ({data_age:.1f}s ago)")
        print(f"  Timestamp: {self.latest_timestamp}")
        print()

        # Acceleration
        print(f"  {Colors.BOLD}Acceleration (m/s²):{Colors.RESET}")
        print(f"    X: {self.format_value(self.latest_accel['x'])}  "
              f"Y: {self.format_value(self.latest_accel['y'])}  "
              f"Z: {self.format_value(self.latest_accel['z'])}")

        # Gyroscope
        print(f"  {Colors.BOLD}Gyroscope (rad/s):{Colors.RESET}")
        print(f"    X: {self.format_value(self.latest_gyro['x'])}  "
              f"Y: {self.format_value(self.latest_gyro['y'])}  "
              f"Z: {self.format_value(self.latest_gyro['z'])}")

        # Rotation
        print(f"  {Colors.BOLD}Rotation Vector:{Colors.RESET}")
        print(f"    X: {self.format_value(self.latest_rotation['x'])}  "
              f"Y: {self.format_value(self.latest_rotation['y'])}  "
              f"Z: {self.format_value(self.latest_rotation['z'])}  "
              f"W: {self.format_value(self.latest_rotation['w'])}")

        # Recording Statistics
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}RECORDING STATISTICS{Colors.RESET}")
        print(f"{Colors.MAGENTA}{'─'*80}{Colors.RESET}")
        print(f"  Total Recordings: {self.total_recordings}")
        print(f"  WALK: {self.action_counts['walk']:>3}  IDLE: {self.action_counts['idle']:>3}  "
              f"PUNCH: {self.action_counts['punch']:>3}")
        print(f"  JUMP: {self.action_counts['jump']:>3}  TURN_L: {self.action_counts['turn_left']:>3}  "
              f"TURN_R: {self.action_counts['turn_right']:>3}")
        print(f"  NOISE: {self.action_counts['noise']:>3}")

        # Footer
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"{Colors.DIM}Press Ctrl+C to stop and save data{Colors.RESET}\n")

    def dashboard_update_loop(self):
        """Background thread that updates dashboard"""
        while self.running:
            try:
                self.draw_dashboard()
                time.sleep(0.5)  # Update twice per second
            except Exception as e:
                # Don't crash dashboard thread on errors
                pass

    def start(self):
        """Start the dashboard and data collector"""
        print(f"\n{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}         DATA COLLECTION DASHBOARD - Starting Up{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*80}{Colors.RESET}\n")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        print(f"🌐 Listening on UDP port {self.udp_port}")
        print(f"⏰ Session start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n🔍 Waiting for connections...")

        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', self.udp_port))
        sock.settimeout(0.5)

        ready_to_start = False

        try:
            # Wait for both connections
            while not ready_to_start:
                try:
                    data, addr = sock.recvfrom(4096)
                    message = data.decode('utf-8')

                    try:
                        msg = json.loads(message)

                        # Check message type
                        if msg.get('sensor'):
                            # Sensor data from watch
                            if not self.watch_connected:
                                self.watch_connected = True
                                print(f"   ✅ Watch connected from {addr[0]}")
                            self.last_watch_data = time.time()
                            self.sensor_data_count += 1
                            self.sensor_rate_window.append(time.time())

                        elif msg.get('type') == 'label_event':
                            # Label event from phone
                            if not self.phone_connected:
                                self.phone_connected = True
                                print(f"   ✅ Phone connected from {addr[0]}")
                            self.last_phone_data = time.time()
                            self.label_event_count += 1
                            self.label_rate_window.append(time.time())

                        # Check if both are ready
                        if self.watch_connected and self.phone_connected and not ready_to_start:
                            print(f"\n🎉 Both devices connected!")
                            print(f"📊 Sensor packets: {self.sensor_data_count}")
                            print(f"📱 Label events: {self.label_event_count}")
                            print(f"\n{'='*80}")
                            print(f"✨ READY TO START DATA COLLECTION ✨")
                            print(f"{'='*80}")
                            print(f"\n👉 Press ENTER to begin collecting data...")

                            input()  # Wait for user

                            print(f"\n🚀 Collection started!")
                            if not self.skip_noise:
                                print(f"📊 Capturing {self.baseline_noise_duration}s baseline noise...")

                            ready_to_start = True
                            self.collection_started = True
                            self.noise_start_time = time.time()

                            # Start dashboard thread
                            self.dashboard_thread = threading.Thread(target=self.dashboard_update_loop, daemon=True)
                            self.dashboard_thread.start()

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

                    try:
                        msg = json.loads(message)
                        self.handle_message(msg, addr)
                    except json.JSONDecodeError:
                        pass

                except socket.timeout:
                    continue

        except KeyboardInterrupt:
            self.running = False
            print("\n\n🛑 Stopping data collector...")
            if self.dashboard_thread:
                self.dashboard_thread.join(timeout=1.0)
            if not self.skip_noise:
                self.segment_and_save_noise()
            self.print_statistics()
        finally:
            self.running = False
            sock.close()

    def handle_message(self, msg, addr):
        """Handle incoming UDP message"""
        msg_type = msg.get('type')

        if msg_type == 'label_event':
            self.handle_label_event(msg, addr)
        elif msg.get('sensor'):
            # Sensor data
            self.last_watch_data = time.time()
            self.sensor_rate_window.append(time.time())

            # Parse sensor values
            values = msg.get('values', {})

            # Extract based on sensor type
            sensor_type = msg.get('sensor', 'unknown')
            self.latest_sensor_type = sensor_type

            if sensor_type == 'linear_acceleration':
                self.latest_accel['x'] = values.get('x', msg.get('accel_x', 0.0))
                self.latest_accel['y'] = values.get('y', msg.get('accel_y', 0.0))
                self.latest_accel['z'] = values.get('z', msg.get('accel_z', 0.0))
                accel_x, accel_y, accel_z = self.latest_accel['x'], self.latest_accel['y'], self.latest_accel['z']
            else:
                accel_x = msg.get('accel_x', 0.0)
                accel_y = msg.get('accel_y', 0.0)
                accel_z = msg.get('accel_z', 0.0)

            if sensor_type == 'gyroscope':
                self.latest_gyro['x'] = values.get('x', msg.get('gyro_x', 0.0))
                self.latest_gyro['y'] = values.get('y', msg.get('gyro_y', 0.0))
                self.latest_gyro['z'] = values.get('z', msg.get('gyro_z', 0.0))
                gyro_x, gyro_y, gyro_z = self.latest_gyro['x'], self.latest_gyro['y'], self.latest_gyro['z']
            else:
                gyro_x = msg.get('gyro_x', 0.0)
                gyro_y = msg.get('gyro_y', 0.0)
                gyro_z = msg.get('gyro_z', 0.0)

            if sensor_type == 'rotation_vector':
                self.latest_rotation['x'] = values.get('x', msg.get('rot_x', 0.0))
                self.latest_rotation['y'] = values.get('y', msg.get('rot_y', 0.0))
                self.latest_rotation['z'] = values.get('z', msg.get('rot_z', 0.0))
                self.latest_rotation['w'] = values.get('w', msg.get('rot_w', 1.0))
                rot_x, rot_y, rot_z, rot_w = (self.latest_rotation['x'], self.latest_rotation['y'],
                                              self.latest_rotation['z'], self.latest_rotation['w'])
            else:
                rot_x = msg.get('rot_x', 0.0)
                rot_y = msg.get('rot_y', 0.0)
                rot_z = msg.get('rot_z', 0.0)
                rot_w = msg.get('rot_w', 1.0)

            self.latest_timestamp = msg.get('timestamp_ns', msg.get('timestamp', time.time() * 1e9))

            sensor_entry = {
                'timestamp': self.latest_timestamp,
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

            with self.lock:
                self.sensor_buffer.append(sensor_entry)

            # Baseline noise capture
            if not self.baseline_noise_captured and self.noise_start_time:
                elapsed = time.time() - self.noise_start_time
                if elapsed <= self.baseline_noise_duration:
                    self.noise_buffer.append(sensor_entry)
                else:
                    self.baseline_noise_captured = True
                    if len(self.noise_buffer) > 0:
                        baseline_file = self.output_dir / f"baseline_noise_{int(time.time())}.csv"
                        self._save_csv(baseline_file, self.noise_buffer)

            # Continue collecting noise in default state
            elif self.active_recording is None and not self.skip_noise:
                self.noise_buffer.append(sensor_entry)

    def handle_label_event(self, msg, addr):
        """Handle label start/end events from button app"""
        action = msg.get('action')
        event = msg.get('event')
        timestamp = msg.get('timestamp_ms')

        self.last_phone_data = time.time()
        self.label_rate_window.append(time.time())

        if event == 'ping':
            return

        if event == 'start':
            with self.lock:
                if self.active_recording:
                    return

                self.active_recording = {
                    'action': action,
                    'start_time': timestamp,
                    'start_buffer_snapshot': list(self.sensor_buffer)  # Snapshot current buffer
                }

        elif event == 'end':
            # Check recording state with lock
            with self.lock:
                if not self.active_recording or self.active_recording['action'] != action:
                    return

                # Copy recording info before releasing lock
                recording_info = {
                    'action': action,
                    'start_time': self.active_recording['start_time'],
                    'end_time': timestamp,
                    'count': msg.get('count', 0)
                }

                # Clear active recording immediately so UI updates
                self.active_recording = None

            # Save recording WITHOUT holding the lock (this takes time)
            self.save_recording(
                action=recording_info['action'],
                start_time=recording_info['start_time'],
                end_time=recording_info['end_time'],
                count=recording_info['count']
            )

            # Update stats
            self.action_counts[action] = self.action_counts.get(action, 0) + 1
            self.total_recordings += 1

    def save_recording(self, action, start_time, end_time, count):
        """Save recording to CSV file"""
        filename = f"{action}_{start_time}_to_{end_time}.csv"
        filepath = self.output_dir / filename

        # Calculate gesture duration in seconds
        duration_ms = end_time - start_time
        duration_sec = duration_ms / 1000.0

        # Save buffer data with some padding (0.5s before + duration + 0.5s after)
        # At 50Hz, each second = ~50 samples
        padding_samples = 25  # 0.5 seconds at 50Hz
        gesture_samples = int(duration_sec * 50)
        total_samples = gesture_samples + (2 * padding_samples)

        # Copy most recent N samples from buffer
        with self.lock:
            # Take the last N samples (most recent data)
            buffer_list = list(self.sensor_buffer)
            if len(buffer_list) >= total_samples:
                recording_data = buffer_list[-total_samples:]
            else:
                recording_data = buffer_list

        # Write to file WITHOUT lock (this is the slow part)
        self._save_csv(filepath, recording_data)
        return filename

    def _save_csv(self, filepath, data):
        """Save data to CSV file"""
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'accel_x', 'accel_y', 'accel_z',
                'gyro_x', 'gyro_y', 'gyro_z',
                'rot_w', 'rot_x', 'rot_y', 'rot_z',
                'sensor', 'timestamp'
            ])

            for entry in data:
                writer.writerow([
                    entry['accel_x'], entry['accel_y'], entry['accel_z'],
                    entry['gyro_x'], entry['gyro_y'], entry['gyro_z'],
                    entry['rot_w'], entry['rot_x'], entry['rot_y'], entry['rot_z'],
                    entry['sensor'],
                    entry['timestamp']
                ])

    def segment_and_save_noise(self):
        """Segment and save noise data"""
        import random

        if not self.noise_buffer:
            return

        # Segment for classifiers
        locomotion_segments = self._segment_noise(self.noise_buffer, duration_sec=5.0, samples_per_sec=50)
        action_segments = self._segment_noise(self.noise_buffer, duration_sec=1.0, samples_per_sec=50)

        # Select 30 samples each
        selected_locomotion = random.sample(locomotion_segments, min(30, len(locomotion_segments)))
        selected_action = random.sample(action_segments, min(30, len(action_segments)))

        # Save
        for i, segment in enumerate(selected_locomotion, 1):
            filename = self.output_dir / f"noise_locomotion_seg_{i:03d}.csv"
            self._save_csv(filename, segment)

        for i, segment in enumerate(selected_action, 1):
            filename = self.output_dir / f"noise_action_seg_{i:03d}.csv"
            self._save_csv(filename, segment)

        self.action_counts['noise'] = len(selected_locomotion) + len(selected_action)

    def _segment_noise(self, noise_data, duration_sec, samples_per_sec):
        """Segment noise data into fixed-duration chunks"""
        segment_size = int(duration_sec * samples_per_sec)
        segments = []

        for i in range(0, len(noise_data) - segment_size + 1, segment_size):
            segment = noise_data[i:i + segment_size]
            if len(segment) == segment_size:
                segments.append(segment)

        return segments

    def print_statistics(self):
        """Print final statistics"""
        duration = datetime.now() - self.session_start

        print("\n" + "="*80)
        print("📈 SESSION STATISTICS")
        print("="*80)
        print(f"Duration: {duration}")
        print(f"Total recordings: {self.total_recordings}")
        print(f"\nAction breakdown:")
        for action, count in sorted(self.action_counts.items()):
            print(f"  {action.upper():12s}: {count:3d} samples")

        print(f"\n💾 Files saved to: {self.output_dir.absolute()}")
        print("="*80)


def main():
    """Main entry point"""
    skip_noise = '--skip-noise' in sys.argv

    collector = DataCollectionDashboard(skip_noise=skip_noise)

    print("""
╔═══════════════════════════════════════════════════════════╗
║       Data Collection Dashboard - Real-time Verification  ║
╚═══════════════════════════════════════════════════════════╝

This dashboard provides live visualization of:
- Connection status for Watch and Phone apps
- Real-time sensor data values
- Data rate metrics
- Recording status

Instructions:
1. Make sure Pixel Watch app is running and streaming
2. Launch Android button grid app on phone
3. Configure phone app with this computer's IP address
4. Watch the dashboard for connection confirmation
5. Press and hold buttons while performing gestures

Press Ctrl+C to stop and see statistics.
""")

    if skip_noise:
        print("⚡ SKIP NOISE MODE ENABLED\n")

    try:
        collector.start()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
