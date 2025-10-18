#!/usr/bin/env python3
"""
Button Data Collector - Python Backend
Receives UDP label events from Android button grid app and saves labeled sensor data.

Usage:
    python button_data_collector.py

Requirements:
    - Watch app streaming sensor data on port 12345
    - Button app sending label events on port 12345
    - All devices on same WiFi network
"""

import json
import socket
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
import csv


class ButtonDataCollector:
    def __init__(self, udp_port=12345, output_dir="data/button_collected"):
        self.udp_port = udp_port
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Buffer for sensor data (keep last 30 seconds at 50Hz = 1500 samples)
        self.sensor_buffer = deque(maxlen=1500)
        
        # Currently recording action (None means NOISE mode - default state)
        self.active_recording = None
        
        # Noise capture
        self.noise_buffer = []
        self.baseline_noise_captured = False
        self.baseline_noise_duration = 30  # seconds
        self.noise_start_time = None
        
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
        
    def start(self):
        """Start the UDP listener"""
        print(f"🎯 Button Data Collector Started")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        print(f"🌐 Listening on UDP port {self.udp_port}")
        print(f"⏰ Session start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n🔇 DEFAULT STATE: NOISE MODE (all data labeled as noise unless button pressed)")
        print(f"📊 Capturing {self.baseline_noise_duration}s baseline noise at session start...\n")
        
        # Start baseline noise capture
        self.noise_start_time = time.time()
        
        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', self.udp_port))
        
        try:
            while True:
                data, addr = sock.recvfrom(4096)
                message = data.decode('utf-8')
                
                # Parse JSON message
                try:
                    msg = json.loads(message)
                    self.handle_message(msg, addr)
                except json.JSONDecodeError:
                    # Not JSON, probably sensor data - ignore for now
                    # In production, parse and buffer sensor data here
                    pass
                    
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping data collector...")
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
            # Sensor data - add to buffer and handle noise capture
            # For MVP, we're not processing sensor data yet
            # In production, parse and add to sensor_buffer
            
            # Capture baseline noise (first 30 seconds)
            if not self.baseline_noise_captured:
                elapsed = time.time() - self.noise_start_time
                if elapsed <= self.baseline_noise_duration:
                    # Still in baseline capture window
                    self.noise_buffer.append({
                        'timestamp': msg.get('timestamp_ns', time.time() * 1e9),
                        'data': msg
                    })
                else:
                    # Baseline complete
                    self.baseline_noise_captured = True
                    print(f"✅ Baseline noise captured ({len(self.noise_buffer)} samples)")
                    print("✋ Ready for button presses...\n")
            
            # If no button pressed (default NOISE state), continue collecting noise
            elif self.active_recording is None:
                self.noise_buffer.append({
                    'timestamp': msg.get('timestamp_ns', time.time() * 1e9),
                    'data': msg
                })
            
            pass
    
    def handle_label_event(self, msg, addr):
        """Handle label start/end events from button app"""
        action = msg.get('action')
        event = msg.get('event')
        timestamp = msg.get('timestamp_ms')
        
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
        
        # For MVP: Create empty CSV with header
        # In production, extract sensor data from buffer and save
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'sensor', 
                'accel_x', 'accel_y', 'accel_z',
                'gyro_x', 'gyro_y', 'gyro_z',
                'rot_x', 'rot_y', 'rot_z', 'rot_w'
            ])
            
            # TODO: Add actual sensor data from buffer
            # For now, just create placeholder
            writer.writerow([start_time, 'placeholder', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        return filename
    
    def segment_and_save_noise(self):
        """Segment noise buffer into fixed-size chunks and save exactly 30 per classifier"""
        import random
        
        if not self.noise_buffer:
            print("⚠️  No noise data collected")
            return
        
        print(f"\n🔇 Processing noise data ({len(self.noise_buffer)} samples)...")
        
        # Validate timestamp integrity
        valid_noise = []
        for sample in self.noise_buffer:
            if sample.get('timestamp') and sample.get('data'):
                valid_noise.append(sample)
        
        print(f"✅ Validated {len(valid_noise)} noise samples with timestamp integrity")
        
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
        """Save a noise segment to CSV"""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'sensor',
                'accel_x', 'accel_y', 'accel_z',
                'gyro_x', 'gyro_y', 'gyro_z',
                'rot_x', 'rot_y', 'rot_z', 'rot_w'
            ])
            
            # Write placeholder data (in production, use actual sensor data)
            for sample in segment:
                timestamp = sample.get('timestamp', 0)
                writer.writerow([timestamp, 'noise', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    def print_progress(self):
        """Print current collection progress"""
        print(f"\n📊 Progress: {self.total_recordings} total recordings")
        
        # Print counts in grid format
        print("   WALK: {walk:2d}  IDLE: {idle:2d}".format(**self.action_counts))
        print("  PUNCH: {punch:2d}  JUMP: {jump:2d}".format(**self.action_counts))
        print("  TURN_L: {turn_left:2d}  TURN_R: {turn_right:2d}".format(**self.action_counts))
        print("  NOISE: {noise:2d}\n".format(**self.action_counts))
    
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
    collector = ButtonDataCollector()
    
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

Press Ctrl+C to stop and see statistics.
""")
    
    try:
        collector.start()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
