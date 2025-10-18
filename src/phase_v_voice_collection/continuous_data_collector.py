"""
Continuous Data Collector for Phase V CNN/LSTM Training

This tool records continuous sensor streams from the smartwatch alongside audio recording.
Users speak gesture commands during recording, which are transcribed post-recording using
Whisper and then aligned with sensor data.

Features:
- Continuous sensor data recording
- Simultaneous audio recording for voice commands
- Live visualization of recording status
- Auto-save with timestamps
- Post-processing script to align voice commands with sensor data

Usage:
    # Record with custom session name (recommended)
    python continuous_data_collector.py --duration 600 --session game_01
    # Creates: data/continuous/20251017_143022_game_01/

    # Record with auto-generated name
    python continuous_data_collector.py --duration 600
    # Creates: data/continuous/20251017_143022_session/

    During recording, speak commands naturally:
    - Say "jump" when performing jump gesture
    - Say "punch" when performing punch gesture
    - Say "turn" when performing turn gesture
    - Say "noise" for noise/unintentional movements
    - Say "walk start" at the beginning to start walking
    - Say "walk" periodically during walking segments
    - Default state between commands = Walk

    Output structure (session-based):
    20251017_143022_game_01/
    ├── audio.wav         # 44.1kHz audio (natural sound)
    ├── audio_16k.wav     # 16kHz audio (for Whisper)
    ├── sensor_data.csv   # IMU sensor data
    ├── metadata.json     # Session info
    └── README.md         # Session documentation

    After recording:
    - Each session gets its own directory with timestamp prefix
    - Run Whisper on audio_16k.wav (optimized for transcription)
    - Use post-processing script to align commands with sensor data
"""

import socket
import json
import time
import csv
import os
import argparse
import threading
from datetime import datetime
from collections import deque
import sys
# Add shared_utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared_utils'))
import network_utils
import sounddevice as sd
import numpy as np
import wave

# ==================== CONFIGURATION ====================

# Sensor types to collect
SENSORS_TO_COLLECT = [
    "rotation_vector",  # Composite orientation sensor
    "linear_acceleration",  # Linear acceleration (gravity removed)
    "gyroscope",  # Angular velocity
]

# Gesture durations (seconds)
GESTURE_DURATIONS = {
    "jump": 0.3,
    "punch": 0.3,
    "turn": 0.5,
    "noise": 1.0,
    "walk": None,  # Default state, no fixed duration
}

# Default recording duration
DEFAULT_DURATION_SEC = 600  # 10 minutes


# ANSI Color codes
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


# ==================== DATA STRUCTURES ====================


class GestureLabel:
    """Represents a labeled gesture in the recording"""

    def __init__(self, timestamp, gesture, duration):
        self.timestamp = timestamp
        self.gesture = gesture
        self.duration = duration


class ContinuousDataCollector:
    """Manages continuous sensor data recording with real-time labeling"""

    def __init__(self, session_name=None, duration_sec=DEFAULT_DURATION_SEC):
        # Generate timestamp prefix for session organization
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Session name with timestamp prefix if user provided name
        if session_name:
            self.session_name = f"{self.timestamp}_{session_name}"
        else:
            self.session_name = f"{self.timestamp}_session"

        self.duration_sec = duration_sec

        # Create session-specific subdirectory
        self.session_dir = os.path.join("data/continuous", self.session_name)
        self.output_dir = self.session_dir

        # Recording state
        self.recording = False
        self.start_time = None
        self.sensor_data = []

        # Network
        self.sock = None
        self.config = None

        # Threading
        self.stop_event = threading.Event()

        # Audio recording
        self.audio_data = []
        self.audio_sample_rate = 44100  # 44.1kHz for quality audio (CD quality)
        self.whisper_sample_rate = 16000  # Downsample to 16kHz for Whisper later
        self.audio_file = None

    def load_config(self):
        """Load configuration from config.json"""
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"{Colors.RED}ERROR: config.json not found!{Colors.RESET}")
            sys.exit(1)

    def setup(self):
        """Set up network connection and output directory"""
        # Auto-detect and update IP
        print(f"{Colors.CYAN}🔍 Auto-detecting IP address...{Colors.RESET}")
        network_utils.update_config_ip()
        self.config = self.load_config()

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"{Colors.GREEN}✓ Output directory: {self.output_dir}{Colors.RESET}")

        # Set up UDP socket
        listen_ip = self.config["network"]["listen_ip"]
        listen_port = self.config["network"]["listen_port"]

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind((listen_ip, listen_port))
            self.sock.setblocking(False)
            print(
                f"{Colors.GREEN}✓ Listening on {listen_ip}:{listen_port}{Colors.RESET}"
            )
        except OSError as e:
            print(
                f"{Colors.RED}ERROR: Could not bind to {listen_ip}:{listen_port}{Colors.RESET}"
            )
            print(
                f"{Colors.YELLOW}Make sure no other listener is running!{Colors.RESET}"
            )
            raise e

        return True

    def check_connection(self, timeout=15):
        """
        Verify watch is sending data before starting recording

        Args:
            timeout: Max seconds to wait for data

        Returns:
            bool: True if connection verified
        """
        print(f"\n{Colors.YELLOW}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.YELLOW}CONNECTION CHECK{Colors.RESET}")
        print(f"{Colors.YELLOW}{'='*70}{Colors.RESET}")
        print(f"{Colors.BLUE}Checking if watch is sending data...{Colors.RESET}")
        print(
            f"{Colors.YELLOW}Please make sure streaming is ON in your watch app.{Colors.RESET}\n"
        )

        start_time = time.time()
        last_update = 0
        packet_count = 0  # Track packets received

        while time.time() - start_time < timeout:
            elapsed = time.time() - start_time
            remaining = timeout - elapsed

            # Update status every 0.5 seconds
            if time.time() - last_update > 0.5:
                status = f"⏳ Waiting for data... {remaining:.1f}s remaining"
                if packet_count > 0:
                    status += f" | {Colors.GREEN}{packet_count} packets received{Colors.RESET}"
                print(f"\r{Colors.YELLOW}{status}{Colors.RESET}", end="", flush=True)
                last_update = time.time()

            try:
                data, _ = self.sock.recvfrom(4096)
                parsed = json.loads(data.decode())

                # Count ANY valid sensor packet
                if "sensor" in parsed and parsed["sensor"] in SENSORS_TO_COLLECT:
                    packet_count += 1

                    # SUCCESS after receiving 3+ packets (more reliable)
                    if packet_count >= 3:
                        print()  # New line
                        print(f"\n{Colors.GREEN}{'='*70}{Colors.RESET}")
                        print(f"{Colors.GREEN}✓ Connection verified!{Colors.RESET}")
                        print(
                            f"{Colors.GREEN}  Received {packet_count} valid packets from watch{Colors.RESET}"
                        )
                        print(f"{Colors.GREEN}{'='*70}{Colors.RESET}\n")
                        self.last_data_time = time.time()
                        # Flush remaining packets in buffer
                        self._flush_socket()
                        return True

            except (BlockingIOError, json.JSONDecodeError, KeyError):
                time.sleep(0.05)  # Reduced from 0.1s for faster response
            except Exception as e:
                print(f"\n{Colors.RED}Unexpected error: {e}{Colors.RESET}")

        # Timeout reached
        print()  # New line
        print(f"\n{Colors.RED}{'='*70}{Colors.RESET}")
        print(f"{Colors.RED}❌ CONNECTION TIMEOUT{Colors.RESET}")

        if packet_count > 0:
            print(
                f"{Colors.YELLOW}⚠️  Received {packet_count} packets but stream seems intermittent{Colors.RESET}"
            )
            print(
                f"{Colors.YELLOW}   This might work - trying to continue...{Colors.RESET}"
            )
            print(f"{Colors.RED}{'='*70}{Colors.RESET}\n")
            self._flush_socket()
            return True  # Allow continuation with partial connection
        else:
            print(f"{Colors.RED}   No sensor data received from watch!{Colors.RESET}")
            print(f"{Colors.YELLOW}   Please ensure:{Colors.RESET}")
            print(
                f"{Colors.BLUE}   1. Watch app is open and streaming is ON{Colors.RESET}"
            )
            print(
                f"{Colors.BLUE}   2. Watch shows 'Connected!' status (green){Colors.RESET}"
            )
            print(f"{Colors.BLUE}   3. Both devices on same WiFi network{Colors.RESET}")
            print(
                f"{Colors.BLUE}   4. Server IP matches: {self.config['network']['listen_ip']}{Colors.RESET}"
            )
            print(f"{Colors.RED}{'='*70}{Colors.RESET}\n")
            return False

    def _flush_socket(self):
        """Clear any pending packets in socket buffer"""
        try:
            while True:
                self.sock.recvfrom(4096)
        except BlockingIOError:
            pass  # Buffer is empty

    def display_instructions(self):
        """Display recording instructions"""
        instructions = f"""
╔══════════════════════════════════════════════════════════════╗
║  {Colors.BOLD}{Colors.GREEN}CONTINUOUS GESTURE RECORDER - Phase V{Colors.RESET}               ║
╚══════════════════════════════════════════════════════════════╝

{Colors.BOLD}Session:{Colors.RESET} {self.session_name}
{Colors.BOLD}Duration:{Colors.RESET} {self.duration_sec} seconds ({self.duration_sec/60:.1f} minutes)

{Colors.BOLD}{Colors.YELLOW}VOICE COMMANDS (speak naturally):{Colors.RESET}
  Say {Colors.GREEN}"jump"{Colors.RESET} when performing jump gesture
  Say {Colors.GREEN}"punch"{Colors.RESET} when performing punch gesture
  Say {Colors.GREEN}"turn"{Colors.RESET} when performing turn gesture
  Say {Colors.GREEN}"noise"{Colors.RESET} for unintentional movements
  Say {Colors.GREEN}"walk start"{Colors.RESET} at the beginning
  Say {Colors.GREEN}"walk"{Colors.RESET} during walking segments

  {Colors.BLUE}Default state = WALK{Colors.RESET}

{Colors.BOLD}{Colors.BLUE}HOW TO USE:{Colors.RESET}
1. Start recording - sensor data and audio recording begin
2. Speak "walk start" and begin moving naturally
3. Speak gesture commands RIGHT WHEN you perform each gesture
4. Continue speaking naturally as you would during gameplay
5. Press Ctrl+C to stop recording early (or let it run full duration)

{Colors.BOLD}{Colors.YELLOW}AFTER RECORDING:{Colors.RESET}
1. Run Whisper on the 16kHz file: {self.session_name}_16k.wav
   (The regular .wav is 44.1kHz for quality playback)
2. Use post-processing script to align voice commands with sensor data
3. Review and validate the generated labels

{Colors.BOLD}{Colors.YELLOW}TIPS FOR NATURAL DATA:{Colors.RESET}
- Play Hollow Knight: Silksong while recording
- Speak commands as you react to gameplay
- Move naturally - match your gestures to game actions
- Don't overthink timing - be natural and reactive
- Variety in speed and intensity is good!

{Colors.BOLD}{Colors.RED}IMPORTANT:{Colors.RESET}
- Microphone should be working and accessible
- Watch should be streaming sensor data
- Audio and sensor data saved automatically at end

──────────────────────────────────────────────────────────────
"""
        print(instructions)

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream - captures audio chunks"""
        if status:
            print(f"\n{Colors.YELLOW}Audio status: {status}{Colors.RESET}")
        # Store audio data
        self.audio_data.append(indata.copy())

    def record_sensor_data(self):
        """Main recording loop for sensor data and audio"""
        print(f"\n{Colors.RED}{Colors.BOLD}🔴 RECORDING STARTED{Colors.RESET}")
        print(
            f"{Colors.YELLOW}🎤 Audio recording active - speak commands naturally!{Colors.RESET}\n"
        )

        self.recording = True
        self.start_time = time.time()
        recording_end = self.start_time + self.duration_sec

        last_status_update = 0
        data_points = 0
        last_data_time = time.time()

        # Start audio recording in separate thread
        audio_stream = sd.InputStream(
            samplerate=self.audio_sample_rate,
            channels=1,
            callback=self.audio_callback,
            dtype="float32",
        )
        audio_stream.start()

        while time.time() < recording_end and not self.stop_event.is_set():
            elapsed = time.time() - self.start_time
            remaining = recording_end - time.time()
            progress_pct = (elapsed / self.duration_sec) * 100

            # Update status display
            if time.time() - last_status_update > 1.0:
                self._display_status(elapsed, remaining, progress_pct, data_points)
                last_status_update = time.time()

            # Receive sensor data
            try:
                data, addr = self.sock.recvfrom(4096)
                parsed = json.loads(data.decode())
                sensor_type = parsed.get("sensor")

                if sensor_type in SENSORS_TO_COLLECT:
                    last_data_time = time.time()

                    # Create record
                    record = {
                        "timestamp": time.time() - self.start_time,
                        "sensor": sensor_type,
                    }

                    # Flatten sensor values
                    if "values" in parsed:
                        vals = parsed["values"]
                        if sensor_type == "rotation_vector":
                            record["rot_x"] = vals.get("x", 0)
                            record["rot_y"] = vals.get("y", 0)
                            record["rot_z"] = vals.get("z", 0)
                            record["rot_w"] = vals.get("w", 0)
                        elif sensor_type == "linear_acceleration":
                            record["accel_x"] = vals.get("x", 0)
                            record["accel_y"] = vals.get("y", 0)
                            record["accel_z"] = vals.get("z", 0)
                        elif sensor_type == "gyroscope":
                            record["gyro_x"] = vals.get("x", 0)
                            record["gyro_y"] = vals.get("y", 0)
                            record["gyro_z"] = vals.get("z", 0)

                    self.sensor_data.append(record)
                    data_points += 1

            except (BlockingIOError, json.JSONDecodeError, KeyError):
                pass

            # Check connection
            if time.time() - last_data_time > 2.0:
                print(
                    f"\r{Colors.RED}⚠️  WARNING: No data received for 2 seconds! Check watch connection.{Colors.RESET}"
                )

        self.recording = False
        audio_stream.stop()
        audio_stream.close()

        print(f"\n\n{Colors.GREEN}✓ Recording complete!{Colors.RESET}")
        print(
            f"{Colors.BLUE}Captured {len(self.sensor_data)} sensor data points{Colors.RESET}"
        )
        print(
            f"{Colors.BLUE}Recorded {len(self.audio_data)} audio chunks{Colors.RESET}"
        )

    def _display_status(self, elapsed, remaining, progress_pct, data_points):
        """Display recording status"""
        # Progress bar
        bar_width = 30
        filled = int(bar_width * progress_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Format time
        elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        total_str = f"{int(self.duration_sec//60):02d}:{int(self.duration_sec%60):02d}"

        # Data rate
        data_rate = data_points / elapsed if elapsed > 0 else 0

        # Audio chunks
        audio_chunks = len(self.audio_data)

        # Display
        status = f"\r⏱️  {elapsed_str}/{total_str} [{bar}] {progress_pct:.0f}% | "
        status += f"📊 {data_points} pts ({data_rate:.0f} pts/s) | "
        status += f"🎤 {audio_chunks} audio chunks"

        print(status, end="", flush=True)

    def save_data(self):
        """Save sensor data and audio to files"""
        print(f"\n{Colors.CYAN}💾 Saving data...{Colors.RESET}")

        # Ensure session directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Use simpler filenames inside the session directory
        # (timestamp is already in the directory name)
        sensor_file = os.path.join(self.output_dir, "sensor_data.csv")
        if self.sensor_data:
            fieldnames = set()
            for record in self.sensor_data:
                fieldnames.update(record.keys())
            fieldnames = sorted(list(fieldnames))

            with open(sensor_file, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.sensor_data)

            print(f"{Colors.GREEN}✓ Sensor data saved: {sensor_file}{Colors.RESET}")

        # Save audio data as WAV file
        # Use clean filenames (timestamp is in directory name)
        audio_file = os.path.join(self.output_dir, "audio.wav")
        audio_file_whisper = os.path.join(self.output_dir, "audio_16k.wav")

        if self.audio_data:
            # Concatenate all audio chunks
            audio_array = np.concatenate(self.audio_data, axis=0)

            # Save high-quality version (44.1kHz) - sounds natural
            with wave.open(audio_file, "wb") as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.audio_sample_rate)
                # Convert float32 to int16
                audio_int16 = (audio_array * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())

            print(f"{Colors.GREEN}✓ Audio saved (44.1kHz): {audio_file}{Colors.RESET}")

            # Downsample to 16kHz for Whisper transcription
            try:
                from scipy import signal

                # Calculate exact number of samples needed to maintain duration
                # Duration = num_samples / sample_rate
                # So: num_samples_16k = duration * 16000
                original_duration = len(audio_array) / self.audio_sample_rate
                target_num_samples = int(original_duration * self.whisper_sample_rate)

                # Use signal.resample for exact duration preservation
                audio_downsampled = signal.resample(
                    audio_array.flatten(), target_num_samples
                )

                # Save 16kHz version for Whisper
                with wave.open(audio_file_whisper, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.whisper_sample_rate)
                    audio_int16_whisper = (audio_downsampled * 32767).astype(np.int16)
                    wf.writeframes(audio_int16_whisper.tobytes())

                print(
                    f"{Colors.GREEN}✓ Audio for Whisper (16kHz): {audio_file_whisper}{Colors.RESET}"
                )
                print(
                    f"{Colors.CYAN}💡 Use *_16k.wav file for Whisper transcription{Colors.RESET}"
                )

            except ImportError:
                print(
                    f"{Colors.YELLOW}⚠️  scipy not installed - skipping 16kHz version{Colors.RESET}"
                )
                print(f"{Colors.YELLOW}   Install: pip install scipy{Colors.RESET}")
                print(
                    f"{Colors.YELLOW}   Or use ffmpeg: ffmpeg -i {audio_file} -ar 16000 {audio_file_whisper}{Colors.RESET}"
                )

        # Save session metadata
        metadata_file = os.path.join(self.output_dir, "metadata.json")
        actual_duration = self.sensor_data[-1]["timestamp"] if self.sensor_data else 0
        audio_duration = (
            len(np.concatenate(self.audio_data)) / self.audio_sample_rate
            if self.audio_data
            else 0
        )

        metadata = {
            "session_name": self.session_name,
            "recording_date": datetime.now().isoformat(),
            "duration_sec": self.duration_sec,
            "actual_duration_sec": actual_duration,
            "audio_duration_sec": audio_duration,
            "sensor_data_points": len(self.sensor_data),
            "audio_sample_rate": self.audio_sample_rate,
            "audio_chunks": len(self.audio_data),
            "sensors_collected": SENSORS_TO_COLLECT,
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"{Colors.GREEN}✓ Metadata saved: {metadata_file}{Colors.RESET}")

        # Create README for the session
        readme_file = os.path.join(self.output_dir, "README.md")
        readme_content = f"""# Session: {self.session_name}

## Recording Information
- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Duration:** {actual_duration:.1f}s ({actual_duration/60:.1f} minutes)
- **Sensor Data Points:** {len(self.sensor_data)}
- **Audio Duration:** {audio_duration:.1f}s

## Files in This Session

| File | Purpose |
|------|---------|
| `audio.wav` | High-quality audio (44.1kHz) - for playback/review |
| `audio_16k.wav` | Downsampled audio (16kHz) - for Whisper transcription |
| `sensor_data.csv` | Raw sensor data (accelerometer, gyroscope, rotation) |
| `metadata.json` | Session metadata and configuration |
| `README.md` | This file |

## Next Steps

### 1. Transcribe Audio with Whisper
```bash
# Use the 16kHz version for best Whisper accuracy
whisper audio_16k.wav --model large-v3-turbo --word_timestamps True --output_format json
```

### 2. Align Voice Commands with Sensor Data
```bash
cd ../../src
python align_voice_labels.py --session {self.session_name} --whisper ../{self.output_dir}/audio_16k.json
```

### 3. Review Generated Labels
Check the generated `labels.csv` file for accuracy.

## Session Directory Structure
```
{self.session_name}/
├── audio.wav          # 44.1kHz audio (natural sound)
├── audio_16k.wav      # 16kHz audio (for Whisper)
├── sensor_data.csv    # IMU sensor data
├── metadata.json      # Session info
└── README.md          # This documentation
```

---
Generated by continuous_data_collector.py
"""
        with open(readme_file, "w") as f:
            f.write(readme_content)

        print(f"{Colors.GREEN}✓ README saved: {readme_file}{Colors.RESET}")

        # Display summary
        print(f"\n{Colors.BOLD}📊 Recording Summary:{Colors.RESET}")
        print(f"  {Colors.BLUE}• Session: {self.session_name}{Colors.RESET}")
        print(f"  {Colors.BLUE}• Sensor duration: {actual_duration:.1f}s{Colors.RESET}")
        print(f"  {Colors.BLUE}• Audio duration: {audio_duration:.1f}s{Colors.RESET}")
        print(
            f"  {Colors.BLUE}• Sensor data points: {len(self.sensor_data)}{Colors.RESET}"
        )
        print(f"  {Colors.BLUE}• Audio chunks: {len(self.audio_data)}{Colors.RESET}")

        print(f"\n{Colors.BOLD}🎤 Next Steps:{Colors.RESET}")
        print(f"  1. Run Whisper on: {self.output_dir}/audio_16k.wav")
        print(f"  2. Use post-processing script to align voice with sensor data")
        print(f"  3. Review generated labels")
        print(f"\n{Colors.CYAN}📁 All files saved to: {self.output_dir}/{Colors.RESET}")

    def cleanup(self):
        """Clean up resources"""
        if self.sock:
            self.sock.close()


# ==================== MAIN ====================


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Continuous gesture data collector with audio recording for Phase V CNN/LSTM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record 10-minute session
  python continuous_data_collector.py --duration 600

  # Record 5-minute session with custom name
  python continuous_data_collector.py --duration 300 --session my_session_01

During recording (speak naturally):
  "jump" = Jump gesture
  "punch" = Punch gesture
  "turn" = Turn gesture
  "noise" = Unintentional movement
  "walk start" = Begin walking
  "walk" = Walking state
        """,
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION_SEC,
        help=f"Recording duration in seconds (default: {DEFAULT_DURATION_SEC})",
    )

    parser.add_argument(
        "--session", type=str, help="Session name (default: auto-generated timestamp)"
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()

    # Create collector
    collector = ContinuousDataCollector(
        session_name=args.session, duration_sec=args.duration
    )

    try:
        # Setup
        collector.setup()
        collector.display_instructions()

        input(
            f"\n{Colors.BOLD}{Colors.GREEN}Press [Enter] to start recording...{Colors.RESET}"
        )

        print(f"{Colors.YELLOW}📱 Checking watch connection...{Colors.RESET}")

        # Check connection before starting
        if not collector.check_connection(timeout=15):
            print(
                f"\n{Colors.RED}❌ Cannot start recording - no connection to watch{Colors.RESET}"
            )
            print(f"{Colors.YELLOW}� Troubleshooting steps:{Colors.RESET}")
            print(f"  1. Open the watch app")
            print(f"  2. Toggle streaming ON")
            print(f"  3. Verify 'Connected!' message appears in green")
            print(f"  4. Try running this script again")
            return

        print(
            f"{Colors.YELLOW}🎤 Microphone will start recording in 3 seconds...{Colors.RESET}"
        )
        print(
            f"{Colors.GREEN}✓ Watch connection verified - ready to record!{Colors.RESET}"
        )
        time.sleep(3)

        # Record
        collector.record_sensor_data()

        # Save
        collector.save_data()

        print(f"\n{Colors.GREEN}✨ Data collection complete!{Colors.RESET}")
        print(f"{Colors.CYAN}Files saved to: {collector.output_dir}/{Colors.RESET}")

    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}⚠️  Recording interrupted by user{Colors.RESET}")
        if collector.sensor_data:
            save = input(
                f"{Colors.YELLOW}Save partial recording? (y/n): {Colors.RESET}"
            )
            if save.lower() == "y":
                collector.save_data()

    except Exception as e:
        print(f"\n{Colors.RED}❌ ERROR: {e}{Colors.RESET}")
        import traceback

        traceback.print_exc()

    finally:
        collector.cleanup()
        print(f"\n{Colors.GREEN}✓ Cleanup complete{Colors.RESET}")


if __name__ == "__main__":
    main()
