"""
Post-Processing Script: Align Voice Commands with Sensor Data

This script takes Whisper or WhisperX transcription output (with word-level timestamps) and
aligns voice commands with sensor data to generate gesture labels.

Usage:
    # Using standard Whisper output
    python align_voice_labels.py --session session_20250101_120000 --whisper whisper_output.json

    # Using WhisperX output (recommended for research)
    python align_voice_labels.py --session session_20250101_120000 --whisper session_20250101_120000_whisperx.json

The Whisper output should be generated using word-level timestamps:
    whisper session_20250101_120000.wav --model large-v3-turbo --word_timestamps True --output_format json

For research-grade timestamps, use WhisperX:
    python whisperx_transcribe.py --audio session_20250101_120000.wav --model large-v3
"""

import json
import csv
import argparse
import os
from datetime import datetime

# Gesture keywords and their durations
GESTURE_KEYWORDS = {
    'jump': 0.3,
    'punch': 0.3,
    'turn': 0.5,
    'noise': 1.0,
    'idle': 2.0,   # Standing still, preparing for gesture
    'rest': 2.0,   # Alternative to idle
    'stop': 2.0,   # Another alternative
    'walk': None  # Will be filled in as default
}

# Keywords that indicate walking
WALK_KEYWORDS = ['walk', 'walking', 'start', 'moving']


def load_whisper_output(whisper_file):
    """Load Whisper transcription with word-level timestamps"""
    with open(whisper_file, 'r') as f:
        data = json.load(f)
    return data


def load_sensor_metadata(session_dir, session_name):
    """Load sensor metadata to get recording duration"""
    metadata_file = os.path.join(session_dir, f"{session_name}_metadata.json")
    with open(metadata_file, 'r') as f:
        return json.load(f)


def extract_gesture_commands(whisper_data):
    """Extract gesture commands from Whisper or WhisperX word-level timestamps

    Supports both standard Whisper format and WhisperX format with forced alignment.
    WhisperX format includes 'score' instead of 'probability' for confidence.

    Returns:
        List of (timestamp, gesture, duration) tuples
    """
    commands = []

    # Check if we have word-level timestamps
    if 'segments' in whisper_data:
        for segment in whisper_data['segments']:
            if 'words' in segment:
                for word_info in segment['words']:
                    word = word_info['word'].strip().lower()
                    timestamp = word_info['start']

                    # Get confidence score (different field names for Whisper vs WhisperX)
                    # WhisperX uses 'score', standard Whisper uses 'probability'
                    confidence = word_info.get('score', word_info.get('probability', 1.0))

                    # Check for gesture keywords (skip 'walk' - handled separately)
                    for gesture, duration in GESTURE_KEYWORDS.items():
                        if gesture == 'walk':
                            continue  # Skip walk, handled below
                        if gesture in word:
                            commands.append({
                                'timestamp': timestamp,
                                'gesture': gesture,
                                'duration': duration,
                                'confidence': confidence
                            })
                            break

                    # Check for walk keywords
                    if any(kw in word for kw in WALK_KEYWORDS):
                        # Mark explicit walk command
                        commands.append({
                            'timestamp': timestamp,
                            'gesture': 'walk',
                            'duration': 1.0,  # Explicit walk marker
                            'confidence': confidence
                        })

    return commands


def generate_complete_labels(commands, total_duration):
    """Generate complete label file with walk as default state

    Args:
        commands: List of gesture commands with timestamps
        total_duration: Total recording duration in seconds

    Returns:
        List of label dictionaries
    """
    # Sort commands by timestamp
    commands.sort(key=lambda x: x['timestamp'])

    labels = []
    current_time = 0.0

    for i, cmd in enumerate(commands):
        # Fill gap with walk
        if current_time < cmd['timestamp']:
            labels.append({
                'timestamp': current_time,
                'gesture': 'walk',
                'duration': cmd['timestamp'] - current_time
            })

        # Add the gesture command
        gesture_duration = cmd['duration']
        labels.append({
            'timestamp': cmd['timestamp'],
            'gesture': cmd['gesture'],
            'duration': gesture_duration
        })

        current_time = cmd['timestamp'] + gesture_duration

    # Fill remaining time with walk
    if current_time < total_duration:
        labels.append({
            'timestamp': current_time,
            'gesture': 'walk',
            'duration': total_duration - current_time
        })

    return labels


def save_labels(labels, output_file):
    """Save labels to CSV file"""
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'gesture', 'duration'])
        writer.writeheader()
        writer.writerows(labels)


def calculate_statistics(labels):
    """Calculate gesture statistics"""
    stats = {}
    for label in labels:
        gesture = label['gesture']
        duration = label['duration']

        if gesture not in stats:
            stats[gesture] = {'count': 0, 'total_duration': 0}

        stats[gesture]['count'] += 1
        stats[gesture]['total_duration'] += duration

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Align Whisper/WhisperX transcription with sensor data to generate gesture labels'
    )
    parser.add_argument('--session', required=True, help='Session name')
    parser.add_argument('--whisper', required=True,
                       help='Whisper or WhisperX JSON output file with word timestamps')
    parser.add_argument('--output-dir', default='data/continuous', help='Output directory')
    parser.add_argument('--min-confidence', type=float, default=0.5,
                       help='Minimum confidence for word detection (0.0-1.0, WhisperX uses score field)')

    args = parser.parse_args()

    print("=" * 70)
    print("Voice Command Label Alignment - Phase V")
    print("=" * 70)
    print()

    # Load Whisper output
    print(f"Loading Whisper transcription: {args.whisper}")
    whisper_data = load_whisper_output(args.whisper)
    print(f"✓ Loaded transcription")
    print()

    # Load sensor metadata
    print(f"Loading sensor metadata for session: {args.session}")
    metadata = load_sensor_metadata(args.output_dir, args.session)
    total_duration = metadata.get('actual_duration_sec', metadata.get('duration_sec', 600))
    print(f"✓ Total recording duration: {total_duration:.1f}s")
    print()

    # Extract gesture commands
    print("Extracting gesture commands from transcription...")
    commands = extract_gesture_commands(whisper_data)

    # Filter by confidence
    filtered_commands = [c for c in commands if c['confidence'] >= args.min_confidence]
    print(f"✓ Found {len(commands)} total commands")
    print(f"✓ {len(filtered_commands)} commands above confidence threshold {args.min_confidence}")
    print()

    # Display detected commands
    print("Detected commands:")
    for cmd in filtered_commands[:20]:  # Show first 20
        print(f"  {cmd['timestamp']:6.2f}s - {cmd['gesture']:8s} (conf: {cmd['confidence']:.2f})")
    if len(filtered_commands) > 20:
        print(f"  ... and {len(filtered_commands) - 20} more")
    print()

    # Generate complete labels
    print("Generating complete label file...")
    labels = generate_complete_labels(filtered_commands, total_duration)
    print(f"✓ Generated {len(labels)} label segments")
    print()

    # Calculate statistics
    stats = calculate_statistics(labels)
    print("Gesture Distribution:")
    for gesture, data in sorted(stats.items()):
        if gesture == 'walk':
            print(f"  {gesture:8s}: {data['total_duration']:7.1f}s ({data['total_duration']/total_duration*100:5.1f}%)")
        else:
            print(f"  {gesture:8s}: {data['count']:3d} events, {data['total_duration']:6.1f}s total")
    print()

    # Save labels
    output_file = os.path.join(args.output_dir, f"{args.session}_labels.csv")
    save_labels(labels, output_file)
    print(f"✓ Labels saved to: {output_file}")
    print()

    # Save alignment metadata
    alignment_metadata = {
        'session_name': args.session,
        'whisper_file': args.whisper,
        'alignment_date': datetime.now().isoformat(),
        'total_duration': total_duration,
        'commands_detected': len(filtered_commands),
        'min_confidence': args.min_confidence,
        'gesture_stats': stats
    }

    alignment_file = os.path.join(args.output_dir, f"{args.session}_alignment.json")
    with open(alignment_file, 'w') as f:
        json.dump(alignment_metadata, f, indent=2)
    print(f"✓ Alignment metadata saved to: {alignment_file}")
    print()

    print("=" * 70)
    print("✅ Label alignment complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Review the generated labels CSV")
    print("2. Validate alignment using visualization tool")
    print("3. Use labeled data for training")


if __name__ == "__main__":
    main()
