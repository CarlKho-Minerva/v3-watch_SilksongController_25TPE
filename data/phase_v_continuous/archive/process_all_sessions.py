#!/usr/bin/env python3
"""
Process All Sessions - Generate Labels from WhisperX Transcriptions

This script automatically processes all continuous data sessions,
extracting gesture labels from WhisperX transcription files.

Usage:
    python process_all_sessions.py

Output:
    - [session_name]_labels.csv for each session
    - Summary report of extracted gestures
"""

import os
import json
import sys
import csv
from pathlib import Path

# Add src to path
sys.path.append('src')
from align_voice_labels import extract_gesture_commands, generate_complete_labels

# Configuration
CONTINUOUS_DATA_DIR = 'src/data/continuous'

def get_audio_duration(session_dir):
    """Get audio duration from metadata.json"""
    metadata_path = os.path.join(session_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            return metadata.get('duration_seconds', 0)
    return 0

def process_session(session_dir):
    """Process a single session directory"""
    session_name = os.path.basename(session_dir)

    # Find WhisperX output file (support multiple naming patterns)
    whisperx_files = [f for f in os.listdir(session_dir)
                      if ('whisperx_output.json' in f or
                          (f.endswith('.json') and session_name in f and
                           f != 'metadata.json'))]

    if not whisperx_files:
        return None, "No WhisperX output file found"

    whisperx_file = os.path.join(session_dir, whisperx_files[0])    # Load WhisperX data
    print(f"\n📂 Processing: {session_name}")
    print(f"   WhisperX file: {whisperx_files[0]}")

    with open(whisperx_file, 'r') as f:
        whisperx_data = json.load(f)

    # Extract gesture commands
    commands = extract_gesture_commands(whisperx_data)
    print(f"   Extracted {len(commands)} gesture commands")

    # Get audio duration
    duration = get_audio_duration(session_dir)
    if duration == 0:
        # Estimate from last word timestamp
        if whisperx_data.get('segments'):
            last_segment = whisperx_data['segments'][-1]
            if last_segment.get('words'):
                duration = last_segment['words'][-1]['end'] + 1.0

    print(f"   Audio duration: {duration:.1f} seconds")

    # Generate complete labels
    labels = generate_complete_labels(commands, duration)
    print(f"   Generated {len(labels)} label segments")

    # Count gestures
    gesture_counts = {}
    for label in labels:
        gesture = label['gesture']
        gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1

    print(f"   Gesture breakdown:")
    for gesture, count in sorted(gesture_counts.items()):
        print(f"     - {gesture}: {count} segments")

    # Save labels to CSV
    labels_file = os.path.join(session_dir, f"{session_name}_labels.csv")
    with open(labels_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'gesture', 'duration'])
        writer.writeheader()
        writer.writerows(labels)

    print(f"   ✅ Saved: {labels_file}")

    return labels, None

def main():
    """Process all sessions in continuous data directory"""
    print("=" * 60)
    print("PROCESSING ALL CONTINUOUS DATA SESSIONS")
    print("=" * 60)

    if not os.path.exists(CONTINUOUS_DATA_DIR):
        print(f"❌ Error: Directory not found: {CONTINUOUS_DATA_DIR}")
        return

    # Find all session directories
    session_dirs = []
    for item in os.listdir(CONTINUOUS_DATA_DIR):
        item_path = os.path.join(CONTINUOUS_DATA_DIR, item)
        if os.path.isdir(item_path) and '_session' in item:
            session_dirs.append(item_path)

    if not session_dirs:
        print(f"❌ No session directories found in {CONTINUOUS_DATA_DIR}")
        return

    print(f"\nFound {len(session_dirs)} session(s) to process\n")

    # Process each session
    results = []
    for session_dir in sorted(session_dirs):
        labels, error = process_session(session_dir)
        if error:
            print(f"   ⚠️  Error: {error}")
            results.append((session_dir, None, error))
        else:
            results.append((session_dir, labels, None))

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r[2] is None]
    failed = [r for r in results if r[2] is not None]

    print(f"\n✅ Successfully processed: {len(successful)}/{len(results)} sessions")

    if successful:
        print("\nReady for training:")
        for session_dir, labels, _ in successful:
            session_name = os.path.basename(session_dir)
            labels_file = f"{session_name}_labels.csv"
            print(f"  • {session_name}/sensor_data.csv")
            print(f"    + {labels_file}")

    if failed:
        print(f"\n⚠️  Failed: {len(failed)} sessions")
        for session_dir, _, error in failed:
            print(f"  • {os.path.basename(session_dir)}: {error}")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Review the generated *_labels.csv files to ensure quality

2. Set up Google Colab for training:
   - Go to: https://colab.research.google.com/
   - Upload Phase_V_Training.ipynb
   - Enable GPU: Runtime > Change runtime type > GPU

3. Upload your data to Google Drive:
   - Create folder: My Drive/silksong_data/
   - Upload all session folders with:
     • sensor_data.csv
     • [session]_labels.csv

4. Run the training notebook in Colab

5. Download the trained model (cnn_lstm_gesture.h5)

6. Test locally with: python src/udp_listener_v3.py
""")

if __name__ == '__main__':
    main()
