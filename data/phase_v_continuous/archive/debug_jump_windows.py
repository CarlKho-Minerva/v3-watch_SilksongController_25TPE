#!/usr/bin/env python3
"""
Debug script to understand why we only get 7 jump windows
when we have 244 jump labels
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Config
DATA_DIR = Path("src/data/continuous")
SESSION_FOLDERS = [
    '20251017_125600_session',
    '20251017_135458_session',
    '20251017_141539_session',
    '20251017_143217_session',
    '20251017_143627_session',
]
WINDOW_SIZE = 50  # 1 second at 50Hz
STRIDE = 25       # 0.5 seconds
SAMPLE_RATE = 50.0

GESTURES = ['jump', 'punch', 'turn', 'walk', 'noise']

def analyze_jump_labels():
    total_jumps_in_csv = 0
    total_jump_duration = 0.0
    total_jump_windows = 0

    for session in SESSION_FOLDERS:
        labels_file = DATA_DIR / session / f"{session}_labels.csv"
        labels_data = pd.read_csv(labels_file)

        # Count jumps
        jump_labels = labels_data[labels_data['gesture'] == 'jump']
        num_jumps = len(jump_labels)
        total_jump_duration_session = jump_labels['duration'].sum()

        print(f"\n{session}:")
        print(f"  Jump labels in CSV: {num_jumps}")
        print(f"  Total jump duration: {total_jump_duration_session:.2f}s")
        print(f"  Average jump duration: {total_jump_duration_session/num_jumps:.2f}s")

        # Calculate how many samples this represents
        jump_samples = total_jump_duration_session * SAMPLE_RATE
        print(f"  Jump samples: {jump_samples:.0f} (out of {WINDOW_SIZE} needed per window)")

        # Estimate windows (rough)
        # A pure jump window needs all 50 samples to be jump
        # But with majority voting, we need at least 26 samples
        possible_pure_windows = int(jump_samples / WINDOW_SIZE)
        possible_majority_windows = int(jump_samples / 26)

        print(f"  Possible pure jump windows: {possible_pure_windows}")
        print(f"  Possible majority-vote windows: {possible_majority_windows}")

        total_jumps_in_csv += num_jumps
        total_jump_duration += total_jump_duration_session
        total_jump_windows += possible_pure_windows

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total jump labels across all sessions: {total_jumps_in_csv}")
    print(f"Total jump duration: {total_jump_duration:.2f}s")
    print(f"Total jump samples: {total_jump_duration * SAMPLE_RATE:.0f}")
    print(f"\nWindow requirements:")
    print(f"  Window size: {WINDOW_SIZE} samples (1.0s)")
    print(f"  Stride: {STRIDE} samples (0.5s)")
    print(f"\nWhy so few jump windows?")
    print(f"  1. Each jump label is only 0.3s = 15 samples")
    print(f"  2. Window needs 50 samples = 3.3 consecutive jumps")
    print(f"  3. Majority vote needs 26 samples = 1.7 consecutive jumps")
    print(f"\n  Problem: Jumps are too SHORT and too SPACED OUT!")
    print(f"  Solution: Need longer jump durations OR consecutive jumps")

    print("\n" + "="*70)
    print("WHAT TO DO")
    print("="*70)
    print("\nOption 1: REDUCE WINDOW SIZE (quick fix)")
    print("  Change WINDOW_SIZE from 50 to 25 (0.5s)")
    print("  This will capture more jump windows")
    print("  Expected result: ~50-100 jump windows instead of 7")

    print("\nOption 2: COLLECT BETTER DATA (best long-term)")
    print("  Record new session where you:")
    print("  - Say 'jump' and hold the gesture for 1-2 seconds")
    print("  - Do multiple jumps in a row without breaks")
    print("  - This creates longer continuous jump segments")

if __name__ == "__main__":
    analyze_jump_labels()
