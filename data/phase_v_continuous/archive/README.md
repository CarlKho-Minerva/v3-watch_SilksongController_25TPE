# Data Archive

This directory contains historical data collection sessions and utility scripts.

## Contents

### Collected Sessions
Voice-labeled gameplay sessions from Phase V data collection:
- `20251017_125600_session/`
- `20251017_135458_session/`
- `20251017_141539_session/`
- `20251017_143217_session/`
- `20251017_143627_session/`

**Total**: 5 sessions, ~5-10 minutes each  
**Format**: CSV sensor data + 16kHz audio + WhisperX JSON transcriptions + aligned labels

Each session directory contains:
- `sensor_data.csv` - Raw IMU data (timestamp, accel_x/y/z, gyro_x/y/z, rot_x/y/z/w)
- `audio_16k.wav` - Voice commands during gameplay
- `metadata.json` - Session info (duration, sample rate, etc.)
- `*_whisperx.json` - WhisperX transcription with word-level timestamps
- `*_labels.csv` - Aligned gesture labels (start_time, end_time, gesture, confidence)

### Utility Scripts
Python scripts used during development and debugging:

- `debug_jump_windows.py` - Analyze jump gesture detection windows
- `diagnose_data.py` - Data quality diagnostics
- `process_all_sessions.py` - Batch processing script for multiple sessions
- `test_data_fixes.py` - Unit tests for data processing fixes

**Status**: These scripts solved specific issues during Phase V development. Kept for reference.

## Data Quality Notes

**Known Issues with Archived Sessions**:
1. **Noisy Labels**: Voice-motion coordination is challenging
2. **Unclear Boundaries**: Walk vs. non-walk classification is messy
3. **Action Transitions**: Difficult to isolate clean gesture windows
4. **Class Imbalance**: Some gestures (walk) over-represented vs. others (jump)

**Recommendation**: Use this data to demonstrate the complete pipeline for first draft, but acknowledge limitations. For second draft, consider:
- Button-grid data collection (see `docs/ALTERNATIVE_DATA_COLLECTION_BRAINSTORM.md`)
- More controlled recording environments
- Larger, more balanced dataset

## Current Data Collection

**Active Directory**: `src/data/continuous/` (parent's sibling)

New sessions should be collected there using:
```bash
cd src
python continuous_data_collector.py --duration 600 --session gameplay_01
```

See `docs/Phase_V/DATA_COLLECTION_GUIDE.md` for current best practices.

## How to Use Archived Data

**For Training**:
1. Copy relevant sessions from archive back to `src/data/continuous/`
2. Process with WhisperX (if not already done)
3. Load in training notebook

**For Analysis**:
- Inspect sensor data patterns
- Study voice label alignment
- Understand data quality challenges
- Learn from troubleshooting approaches

**For Comparison**:
- Baseline for improved collection methods
- A/B testing different labeling approaches

---

**Last Updated**: October 18, 2025  
**Total Data**: ~40-50 minutes of labeled sensor data  
**Recommendation**: Use for first draft demonstration, iterate on collection method for future drafts
