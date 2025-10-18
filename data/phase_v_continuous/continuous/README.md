# Continuous Data Collection Directory

This directory stores continuous training data sessions for Phase V CNN/LSTM model training.

## Directory Structure

Each recording session gets its own timestamped directory:

```
continuous/
├── 20251017_143022_game_01/          # Session 1
│   ├── audio.wav                      # Original 44.1kHz audio
│   ├── audio_16k.wav                  # 16kHz audio for WhisperX
│   ├── sensor_data.csv                # IMU sensor data (accel, gyro, rotation)
│   ├── metadata.json                  # Session metadata (duration, timestamp, etc.)
│   ├── README.md                      # Session documentation
│   ├── 20251017_143022_game_01_whisperx.json       # WhisperX transcription output
│   ├── 20251017_143022_game_01_whisperx_summary.txt # Transcription summary
│   ├── 20251017_143022_game_01_labels.csv          # Generated gesture labels
│   └── 20251017_143022_game_01_alignment.json      # Alignment metadata
├── 20251017_150000_game_02/          # Session 2
│   └── ... (same structure)
└── 20251017_153000_game_03/          # Session 3
    └── ... (same structure)
```

## Workflow

### 1. Data Collection

Record a continuous session using the data collector:

```bash
cd src
python continuous_data_collector.py --duration 600 --session game_01
```

This creates a new session directory with:
- `audio.wav` - Original audio recording
- `audio_16k.wav` - Downsampled for WhisperX
- `sensor_data.csv` - Sensor readings
- `metadata.json` - Session information
- `README.md` - Session notes

### 2. Post-Processing

Process the recorded session:

```bash
# Option 1: Use automated script (recommended)
./process_transcripts.sh YYYYMMDD_HHMMSS_session_name

# Option 2: Manual processing
# Step 2a: Run WhisperX transcription
python src/whisperx_transcribe.py \
  --audio src/data/continuous/YYYYMMDD_HHMMSS_session_name/audio_16k.wav

# Step 2b: Align voice commands with sensor data
python src/align_voice_labels.py \
  --session YYYYMMDD_HHMMSS_session_name \
  --whisper src/data/continuous/YYYYMMDD_HHMMSS_session_name/YYYYMMDD_HHMMSS_session_name_whisperx.json
```

This generates:
- `*_whisperx.json` - Transcription with word-level timestamps
- `*_labels.csv` - Gesture labels timeline
- `*_alignment.json` - Alignment statistics

### 3. Training

Use the labeled data for model training:

```bash
# Load all sessions for training
python notebooks/Phase_V_Training.ipynb
```

## File Formats

### sensor_data.csv

```csv
timestamp,sensor_type,x,y,z,accuracy
1697543822.123,rotation_vector,0.123,0.456,0.789,3
1697543822.143,linear_acceleration,0.012,0.034,-9.8,3
1697543822.143,gyroscope,0.001,0.002,0.003,3
...
```

### *_labels.csv

```csv
timestamp,gesture,duration
0.0,walk,15.2
15.2,jump,0.3
15.5,walk,14.6
30.1,punch,0.3
...
```

### metadata.json

```json
{
  "session_name": "20251017_143022_game_01",
  "start_time": "2025-10-17T14:30:22.123456",
  "duration_sec": 600,
  "actual_duration_sec": 598.5,
  "audio_sample_rate": 44100,
  "sensors": ["rotation_vector", "linear_acceleration", "gyroscope"]
}
```

## Data Quality Guidelines

### Good Session Characteristics

✅ **Duration**: 5-10 minutes (300-600 seconds)  
✅ **Gesture Variety**: 20-40 jumps, 20-40 punches, 10-20 turns  
✅ **Audio Quality**: Clear voice commands, minimal background noise  
✅ **Sensor Quality**: Continuous stream, no dropouts  
✅ **Natural Motion**: Realistic gameplay patterns, smooth transitions  

### Warning Signs

⚠️ **Too Short**: <3 minutes (insufficient data)  
⚠️ **Too Long**: >15 minutes (fatigue affects quality)  
⚠️ **Few Gestures**: <10 of any type (poor variety)  
⚠️ **Poor Audio**: Commands not recognized by WhisperX  
⚠️ **Sensor Gaps**: Missing data in sensor_data.csv  

## Best Practices

### Recording
- **Environment**: Quiet room, minimal background noise
- **Microphone**: 1-2 feet away, unobstructed
- **Speaking**: Clear commands, natural timing
- **Motion**: Natural gameplay patterns, avoid exaggeration

### Post-Processing
- **Use WhisperX**: More accurate than standard Whisper
- **Check Transcription**: Verify commands were recognized
- **Validate Labels**: Review alignment statistics
- **Minimum Confidence**: Use 0.5-0.7 threshold

### Training Data
- **Multiple Sessions**: Collect 5-10 sessions minimum
- **Total Duration**: Aim for 30-60 minutes total
- **Diverse Conditions**: Different times, orientations
- **Regular Validation**: Check label quality after each session

## Troubleshooting

### No audio files generated
- Check microphone permissions
- Verify sounddevice installation: `pip install sounddevice`
- Test microphone: `python -c "import sounddevice; print(sounddevice.query_devices())"`

### WhisperX not working
- Install WhisperX: `pip install whisperx`
- Check CUDA/GPU availability (optional but faster)
- Try smaller model: `--model medium`

### Labels look wrong
- Review WhisperX transcription output
- Check if you spoke commands during recording
- Adjust confidence threshold: `--min-confidence 0.3`
- Re-record session if necessary

### Sensor data gaps
- Check watch connection stability
- Verify UDP listener is receiving data
- Use `test_connection.py` to diagnose issues

## See Also

- [DATA_COLLECTION_GUIDE.md](../../docs/Phase_V/DATA_COLLECTION_GUIDE.md) - Detailed recording instructions
- [POST_PROCESSING.md](../../docs/Phase_V/POST_PROCESSING.md) - Post-processing workflow
- [WHISPERX_GUIDE.md](../../docs/Phase_V/WhisperX/WHISPERX_GUIDE.md) - WhisperX documentation
- [CNN_LSTM_ARCHITECTURE.md](../../docs/Phase_V/CNN_LSTM_ARCHITECTURE.md) - Model training guide

---

**Last Updated**: October 17, 2025  
**Directory Status**: Ready for data collection 📁
