# Phase V: Voice-Labeled Data Collection

**Natural gameplay data collection with voice command labeling**

This phase implements a more advanced data collection method where you play naturally and speak gesture names, then post-process with WhisperX for automated labeling.

---

## Scripts in This Phase

### `continuous_data_collector.py` ⭐
**Purpose**: Continuous sensor data recording with simultaneous audio capture

**Usage**:
```bash
python continuous_data_collector.py --duration 600 --session game_01
```

**Arguments**:
- `--duration`: Recording duration in seconds (default: 300)
- `--session`: Session name for organizing files (default: auto-generated timestamp)

**Features**:
- ✅ Continuous 50Hz sensor data streaming
- ✅ Simultaneous audio recording (16kHz)
- ✅ Voice command capture ("jump", "punch", "turn", "walk", "idle")
- ✅ Automatic file organization
- ✅ Metadata tracking

**Output**:
```
../../data/continuous/YYYYMMDD_HHMMSS_session/
├── sensor_data.csv          # Raw IMU data
├── audio_16k.wav            # Voice commands
└── metadata.json            # Session info
```

**When to use**: Collecting large amounts of natural gameplay data

---

### `whisperx_transcribe.py`
**Purpose**: Transcribe audio using WhisperX with word-level timestamps

**Usage**:
```bash
python whisperx_transcribe.py --audio ../data/continuous/SESSION/audio_16k.wav
```

**Features**:
- ✅ Word-level timestamp accuracy
- ✅ Speaker diarization
- ✅ Automatic language detection
- ✅ GPU acceleration support

**Output**: `SESSION_whisperx.json` with word segments

**Requirements**: 
```bash
pip install whisperx
```

**When to use**: After collecting continuous data, before training

---

### `align_voice_labels.py`
**Purpose**: Align transcribed voice commands to sensor timestamps

**Usage**:
```bash
python align_voice_labels.py --session_dir ../data/continuous/SESSION/
```

**Features**:
- ✅ Matches gesture keywords from transcription
- ✅ Aligns to sensor data timestamps
- ✅ Generates training labels with confidence scores
- ✅ Handles timing offsets

**Output**: `SESSION_labels.csv` with gesture labels

**Format**:
```csv
timestamp,gesture,duration,confidence
218856645784104,jump,1000,0.95
218857645784104,punch,800,0.90
...
```

**When to use**: After WhisperX transcription, before training

---

## Complete Workflow

### 1. Collect Data
```bash
python continuous_data_collector.py --duration 600 --session game_01
```

Speak gesture names naturally while performing them:
- "jump" when jumping
- "punch" or "attack" when punching
- "turn left" or "turn right" when turning
- "walk" when walking
- "idle" when idle

### 2. Transcribe Audio
```bash
# Using the helper script (recommended)
../../docs/Phase_V/process_transcripts.sh game_01

# Or manually
python whisperx_transcribe.py --audio ../data/continuous/20241018_123456_game_01/audio_16k.wav
```

### 3. Align Labels
```bash
python align_voice_labels.py --session_dir ../data/continuous/20241018_123456_game_01/
```

### 4. Verify Data
```bash
# Check if labels were generated correctly
cat ../data/continuous/20241018_123456_game_01/20241018_123456_game_01_labels.csv | head
```

### 5. Upload to Google Drive
```
My Drive/silksong_data/
└── game_01/
    ├── sensor_data.csv
    └── game_01_labels.csv
```

### 6. Train CNN-LSTM Model
Use the Colab notebook with continuous data for CNN-LSTM training.

---

## Advantages Over Phase II

**Phase II (Manual)**:
- ❌ Button-press interrupts gameplay
- ❌ Time-consuming (30+ button presses)
- ❌ Limited to isolated gestures
- ✅ Simple and direct

**Phase V (Voice-labeled)**:
- ✅ Natural gameplay (no interruptions)
- ✅ Collect hundreds of samples quickly
- ✅ Captures gesture sequences and transitions
- ✅ Better for training CNN-LSTM models
- ❌ Requires post-processing

---

## Voice Command Guidelines

### Supported Keywords
- **Jump**: "jump", "jumping"
- **Punch**: "punch", "punching", "attack", "attacking"
- **Turn**: "turn", "turning", "left", "right"
- **Walk**: "walk", "walking", "run", "running"
- **Idle**: "idle", "wait", "waiting", "stop"

### Tips for Good Labels
1. **Speak clearly**: Enunciate gesture names
2. **Timing**: Say the word AS you start the gesture
3. **Consistency**: Use the same words for same gestures
4. **Natural**: Don't overthink it - speak naturally
5. **Repetition**: Multiple gestures of same type are good!

---

## WhisperX Setup

### Installation
```bash
# Install WhisperX
pip install git+https://github.com/m-bain/whisperx.git

# For GPU support (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Configuration
WhisperX automatically:
- Downloads models on first run (~1GB)
- Detects language (English assumed)
- Provides word-level timestamps (±0.1s accuracy)

---

## Data Quality

### Expected Output
From 10 minutes of natural gameplay:
- **Sensor samples**: ~30,000 readings (50Hz × 600s)
- **Gesture labels**: 50-200 labeled segments
- **Audio size**: ~10MB (16kHz WAV)
- **CSV size**: ~5-10MB

### Validation
```bash
# Check if data is valid
python ../../shared_utils/inspect_csv_data.py ../data/continuous/SESSION/sensor_data.csv

# Check label count
wc -l ../data/continuous/SESSION/SESSION_labels.csv
```

---

## Troubleshooting

### "Audio not recording"
**Solution**: Check microphone permissions and `sounddevice` installation

### "WhisperX not installed"
**Solution**: 
```bash
pip install git+https://github.com/m-bain/whisperx.git
```

### "No labels generated"
**Solution**: 
- Check if WhisperX transcription succeeded
- Verify you spoke gesture keywords clearly
- Check `SESSION_whisperx.json` for transcribed text

### "Poor label alignment"
**Solution**:
- Speak gesture names EXACTLY when you start the gesture
- Reduce background noise
- Use a better microphone

---

## Next Steps

After collecting voice-labeled data:
1. **Upload to Google Drive**: Organize for Colab training
2. **Train CNN-LSTM**: Use continuous data for deep learning
3. **Deploy**: Use trained model with Phase IV controller

---

See [../../docs/Phase_V/README.md](../../docs/Phase_V/README.md) for more details and the complete training pipeline.
