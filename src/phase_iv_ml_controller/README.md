# Phase IV: ML-Powered Real-Time Controller

**Multi-threaded gesture recognition controller with trained ML models**

This phase implements the real-time controller that uses trained machine learning models to recognize gestures and control the game.

---

## Scripts in This Phase

### `udp_listener.py` ⭐ Main Controller
**Purpose**: Real-time ML-powered gesture controller (Phase IV architecture)

**Usage**:
```bash
python udp_listener.py
```

**Features**:
- ✅ Multi-threaded architecture (Collector → Predictor → Actor)
- ✅ SVM-based gesture classification
- ✅ <500ms latency
- ✅ Confidence gating (requires 5 consecutive predictions)
- ✅ Micro-window processing (0.3s windows)

**Requirements**:
- Trained SVM model in `../../models/`:
  - `gesture_classifier.pkl`
  - `feature_scaler.pkl`
  - `feature_names.pkl`

**Architecture**:
```
UDP Stream (50Hz) → Collector Thread → Queue
                                       ↓
                    Predictor Thread → ML Model → Queue
                                                  ↓
                                  Actor Thread → Keyboard
```

**When to use**: Primary controller for gameplay with ML gestures

---

### `udp_listener_v3.py`
**Purpose**: Earlier version of ML controller (reference implementation)

**Usage**:
```bash
python udp_listener_v3.py
```

**Features**:
- ✅ Single-threaded implementation
- ✅ SVM-based classification
- ✅ Simpler architecture

**Differences from `udp_listener.py`**:
- ❌ Higher latency (~1s vs <500ms)
- ❌ Blocking operations
- ✅ Easier to understand

**When to use**: Reference implementation or debugging

---

### `calibrate.py`
**Purpose**: Calibration utility for threshold-based detection

**Usage**:
```bash
python calibrate.py
```

**Features**:
- ✅ Records sensor data for calibration
- ✅ Calculates optimal thresholds
- ✅ Updates `config.json`

**When to use**: Fine-tuning fallback thresholds (when ML models not available)

**Note**: Phase IV uses ML exclusively; calibration is for fallback only.

---

## Architecture Details

### Multi-Threading Pattern

**Thread 1: Collector**
- Reads UDP packets at network speed
- Zero processing (just queuing)
- Never blocks on ML or keyboard

**Thread 2: Predictor**
- Runs ML on 0.3s micro-windows
- Continuous prediction (not fixed intervals)
- Outputs to action queue

**Thread 3: Actor**
- Executes keyboard actions
- Confidence gating (5 consecutive matches)
- Prevents jitter and false positives

### Latency Optimization

**Total latency: <500ms**
- UDP transmission: ~20ms
- ML inference: ~10-30ms
- Confidence gating: ~100-150ms (5 predictions at 50Hz)
- Keyboard execution: <10ms

**Comparison to single-threaded**:
- Single-threaded: 1000-1500ms (blocking operations)
- Multi-threaded: <500ms (parallel processing)

---

## Configuration

Edit `../../config.json` for:
- Keyboard mappings
- Confidence thresholds
- Cooldown timers
- Fallback thresholds (when ML unavailable)

**Example**:
```json
{
  "ml_controller": {
    "confidence_threshold": 0.7,
    "consecutive_predictions": 5,
    "micro_window_size": 0.3
  },
  "keyboard": {
    "jump": "z",
    "attack": "x",
    "left": "left",
    "right": "right"
  }
}
```

---

## Required Models

Place trained models in `../../models/`:

```
models/
├── gesture_classifier.pkl    # SVM model
├── feature_scaler.pkl         # Feature scaling
└── feature_names.pkl          # Feature order
```

**Training models**: See [../../docs/COLAB_TRAINING_GUIDE.md](../../docs/COLAB_TRAINING_GUIDE.md)

---

## Quick Start

1. **Train models**: Use Phase II data with Colab notebook
2. **Place models**: Copy `.pkl` files to `../../models/`
3. **Start watch**: Run Android watch app
4. **Run controller**: `python udp_listener.py`
5. **Test gestures**: Perform jump, punch, turn, walk

---

## Troubleshooting

### "Model files not found"
**Solution**: Train models using Colab notebook or use Phase III training script

### "High latency / sluggish"
**Solution**: 
- Check if using `udp_listener.py` (not `udp_listener_v3.py`)
- Verify multi-threading is active (check console logs)

### "False positives / wrong gestures"
**Solution**:
- Increase `consecutive_predictions` in config.json
- Re-train model with more balanced data

### "Connection lost"
**Solution**: 
- Check WiFi connection
- Restart watch app
- Run `../../shared_utils/test_connection.py`

---

## Performance Metrics

**Expected performance** (with well-trained SVM):
- Accuracy: 85-95%
- Latency: <500ms
- False positive rate: <5%
- CPU usage: 10-20%

---

## Next Steps

### If accuracy is low:
1. Collect more training data (Phase II)
2. Re-train model on Colab
3. Verify feature extraction matches training

### For even better results:
- Try **Phase V** voice-labeled collection
- Train **CNN-LSTM** model instead of SVM
- See [../../docs/Phase_V/README.md](../../docs/Phase_V/README.md)

---

See [../../docs/Phase_IV/README.md](../../docs/Phase_IV/README.md) for detailed architecture documentation.
