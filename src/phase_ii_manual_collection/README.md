# Phase II: Manual Data Collection

**Manual gesture data collection using button-press UI**

This phase focuses on collecting labeled gesture samples through manual UI interaction.

---

## Scripts in This Phase

### `button_data_collector.py`
**Purpose**: Simplified button-based data collection for each gesture

**Usage**:
```bash
python button_data_collector.py
```

**Features**:
- ✅ One-button interface for each gesture
- ✅ Real-time data visualization
- ✅ Automatic CSV export
- ✅ Live connection status

**Output**: CSV files in `data/button_collected/`

**When to use**: Quick data collection for training SVM models

---

### `data_collector.py`
**Purpose**: Original comprehensive data collector with guided workflow

**Usage**:
```bash
python data_collector.py
```

**Features**:
- ✅ Guided 8-gesture recording sequence
- ✅ Snippet mode for atomic gestures
- ✅ Continuous mode for state-based gestures
- ✅ 40 samples per atomic gesture

**Output**: Organized data folders with metadata

**When to use**: First-time data collection with structured guidance

---

### `data_collection_dashboard.py`
**Purpose**: Real-time dashboard for monitoring data collection

**Usage**:
```bash
python data_collection_dashboard.py
```

**Features**:
- ✅ Live sensor data visualization
- ✅ Connection status for Watch and Phone
- ✅ Data rate monitoring
- ✅ Real-time graphs

**Output**: Visual feedback during collection

**When to use**: Debugging connection issues or monitoring data quality

---

## Data Format

All collectors output CSV files with this structure:

```csv
accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,rot_w,rot_x,rot_y,rot_z,sensor,timestamp
```

**Sensors**:
- `linear_acceleration`: Movement acceleration (m/s²)
- `gyroscope`: Rotation speed (rad/s)
- `rotation_vector`: Device orientation (quaternion)

---

## Quick Start

1. **Start watch app**: Android/app → Run on Pixel Watch
2. **Run collector**: `python button_data_collector.py`
3. **Perform gestures**: Click buttons and perform corresponding gestures
4. **Export data**: Files saved to `data/button_collected/`

---

## Troubleshooting

See [../../docs/BUTTON_PROTOCOL_QUICK_START.md](../../docs/BUTTON_PROTOCOL_QUICK_START.md) for detailed troubleshooting.

**Common issues**:
- **No data received**: Check if watch and computer on same WiFi
- **Connection drops**: Restart watch app and collector
- **Empty CSV files**: Perform gestures longer (1-2 seconds)

---

## Next Steps

After collecting data:
1. **Verify data quality**: `python ../../shared_utils/inspect_csv_data.py data/button_collected/*.csv`
2. **Train model**: Upload to Google Colab (see [../../docs/COLAB_TRAINING_GUIDE.md](../../docs/COLAB_TRAINING_GUIDE.md))
3. **Deploy**: Use trained model with Phase IV controller

---

See [../../docs/Phase_II/README.md](../../docs/Phase_II/README.md) for more details.
