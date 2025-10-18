# Shared Utilities

**Common utilities used across multiple phases**

These tools are shared by different phases of the pipeline.

---

## Scripts in This Directory

### `network_utils.py`
**Purpose**: Network Service Discovery (NSD) and UDP utilities

**Functions**:
- `discover_service()`: Find watch service via NSD
- `create_udp_socket()`: Create UDP listener socket
- `parse_sensor_packet()`: Parse incoming UDP JSON packets

**Used by**:
- Phase II data collectors
- Phase IV ML controllers
- Phase V continuous collector

**When to modify**: Changing network protocol or adding new sensor types

---

### `feature_extractor.py`
**Purpose**: Extract ~60+ features from sensor data windows

**Functions**:
- `extract_window_features(df)`: Main feature extraction
- Time-domain features: mean, std, min, max, range, median, skew, kurtosis
- Frequency-domain features: FFT max, dominant frequency
- Cross-sensor features: magnitude calculations

**Used by**:
- Phase III SVM training
- Phase IV ML controller (real-time)
- Training notebooks

**Feature list must match**:
- Training feature names
- Real-time extraction order
- Stored in `feature_names.pkl`

**When to modify**: Adding new features (must retrain model!)

---

### `test_connection.py`
**Purpose**: Test UDP connection between watch and computer

**Usage**:
```bash
python test_connection.py
```

**Features**:
- ✅ Tests NSD discovery
- ✅ Verifies UDP reception
- ✅ Shows packet rate
- ✅ Displays sensor data

**When to use**: 
- Debugging connection issues
- Verifying watch is transmitting
- Testing network setup

**Output**:
```
✅ NSD discovery successful
✅ UDP socket created
📡 Receiving packets at 48 Hz
🎯 Latest sensor data:
   accel: [1.2, -0.5, 9.8]
   gyro: [0.1, 0.2, -0.3]
```

---

### `inspect_csv_data.py`
**Purpose**: Validate and inspect collected CSV data

**Usage**:
```bash
python inspect_csv_data.py path/to/data.csv
python inspect_csv_data.py ../data/button_collected/*.csv
```

**Features**:
- ✅ Shows CSV structure
- ✅ Counts samples per sensor
- ✅ Displays statistics
- ✅ Validates format

**When to use**:
- After data collection
- Before uploading to Colab
- Debugging data quality issues

**Output**:
```
File: jump_sample_01.csv
Rows: 120
Duration: 2.4s
Sensors:
  linear_acceleration: 40 samples
  gyroscope: 40 samples
  rotation_vector: 40 samples
✅ Valid format
```

---

## Import Structure

### From Phase II Scripts
```python
from shared_utils.network_utils import discover_service, create_udp_socket
from shared_utils.feature_extractor import extract_window_features
```

### From Phase IV Controller
```python
from shared_utils.network_utils import parse_sensor_packet
from shared_utils.feature_extractor import extract_window_features
```

### From Training Scripts
```python
from shared_utils.feature_extractor import extract_window_features
```

---

## Modifying Shared Utilities

### Adding New Features

If modifying `feature_extractor.py`:

1. **Add feature extraction code**
2. **Update feature names list**
3. **Re-train ALL models** (SVM and CNN-LSTM)
4. **Update Phase IV controller** to use new features
5. **Document in feature list**

**Warning**: Feature changes break existing models!

### Changing Network Protocol

If modifying `network_utils.py`:

1. **Update Android watch app** to match new protocol
2. **Update all data collectors**
3. **Test with `test_connection.py`**
4. **Update documentation**

---

## Testing Utilities

### Test Network Utils
```bash
python test_connection.py
```

### Test Feature Extraction
```bash
python -c "from feature_extractor import extract_window_features; print('✅ Import successful')"
```

### Test CSV Inspection
```bash
python inspect_csv_data.py ../data/button_collected/jump*.csv
```

---

## Dependencies

### network_utils.py
```
- zeroconf (NSD)
- socket (UDP)
- json (packet parsing)
```

### feature_extractor.py
```
- pandas
- numpy
- scipy (stats, fft)
```

### test_connection.py
```
- network_utils
- socket
- json
```

### inspect_csv_data.py
```
- pandas
- argparse
```

---

## Directory Structure

```
shared_utils/
├── README.md (this file)
├── network_utils.py
├── feature_extractor.py
├── test_connection.py
└── inspect_csv_data.py
```

---

## Best Practices

1. **Don't duplicate code**: If a function is used by multiple phases, put it here
2. **Keep imports clean**: Import only what you need
3. **Test changes**: Use test scripts after modifications
4. **Document changes**: Update this README when adding utilities
5. **Version compatibility**: Ensure changes work with all dependent phases

---

## Quick Reference

**Connection issues?** → `test_connection.py`

**Data quality check?** → `inspect_csv_data.py`

**Adding features?** → Modify `feature_extractor.py` + retrain

**Network changes?** → Update `network_utils.py` + Android app

---

See phase-specific READMEs for usage in context:
- [Phase II](../phase_ii_manual_collection/README.md)
- [Phase IV](../phase_iv_ml_controller/README.md)
- [Phase V](../phase_v_voice_collection/README.md)
