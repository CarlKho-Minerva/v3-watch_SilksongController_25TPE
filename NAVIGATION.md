# Quick Navigation Guide

**Finding your way around the reorganized project**

The project is now organized by development phases for clarity. This guide helps you find what you need quickly.

---

## 📍 Where to Start

### I want to collect gesture data
→ **Go to**: `src/phase_ii_manual_collection/`  
→ **Run**: `python button_data_collector.py`  
→ **Guide**: [Phase II README](src/phase_ii_manual_collection/README.md)

### I want to train a model
→ **Go to**: `notebooks/Silksong_Complete_Training_Colab.ipynb`  
→ **Upload to**: Google Colab  
→ **Guide**: [COLAB_TRAINING_GUIDE.md](docs/COLAB_TRAINING_GUIDE.md)

### I want to run the controller
→ **Go to**: `src/phase_iv_ml_controller/`  
→ **Run**: `python udp_listener.py`  
→ **Guide**: [Phase IV README](src/phase_iv_ml_controller/README.md)

### I want to test the connection
→ **Go to**: `src/shared_utils/`  
→ **Run**: `python test_connection.py`  
→ **Guide**: [Shared Utils README](src/shared_utils/README.md)

---

## 🗂️ Directory Structure

### Source Code (`src/`)

```
src/
├── phase_ii_manual_collection/    # Collect data with button UI
├── phase_iv_ml_controller/        # Run ML-powered controller
├── phase_v_voice_collection/      # Advanced voice-labeled collection
└── shared_utils/                  # Common tools (connection test, etc.)
```

### Data (`data/`)

```
data/
├── phase_ii_button_collected/     # Button-collected gesture samples
└── phase_v_continuous/            # Voice-labeled gameplay sessions
```

### Notebooks (`notebooks/`)

```
notebooks/
├── Silksong_Complete_Training_Colab.ipynb  # ⭐ All-in-one training
└── watson_Colab_CNN_LSTM_Training.ipynb    # Original CNN-LSTM
```

### Documentation (`docs/`)

```
docs/
├── COLAB_TRAINING_GUIDE.md        # ⭐ How to train models
├── SESSION_BUTTON_PIVOT_NARRATIVE.md  # Recent improvements
├── Phase_II/                       # Manual collection docs
├── Phase_III/                      # SVM training docs
├── Phase_IV/                       # Controller architecture docs
└── Phase_V/                        # Voice collection docs
```

---

## 🎯 Common Tasks

### Collect 30 samples for training
```bash
cd src/phase_ii_manual_collection
python button_data_collector.py
```
**Output**: `data/phase_ii_button_collected/`

### Check if watch is connected
```bash
cd src/shared_utils
python test_connection.py
```

### Verify collected data quality
```bash
cd src/shared_utils
python inspect_csv_data.py ../../data/phase_ii_button_collected/*.csv
```

### Train model on Google Colab
1. Upload CSVs to Google Drive: `My Drive/silksong_data/`
2. Open `notebooks/Silksong_Complete_Training_Colab.ipynb` in Colab
3. Enable GPU (Runtime → Change runtime type → GPU)
4. Run all cells
5. Download trained model

### Run controller with trained model
```bash
# Place models in models/ directory:
# - gesture_classifier.pkl
# - feature_scaler.pkl
# - feature_names.pkl

cd src/phase_iv_ml_controller
python udp_listener.py
```

### Collect advanced voice-labeled data
```bash
cd src/phase_v_voice_collection
python continuous_data_collector.py --duration 600 --session game_01
```
**See**: [Phase V README](src/phase_v_voice_collection/README.md) for full workflow

---

## 📚 Key Documentation

### For Beginners
1. [COLAB_TRAINING_GUIDE.md](docs/COLAB_TRAINING_GUIDE.md) - Train your first model
2. [Phase II README](src/phase_ii_manual_collection/README.md) - Collect data
3. [Phase IV README](src/phase_iv_ml_controller/README.md) - Run controller

### For Advanced Users
1. [Phase V README](src/phase_v_voice_collection/README.md) - Voice-labeled collection
2. [SESSION_BUTTON_PIVOT_NARRATIVE.md](docs/SESSION_BUTTON_PIVOT_NARRATIVE.md) - Recent improvements
3. [Shared Utils README](src/shared_utils/README.md) - Utilities documentation

### Reference
1. [Main README](README.md) - Project overview
2. [CHRONOLOGICAL_NARRATIVE.md](docs/CHRONOLOGICAL_NARRATIVE.md) - Project history
3. Phase-specific docs in `docs/Phase_*`

---

## 🔧 Troubleshooting

### "Import error: No module named 'network_utils'"
**Cause**: Running script from wrong directory  
**Solution**: Scripts must be run from their own directory (imports are path-relative)

**Example**:
```bash
# ❌ Wrong
python src/phase_iv_ml_controller/udp_listener.py

# ✅ Correct
cd src/phase_iv_ml_controller
python udp_listener.py
```

### "No such file or directory: config.json"
**Cause**: Running from wrong directory  
**Solution**: Run from the script's directory, or use absolute paths

### "Cannot find data files"
**Data locations have changed**:
- Button-collected: `data/phase_ii_button_collected/`
- Voice-labeled: `data/phase_v_continuous/`

### "My old commands don't work"
**Scripts have moved to phase directories**. Update your commands:

**Old**:
```bash
cd src
python udp_listener.py
python data_collector.py
python test_connection.py
```

**New**:
```bash
cd src/phase_iv_ml_controller && python udp_listener.py
cd src/phase_ii_manual_collection && python data_collector.py
cd src/shared_utils && python test_connection.py
```

---

## 💡 Tips

1. **Always run scripts from their directory** - Import paths are relative
2. **Check phase READMEs first** - Each phase has comprehensive documentation
3. **Use test_connection.py** - Verify watch connection before collecting data
4. **Start with Phase II** - Button collector is the easiest way to begin
5. **Read COLAB_TRAINING_GUIDE** - Step-by-step training instructions

---

## 🚀 Quick Reference

| Task | Command | Directory |
|------|---------|-----------|
| Collect data | `python button_data_collector.py` | `src/phase_ii_manual_collection/` |
| Test connection | `python test_connection.py` | `src/shared_utils/` |
| Inspect data | `python inspect_csv_data.py FILE` | `src/shared_utils/` |
| Run controller | `python udp_listener.py` | `src/phase_iv_ml_controller/` |
| Train model | Upload notebook to Colab | `notebooks/` |

---

**Still confused?** Open an issue or check the [Main README](README.md) for the full project overview.
