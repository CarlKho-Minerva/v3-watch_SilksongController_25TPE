# Migration Guide: Old → New Structure

**Guide for users migrating from the old flat structure to the new phase-based organization**

If you've been using this project before the October 2025 reorganization, this guide will help you update your workflows.

---

## What Changed?

### Scripts Moved to Phase Directories

**Before**: All scripts in `src/`  
**After**: Scripts organized by phase in subdirectories

### Data Folders Renamed

**Before**: `data/button_collected/`, `src/data/continuous/`  
**After**: `data/phase_ii_button_collected/`, `data/phase_v_continuous/`

---

## Quick Migration Reference

### Script Locations

| Old Location | New Location | Notes |
|--------------|--------------|-------|
| `src/udp_listener.py` | `src/phase_iv_ml_controller/udp_listener.py` | Main controller |
| `src/data_collector.py` | `src/phase_ii_manual_collection/data_collector.py` | Original collector |
| `src/button_data_collector.py` | `src/phase_ii_manual_collection/button_data_collector.py` | Button UI |
| `src/data_collection_dashboard.py` | `src/phase_ii_manual_collection/data_collection_dashboard.py` | Dashboard |
| `src/continuous_data_collector.py` | `src/phase_v_voice_collection/continuous_data_collector.py` | Voice collection |
| `src/whisperx_transcribe.py` | `src/phase_v_voice_collection/whisperx_transcribe.py` | Transcription |
| `src/align_voice_labels.py` | `src/phase_v_voice_collection/align_voice_labels.py` | Label alignment |
| `src/calibrate.py` | `src/phase_iv_ml_controller/calibrate.py` | Calibration |
| `src/udp_listener_v3.py` | `src/phase_iv_ml_controller/udp_listener_v3.py` | CNN-LSTM version |
| `src/network_utils.py` | `src/shared_utils/network_utils.py` | Network utilities |
| `src/feature_extractor.py` | `src/shared_utils/feature_extractor.py` | Feature extraction |
| `src/test_connection.py` | `src/shared_utils/test_connection.py` | Connection test |
| `src/inspect_csv_data.py` | `src/shared_utils/inspect_csv_data.py` | Data inspector |

### Data Locations

| Old Location | New Location |
|--------------|--------------|
| `data/button_collected/` | `data/phase_ii_button_collected/` |
| `src/data/continuous/` | `data/phase_v_continuous/continuous/` |
| `src/data/archive/` | `data/phase_v_continuous/archive/` |

---

## Updating Your Workflows

### 1. Data Collection

**Old**:
```bash
cd src
python button_data_collector.py
```

**New**:
```bash
cd src/phase_ii_manual_collection
python button_data_collector.py
```

### 2. Running Controller

**Old**:
```bash
cd src
python udp_listener.py
```

**New**:
```bash
cd src/phase_iv_ml_controller
python udp_listener.py
```

### 3. Testing Connection

**Old**:
```bash
cd src
python test_connection.py
```

**New**:
```bash
cd src/shared_utils
python test_connection.py
```

### 4. Voice Collection

**Old**:
```bash
cd src
python continuous_data_collector.py --duration 600 --session game_01
```

**New**:
```bash
cd src/phase_v_voice_collection
python continuous_data_collector.py --duration 600 --session game_01
```

---

## Updating Scripts and Notebooks

### Import Statements

If you have custom scripts that import project modules:

**Old**:
```python
import network_utils
import feature_extractor
```

**New**:
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared_utils'))
import network_utils
from feature_extractor import extract_window_features
```

Or use absolute imports from project root:
```python
from src.shared_utils import network_utils
from src.shared_utils.feature_extractor import extract_window_features
```

### Data Paths in Code

**Old**:
```python
data_dir = "data/button_collected/"
continuous_dir = "src/data/continuous/"
```

**New**:
```python
data_dir = "data/phase_ii_button_collected/"
continuous_dir = "data/phase_v_continuous/continuous/"
```

---

## Configuration Files

### config.json Location

`config.json` is still in `src/` but now also in `src/phase_iv_ml_controller/` and other phase directories as needed.

**If you customized config.json**:
1. Copy your customizations to the relevant phase directory
2. Or use absolute paths to point to a central config

---

## Trained Models

### Model Files Location

Models are still in `models/` directory at project root. No changes needed.

**Model files**:
- `models/gesture_classifier.pkl` (SVM)
- `models/feature_scaler.pkl`
- `models/feature_names.pkl`
- `models/cnn_lstm_gesture.h5` (CNN-LSTM)

---

## Shell Scripts and Aliases

### Update Bash Aliases

If you have aliases in `.bashrc` or `.zshrc`:

**Old**:
```bash
alias silksong-collect='cd ~/projects/silksong/src && python button_data_collector.py'
alias silksong-run='cd ~/projects/silksong/src && python udp_listener.py'
```

**New**:
```bash
alias silksong-collect='cd ~/projects/silksong/src/phase_ii_manual_collection && python button_data_collector.py'
alias silksong-run='cd ~/projects/silksong/src/phase_iv_ml_controller && python udp_listener.py'
```

### Update Shell Scripts

If you have custom shell scripts:

**Example update**:
```bash
# Old
cd src
python data_collector.py --gestures jump,punch
python udp_listener.py

# New
cd src/phase_ii_manual_collection
python data_collector.py --gestures jump,punch
cd ../phase_iv_ml_controller
python udp_listener.py
```

---

## IDE and Editor Configuration

### VS Code Launch Configurations

Update `.vscode/launch.json`:

**Old**:
```json
{
  "name": "Run Controller",
  "type": "python",
  "request": "launch",
  "program": "${workspaceFolder}/src/udp_listener.py",
  "cwd": "${workspaceFolder}/src"
}
```

**New**:
```json
{
  "name": "Run Controller",
  "type": "python",
  "request": "launch",
  "program": "${workspaceFolder}/src/phase_iv_ml_controller/udp_listener.py",
  "cwd": "${workspaceFolder}/src/phase_iv_ml_controller"
}
```

### PyCharm Run Configurations

1. Edit existing run configurations
2. Update "Script path" to new location
3. Update "Working directory" to script's directory

---

## Git and Version Control

### .gitignore Updates

No changes needed - `.gitignore` patterns still work:
- `data/` ignores all data folders
- `models/*.pkl` ignores model files
- `__pycache__/` ignores Python cache

### Pull Latest Changes

```bash
git fetch origin
git pull origin main  # or your branch
```

Your existing data won't be affected - it's not in git.

---

## Troubleshooting Migration Issues

### "Module not found" errors

**Cause**: Running script from wrong directory  
**Solution**: Always `cd` into the script's directory before running

### "File not found" for data files

**Cause**: Old data paths in code  
**Solution**: Update paths to new locations:
- `data/phase_ii_button_collected/`
- `data/phase_v_continuous/`

### "Config file not found"

**Cause**: Running from wrong directory  
**Solution**: `config.json` is in each phase directory or use absolute path

### Old notebooks don't work

**Training notebooks have been updated**:
- Use `notebooks/Silksong_Complete_Training_Colab.ipynb` (new all-in-one)
- Old notebooks moved to `notebooks/archive/`

---

## Benefits of New Structure

### Before (Flat Structure)
❌ Hard to find relevant scripts  
❌ Unclear which phase/workflow scripts belong to  
❌ Mixed data from different collection methods  
❌ No clear starting point for beginners

### After (Phase-Based Structure)
✅ Clear workflow progression (Phase II → III → IV → V)  
✅ Each phase has its own README  
✅ Shared utilities clearly separated  
✅ Data organized by collection method  
✅ Easy to navigate with NAVIGATION.md

---

## Getting Help

### Resources
1. [NAVIGATION.md](NAVIGATION.md) - Quick reference for new structure
2. [README.md](README.md) - Updated project overview
3. Phase READMEs - Documentation for each phase

### Still Having Issues?

1. Check [NAVIGATION.md](NAVIGATION.md) for common tasks
2. Read the phase README for your specific task
3. Open an issue on GitHub with your error message

---

## Quick Checklist

- [ ] Updated bookmarks/shortcuts to new script locations
- [ ] Updated shell aliases to new paths
- [ ] Updated custom scripts with new import statements
- [ ] Updated data paths in notebooks
- [ ] Updated IDE/editor run configurations
- [ ] Read [NAVIGATION.md](NAVIGATION.md) for quick reference
- [ ] Tested your workflow with new structure

---

**Migration complete?** Check out the new features in [SESSION_BUTTON_PIVOT_NARRATIVE.md](docs/SESSION_BUTTON_PIVOT_NARRATIVE.md)!
