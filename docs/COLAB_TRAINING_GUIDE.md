# Google Colab Training Guide

**Complete guide for training gesture recognition models on Google Colab**

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Prepare Your Data

Your button-collected data is already in the right CSV format! Just organize it by gesture type.

**Current data location**: `data/button_collected/`

**What you have**:
- `jump_*.csv` - Jump gesture samples
- `punch_*.csv` - Punch gesture samples  
- `turn_left_*.csv` and `turn_right_*.csv` - Turn gesture samples
- `walk_*.csv` - Walk gesture samples
- `idle_*.csv` - Idle state samples

### Step 2: Upload to Google Drive

Create this folder structure on Google Drive:

```
My Drive/
└── silksong_data/
    ├── jump/
    │   └── (copy all jump_*.csv files here)
    ├── punch/
    │   └── (copy all punch_*.csv files here)
    ├── turn/
    │   └── (copy all turn_*.csv files here)
    ├── walk/
    │   └── (copy all walk_*.csv files here)
    └── idle/
        └── (copy all idle_*.csv files here)
```

**Note**: Combine `turn_left` and `turn_right` into the same `turn/` folder.

### Step 3: Open the Notebook

1. Upload `notebooks/Silksong_Complete_Training_Colab.ipynb` to Google Colab
2. Or open directly: File → Open notebook → Upload

### Step 4: Enable GPU

1. In Colab: **Runtime** → **Change runtime type**
2. Hardware accelerator: **GPU**
3. GPU type: **T4** (free tier) or better if you have Colab Pro

### Step 5: Run All Cells

1. Click **Runtime** → **Run all**
2. When prompted, authorize Google Drive access
3. Wait 20-40 minutes for training to complete
4. Download your trained model!

---

## 📊 Understanding Your Data

### CSV Format

Each CSV file contains sensor readings from your Pixel Watch:

```csv
accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,rot_w,rot_x,rot_y,rot_z,sensor,timestamp
```

**Sensors included**:
- **linear_acceleration**: Movement acceleration (m/s²)
- **gyroscope**: Rotation speed (rad/s)
- **rotation_vector**: Device orientation (quaternion)

### Sample Requirements

**Minimum**: 30 samples per gesture (150 total)

**Your current data**:
- Jump: ~1-2 samples ⚠️ **Need more!**
- Punch: ~1-2 samples ⚠️ **Need more!**
- Turn: ~2-4 samples ⚠️ **Need more!**
- Walk: ~1-2 samples ⚠️ **Need more!**
- Idle: ~1-2 samples ⚠️ **Need more!**

**Action needed**: Use `src/button_data_collector.py` to collect more samples!

```bash
# Collect more data
cd src
python button_data_collector.py
```

See [BUTTON_PROTOCOL_QUICK_START.md](BUTTON_PROTOCOL_QUICK_START.md) for detailed instructions.

---

## 🎯 Model Selection Guide

The notebook supports two model types:

### Option A: SVM (Support Vector Machine)

**Recommended for**: First-time training, quick experiments

**Pros**:
- ✅ Fast training (5-10 minutes)
- ✅ Works without GPU
- ✅ Good accuracy (85-95%)
- ✅ Small model size (~100 KB)

**Cons**:
- ❌ Requires hand-crafted features
- ❌ Less flexible than deep learning

**When to use**: 
- You want quick results
- You don't have GPU access
- You have limited data (<100 samples)

### Option B: CNN-LSTM (Deep Learning)

**Recommended for**: Best performance, have GPU

**Pros**:
- ✅ Higher accuracy (90-98%)
- ✅ Learns features automatically
- ✅ Better generalization
- ✅ Handles complex patterns

**Cons**:
- ❌ Slower training (20-40 min with GPU)
- ❌ Needs more data (200+ samples ideal)
- ❌ Larger model size (~850 KB)

**When to use**:
- You have GPU enabled
- You want best accuracy
- You have enough data (150+ samples)

---

## 🔧 GPU Configuration

### Free Tier (Google Colab)

**Available GPUs**:
- **T4**: Standard free GPU (recommended)
- **K80**: Older, slower (if T4 unavailable)

**Training times with T4**:
- SVM: 5-10 minutes
- CNN-LSTM: 20-40 minutes

### Colab Pro

**Available GPUs**:
- **T4**: Standard
- **V100**: 2-3x faster than T4
- **A100**: 4-5x faster than T4

**Training times with V100**:
- SVM: 2-5 minutes
- CNN-LSTM: 8-15 minutes

**Recommendation**: **T4 is sufficient** for this project. Only use V100/A100 if you're training on large datasets (1000+ samples).

---

## 📥 Exporting and Using Models

### After Training

The notebook saves models to: `My Drive/silksong_models/`

**For SVM**, you'll get 3 files:
```
gesture_classifier_svm_YYYYMMDD_HHMMSS.pkl
feature_scaler_YYYYMMDD_HHMMSS.pkl
feature_names_YYYYMMDD_HHMMSS.pkl
```

**For CNN-LSTM**, you'll get 1 file:
```
gesture_classifier_cnn_lstm_YYYYMMDD_HHMMSS.h5
```

### Download Models

1. Go to Google Drive → `silksong_models/`
2. Right-click each file → Download
3. Move files to your project's `models/` directory

### Rename for Controller

**For SVM**:
```bash
cd models
mv gesture_classifier_svm_*.pkl gesture_classifier.pkl
mv feature_scaler_*.pkl feature_scaler.pkl
mv feature_names_*.pkl feature_names.pkl
```

**For CNN-LSTM**:
```bash
cd models
mv gesture_classifier_cnn_lstm_*.h5 cnn_lstm_gesture.h5
```

### Using with Controller

**SVM** (default):
```bash
cd src
python udp_listener.py
```

**CNN-LSTM** (requires code update):
See [Phase_V/README.md](Phase_V/README.md) for CNN-LSTM integration guide.

---

## 🐛 Troubleshooting

### "Data directory not found"

**Problem**: Notebook can't find `silksong_data/` folder

**Solution**:
1. Make sure you created the folder in "My Drive" (not "Shared with me")
2. Check spelling: `silksong_data` (all lowercase, underscore)
3. Re-run the "Mount Google Drive" cell

### "No GPU detected"

**Problem**: GPU not enabled in Colab

**Solution**:
1. Runtime → Change runtime type
2. Hardware accelerator → GPU
3. Click "Save"
4. Re-run notebook from beginning

### "Not enough data"

**Problem**: Less than 30 samples per gesture

**Solution**: Collect more data using button data collector:
```bash
cd src
python button_data_collector.py
```

### "Low accuracy (<70%)"

**Possible causes**:
1. **Not enough data**: Collect 30+ samples per gesture
2. **Unbalanced data**: Some gestures have way more samples than others
3. **Similar gestures**: Turn left/right might be too similar

**Solutions**:
1. Collect more balanced data
2. Try combining similar gestures (e.g., turn_left + turn_right = turn)
3. Switch to CNN-LSTM for better feature learning

### "Training is slow"

**Problem**: No GPU or using CPU

**Solution**:
1. Enable GPU (see above)
2. If GPU is enabled but still slow, try SVM instead of CNN-LSTM
3. Reduce data size for faster experimentation

---

## 📈 Next Steps After Training

### 1. Test Your Model

```bash
cd src
python udp_listener.py
```

Perform gestures with your watch and see if they're recognized correctly.

### 2. Collect More Data

If accuracy is low, collect more samples:
```bash
cd src  
python button_data_collector.py
```

### 3. Iterate

1. Collect more data
2. Re-train model on Colab
3. Test again
4. Repeat until accuracy is satisfactory

### 4. Advanced Training

For even better results, consider:
- **Phase V**: Voice-labeled continuous data collection
- See [Phase_V/README.md](Phase_V/README.md) for details

---

## 💡 Tips for Best Results

### Data Collection
1. **Vary your wrist position**: Don't always start from the same position
2. **Natural movements**: Perform gestures as you would during gameplay
3. **Consistent duration**: ~1-2 seconds per gesture sample
4. **Clear separation**: Wait between gestures (idle states help!)

### Training
1. **Start with SVM**: Get quick feedback before investing in CNN-LSTM
2. **Check class balance**: Make sure all gestures have similar sample counts
3. **Use validation metrics**: Don't just trust training accuracy
4. **Save old models**: Keep previous versions in case new training performs worse

### Deployment
1. **Test extensively**: Try all gestures multiple times
2. **Adjust thresholds**: Edit `config.json` if needed
3. **Monitor false positives**: Track which gestures get confused
4. **Iterate quickly**: The faster you test → collect → train cycle, the better

---

## 📚 Additional Resources

- **Main README**: [../README.md](../README.md)
- **Button Collection Guide**: [BUTTON_PROTOCOL_QUICK_START.md](BUTTON_PROTOCOL_QUICK_START.md)
- **Phase IV ML Controller**: [Phase_IV/README.md](Phase_IV/README.md)
- **Phase V Advanced Training**: [Phase_V/README.md](Phase_V/README.md)

---

**Questions?** Check the [troubleshooting section](#-troubleshooting) or open an issue on GitHub.

**Happy training! 🎓**
