# Silksong Motion Controller - Machine Learning Pipeline (First Draft)

**Course**: CS156 Machine Learning  
**Project**: Motion Gesture Recognition for Game Control  
**Hardware**: Pixel Watch (Wear OS) + Python Controller

Control Hollow Knight: Silksong using motion gestures from a smartwatch. This project demonstrates a complete machine learning pipeline from data collection through model deployment, using real-time IMU sensor data streamed over UDP.

Building on top of [V2](https://github.com/CarlKho-Minerva/v2_SilksongController_25TPE)

---

**⚠️ Academic Context**: This is a **first draft** machine learning pipeline for educational purposes. The focus is on demonstrating ML fundamentals (data collection, feature engineering, model training, deployment) rather than achieving production-ready performance. Second and final draft iterations will follow.

---

## 🆕 NEW: Button Data Collection with Dashboard

**Having trouble collecting data?** We've added a comprehensive data collection system with real-time verification:

📊 **[Quick Start Guide](DASHBOARD_QUICK_START.md)** - Get started with the new dashboard in 5 minutes

🔧 **[Troubleshooting Guide](DATA_COLLECTION_VERIFICATION.md)** - Step-by-step verification and problem solving

🤝 **[App Coordination Guide](ANDROID_COORDINATION_GUIDE.md)** - How Watch and Phone apps work together

### Quick Commands

```bash
# Start the real-time dashboard (RECOMMENDED)
cd src/phase_ii_manual_collection
python data_collection_dashboard.py

# Collect gesture data with button UI
cd src/phase_ii_manual_collection
python button_data_collector.py

# Verify data quality after collection
cd src/shared_utils
python inspect_csv_data.py ../../data/phase_ii_button_collected/*.csv

# Test watch connection separately
cd src/shared_utils
python test_connection.py

# Run ML controller (after training models)
cd src/phase_iv_ml_controller
python udp_listener.py
```

**Key Features**:
- ✅ Real-time sensor data visualization
- ✅ Live connection status for Watch and Phone apps
- ✅ Data rate monitoring
- ✅ CSV quality inspector
- ✅ Comprehensive troubleshooting

---

## 📹 Demo Videos

**Watch Data Transmission**: [Loom Video](https://www.loom.com/share/175721940a354cb98fe0ec2a13e2bddf) - NSD discovery and UDP streaming  
**SVM Controller Demo**: [Loom Video](https://www.loom.com/share/dfb0398e038c409084696484e159a588) - Phase III real-time gesture recognition  
**Voice-Labeled Data Collection**: [Loom Video](https://www.loom.com/share/db304cbfea1d4fa4914256f097d4a166) - Phase V collection process

## 🎓 Learning Objectives

This project demonstrates:

1. **Data Collection**: Voice-labeled continuous recording with WhisperX post-processing
2. **Feature Engineering**: Time-series features from IMU data (accel, gyro, rotation)
3. **Model Training**: SVM (local) and CNN-LSTM (Colab) architectures
4. **Real-time Deployment**: Multi-threaded Python controller with <500ms latency
5. **Iteration & Design Thinking**: Documented exploration of data collection methods

## 🎮 Key Features

- **Smartwatch Sensors**: Accelerometer, gyroscope, rotation vector, step detector
- **Automatic Discovery**: NSD (Network Service Discovery) - no manual IP setup
- **Real-time UDP Streaming**: ~50Hz sensor data transmission
- **ML-Powered Gestures**: Walk, jump, attack, turn detection via trained models
- **Voice-Labeled Training Data**: Natural gameplay with retrospective labeling

## 📁 Project Structure

```text
v3-watch_SilksongController_25TPE/
├── Android/                         # Wear OS app (streams sensor data via UDP)
│   └── app/src/main/java/com/cvk/silksongcontroller/
│       └── MainActivity.kt          # Sensor streaming + NSD discovery
├── src/                             # Python ML pipeline (organized by phase)
│   ├── phase_ii_manual_collection/  # Button-based data collection
│   │   ├── button_data_collector.py # Simplified one-button UI
│   │   ├── data_collector.py        # Original guided collector
│   │   ├── data_collection_dashboard.py # Real-time monitoring
│   │   └── README.md                # Phase II documentation
│   ├── phase_iv_ml_controller/      # Real-time ML controller
│   │   ├── udp_listener.py          # Main ML controller (multi-threaded)
│   │   ├── udp_listener_v3.py       # CNN-LSTM version
│   │   ├── calibrate.py             # Threshold calibration
│   │   └── README.md                # Phase IV documentation
│   ├── phase_v_voice_collection/    # Voice-labeled data collection
│   │   ├── continuous_data_collector.py # Natural gameplay recording
│   │   ├── whisperx_transcribe.py   # Audio transcription
│   │   ├── align_voice_labels.py    # Label alignment
│   │   └── README.md                # Phase V documentation
│   ├── shared_utils/                # Common utilities
│   │   ├── network_utils.py         # NSD + UDP helpers
│   │   ├── feature_extractor.py     # Feature engineering
│   │   ├── test_connection.py       # Connection testing
│   │   ├── inspect_csv_data.py      # Data validation
│   │   └── README.md                # Utilities documentation
│   ├── models/                      # ML model code
│   │   └── cnn_lstm_model.py        # Deep learning architecture
│   └── config.json                  # Runtime configuration
├── data/                            # Training data (organized by phase)
│   ├── phase_ii_button_collected/   # Button-collected samples
│   └── phase_v_continuous/          # Voice-labeled sessions
│       ├── continuous/              # Active sessions
│       └── archive/                 # Historical data + scripts
├── notebooks/                       # Training notebooks
│   ├── Silksong_Complete_Training_Colab.ipynb  # All-in-one training
│   ├── watson_Colab_CNN_LSTM_Training.ipynb    # Original CNN-LSTM
│   └── archive/                     # Previous experiments
├── models/                          # Trained model artifacts (generated)
│   ├── README.md                    # Model documentation
│   └── archive/                     # Previous model versions
├── docs/                            # Documentation organized by phase
│   ├── Phase_II/                    # Manual data collection
│   ├── Phase_III/                   # SVM training (local)
│   ├── Phase_IV/                    # Multi-threaded ML controller
│   ├── Phase_V/                     # Voice-labeled collection + WhisperX
│   ├── COLAB_TRAINING_GUIDE.md      # Google Colab training guide
│   ├── CHRONOLOGICAL_NARRATIVE.md   # Project history
│   ├── SESSION_BUTTON_PIVOT_NARRATIVE.md # Recent improvements
│   └── archive/                     # Historical docs
└── requirements.txt                 # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- **Hardware**: Pixel Watch (Wear OS)
- **Software**:
  - Python 3.8+ with pip
  - Android Studio (for watch app)
  - Google Colab (for CNN-LSTM training)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/CarlKho-Minerva/v3-watch_SilksongController_25TPE.git
cd v3-watch_SilksongController_25TPE

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Build Android app (see Android Studio instructions in installer/INSTALLATION_GUIDE.md)

# 4. Collect training data
cd src/phase_ii_manual_collection
python button_data_collector.py

# 5. Train model on Google Colab (see docs/COLAB_TRAINING_GUIDE.md)
# Upload data to Google Drive and use notebooks/Silksong_Complete_Training_Colab.ipynb

# 6. Run the controller with trained models
cd src/phase_iv_ml_controller
python udp_listener.py
```

The watch app will automatically discover your computer via NSD - no manual IP configuration needed!

## 📊 Machine Learning Pipeline Overview

This project evolved through multiple phases, demonstrating different approaches to gesture recognition:

### Phase II: Manual Data Collection ✅
Button-based and guided recording system for collecting labeled gesture samples.

**Tools**: 
- `src/phase_ii_manual_collection/button_data_collector.py` - Simplified one-button UI
- `src/phase_ii_manual_collection/data_collection_dashboard.py` - Real-time monitoring
- `src/phase_ii_manual_collection/data_collector.py` - Original guided collector

**Output**: CSV files in `data/phase_ii_button_collected/`  
**Documentation**: `src/phase_ii_manual_collection/README.md`

### Phase III: SVM Classifier ✅
Traditional ML pipeline using hand-engineered features (~60 features from time/frequency domains) with SVM classification.

**Training**: Google Colab notebook with SVM option  
**Notebook**: `notebooks/Silksong_Complete_Training_Colab.ipynb`  
**Features**: Time-domain stats, FFT features, magnitude calculations  
**Performance**: 85-95% accuracy on test set  
**Guide**: `docs/COLAB_TRAINING_GUIDE.md`

### Phase IV: Multi-Threaded Controller ✅ **(Current Production)**
Real-time ML deployment with decoupled architecture for low-latency performance.

**Architecture**: Collector → Predictor → Actor (three threads with queues)  
**Latency**: <500ms using 0.3s micro-windows  
**Confidence Gating**: 5 consecutive predictions for state changes  
**Code**: `src/phase_iv_ml_controller/udp_listener.py`

See `src/phase_iv_ml_controller/README.md` for architecture details.

### Phase V: Voice-Labeled Data Collection ✅ **(Advanced)**
Natural gameplay recording with voice commands, retrospectively processed using WhisperX for word-level timestamp alignment.

**Tools**:
- `src/phase_v_voice_collection/continuous_data_collector.py` - Natural gameplay recording
- `src/phase_v_voice_collection/whisperx_transcribe.py` - Audio transcription
- `src/phase_v_voice_collection/align_voice_labels.py` - Label alignment

**Workflow**:
```bash
# 1. Record session (natural gameplay + voice labels)
cd src/phase_v_voice_collection
python continuous_data_collector.py --duration 600 --session gameplay_01

# 2. Post-process with WhisperX
python whisperx_transcribe.py --audio ../../data/phase_v_continuous/SESSION/audio_16k.wav

# 3. Align labels
python align_voice_labels.py --session_dir ../../data/phase_v_continuous/SESSION/
```

**Advantages**: Hands-free, natural motion, word-level timestamps, captures transitions  
**Use Case**: Training CNN-LSTM models with large datasets  
**Documentation**: `src/phase_v_voice_collection/README.md`

### Phase VI: Button Data Collection *(Proposed)*
Alternative data collection approach using button-grid Android app for more controlled, precise labeling.

📄 **`docs/BUTTON_DATA_COLLECTION_PROTOCOL.md`** - Complete implementation specification
- Press-and-hold interaction model with exact timestamps
- 2x3 button grid layout with real-time count display
- Dual classifier architecture (walk/idle vs. actions)
- Three-stage streaming pipeline (Watch → Phone → MacBook)
- Data integrity and noise handling strategies
- MVP implementation guide and testing protocol

📄 **`docs/ALTERNATIVE_DATA_COLLECTION_BRAINSTORM.md`** - Design thinking and expert analysis
- Analysis of current data quality issues
- Expert panel discussion (Ng, Li, Jordan, Norman, Ries)
- Trade-offs: organic vs. controlled data collection
- Recommendations for academic pipeline drafts

## 💾 Current Data Status

**Collected Sessions**: 5 voice-labeled gameplay sessions (archived in `src/data/archive/`)
- Session duration: 5-10 minutes each
- Gestures: walk, punch, jump, turn, idle
- Format: CSV sensor data + audio recordings + WhisperX transcriptions

**Known Issues**:
- Labels are noisy (voice-motion coordination challenges)
- Walk vs. non-walk classification has unclear boundaries
- Action transitions are difficult to isolate cleanly

**Recommendation**: Use existing data to demonstrate the complete pipeline in first draft, acknowledge data quality limitations in write-up. For second draft, consider implementing button-grid collection method (see brainstorming doc).

## 📚 Documentation Structure

- **`docs/Phase_II/`** - Manual data collection (archived approach)
- **`docs/Phase_III/`** - SVM training pipeline (archived)
- **`docs/Phase_IV/`** - Multi-threaded controller architecture (current)
- **`docs/Phase_V/`** - Voice-labeled data collection (current)
- **`docs/BUTTON_DATA_COLLECTION_PROTOCOL.md`** - Button grid protocol specification (proposed)
- **`docs/ALTERNATIVE_DATA_COLLECTION_BRAINSTORM.md`** - Design exploration and expert analysis
- **`docs/CHRONOLOGICAL_NARRATIVE.md`** - Complete project history
- **`docs/archive/`** - Historical troubleshooting guides and training docs

## 🔧 Development Notes

**Android App**: 
- Package: `com.cvk.silksongcontroller`
- Main code: `Android/app/src/main/java/com/cvk/silksongcontroller/MainActivity.kt`
- Streams sensor data over UDP with NSD discovery

**Python Controller**:
- Entry point: `src/udp_listener.py` (Phase IV multi-threaded architecture)
- Data collection: `src/continuous_data_collector.py` (Phase V voice labeling)
- Feature extraction: `src/feature_extractor.py` (~60 engineered features)

**Training**:
- Local SVM: `notebooks/archive/CS156_Silksong_Watch.ipynb`
- Colab CNN-LSTM: `notebooks/watson_Colab_CNN_LSTM_Training.ipynb`

## 🎓 Academic Context

This project is part of CS156 Machine Learning coursework, demonstrating:
- Complete ML pipeline implementation
- Design thinking and iteration (documented in brainstorming doc)
- Trade-off analysis (data quality vs. collection methodology)
- Real-world deployment considerations

**Note**: This is a **first draft** pipeline. Emphasis is on demonstrating understanding of ML fundamentals rather than achieving state-of-the-art performance. Future drafts will iterate on data quality and model architecture.

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built on [V2 SilksongController](https://github.com/CarlKho-Minerva/v2_SilksongController_25TPE) with significant enhancements:
- Pixel Watch integration with NSD discovery
- ML-powered gesture recognition (SVM and CNN-LSTM)
- Voice-labeled data collection pipeline
- Multi-threaded real-time controller architecture
