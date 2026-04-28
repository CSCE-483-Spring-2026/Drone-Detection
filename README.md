# 🚁 Drone Detection System

> An acoustic drone detection system using FFT-based signal processing on embedded hardware and machine learning classifiers (CNN and Logistic Regression) for real-time drone identification.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [How It Works](#how-it-works)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Getting Started](#getting-started)
  - [Arduino / Embedded Setup](#arduino--embedded-setup)
  - [Python ML Pipeline](#python-ml-pipeline)
- [Module Documentation](#module-documentation)
  - [filterArduino.ino](#filterarduinoino)
  - [arduino/](#arduino)
  - [cnn_prediction/](#cnn_prediction)
  - [lr_prediction/](#lr_prediction)
  - [filter_programs/](#filter_programs)
  - [at_home_recording/](#at_home_recording)
  - [drone_data/](#drone_data)
  - [visualization/](#visualization)
- [Signal Processing Details](#signal-processing-details)
- [Communication Protocol](#communication-protocol)
- [Contributing](#contributing)

---

## Overview

This project detects drones acoustically by analyzing microphone input for frequency signatures characteristic of drone motors and propellers. The system operates in two phases:

1. **Embedded Detection (Arduino/ESP32):** Continuously listens to audio via an I2S microphone. When a frequency peak in the drone range (175–500 Hz) is detected using FFT analysis, it triggers a 3-second WAV recording and streams it over serial to a host computer.

2. **ML Classification (Python):** The host computer receives the audio and runs it through either a Convolutional Neural Network (CNN) or a Logistic Regression (LR) classifier to confirm drone presence.

---

## Repository Structure

```
Drone-Detection/
│
├── filterArduino.ino         # Main Arduino sketch: I2S capture, FFT detection, serial streaming
│
├── arduino/                  # Additional Arduino sketches and hardware utilities
│
├── cnn_prediction/           # CNN-based drone audio classifier (Python)
│
├── lr_prediction/            # Logistic Regression-based classifier (Python)
│
├── filter_programs/          # DSP filter utilities and preprocessing scripts
│
├── at_home_recording/        # Scripts for capturing training/test audio at home
│
├── drone_data/               # Raw and processed audio dataset
│
├── visualization/            # Data visualization and analysis notebooks/scripts
│
└── __pycache__/              # Python bytecode cache (auto-generated)
```

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                        ESP32 / Arduino                      │
│                                                             │
│  I2S Mic ──► 8192-sample Window ──► FFT ──► Peak in        │
│                                             175–500 Hz?     │
│                                                ↓ YES        │
│                                        Stream 3s WAV        │
│                                        over Serial (921600) │
└────────────────────────────────────┬────────────────────────┘
                                     │ USB Serial
                                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      Host Computer (Python)                 │
│                                                             │
│  Receive WAV ──► Preprocess ──► CNN / LR Model ──► DRONE   │
│                                                    or NOT   │
└─────────────────────────────────────────────────────────────┘
```

---

## Hardware Requirements

| Component | Details |
|-----------|---------|
| Microcontroller | ESP32 (tested; any board supporting I2S should work) |
| Microphone | I2S MEMS microphone (e.g., INMP441, SPH0645) |
| Wiring | WS → GPIO 18, SD → GPIO 38, SCK → GPIO 17 |
| USB Cable | For serial communication with host (baud: 921600) |

### Wiring Diagram

```
I2S Mic Pin    →    ESP32 GPIO
─────────────────────────────
WS  (Word Select)  → GPIO 18
SD  (Serial Data)  → GPIO 38
SCK (Bit Clock)    → GPIO 17
VDD                → 3.3V
GND                → GND
```

---

## Software Requirements

### Arduino
- [Arduino IDE](https://www.arduino.cc/en/software) or PlatformIO
- [arduinoFFT library](https://github.com/kosme/arduinoFFT) — install via Arduino Library Manager
- ESP32 board support package

### Python
- Python 3.8+
- Dependencies (install via pip):

```bash
pip install numpy scipy librosa scikit-learn tensorflow keras matplotlib
```

---

## Getting Started

### Arduino / Embedded Setup

1. **Install dependencies** in Arduino IDE:
   - Go to **Sketch → Include Library → Manage Libraries**
   - Search for `arduinoFFT` and install it
   - Install the ESP32 board package from Board Manager

2. **Wire your microphone** to the ESP32 per the wiring table above.

3. **Open `filterArduino.ino`** in Arduino IDE.

4. **Upload** to your ESP32. Open the Serial Monitor at **921600 baud** to observe status messages.

5. The device starts in **ARMED** mode automatically and prints:
   ```
   READY
   ARMED
   Listening for peak in 175 - 500 Hz
   ```

6. On each 8192-sample analysis window, it prints the detected peak frequency:
   ```
   PEAK 312.5
   ```
   When a drone is detected, it prints detection info and begins streaming WAV data.

---

### Python ML Pipeline

1. **Clone the repository:**
   ```bash
   git clone https://github.com/CSCE-483-Spring-2026/Drone-Detection.git
   cd Drone-Detection
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Record or collect audio data** using scripts in `at_home_recording/`.

4. **Train a classifier:**
   - For CNN: see `cnn_prediction/`
   - For Logistic Regression: see `lr_prediction/`

5. **Run inference** on incoming serial audio or saved WAV files using the appropriate prediction script.

---

## Module Documentation

### `filterArduino.ino`

The primary embedded firmware. This sketch runs the full real-time detection pipeline on an ESP32.

**Key Constants:**

| Constant | Value | Purpose |
|----------|-------|---------|
| `SAMPLING_RATE` | 16000 Hz | I2S audio sample rate |
| `WINDOW_SIZE` | 8192 | FFT window length (samples) |
| `I2S_BUFFER_LEN` | 256 | I2S read buffer size |
| `DRONE_LOW` | 175.0 Hz | Low end of drone frequency band |
| `DRONE_HIGH` | 500.0 Hz | High end of drone frequency band |
| `RECORD_SECONDS` | 3 | Duration of triggered recording |

**States:**

- `ARMED` — Listening and running FFT analysis on incoming audio windows.
- `RECORDING` — Drone detected; streaming a WAV file over serial.
- `STOPPED` — Idle; waiting for an `arm` command.

**Serial Commands:**

| Command | Effect |
|---------|--------|
| `c` | Arm the detector (start listening) |
| `s` | Stop the detector |

**Serial Output Messages:**

| Message | Meaning |
|---------|---------|
| `READY` | Device initialized |
| `ARMED` | Now listening for drones |
| `STOPPED` | Detection halted |
| `PEAK <freq>` | Current window peak frequency (Hz) |
| `DETECTED <freq>` | Drone signature found at this frequency |
| `WAVSIZE <bytes>` | Size of incoming WAV data in bytes |
| `DONE` | Recording complete; re-arming |

**Core Functions:**

- `detectDroneWindow()` — Applies DC removal, Hamming windowing, FFT, and checks if the peak falls in the drone frequency range.
- `startTriggeredRecording()` — Writes a WAV header to serial and begins streaming the current window + future audio.
- `pushRecordedSample()` — Converts 24-bit I2S samples to 16-bit PCM and sends them over serial.
- `armDetector()` / `stopDetector()` — State management.

---

### `arduino/`

Contains supplementary Arduino sketches, such as raw I2S capture utilities, calibration tools, or alternative filter implementations.

---

### `cnn_prediction/`

Python module for training and running a Convolutional Neural Network to classify drone audio.

- Loads audio segments (likely Mel spectrograms or MFCCs as input features)
- Trains a CNN model using Keras/TensorFlow
- Outputs a binary classification: **drone** or **non-drone**

---

### `lr_prediction/`

Python module for a Logistic Regression classifier — a lighter-weight alternative to the CNN.

- Uses handcrafted audio features (e.g., FFT statistics, spectral centroid, zero-crossing rate)
- Trained with scikit-learn
- Useful for fast inference or resource-constrained environments

---

### `filter_programs/`

DSP utility scripts for signal preprocessing, including:

- Bandpass filtering around the drone frequency range
- Noise reduction / smoothing
- Feature extraction pipelines for ML training

---

### `at_home_recording/`

Scripts to record labeled audio samples for the training dataset. Useful for capturing drone audio (positive samples) and ambient noise (negative samples) in different environments.

---

### `drone_data/`

The audio dataset directory. Likely contains:

- Labeled WAV files (drone / non-drone)
- Possibly organized into `train/`, `test/`, or `val/` subdirectories
- May include metadata files describing each recording

---

### `visualization/`

Scripts and notebooks for exploring and visualizing:

- FFT spectra of drone vs. non-drone audio
- Spectrogram comparisons
- Model performance metrics (confusion matrices, ROC curves)

---

## Signal Processing Details

The embedded FFT pipeline operates as follows:

1. **I2S Capture:** Audio is read from the MEMS microphone at 16 kHz, 32-bit stereo (left channel used as mono).
2. **Windowing:** 8192 samples are accumulated before analysis begins.
3. **DC Removal:** Mean offset is subtracted to remove bias.
4. **Hamming Window:** Applied to reduce spectral leakage.
5. **FFT:** Frequency spectrum is computed via `ArduinoFFT`.
6. **Peak Detection:** The dominant frequency (`majorPeak()`) is extracted.
7. **Threshold Check:** If the peak falls between 175 Hz and 500 Hz, a drone is declared detected.

The 8192-point FFT at 16 kHz gives a **frequency resolution of ~1.95 Hz per bin**, providing good precision for identifying drone motor harmonics.

---

## Communication Protocol

When a drone is detected, the ESP32 streams a complete WAV file over serial at **921600 baud**:

```
DETECTED <peak_freq>\n
WAVSIZE <byte_count>\n
[44-byte WAV header]
[raw 16-bit PCM audio, RECORD_SAMPLES * 2 bytes]
DONE\n
```

The host Python script should:
1. Listen on the serial port for `DETECTED` messages.
2. Read `WAVSIZE` to know how many bytes to receive.
3. Collect the WAV header + PCM data.
4. Pass the reconstructed WAV to the classifier.

---

## Contributing

This project is part of **CSCE 483 – Spring 2026** at Texas A&M University.

To contribute:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to your branch: `git push origin feature/your-feature`
5. Open a Pull Request.

Please keep code well-commented and consistent with the existing style. For major changes, open an issue first to discuss your approach.

---

*Built for CSCE 483 – Spring 2026 | Texas A&M University*
