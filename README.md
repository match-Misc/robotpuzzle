# 🤖 Robot Puzzle Solver 🧩

A computer vision system that automatically solves jigsaw puzzles using webcam input and advanced image processing techniques. 🚀

## 🎯 What It Does

This project provides an automated puzzle-solving assistant that can:
- **🔍 Detect puzzle pieces** in real-time from webcam feed
- **🔗 Match pieces** to their correct positions using shape analysis
- **👨‍🏫 Guide users** through puzzle assembly with visual feedback
- **📏 Support multiple puzzle sizes** (12-piece and 24-piece puzzles)
- **🔄 Provide live solving mode** for continuous assistance

The system uses computer vision algorithms to analyze puzzle piece shapes, calculate their orientations, and match them to pre-computed solution data. 🧠

## ⚙️ How It Works

### 🏗️ Core Components

1. **📷 Camera Calibration** (`puzzle_capture.py`)
    - Calibrates the webcam to correct perspective distortion 📐
    - Allows manual selection of puzzle boundary corners 👆
    - Saves calibration data to `configs/config.json` 💾

2. **🔎 Piece Detection** (`detect_pieces.py`)
    - Processes puzzle images to identify individual pieces 🖼️
    - Uses contour analysis and morphological operations 🔬
    - Computes Hu moments for shape matching 📊
    - Calculates centroids and orientations 🧭
    - Outputs piece data in JSON format (`configs/Puzzle_N.json`) 📄

3. **🎮 Live Solving** (`puzzle_solver.py`)
    - GUI application built with CustomTkinter 🖥️
    - Real-time webcam capture and processing 📹
    - Piece detection and matching using Hungarian algorithm 🧮
    - Interactive hover feedback showing target positions 🖱️
    - Live mode for continuous puzzle assistance 🔄

### 🛠️ Technical Approach

The system employs several computer vision techniques:

- **🔄 Perspective Transformation**: Corrects camera angle distortion
- **📏 Contour Detection**: Identifies piece boundaries using edge detection
- **📐 Shape Descriptors**: Hu moments provide rotation-invariant shape matching
- **🎯 Assignment Problem**: Hungarian algorithm optimally matches detected pieces to solution
- **⚡ Real-time Processing**: Optimized for live webcam input at 30+ FPS

### 📊 Data Flow

```
Webcam Feed → Perspective Transform → Piece Detection → Shape Matching → Solution Display
     ↓              ↓                      ↓              ↓              ↓
  Raw Frame    Corrected View        Contours      Hu Moments    Target Positions
```

## 📋 Requirements

- 🐍 Python 3.7+
- 📹 OpenCV (`cv2`)
- 🔢 NumPy
- 🎨 CustomTkinter
- 🔬 SciPy (for Hungarian algorithm)

## 🚀 Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/robotpuzzle.git
    cd robotpuzzle
    ```

2. Install dependencies:
    ```bash
    pip install opencv-python numpy customtkinter scipy
    ```

## 📖 Usage

### 1. 🛠️ Prepare Solution Data

First, create solution data for your puzzle:

```bash
python detect_pieces.py path/to/solved_puzzle_image.png 24
```

This generates `configs/Puzzle_24.json` with piece data. 📄

### 2. 📷 Calibrate Camera

```bash
python puzzle_capture.py
```

- Press 'c' to capture a frame 📸
- Click the 4 corners of your puzzle area 👆
- Press 's' to save calibration 💾

### 3. 🧩 Solve Puzzles

Launch the GUI solver:

```bash
python puzzle_solver.py
```

- Select puzzle size (12 or 24 pieces) 📏
- Click "Calibrate Camera" if needed 📷
- Click "Capture & Solve" for single-shot solving 📸
- Use "Start Live Mode" for continuous assistance 🔄

## 📁 Project Structure

```
robotpuzzle/
├── puzzle_solver.py          # Main GUI application 🖥️
├── puzzle_capture.py         # Camera calibration tool 📷
├── detect_pieces.py          # Piece detection from images 🔍
├── live_puzzle_highlighter.py # Basic live piece highlighting ✨
├── live_puzzle_solver.py     # Command-line live solver 💻
├── configs/                  # Configuration files directory ⚙️
│   ├── config.json           # Camera calibration data 📄
│   ├── Puzzle_12.json        # 12-piece puzzle solution data 🧩
│   └── Puzzle_24.json        # 24-piece puzzle solution data 🧩
├── Solutions/               # Solved puzzle images 🖼️
├── examples/                # Example images and scripts 📚
└── README.md               # This file 📖
```

## 🔬 Algorithm Details

### 🧩 Piece Matching

Pieces are matched using Hu moment invariants, which are:
- **📍 Translation invariant**: Position doesn't affect matching
- **📏 Scale invariant**: Size differences are normalized
- **🔄 Rotation invariant**: Shape matching works regardless of orientation

The Hungarian algorithm ensures optimal assignment of detected pieces to solution positions. 🧮

### 🧭 Orientation Calculation

Piece orientation uses moment-based analysis of filled contours, providing robust angle estimation even for irregular shapes. 📐