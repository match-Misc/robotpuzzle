# 🤖 Robot Puzzle Solver - Technical Specification

## 📋 Overview

The Robot Puzzle Solver is an automated system that uses computer vision and robotic manipulation to solve jigsaw puzzles. The system detects puzzle pieces from webcam input, matches them to their correct positions using advanced image processing algorithms, and guides a robotic arm to pick and place the pieces, creating a competitive puzzle-solving experience against human players.

### 🎯 Core Functionality

- **Real-time Piece Detection**: Processes webcam feed to identify and locate puzzle pieces
- **Shape Matching**: Uses Hu moment invariants and Hungarian algorithm for optimal piece-to-position matching
- **Robotic Control**: Communicates pose data to a robotic arm for automated piece placement
- **Game Management**: Provides NFC-based player identification, timing, and scoring system
- **Live Assistance**: Offers continuous visual feedback during puzzle solving

### 🏗️ System Architecture

The application consists of three main components:

1. **Computer Vision Module** (`puzzle_gui.py`)
   - Webcam capture and image processing
   - Piece detection and shape analysis
   - Solution matching and pose calculation

2. **Robot Communication Module** (`send_pose.py`)
   - TCP socket server for robot communication
   - Pose data transmission and feedback handling

3. **Game Management Module** (`gamemanager.py`)
   - Player registration via NFC
   - Game timing and control
   - Result submission to external server

## 📋 Requirements

### 🔧 Hardware Requirements

- **Camera**: High-resolution webcam (4K recommended) for piece detection
- **Computer**: Windows/Linux system with webcam support
- **Robot Arm**: TCP/IP controllable robotic manipulator (UR series compatible)
- **NFC Reader**: USB NFC reader for player identification (optional for basic operation)
- **Network**: Ethernet connection for robot communication

### 💻 Software Requirements

- **Operating System**: Windows 11, Linux (Ubuntu 20.04+)
- **Python**: Version 3.13 or higher
- **Dependencies**:
  - `opencv-python` (4.x) - Computer vision processing
  - `numpy` (1.21+) - Numerical computations
  - `customtkinter` - Modern GUI framework
  - `Pillow` (9.x) - Image processing
  - `scipy` (1.7+) - Scientific computing (Hungarian algorithm)
  - `pyserial` (3.5+) - Serial communication for NFC
  - `requests` (2.32.5+) - HTTP communication for game results

### 📊 Data Requirements

- **Calibration Data**: Camera calibration matrix stored in `configs/config.json`
- **Solution Data**: Pre-computed puzzle solutions in `configs/Puzzle_N.json` format
- **Piece Masks**: Binary masks for each puzzle piece in `configs/piece_N_M_mask.png`
- **Offset Maps**: Collision avoidance offsets in `configs/offsetmap_N.json`
- **Solved Images**: Reference images in `solutions/Puzzle_N.png`

## 🔄 System Workflow

### 1. 🛠️ Setup Phase

#### Camera Calibration
```bash
python puzzle_calibration.py
```
- Capture reference frame of puzzle area
- Manually select 4 corner points of puzzle boundary
- Compute perspective transformation matrix
- Save calibration to `configs/config.json`

#### Solution Data Preparation
```bash
python puzzle_solution.py path/to/solved_puzzle.png N
```
- Process solved puzzle image
- Extract individual piece contours and shapes
- Compute Hu moments for shape matching
- Generate piece masks and solution data
- Output: `configs/Puzzle_N.json`, piece mask images

### 2. 🎮 Game Phase

#### Player Registration
- NFC tag scanning for player identification
- Name assignment and storage on external server
- Difficulty selection (12 or 24 pieces)

#### Puzzle Solving
1. **Image Capture**: Capture current puzzle state via webcam
2. **Perspective Correction**: Apply calibration transform to correct camera angle
3. **Piece Detection**:
   - Convert to grayscale and apply thresholding
   - Find contours using OpenCV
   - Filter by area and compute shape descriptors
   - Calculate piece orientations using PCA
4. **Shape Matching**:
   - Compare detected pieces to solution database using Hu moments
   - Solve assignment problem with Hungarian algorithm
   - Handle piece orientation ambiguity (normal vs 180° rotation)
5. **Pose Calculation**:
   - Transform piece positions from camera coordinates to robot coordinates
   - Apply collision avoidance offsets
   - Generate pickup and placement poses

#### Robot Execution
- Establish TCP connection to robot controller
- Send pose pairs (pickup → target) sequentially
- Wait for placement confirmation before next piece
- Handle communication timeouts and errors

### 3. 📊 Results Phase

- Track completion time and piece placement progress
- Submit results to external server with player data
- Reset system for next game

## 🔧 Technical Specifications

### 📷 Computer Vision Pipeline

#### Image Processing Parameters
- **Resolution**: 3840×2160 (4K) input, processed at 1920×1080
- **Frame Rate**: 30 FPS for live mode
- **Thresholding**: Adaptive Gaussian thresholding with manual override
- **Morphology**: 5×5 kernel morphological operations

#### Shape Analysis
- **Hu Moments**: 7 invariant moments for rotation/scale/translation invariance
- **Orientation**: PCA-based angle calculation with skewness disambiguation
- **IoU Validation**: Intersection-over-union comparison for orientation verification

#### Coordinate Systems
- **Camera Space**: Pixel coordinates from webcam
- **Pickup Space**: Millimeters relative to pickup frame (520×325mm)
- **Target Space**: Millimeters relative to DIN A4 solution (297×210mm)

### 🤖 Robot Communication Protocol

#### TCP Socket Interface
- **Port**: 30020
- **Protocol**: ASCII text commands
- **Message Format**: `(pickup_x, pickup_y, pickup_angle, target_x, target_y, target_angle)`

#### Communication Flow
```
Robot → Server: "ok" (ready signal)
Server → Robot: "(px, py, pa, tx, ty, ta)" (pose command)
Robot → Server: "placed" (completion confirmation)
... (repeat for each piece)
Robot → Server: "human" (human intervention signal)
```

### 🎯 Game Management

#### Difficulty Levels
- **Easy (leicht)**: 12 pieces
- **Medium (mittel)**: 24 pieces
- **Hard (schwer)**: 24 pieces (same as medium)

#### Timing System
- **Countdown**: 3-second preparation period
- **Stopwatch**: Centisecond precision timing
- **Auto-stop**: Triggers on robot completion or human intervention

#### Scoring API
```json
{
  "nfc_id": "string",
  "time": 123.45,
  "difficulty": "Leicht|Mittel|Schwer"
}
```

## 📁 Data Formats

### Camera Calibration (`configs/config.json`)
```json
{
  "M": [[float, float, float], [float, float, float]],
  "output_size": [int, int]
}
```

### Puzzle Solution (`configs/Puzzle_N.json`)
```json
[
  {
    "id": int,
    "centroid": [float, float],
    "orientation": float,
    "hu_moments": [float, float, float, float, float, float, float],
    "area": float
  }
]
```

### Solved Puzzle Data (`solved_puzzle_N_with_offsets.json`)
```json
[
  {
    "id": int,
    "pickup_x": float,
    "pickup_y": float,
    "pickup_angle": float,
    "target_x": float,
    "target_y": float,
    "target_angle": float,
    "offset_x": float,
    "offset_y": float
  }
]
```

### Offset Map (`configs/offsetmap_N.json`)
```json
{
  "map": [[int, int, ...], [int, int, ...], ...],
  "offset": float
}
```

## 🚀 Usage Instructions

### Basic Operation
1. Calibrate camera: `python puzzle_calibration.py`
2. Launch GUI: `python puzzle_gui.py`
3. Select puzzle size and calibrate if needed
4. Capture image and solve, or use live mode

### Game Mode
1. Launch game manager: `python gamemanager.py`
2. Scan NFC tag and enter player name
3. Select difficulty and start game
4. Monitor progress and timing
5. Record winner and submit results

### Robot Integration
- Ensure robot controller connects to PC on port 30020
- Robot must send "ok" when ready
- Robot must send "placed" after each successful placement
- Robot can send "human" to indicate human intervention

## 🔍 Troubleshooting

### Common Issues
- **Camera not detected**: Check webcam permissions and drivers
- **Poor detection**: Adjust threshold values, ensure good lighting
- **Robot connection fails**: Verify network settings and firewall
- **NFC not working**: Check USB connection and driver installation

### Performance Optimization
- Use consistent lighting conditions
- Ensure puzzle pieces are well-separated
- Calibrate camera regularly
- Monitor system resources during live mode

## 🔮 Future Enhancements

- Multi-camera support for larger puzzles
- Machine learning-based piece recognition
- Advanced collision avoidance algorithms
- Real-time difficulty adjustment
- Multiplayer competitive modes