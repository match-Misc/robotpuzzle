# 🤖 Robot Puzzle Solver - System Agents

This document describes the key agents and actors in the Robot Puzzle Solver system, their roles, responsibilities, and interactions.

## 📋 Overview

The Robot Puzzle Solver is a multi-agent system that combines computer vision, robotic manipulation, and game management to create an interactive puzzle-solving experience. The system consists of several specialized agents that work together to detect, solve, and place jigsaw puzzle pieces.

## 🎭 System Agents

### 1. **Human Player** 👤
**Role**: Primary user and game participant

**Responsibilities**:
- Initiates game by scanning NFC tag
- Provides player name (if not already registered)
- Selects difficulty level (leicht/mittel/schwer)
- Physically arranges puzzle pieces on the pickup area
- Competes against the robot to complete the puzzle
- Indicates when they have won the game

**Interactions**:
- → NFC Reader Agent (scans NFC tag)
- → Game Manager Agent (provides name, selects difficulty)
- → Robot Agent (competes for puzzle completion)

**Key Data**:
- NFC ID (unique identifier)
- Player name
- Selected difficulty
- Completion time

---

### 2. **Computer Vision Agent** 👁️
**Primary Implementation**: `puzzle_gui.py`

**Role**: Vision processing and puzzle analysis expert

**Responsibilities**:
- Captures real-time webcam feed
- Applies perspective transformation for distortion correction
- Detects individual puzzle pieces using contour analysis
- Computes piece orientations using PCA (Principal Component Analysis)
- Matches detected pieces to solution data using Hu moments
- Provides visual feedback through GUI
- Supports both single-shot and live mode processing

**Key Technologies**:
- OpenCV for image processing
- Hu moment invariants for shape matching
- Hungarian algorithm for optimal piece assignment
- PCA-based orientation calculation

**Interactions**:
- ← Human Player (receives raw puzzle images)
- → Game Manager Agent (provides solved puzzle data)
- → Robot Agent (provides piece poses and orientations)

**Data Flow**:
```
Webcam Feed → Perspective Transform → Piece Detection → Shape Matching → Solution Data
```

---

### 3. **Game Manager Agent** 🎮
**Primary Implementation**: `gamemanager.py`

**Role**: Game orchestration and state management

**Responsibilities**:
- Manages overall game flow and state
- Handles NFC tag scanning and player registration
- Coordinates difficulty selection and puzzle size determination
- Controls game timing (countdown, stopwatch)
- Monitors pose transmission progress
- Handles winner selection and result submission
- Manages pose server lifecycle

**Key Features**:
- 3-second countdown before game start
- Real-time stopwatch during gameplay
- Progress tracking (pieces sent/placed)
- Winner determination (human vs robot)
- Result submission to external server

**Interactions**:
- ← Human Player (receives game inputs)
- → NFC Reader Agent (requests NFC scanning)
- → Computer Vision Agent (receives solved puzzle data)
- → Pose Communication Agent (sends poses to robot)
- → External Server (submits game results)

**Game States**:
1. Player Setup (NFC scan, name entry, difficulty selection)
2. Countdown (3-second preparation)
3. Active Game (pose transmission, timing)
4. Completion (winner selection, result submission)

---

### 4. **NFC Reader Agent** 📱
**Primary Implementation**: `nfc_reader.py`

**Role**: Identity management and player authentication

**Responsibilities**:
- Reads NFC tags via serial communication with D1 Mini device
- Parses NFC data formats (Arduino format, raw hex)
- Communicates with external server for player lookup
- Handles new NFC chip registration
- Manages player name assignment
- Provides player information to game manager

**Key Technologies**:
- Serial communication (COM11, 9600 baud)
- HTTP requests to external server
- NFC data parsing and validation

**Interactions**:
- ← Human Player (reads NFC tags)
- ↔ External Server (player data synchronization)
- → Game Manager Agent (provides player information)

**Server Endpoints**:
- `POST /api/nfc_scan` - Lookup player by NFC ID
- `POST /admin/add_nfc` - Register new NFC chip
- `POST /admin/assign_name` - Assign name to NFC ID

---

### 5. **Pose Communication Agent** 📡
**Primary Implementation**: `send_pose.py`

**Role**: Robotic communication and pose transmission

**Responsibilities**:
- Establishes TCP socket server for robot communication
- Manages robot connection and readiness state
- Transmits piece pickup and placement poses
- Handles robot feedback messages ("ok", "placed", "human")
- Applies collision avoidance offsets to target positions
- Sends zero pose to signal game completion
- Provides callbacks for pose transmission events

**Key Technologies**:
- TCP socket server (port 30020)
- Asynchronous communication handling
- Pose data formatting (6-tuple: x,y,angle pickup + x,y,angle target)

**Interactions**:
- ← Game Manager Agent (receives pose transmission commands)
- ↔ Robot Agent (bidirectional communication)
- → Game Manager Agent (provides transmission status)

**Communication Protocol**:
- Robot → Server: "ok" (ready), "placed" (piece placed), "human" (human won)
- Server → Robot: Pose strings "(px,py,pa,tx,ty,ta)" or predefined responses

---

### 6. **Robot Agent** 🤖
**Role**: Physical manipulation and puzzle assembly

**Responsibilities**:
- Receives pose instructions from pose server
- Moves to pickup locations and grasps puzzle pieces
- Transports pieces to target positions
- Places pieces with correct orientation
- Provides feedback on operation status
- Detects when human player has completed puzzle
- Signals game completion

**Interactions**:
- ↔ Pose Communication Agent (receives commands, sends status)
- ← Human Player (competes for puzzle completion)
- → Game Manager Agent (signals game end conditions)

**Key Capabilities**:
- Precise positioning (x,y coordinates in mm)
- Orientation control (rotation in degrees)
- Pickup and placement operations
- Status feedback ("ok", "placed", "human")

---

### 7. **External Server Agent** 🌐
**Role**: Data persistence and leaderboard management

**Responsibilities**:
- Stores player information (NFC ID ↔ player name mapping)
- Maintains game results and leaderboards
- Provides player lookup by NFC ID
- Accepts game result submissions
- Manages administrative functions (add NFC chips, assign names)

**Interactions**:
- ↔ NFC Reader Agent (player data operations)
- ← Game Manager Agent (game result submissions)

**API Endpoints**:
- `POST /api/nfc_scan` - Player lookup
- `POST /api/puzzle` - Game result submission
- `POST /admin/add_nfc` - Add NFC chip
- `POST /admin/assign_name` - Assign player name

---

## 🔄 System Interactions Flow

### Game Initialization:
```
Human Player → NFC Reader → External Server → Game Manager
                    ↓
Difficulty Selection → Puzzle Size Determination
```

### Puzzle Solving:
```
Computer Vision Agent → Piece Detection → Pose Generation → Pose Communication Agent → Robot Agent
```

### Game Completion:
```
Robot Agent / Human Player → Game Manager → Winner Selection → External Server
```

## 📊 Data Structures

### Piece Pose Data:
```json
{
  "id": 1,
  "pickup_x": 150.5,
  "pickup_y": 200.3,
  "pickup_angle": 45.2,
  "target_x": 297.0,
  "target_y": 210.0,
  "target_angle": 90.0,
  "offset_x": 5.0,
  "offset_y": 3.0
}
```

### Player Data:
```json
{
  "nfc_id": "A1B2C3D4E5F6",
  "player_name": "Alice",
  "exists": true,
  "has_name": true
}
```

### Game Result:
```json
{
  "nfc_id": "A1B2C3D4E5F6",
  "time": 45.67,
  "difficulty": "Mittel"
}
```

## 🎯 Agent Collaboration Patterns

1. **Sequential Processing**: Computer Vision → Pose Generation → Robot Execution
2. **Event-Driven Communication**: Pose server callbacks for status updates
3. **State Synchronization**: Game manager coordinates all agent states
4. **External Data Management**: Server handles persistent player and result data
5. **Real-time Feedback**: Live mode provides continuous visual updates

## 🔧 Technical Architecture

- **Language**: Python 3.7+
- **GUI Framework**: CustomTkinter
- **Computer Vision**: OpenCV
- **Networking**: TCP sockets, HTTP requests
- **Hardware Communication**: Serial (NFC), TCP (Robot)
- **Algorithm**: Hungarian algorithm for assignment, PCA for orientation

This multi-agent architecture enables a seamless interactive experience where computer vision, robotic manipulation, and game management work together to create an engaging human-robot puzzle competition.