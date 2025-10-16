# SPEC.md: Python Jigsaw Puzzle Solver

## 1\. Overview

This document specifies a Python application designed to solve a 2D jigsaw puzzle. The system operates by first analyzing a **target layout** (an image of the solved puzzle) to create a map of all piece shapes and their final positions. It then analyzes an **input image** (containing all pieces scattered randomly) to detect and extract each piece.

Using computer vision techniques, the application will match the shapes of the scattered pieces to the shapes in the target layout. The final output will be a mapping that provides the current (scattered) position and the required target (solved) position for each piece.

A simple graphical user interface (GUI) will be provided to visualize these steps.

## 2\. Key Features

  * **Target Layout Analysis:** One-time processing of a "solution" image to build a database of piece shapes, their contours, and their final (x, y) coordinates.
  * **Input Piece Detection:** Ingest an image of scattered pieces and use segmentation and contour detection to isolate each individual piece.
  * **Shape Matching:** Compare each detected piece's shape against the target layout database to find its correct identity and orientation.
  * **Position Mapping:** Output a structured data file (JSON) that lists each piece, its current coordinates, and its target coordinates.
  * **Visualization:** A GUI to upload images, run the analysis, and visualize the matching process and final solution.

## 3\. Technical Stack

  * **Language:** Python 3.9+
  * **Computer Vision:** `opencv-python` (OpenCV) for image processing, thresholding, Canny edge detection, contour finding (`cv2.findContours`), and shape matching (`cv2.matchShapes`).
  * **Numerical & Array Ops:** `numpy` for all image array manipulations.
  * **User Interface:** `Streamlit` for a simple, web-based interactive GUI to visualize the steps.
  * **Data Interchange:** `JSON` for storing the output map.

## 4\. Application Workflow

The application will operate in three distinct phases.

### Phase 1: Target Layout Analysis (Setup)

This phase is run once per puzzle layout to create the "ground truth" solution map.

1.  **Input:** A clean, high-contrast image of the *solved* puzzle (e.g., `target_layout.png`).
2.  **Process:**
      * Load the image in grayscale.
      * Apply Canny edge detection or adaptive thresholding to clearly identify the "seams" between pieces.
      * Use `cv2.findContours` to find the outline of *every* piece in the solved layout.
      * For each contour (piece) found:
          * Calculate a unique shape descriptor (e.g., Hu Moments via `cv2.HuMoments`).
          * Calculate its centroid or bounding box center. This is its `target_position`.
          * Assign a unique `target_piece_id` (e.g., "piece\_001").
      * 
3.  **Output:** A JSON file (`target_map.json`) containing the database of all target pieces.

### Phase 2: Input Piece Detection (Runtime)

This phase is run every time a new "unsolved" image is provided.

1.  **Input:** An image of all puzzle pieces scattered, ideally on a uniform, high-contrast background (e.g., `input_pieces.png`).
2.  **Process:**
      * Load the image.
      * Apply segmentation (e.g., binary thresholding) to isolate the pieces from the background.
      * Use `cv2.findContours` to find the outline of *each* individual scattered piece.
      * 
      * For each contour (piece) found:
          * Calculate its shape descriptor (using the *same method* as in Phase 1).
          * Calculate its current centroid or bounding box center. This is its `current_position`.
          * Store this information temporarily in a list.

### Phase 3: Matching & Mapping (Runtime)

This phase connects the detected pieces (Phase 2) to the target layout (Phase 1).

1.  **Input:** The `target_map.json` and the list of detected pieces from Phase 2.
2.  **Process:**
      * Create an empty list for the final `solution_map`.
      * For each detected piece from the input image:
          * Compare its shape descriptor against *all* unused shape descriptors in the `target_map.json` using `cv2.matchShapes`.
          * The `target_piece_id` with the lowest match score (closest match) is considered the correct identity.
          * Create a new record containing:
              * `detected_piece_id` (e.g., "input\_001")
              * `current_position` (from Phase 2)
              * `matched_target_id` (the best match from `target_map.json`)
              * `target_position` (from the matched target piece)
              * `match_confidence` (the `cv2.matchShapes` score)
          * Add this record to the `solution_map`.
3.  **Output:** The final `solution_map.json` file.

## 5\. Data Structures (I/O)

### `target_map.json` (Output of Phase 1)

A list of objects, where each object represents one piece in the solved puzzle.

```json
[
  {
    "target_piece_id": "piece_001",
    "target_position": { "x": 50, "y": 50 },
    "shape_descriptor": [0.12, 0.004, ...],
    "contour_data": [[20, 20], [21, 20], ...]
  },
  {
    "target_piece_id": "piece_002",
    "target_position": { "x": 150, "y": 50 },
    "shape_descriptor": [0.15, 0.007, ...],
    "contour_data": [[120, 20], [121, 20], ...]
  }
]
```

### `solution_map.json` (Final Output of Phase 3)

This directly addresses the user requirement to map current and target positions.

```json
[
  {
    "detected_piece_id": "input_A",
    "current_position": { "x": 845, "y": 620 },
    "matched_target_id": "piece_002",
    "target_position": { "x": 150, "y": 50 },
    "match_confidence": 0.0012
  },
  {
    "detected_piece_id": "input_B",
    "current_position": { "x": 112, "y": 910 },
    "matched_target_id": "piece_001",
    "target_position": { "x": 50, "y": 50 },
    "match_confidence": 0.0009
  }
]
```

## 6\. Visualization (GUI)

The `Streamlit` interface will provide a simple step-by-step workflow:

1.  **Home Page:**

      * `st.title("Jigsaw Puzzle Solver")`
      * **File Uploader 1:** "Upload Target Layout (Solved Puzzle)" (Default: `puzzle_solved.png`)
      * **File Uploader 2:** "Upload Input Image (Scattered Pieces)" (Allow selection of `puzzle_test01.png`, `puzzle_test02.png`, or custom upload)
      * **Button:** "Run Analysis"

2.  **Analysis Page (After 'Run Analysis' is clicked):**

      * `st.header("Step 1: Target Layout Analysis")`
      * `st.image(target_image_with_contours)` (Displays the solved puzzle with all detected piece outlines drawn on it).
      * `st.header("Step 2: Input Piece Detection")`
      * `st.image(input_image_with_contours)` (Displays the scattered pieces with their detected outlines).
      * `st.header("Step 3: Solution Map")`
      * `st.dataframe(solution_map_df)` (Displays the final `solution_map` in a clean, sortable table).
      * **(Optional) Visual Match:** Display the `target_layout.png` with arrows drawn from each `current_position` to its corresponding `target_position`.

## 7\. Test Data

The application will be developed and tested using the following provided image files:

  * **Target Layout:** `puzzle_solved.png`
  * **Input Test Case 1:** `puzzle_test01.png`
  * **Input Test Case 2:** `puzzle_test02.png`

## 8\. Key Assumptions

  * **Non-overlapping Pieces:** The pieces in the `input_pieces.png` (and test files) are fully separated and do not overlap.
  * **Uniform Background:** The input pieces are placed on a high-contrast, uniform-color background (e.g., white pieces on a black table).
  * **2D Puzzle:** The puzzle is a standard 2D jigsaw. 3D or complex-shaped puzzles are out of scope.
  * **Rotation:** The `cv2.matchShapes` (using Hu Moments) is rotation-invariant. If pieces in the input are rotated, this method should still work. If a non-invariant descriptor is used, rotation handling must be added.
  * **No "False" Pieces:** The input image contains *only* the pieces belonging to the target puzzle and no other artifacts.