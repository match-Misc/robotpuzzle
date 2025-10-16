# SPEC.md: Simple OpenCV Jigsaw Puzzle Solver

## 1\. Overview

This document specifies a simple Python script that solves a 2D jigsaw puzzle. The script uses **OpenCV** to perform all image processing and visualization.

The process is:

1.  Analyze a "solved" puzzle image (`puzzle_solved.png`) to find the shape and target position of every piece.
2.  Analyze an "unsolved" puzzle image (e.g., `puzzle_test01.png`) to find all scattered pieces.
3.  Match the scattered pieces to the target pieces based on their shape.
4.  Show all steps visually using `cv2.imshow` windows and print the final position map to the console.

## 2\. Technical Stack

  * **Language:** Python 3
  * **Libraries:**
      * `opencv-python` (for all image processing, contour detection, shape matching, and visualization)
      * `numpy` (for image array manipulation)

## 3\. Application Workflow

The script will run as a single process, showing the visualization steps one by one.

### Phase 1: Target Layout Analysis

1.  **Input:** Load the target image (`puzzle_solved.png`).
2.  **Processing:**
      * Convert the image to grayscale (`cv2.cvtColor`).
      * Apply an inverted binary threshold to get the piece outlines from the white background. Use `cv2.threshold` with `cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU`.
      * Find all piece contours using `cv2.findContours` with `cv2.RETR_EXTERNAL`.
      * For each `target_contour` found:
          * Calculate its shape descriptor using Hu Moments (`cv2.HuMoments`).
          * Calculate its center position (centroid) using `cv2.moments`.
          * Store this `(target_contour, target_centroid, target_shape)` in a `target_pieces` list.
3.  **Visualization (Step 1):**
      * Draw all `target_contours` in green on a copy of the original `puzzle_solved.png`.
      * Display this image in an OpenCV window: `cv2.imshow("1 - Target Layout", image_with_contours)`.

### Phase 2: Input Piece Detection

1.  **Input:** Load the input image (e.g., `puzzle_test01.png`).
2.  **Processing:**
      * Repeat the exact same processing steps as in Phase 1 (grayscale, threshold, find contours).
      * For each `input_contour` found:
          * Calculate its shape descriptor (`cv2.HuMoments`).
          * Calculate its centroid (`cv2.moments`).
          * Store this `(input_contour, input_centroid, input_shape)` in an `input_pieces` list.
3.  **Visualization (Step 2):**
      * Draw all `input_contours` in red on a copy of the original `puzzle_test01.png`.
      * Display this image: `cv2.imshow("2 - Input Pieces", image_with_contours)`.

### Phase 3: Shape Matching & Output

1.  **Matching:**
      * Create an empty list called `solution_map`.
      * Loop through each `input_piece` in the `input_pieces` list.
      * Inside the loop, compare its `input_shape` to *all* shapes in the `target_pieces` list using `cv2.matchShapes`.
      * Find the `target_piece` that has the lowest match score (best match).
      * Add the matched pair to the `solution_map`: `solution_map.append((input_centroid, target_centroid, input_contour))`
2.  **Console Output:**
      * Loop through the `solution_map`.
      * Print the required mapping:
        ```
        Piece at (Current_X, Current_Y) -> maps to -> (Target_X, Target_Y)
        Piece at (812, 604) -> maps to -> (150, 50)
        ...
        ```
3.  **Visualization (Step 3):**
      * Create a new, blank white image the same size as the target image.
      * Iterate through the `solution_map`. For each `(input_centroid, target_centroid, input_contour)`:
          * Calculate the position offset needed: `delta = target_centroid - input_centroid`.
          * Translate the `input_contour` by this `delta`.
          * Draw the translated `input_contour` (filled) onto the blank white image.
      * Display the final reconstructed puzzle: `cv2.imshow("3 - Solved Puzzle", solved_image)`.
      * Call `cv2.waitKey(0)` to keep all windows open.

## 4\. Test Data

  * **Target Layout:** `puzzle_solved.png`
  * **Input Test Case 1:** `puzzle_test01.png`
  * **Input Test Case 2:** `puzzle_test02.png`

## 5\. Key Assumptions

  * **High Contrast:** Pieces are on a uniform, high-contrast (mostly white) background.
  * **No Overlap:** Pieces in the input images do not touch or overlap.
  * **Rotation Invariance:** `cv2.HuMoments` is used, which is invariant to rotation, scale, and translation.
  * **Unique Shapes:** Each piece has a unique enough shape to be distinguished by `cv2.matchShapes`.