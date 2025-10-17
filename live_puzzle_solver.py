import json
import sys

import cv2
import numpy as np


def load_config():
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(
            "Error: config.json not found. Please run puzzle_capture.py first to calibrate."
        )
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: Invalid config.json file.")
        sys.exit(1)


def set_webcam_settings(cap, brightness=50, contrast=100, saturation=0):
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    cap.set(cv2.CAP_PROP_CONTRAST, contrast)
    cap.set(cv2.CAP_PROP_SATURATION, saturation)
    # Set resolution to Full HD
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


def apply_perspective_transform(frame, M, output_size):
    if M is None or output_size is None:
        return None
    # Apply perspective warp
    warped = cv2.warpPerspective(frame, np.array(M), tuple(output_size))
    return warped


def stretch_to_16_10(image):
    height, width = image.shape[:2]
    target_aspect = 16 / 10  # 1.6
    current_aspect = width / height

    if current_aspect > target_aspect:
        # Image is too wide, stretch height
        new_height = int(width / target_aspect)
        stretched = cv2.resize(
            image, (width, new_height), interpolation=cv2.INTER_LINEAR
        )
    else:
        # Image is too tall, stretch width
        new_width = int(height * target_aspect)
        stretched = cv2.resize(
            image, (new_width, height), interpolation=cv2.INTER_LINEAR
        )

    return stretched


def detect_pieces(image, num_pieces):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection with fixed thresholds
    edges = cv2.Canny(gray, 50, 150)

    # Apply morphological closing to connect gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter contours by minimum area to exclude small edge fragments
    min_area = 1000
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Sort by area descending and take top num_pieces
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:num_pieces]

    # Compute Hu moments, centroids, and orientations for each contour
    detected_pieces = []
    for cnt in contours:
        M = cv2.moments(cnt)
        hu_moments = cv2.HuMoments(M).flatten()
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        # Calculate orientation using minAreaRect
        rect = cv2.minAreaRect(cnt)
        angle = rect[2]  # Angle of rotation
        detected_pieces.append((cnt, hu_moments, (cx, cy), angle))

    return detected_pieces


def match_pieces(detected_pieces, target_pieces):
    solution_map = []
    used_targets = set()

    for detected_cnt, detected_hu, detected_centroid, detected_angle in detected_pieces:
        best_match = None
        best_score = float("inf")
        best_target_centroid = None
        best_target_angle = None

        for target in target_pieces:
            if target["id"] in used_targets:
                continue
            target_hu = np.array(target["hu_moments"])
            # Compare Hu moments using Euclidean distance
            score = np.linalg.norm(detected_hu - target_hu)
            if score < best_score:
                best_score = score
                best_match = target["id"]
                best_target_centroid = target["centroid"]
                best_target_angle = target["orientation"]

        if best_match is not None:
            used_targets.add(best_match)
            # Compute translation and rotation
            translation = (
                best_target_centroid[0] - detected_centroid[0],
                best_target_centroid[1] - detected_centroid[1],
            )
            rotation = best_target_angle - detected_angle
            solution_map.append(
                (detected_centroid, best_target_centroid, translation, rotation)
            )

    return solution_map


def highlight_pieces(image, detected_pieces):
    # Generate unique colors using HSV color space for distinct hues
    num_pieces = len(detected_pieces)
    colors = []
    for i in range(num_pieces):
        hue = int(180 * i / num_pieces)  # Hue from 0 to 180
        hsv_color = np.uint8([[[hue, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])))

    # Create a copy of the original image for highlighting
    highlighted = image.copy()

    # Draw each contour filled with its assigned unique color
    for i, (cnt, _, _, _) in enumerate(detected_pieces):
        cv2.drawContours(highlighted, [cnt], -1, colors[i], -1)  # -1 for filled

    return highlighted, num_pieces


def load_target_pieces(puzzle_size):
    try:
        with open(f"Puzzle_{puzzle_size}.json", "r") as f:
            target_pieces = json.load(f)
        return target_pieces
    except FileNotFoundError:
        print(f"Error: Puzzle_{puzzle_size}.json not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid Puzzle_{puzzle_size}.json file.")
        sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print("Usage: python live_puzzle_solver.py <puzzle_size>")
        sys.exit(1)

    puzzle_size = int(sys.argv[1])

    # Load target pieces data
    target_pieces = load_target_pieces(puzzle_size)

    # Load calibration data
    config = load_config()
    M = np.array(config["M"]) if config["M"] else None
    output_size = tuple(config["output_size"]) if config["output_size"] else None

    if M is None or output_size is None:
        print("Error: Invalid calibration data in config.json")
        return

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # Set webcam settings
    set_webcam_settings(cap, brightness=50, contrast=100, saturation=0)

    print("Starting live puzzle highlighting. Press 'q' to quit.")

    cv2.namedWindow("Original Feed")
    cv2.namedWindow("Transformed Puzzle")
    cv2.namedWindow("Highlighted Pieces")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # Show original feed
        cv2.imshow("Original Feed", frame)

        # Apply perspective transform
        transformed = apply_perspective_transform(frame, M, output_size)
        if transformed is not None:
            cv2.imshow("Transformed Puzzle", transformed)

            # Stretch to 16:10 aspect ratio
            stretched = stretch_to_16_10(transformed)

            # Detect pieces and compute Hu moments
            detected_pieces = detect_pieces(stretched, puzzle_size)

            # Match detected pieces to target pieces
            solution_map = match_pieces(detected_pieces, target_pieces)

            # Output the centroid mapping to console
            print("Current piece mappings:")
            for (
                current_centroid,
                target_centroid,
                translation,
                rotation,
            ) in solution_map:
                print(
                    f"Piece at ({current_centroid[0]}, {current_centroid[1]}) -> maps to -> ({target_centroid[0]}, {target_centroid[1]}) | Translate by ({translation[0]}, {translation[1]}) | Rotate by {rotation:.2f} degrees"
                )

            # Highlight pieces
            highlighted, num_pieces = highlight_pieces(stretched, detected_pieces)

            # Show highlighted pieces
            cv2.imshow("Highlighted Pieces", highlighted)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
