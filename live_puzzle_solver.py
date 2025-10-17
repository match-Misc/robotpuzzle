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


def highlight_pieces(image):
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

    # Sort by area descending and take top 24 (assuming largest are pieces)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:24]

    # Generate 24 unique colors using HSV color space for distinct hues
    colors = []
    for i in range(24):
        hue = int(180 * i / 24)  # Hue from 0 to 180
        hsv_color = np.uint8([[[hue, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])))

    # Create a copy of the original image for highlighting
    highlighted = image.copy()

    # Draw each contour filled with its assigned unique color
    for i, cnt in enumerate(contours):
        cv2.drawContours(highlighted, [cnt], -1, colors[i], -1)  # -1 for filled

    return highlighted, len(contours)


def main():
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

            # Highlight pieces
            highlighted, num_pieces = highlight_pieces(stretched)

            # Show highlighted pieces
            cv2.imshow("Highlighted Pieces", highlighted)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
