import sys

import cv2
import numpy as np


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        sys.exit(1)
    return image


def process_image(image_path):
    # Load image
    img = load_image(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
    highlighted = img.copy()

    # Draw each contour filled with its assigned unique color
    for i, cnt in enumerate(contours):
        cv2.drawContours(highlighted, [cnt], -1, colors[i], -1)  # -1 for filled

    # Display the highlighted image
    window_name = f"Highlighted Pieces - {image_path}"
    cv2.imshow(window_name, highlighted)

    # Print details to console
    print(f"Image: {image_path} - Identified {len(contours)} pieces")
    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            print(f"Piece {i + 1}: Centroid at ({cx}, {cy})")


def main():
    # List of images to process
    images = [
        "puzzle_monitor.png",
        "puzzle_test01.png",
        "puzzle_test02.png",
        "solved.png",
    ]

    # Process each image
    for img_path in images:
        process_image(img_path)

    # Wait for user to close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
