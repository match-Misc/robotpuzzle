import sys

import cv2
import numpy as np


def get_robust_orientation(contour, image_shape):
    """
    Calculate orientation using moments of the filled shape's mask.
    This is generally more robust than fitting an ellipse to the contour points.
    """
    # 1. Create a blank mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # 2. Draw the contour filled in on the mask
    cv2.drawContours(mask, [contour], -1, 255, -1)

    # 3. Calculate moments from the mask
    M = cv2.moments(mask)

    # 4. Calculate orientation from second-order central moments
    # These are pre-calculated in the moments dictionary as 'mu'
    mu20 = M["mu20"]
    mu02 = M["mu02"]
    mu11 = M["mu11"]

    angle_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    angle_deg = np.degrees(angle_rad)

    # Normalize to 0-180 range for consistency
    # if angle_deg < 0:
    #     angle_deg += 180

    return angle_deg


def detect_pieces(image, num_pieces):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth edges and reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding for better handling of varying backgrounds
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        closed_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    min_area = 500  # Lower min area for adaptive thresholding
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:num_pieces]

    detected_pieces = []
    for cnt in contours:
        M = cv2.moments(cnt)
        hu_moments = cv2.HuMoments(M).flatten()
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        angle = get_robust_orientation(cnt, gray.shape)
        detected_pieces.append((cnt, hu_moments, (cx, cy), angle))

    return detected_pieces


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python highlight_pieces.py <image_path> <num_pieces>")
        sys.exit(1)

    image_path = sys.argv[1]
    num_pieces = int(sys.argv[2])

    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        sys.exit(1)

    detected_pieces = detect_pieces(image, num_pieces)

    # Create highlighted image
    img_height, img_width = image.shape[:2]
    highlighted = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    # Generate unique colors
    num_detected = len(detected_pieces)
    colors = []
    for i in range(num_detected):
        hue = int(180 * i / num_detected)
        hsv_color = np.uint8([[[hue, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])))

    # Draw filled contours on black background
    for i, (cnt, _, _, _) in enumerate(detected_pieces):
        cv2.drawContours(highlighted, [cnt], -1, colors[i], -1)

    output_path = "highlighted_pieces.png"
    cv2.imwrite(output_path, highlighted)
    print(f"Highlighted image saved to {output_path}")
