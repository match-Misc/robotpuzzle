import sys

import cv2
import numpy as np

# Global variables
gray = None
window_name = "Canny Edge Detection Tuner"


def update_canny(x):
    global gray
    if gray is not None:
        try:
            threshold1 = cv2.getTrackbarPos("Threshold1", window_name)
            threshold2 = cv2.getTrackbarPos("Threshold2", window_name)
        except cv2.error:
            # Trackbars not yet created, skip update
            return

        # Ensure threshold1 < threshold2
        if threshold1 >= threshold2:
            threshold1 = threshold2 - 1

        # Apply Canny edge detection
        edges = cv2.Canny(gray, threshold1, threshold2)

        # Display the result
        cv2.imshow(window_name, edges)


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        sys.exit(1)
    return image


def main():
    if len(sys.argv) != 2:
        print("Usage: python canny_tuner.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = load_image(image_path)

    # Convert to grayscale
    global gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a window
    cv2.namedWindow(window_name)

    # Create trackbars for threshold1 and threshold2 with initial values
    cv2.createTrackbar("Threshold1", window_name, 50, 255, update_canny)
    cv2.createTrackbar("Threshold2", window_name, 150, 255, update_canny)

    # Apply initial Canny edge detection
    update_canny(0)

    while True:
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # Exit on 'q' or ESC
        if key == ord("q") or key == 27:
            break

        # Print current parameters on 'p'
        if key == ord("p"):
            threshold1 = cv2.getTrackbarPos("Threshold1", window_name)
            threshold2 = cv2.getTrackbarPos("Threshold2", window_name)
            print(
                f"Current parameters: threshold1={threshold1}, threshold2={threshold2}"
            )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
