import json

import cv2
import numpy as np

# Global variables for manual selection
points = []
selecting = False
calibrated = False
M = None  # Perspective transform matrix
output_size = None


def mouse_callback(event, x, y, flags, param):
    global points, selecting, frame_copy
    if event == cv2.EVENT_LBUTTONDOWN and selecting:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")
        if len(points) == 4:
            selecting = False
            print("All 4 points selected. Processing...")


def set_webcam_settings(cap, brightness=50, contrast=100, saturation=0):
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    cap.set(cv2.CAP_PROP_CONTRAST, contrast)
    cap.set(cv2.CAP_PROP_SATURATION, saturation)
    # Set resolution to Full HD
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        return None
    return frame


def calibrate_perspective(pts):
    global M, output_size
    if len(pts) != 4:
        return False

    # Convert points to numpy array
    rect = np.array(pts, dtype="float32")

    # Order the points: top-left, top-right, bottom-right, bottom-left
    pts = rect.reshape(4, 2)
    sorted_pts = np.zeros((4, 2), dtype="float32")

    # Sum of coordinates
    s = pts.sum(axis=1)
    sorted_pts[0] = pts[np.argmin(s)]  # Top-left
    sorted_pts[2] = pts[np.argmax(s)]  # Bottom-right

    # Difference of coordinates
    diff = np.diff(pts, axis=1)
    sorted_pts[1] = pts[np.argmin(diff)]  # Top-right
    sorted_pts[3] = pts[np.argmax(diff)]  # Bottom-left

    # Define destination points for perspective transform (axis-aligned rectangle)
    (tl, tr, br, bl) = sorted_pts
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(sorted_pts, dst)
    output_size = (maxWidth, maxHeight)
    return True


def apply_perspective_transform(frame):
    global M, output_size
    if M is None or output_size is None:
        return None
    # Apply perspective warp
    warped = cv2.warpPerspective(frame, M, output_size)
    return warped


def main():
    global points, selecting, calibrated, M, output_size

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # Set webcam settings
    set_webcam_settings(cap, brightness=50, contrast=100, saturation=0)

    print(
        "First, calibrate by pressing 'c' to capture frame, then click 4 corners of the white rectangle."
    )
    print(
        "After calibration, press 's' to start capturing transformed frames, 'q' to quit"
    )

    cv2.namedWindow("Webcam Feed")
    cv2.setMouseCallback("Webcam Feed", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        display_frame = frame.copy()

        if not calibrated:
            # Draw selected points with highlighting during calibration
            for i, pt in enumerate(points):
                cv2.circle(display_frame, pt, 10, (0, 255, 0), 2)
                cv2.circle(display_frame, pt, 5, (0, 255, 0), -1)
                cv2.putText(
                    display_frame,
                    str(i + 1),
                    (pt[0] + 15, pt[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            # Draw lines connecting the points if all 4 are selected
            if len(points) == 4:
                cv2.polylines(display_frame, [np.array(points)], True, (255, 0, 0), 2)

        cv2.imshow("Webcam Feed", display_frame)

        if calibrated:
            # Show transformed frame continuously
            transformed = apply_perspective_transform(frame)
            if transformed is not None:
                cv2.imshow("Transformed Puzzle", transformed)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c") and not selecting and not calibrated:
            # Capture frame for calibration
            points = []
            selecting = True
            print("Frame captured. Click 4 corners of the white rectangle.")
        elif key == ord("s") and len(points) == 4 and not calibrated:
            # Calibrate with selected points
            if calibrate_perspective(points):
                calibrated = True
                # Save calibration data to config.json
                config_data = {
                    "points": points,
                    "M": M.tolist() if M is not None else None,
                    "output_size": output_size,
                }
                with open("config.json", "w") as f:
                    json.dump(config_data, f, indent=4)
                print("Calibration data saved to config.json")
                print(
                    "Calibration complete. Press 's' again to start/stop live transform."
                )
            else:
                print("Calibration failed")
        elif key == ord("s") and calibrated:
            # Toggle live transform display
            calibrated = not calibrated
            if calibrated:
                print("Live transform started")
            else:
                cv2.destroyWindow("Transformed Puzzle")
                print("Live transform stopped")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
