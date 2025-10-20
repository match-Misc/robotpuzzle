import datetime
import json

import cv2
import depthai as dai
import numpy as np

# Global variables for manual selection
points = []
selecting = False
calibrated = False
M = None  # Perspective transform matrix
output_size = None
full_frame = None  # Store full resolution frame for processing
display_scale = 0.5  # Scale factor for display (50% of original)


def mouse_callback(event, x, y, flags, param):
    global points, selecting, frame_copy, display_scale
    if event == cv2.EVENT_LBUTTONDOWN and selecting:
        # Scale coordinates back to full resolution
        full_x = int(x / display_scale)
        full_y = int(y / display_scale)
        points.append((full_x, full_y))
        print(f"Point {len(points)}: ({full_x}, {full_y})")
        if len(points) == 4:
            selecting = False
            print("All 4 points selected. Processing...")


def capture_frame(device, q_rgb):
    in_rgb = q_rgb.tryGet()
    if in_rgb is not None:
        return in_rgb.getCvFrame()
    return None


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
    global points, selecting, calibrated, M, output_size, full_frame, display_scale

    # Create pipeline
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)

    # Create XLinkIn for camera control
    control_in = pipeline.create(dai.node.XLinkIn)
    control_in.setStreamName("control")
    control_in.out.link(cam_rgb.inputControl)

    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
    cam_rgb.setFps(6)
    cam_rgb.setVideoSize(3840, 2160)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb_4k")
    cam_rgb.video.link(xout_rgb.input)

    # Set initial camera controls
    control = dai.CameraControl()
    control.setBrightness(1)
    control.setContrast(1)
    control.setManualExposure(10000, 300)
    control.setManualFocus(123)
    control.setLumaDenoise(4)
    control.setChromaDenoise(4)
    control.setAutoExposureLock(True)
    control.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
    control.setAutoWhiteBalanceLock(True)
    control.setAntiBandingMode(dai.CameraControl.AntiBandingMode.MAINS_50_HZ)

    # Start device
    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue(name="rgb_4k", maxSize=4, blocking=False)

        # Send initial control message
        control_queue = device.getInputQueue("control")
        control_queue.send(control)

        print(
            "First, calibrate by pressing 'c' to capture frame, then click 4 corners of the white rectangle."
        )
        print(
            "After calibration, press 's' to start/stop live transform, 'p' to save calibrated frame as PNG, 'q' to quit"
        )

        cv2.namedWindow("Camera Feed")
        cv2.setMouseCallback("Camera Feed", mouse_callback)

        while True:
            frame = capture_frame(device, q_rgb)
            if frame is None:
                continue

            # Store full resolution frame for processing
            full_frame = frame.copy()

            # Resize for display (50% of original for easier calibration)
            height, width, _ = frame.shape
            resized_frame = cv2.resize(
                frame,
                (int(width * display_scale), int(height * display_scale)),
                interpolation=cv2.INTER_AREA,
            )
            display_frame = resized_frame.copy()

            if not calibrated:
                # Draw selected points with highlighting during calibration
                # Scale points down for display on resized frame
                for i, pt in enumerate(points):
                    display_pt = (
                        int(pt[0] * display_scale),
                        int(pt[1] * display_scale),
                    )
                    cv2.circle(display_frame, display_pt, 10, (0, 255, 0), 2)
                    cv2.circle(display_frame, display_pt, 5, (0, 255, 0), -1)
                    cv2.putText(
                        display_frame,
                        str(i + 1),
                        (display_pt[0] + 15, display_pt[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                # Draw lines connecting the points if all 4 are selected
                if len(points) == 4:
                    display_points = [
                        (int(pt[0] * display_scale), int(pt[1] * display_scale))
                        for pt in points
                    ]
                    cv2.polylines(
                        display_frame, [np.array(display_points)], True, (255, 0, 0), 2
                    )

            cv2.imshow("Camera Feed", display_frame)

            if calibrated:
                # Show transformed frame continuously (use full resolution frame)
                transformed = apply_perspective_transform(full_frame)
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
            elif key == ord("p") and calibrated:
                # Save the current transformed frame as PNG
                transformed = apply_perspective_transform(full_frame)
                if transformed is not None:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"calibrated_frame_{timestamp}.png"
                    cv2.imwrite(filename, transformed)
                    print(f"Calibrated frame saved as {filename}")
                else:
                    print("No transformed frame available to save")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
