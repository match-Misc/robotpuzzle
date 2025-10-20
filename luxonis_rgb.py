import cv2
import depthai as dai

# --- Pipeline Setup ---
# 1. Create a new pipeline
pipeline = dai.Pipeline()

# 2. Define a source node for the color camera
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)

# Create XLinkIn for camera control
control_in = pipeline.create(dai.node.XLinkIn)
control_in.setStreamName("control")
control_in.out.link(cam_rgb.inputControl)

# --- KEY CHANGE HERE ---
# The sensor on the OAK-D Pro is 12MP (IMX378), which is used to output 4K video.
# The correct enum is THE_12_MP, not THE_4K_P.
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
# -----------------------

cam_rgb.setFps(6)
# Make sure you are using the 'video' output for full resolution
cam_rgb.setVideoSize(3840, 2160)

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

# 3. Create an XLinkOut node to send frames to the host
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb_4k")

# 4. Link the camera's 'video' output for full resolution
cam_rgb.video.link(xout_rgb.input)


# --- Device Connection and Main Loop ---
with dai.Device(pipeline) as device:
    print("Connected to OAK-D Pro. Press 'q' to quit.")

    q_rgb = device.getOutputQueue(name="rgb_4k", maxSize=4, blocking=False)

    # Send initial control message
    control_queue = device.getInputQueue("control")
    control_queue.send(control)

    while True:
        in_rgb = q_rgb.get()

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()

            # For display, we resize the 4K frame to 1080p (25% of original)
            # The 'frame' variable still contains the full 4K image if you need it.
            height, width, _ = frame.shape
            resized_frame = cv2.resize(
                frame, (width // 2, height // 2), interpolation=cv2.INTER_AREA
            )

            cv2.imshow("OAK-D Pro 4K Capture (Resized)", resized_frame)

        if cv2.waitKey(166) == ord("q"):
            break

cv2.destroyAllWindows()
print("Stream stopped.")
