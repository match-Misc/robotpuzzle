import json
import os
import subprocess
import sys
import threading
import time

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

# Physical dimensions
PICKUP_FRAME_WIDTH_MM = 520
PICKUP_FRAME_HEIGHT_MM = 325
TARGET_FRAME_WIDTH_MM = 297  # DIN A4 width
TARGET_FRAME_HEIGHT_MM = 210  # DIN A4 height
# Pixel resolution of the target (solved) puzzle image frame.
# Corresponds to DIN A4 dimensions at 300 DPI:
# Width: 297mm × 300 DPI ÷ 25.4 ≈ 3510 pixels
# Height: 210mm × 300 DPI ÷ 25.4 ≈ 2482 pixels
# Used for coordinate transformations between millimeters and pixels in the solution image.
TARGET_FRAME_RESOLUTION = (3510, 2482)  # pixels


def get_robust_orientation(contour, image_shape):
    """
    Calculate orientation using PCA (Principal Component Analysis) on the complete mask points.
    This method finds the main axis of the shape without assuming symmetry.
    Uses skewness-based disambiguation for consistent orientation.
    """
    # Create a filled mask from the contour
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    # Extract all points from the mask (where mask == 255)
    y_coords, x_coords = np.where(mask == 255)
    points = np.column_stack((x_coords, y_coords)).astype(np.float64)  # (x, y) format

    # Calculate mean
    mean = np.mean(points, axis=0)

    # Center the points
    centered = points - mean

    # Calculate covariance matrix
    cov = np.cov(centered.T)

    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort eigenvectors by eigenvalues in descending order
    sort_indices = np.argsort(eigenvalues)[::-1]
    v1 = eigenvectors[:, sort_indices[0]]
    v2 = eigenvectors[:, sort_indices[1]]

    # Ensure consistent basis: right-handed coordinate system
    det = np.linalg.det(np.column_stack([v1, v2]))
    if det < 0:
        v2 = -v2

    # Calculate skewness along both axes
    projections_v1 = centered.dot(v1)
    skew_v1 = np.sum(projections_v1**3)
    projections_v2 = centered.dot(v2)
    skew_v2 = np.sum(projections_v2**3)

    # Determine the dominant axis and disambiguate
    if abs(skew_v1) > abs(skew_v2):
        # Primary axis is dominant
        if skew_v1 < 0:
            v1 = -v1
    else:
        # Secondary axis is dominant
        if skew_v2 < 0:
            # Flip the entire coordinate system 180 degrees
            v1 = -v1

    main_axis = v1

    # Calculate angle from the horizontal
    angle_rad = np.arctan2(main_axis[1], main_axis[0])
    angle_deg = np.degrees(angle_rad)

    # Ensure angle is in [0, 180) range
    # if angle_deg < 0:
    #     angle_deg += 180

    return angle_deg


# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class PuzzleSolverGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Puzzle Solver")
        self.root.geometry("1200x800")

        # Variables
        self.num_pieces = ctk.StringVar(value="24")
        self.solution_exists = False
        self.captured_image = None
        self.warped_image = None
        self.threshold_image = None
        self.stretched_image = None
        self.solution_image = None
        self.detected_pieces = []
        self.solution_map = []
        self.target_pieces = []
        self.piece_colors = []
        self.hovered_piece = None
        self.threshold_value = ctk.DoubleVar(value=127)
        self.use_adaptive_threshold = ctk.BooleanVar(value=False)

        # Webcam variables
        self.cap = None
        self.is_capturing = False
        self.current_frame = None

        # Live mode variables
        self.is_live_mode = False
        self.live_thread = None
        self.live_stop_event = threading.Event()

        # Create GUI elements
        self.create_widgets()

        # Load solution image if exists
        self.check_solution_exists()

    def create_widgets(self):
        # Main frame
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Top control panel
        control_frame = ctk.CTkFrame(main_frame)
        control_frame.pack(fill="x", padx=10, pady=10)

        # Number of pieces input
        pieces_label = ctk.CTkLabel(control_frame, text="Number of Pieces:")
        pieces_label.pack(side="left", padx=(10, 5))

        self.pieces_entry = ctk.CTkEntry(
            control_frame, textvariable=self.num_pieces, width=100
        )
        self.pieces_entry.pack(side="left", padx=(0, 10))

        # Check solution button
        self.check_btn = ctk.CTkButton(
            control_frame, text="Check Solution", command=self.check_solution_exists
        )
        self.check_btn.pack(side="left", padx=(0, 10))

        # Calibration button
        self.calibrate_btn = ctk.CTkButton(
            control_frame, text="Calibrate Camera", command=self.launch_calibration
        )
        self.calibrate_btn.pack(side="left", padx=(0, 10))

        # Capture button
        self.capture_btn = ctk.CTkButton(
            control_frame,
            text="Capture & Solve",
            command=self.capture_and_solve,
            fg_color="green",
            hover_color="dark green",
        )
        self.capture_btn.pack(side="left", padx=(0, 10))

        # Live mode button
        self.live_btn = ctk.CTkButton(
            control_frame,
            text="Start Live Mode",
            command=self.toggle_live_mode,
            fg_color="blue",
            hover_color="dark blue",
        )
        self.live_btn.pack(side="left", padx=(0, 10))

        # Threshold controls - compact horizontal layout
        threshold_label = ctk.CTkLabel(control_frame, text="Threshold:")
        threshold_label.pack(side="left", padx=(0, 5))

        # Adaptive threshold checkbox
        self.adaptive_checkbox = ctk.CTkCheckBox(
            control_frame,
            text="Adaptive",
            variable=self.use_adaptive_threshold,
            command=self.on_threshold_mode_change,
            width=70,
        )
        self.adaptive_checkbox.pack(side="left", padx=(0, 10))

        # Threshold slider
        self.threshold_slider = ctk.CTkSlider(
            control_frame,
            from_=0,
            to=255,
            variable=self.threshold_value,
            command=self.on_threshold_change,
            width=120,
            height=16,
        )
        self.threshold_slider.pack(side="left", padx=(0, 5))

        # Threshold value label
        self.threshold_label = ctk.CTkLabel(
            control_frame,
            text="127",
            font=ctk.CTkFont(size=11, weight="bold"),
            width=30,
        )
        self.threshold_label.pack(side="left", padx=(0, 10))

        # Status label
        self.status_label = ctk.CTkLabel(control_frame, text="Ready", text_color="gray")
        self.status_label.pack(side="right", padx=(10, 0))

        # Image display area
        display_frame = ctk.CTkFrame(main_frame)
        display_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Left panel - Images in two rows
        left_panel = ctk.CTkFrame(display_frame)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 5))

        # First row - Raw, Warped, Threshold images
        first_row = ctk.CTkFrame(left_panel)
        first_row.pack(fill="both", expand=True, pady=(10, 5))

        # Raw image section
        raw_frame = ctk.CTkFrame(first_row)
        raw_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        raw_label = ctk.CTkLabel(
            raw_frame, text="Raw Puzzle", font=ctk.CTkFont(size=12, weight="bold")
        )
        raw_label.pack(pady=(5, 5))

        self.raw_canvas = ctk.CTkCanvas(raw_frame, bg="gray20", highlightthickness=0)
        self.raw_canvas.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        # Warped image section
        warped_frame = ctk.CTkFrame(first_row)
        warped_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        warped_label = ctk.CTkLabel(
            warped_frame,
            text="Warped Puzzle",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        warped_label.pack(pady=(5, 5))

        self.warped_canvas = ctk.CTkCanvas(
            warped_frame, bg="gray20", highlightthickness=0
        )
        self.warped_canvas.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        # Threshold image section
        threshold_frame = ctk.CTkFrame(first_row)
        threshold_frame.pack(side="left", fill="both", expand=True, padx=(0, 0))

        threshold_label = ctk.CTkLabel(
            threshold_frame,
            text="Threshold",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        threshold_label.pack(pady=(5, 5))

        self.threshold_canvas = ctk.CTkCanvas(
            threshold_frame, bg="gray20", highlightthickness=0
        )
        self.threshold_canvas.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        # Second row - Detected pieces
        second_row = ctk.CTkFrame(left_panel)
        second_row.pack(fill="both", expand=True, pady=(0, 10))

        highlighted_label = ctk.CTkLabel(
            second_row,
            text="Detected Pieces",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        highlighted_label.pack(pady=(5, 5))

        self.captured_canvas = ctk.CTkCanvas(
            second_row, bg="gray20", highlightthickness=0
        )
        self.captured_canvas.pack(fill="both", expand=True, padx=10, pady=(0, 5))

        # Right panel - Solution image
        right_panel = ctk.CTkFrame(display_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=(5, 0))

        solution_label = ctk.CTkLabel(
            right_panel, text="Solution", font=ctk.CTkFont(size=16, weight="bold")
        )
        solution_label.pack(pady=(10, 5))

        self.solution_canvas = ctk.CTkCanvas(
            right_panel, bg="gray20", highlightthickness=0
        )
        self.solution_canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Bind mouse events for hover (only on captured canvas)
        self.captured_canvas.bind("<Motion>", self.on_mouse_move)
        self.captured_canvas.bind("<Leave>", self.on_mouse_leave)

        # Bind window resize event to rescale images
        self.root.bind("<Configure>", self.on_resize)

        # Info panel at bottom
        info_frame = ctk.CTkFrame(main_frame)
        info_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.info_label = ctk.CTkLabel(
            info_frame,
            text="Hover over pieces to see target positions",
            font=ctk.CTkFont(size=12),
        )
        self.info_label.pack(pady=10)

    def on_threshold_change(self, value):
        """Callback when threshold slider value changes."""
        # Update the threshold label
        self.threshold_label.configure(text=f"{int(float(value))}")

        # Update threshold display if we have an image
        if self.stretched_image is not None:
            self.threshold_image = self.compute_threshold(self.stretched_image)
            self.display_threshold_image()

            # Update piece detection if we have target pieces loaded
            if self.target_pieces:
                num_pieces = int(self.num_pieces.get())
                self.detected_pieces = self.detect_pieces(
                    self.stretched_image, num_pieces
                )
                self.solution_map = self.match_pieces(
                    self.detected_pieces, self.target_pieces
                )
                self.display_captured_image()
                self.display_solution_image()

            # In live mode, also update piece detection
            if self.is_live_mode:
                num_pieces = int(self.num_pieces.get())
                self.detected_pieces = self.detect_pieces(
                    self.stretched_image, num_pieces
                )
                if self.target_pieces:
                    self.solution_map = self.match_pieces(
                        self.detected_pieces, self.target_pieces
                    )
                self.display_captured_image()
                self.display_solution_image()

    def on_threshold_mode_change(self):
        """Callback when adaptive checkbox is toggled."""
        # Enable/disable slider based on checkbox
        is_adaptive = self.use_adaptive_threshold.get()
        self.threshold_slider.configure(state="disabled" if is_adaptive else "normal")

        # Update threshold display and piece detection if we have an image
        if self.stretched_image is not None:
            self.on_threshold_change(self.threshold_value.get())

    def check_solution_exists(self):
        try:
            num_pieces = int(self.num_pieces.get())
            json_path = f"configs/Puzzle_{num_pieces}.json"
            png_path = f"Solutions/Puzzle_{num_pieces}.png"

            json_exists = os.path.exists(json_path)
            png_exists = os.path.exists(png_path)

            if json_exists and png_exists:
                self.solution_exists = True
                self.status_label.configure(
                    text=f"Solution for {num_pieces} pieces found", text_color="green"
                )
                self.load_solution_image(num_pieces)
            else:
                self.solution_exists = False
                missing = []
                if not json_exists:
                    missing.append("JSON")
                if not png_exists:
                    missing.append("PNG")
                self.status_label.configure(
                    text=f"Missing: {', '.join(missing)}", text_color="red"
                )

        except ValueError:
            self.status_label.configure(
                text="Invalid number of pieces", text_color="red"
            )

    def load_solution_image(self, num_pieces):
        try:
            png_path = f"Solutions/Puzzle_{num_pieces}.png"
            self.solution_image = cv2.imread(png_path)
            if self.solution_image is not None:
                self.display_solution_image()
        except Exception:
            pass

    def display_solution_image(self):
        if self.solution_image is not None:
            # Start with the solution image
            overlay = self.solution_image.copy()

            # Overlay detected pieces if available
            if self.detected_pieces and self.solution_map and self.piece_colors:
                for i, (cnt, _, _, _, pickup_pose) in enumerate(self.detected_pieces):
                    if i < len(self.solution_map):
                        _, target_pose, _, _, _ = self.solution_map[i]
                        # Transform contour to target space with detected orientation (no scaling for initial display)
                        transformed_cnt = self.transform_contour(
                            cnt,
                            pickup_pose,
                            target_pose,
                            use_target_orientation=False,
                            apply_scaling=False,
                        )
                        # Draw the transformed contour on the overlay
                        cv2.drawContours(
                            overlay, [transformed_cnt], -1, self.piece_colors[i], -1
                        )

            # Draw PCA axes for target pieces
            if self.target_pieces:
                for piece in self.target_pieces:
                    centroid_px = piece["centroid"]
                    orientation = piece["orientation"]
                    cx, cy = centroid_px
                    length = 100  # pixels, longer for visibility
                    end_x = int(cx + length * np.cos(np.radians(orientation)))
                    end_y = int(cy + length * np.sin(np.radians(orientation)))
                    cv2.arrowedLine(
                        overlay,
                        (cx, cy),
                        (end_x, end_y),
                        (0, 0, 0),
                        5,
                        tipLength=0.1,
                    )

            # Resize to fit canvas
            canvas_width = self.solution_canvas.winfo_width()
            canvas_height = self.solution_canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                img_height, img_width = overlay.shape[:2]
                scale = min(canvas_width / img_width, canvas_height / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)

                resized = cv2.resize(overlay, (new_width, new_height))
                img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                self.solution_photo = ImageTk.PhotoImage(img)

                # Center the image
                x = (canvas_width - new_width) // 2
                y = (canvas_height - new_height) // 2

                self.solution_canvas.create_image(
                    x, y, anchor="nw", image=self.solution_photo
                )

    def launch_calibration(self):
        try:
            subprocess.Popen([sys.executable, "puzzle_calibration.py"])
            self.status_label.configure(text="Calibration launched", text_color="blue")
        except Exception as e:
            self.status_label.configure(
                text=f"Failed to launch calibration: {e}", text_color="red"
            )

    def capture_and_solve(self):
        if not self.solution_exists:
            self.status_label.configure(text="No solution available", text_color="red")
            return

        try:
            num_pieces = int(self.num_pieces.get())
            self.status_label.configure(text="Capturing image...", text_color="blue")

            # Capture single frame
            self.capture_single_frame()

            if self.captured_image is not None:
                self.status_label.configure(text="Processing...", text_color="blue")

                # Process in thread to avoid blocking GUI
                threading.Thread(
                    target=self.process_puzzle, args=(num_pieces,), daemon=True
                ).start()

        except ValueError:
            self.status_label.configure(
                text="Invalid number of pieces", text_color="red"
            )

    def capture_single_frame(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.configure(text="Cannot open webcam", text_color="red")
            return

        # Set webcam settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 100)
        self.cap.set(cv2.CAP_PROP_SATURATION, 0)

        ret, frame = self.cap.read()
        if ret:
            self.captured_image = frame.copy()
        else:
            self.status_label.configure(
                text="Failed to capture frame", text_color="red"
            )

        self.cap.release()

    def process_puzzle(self, num_pieces):
        try:
            # Load calibration
            config = self.load_config()
            if config is None:
                self.root.after(
                    0,
                    lambda: self.status_label.configure(
                        text="No calibration found", text_color="red"
                    ),
                )
                return

            M = np.array(config["M"])
            output_size = tuple(config["output_size"])

            # Apply perspective transform
            transformed = cv2.warpPerspective(self.captured_image, M, output_size)
            self.warped_image = transformed

            # Stretch to 16:10
            stretched = self.stretch_to_16_10(transformed)
            self.stretched_image = stretched

            # Detect pieces
            self.detected_pieces = self.detect_pieces(stretched, num_pieces)

            # Load target pieces
            self.target_pieces = self.load_target_pieces(num_pieces)

            # Match pieces
            self.solution_map = self.match_pieces(
                self.detected_pieces, self.target_pieces
            )

            # Update GUI
            self.root.after(0, self.update_display)

            self.root.after(
                0,
                lambda: self.status_label.configure(
                    text="Processing complete", text_color="green"
                ),
            )

        except Exception as e:
            error_msg = str(e)
            self.root.after(
                0,
                lambda: self.status_label.configure(
                    text=f"Processing failed: {error_msg}", text_color="red"
                ),
            )

    def load_config(self):
        try:
            with open("configs/config.json", "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def stretch_to_16_10(self, image):
        height, width = image.shape[:2]
        target_aspect = 16 / 10
        current_aspect = width / height

        if current_aspect > target_aspect:
            new_height = int(width / target_aspect)
            stretched = cv2.resize(
                image, (width, new_height), interpolation=cv2.INTER_LINEAR
            )
        else:
            new_width = int(height * target_aspect)
            stretched = cv2.resize(
                image, (new_width, height), interpolation=cv2.INTER_LINEAR
            )

        return stretched

    def compute_threshold(self, image):
        """Compute threshold image using either manual or adaptive thresholding."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.use_adaptive_threshold.get():
            # Use adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2,
            )
        else:
            # Use manual thresholding
            threshold_value = int(self.threshold_value.get())
            _, thresh = cv2.threshold(
                blurred, threshold_value, 255, cv2.THRESH_BINARY_INV
            )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        return closed_thresh

    def transform_contour(
        self,
        contour,
        pickup_pose,
        target_pose,
        use_target_orientation=True,
        apply_scaling=False,
        target_piece_id=None,
    ):
        """
        Transform a contour from pickup space to target space with optional re-orientation and scaling.
        """
        # Extract poses
        pickup_x, pickup_y, _, pickup_angle = pickup_pose
        target_x, target_y, _, target_angle = target_pose

        # Convert mm to pixels for pickup frame
        pickup_x_px = (pickup_x / PICKUP_FRAME_WIDTH_MM) * self.stretched_image.shape[1]
        pickup_y_px = (
            self.stretched_image.shape[0]
            - (pickup_y / PICKUP_FRAME_HEIGHT_MM) * self.stretched_image.shape[0]
        )

        # Convert mm to pixels for target frame
        target_x_px = (target_x / TARGET_FRAME_WIDTH_MM) * TARGET_FRAME_RESOLUTION[0]
        target_y_px = (
            TARGET_FRAME_RESOLUTION[1]
            - (target_y / TARGET_FRAME_HEIGHT_MM) * TARGET_FRAME_RESOLUTION[1]
        )

        # Translation vector
        tx = target_x_px - pickup_x_px
        ty = target_y_px - pickup_y_px

        # Rotation angle (use target orientation if specified, otherwise keep detected)
        # angle = target_angle if use_target_orientation else pickup_angle

        angle = target_angle - pickup_angle

        # Scaling factor (only if apply_scaling is True)
        scale_factor = 1.0
        if apply_scaling and target_piece_id is not None:
            # Find the target piece to get its area
            for piece in self.target_pieces:
                if piece["id"] == target_piece_id:
                    target_area = piece.get(
                        "area", 1.0
                    )  # Use area from JSON if available
                    detected_area = cv2.contourArea(contour)
                    if detected_area > 0:
                        scale_factor = np.sqrt(target_area / detected_area)
                    break

        # Build transformation matrix with rotation and optional scaling
        rotation_matrix = cv2.getRotationMatrix2D(
            (pickup_x_px, pickup_y_px), -angle, scale_factor
        )

        # Apply translation
        rotation_matrix[0, 2] += tx
        rotation_matrix[1, 2] += ty

        # Transform contour
        transformed_contour = cv2.transform(
            contour.reshape(-1, 1, 2).astype(np.float32), rotation_matrix
        ).astype(np.int32)

        return transformed_contour

    def detect_pieces(self, image, num_pieces):
        # Compute threshold using the new method
        self.threshold_image = self.compute_threshold(image)

        contours, _ = cv2.findContours(
            self.threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Get gray image for orientation calculation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.GaussianBlur(gray, (5, 5), 0)

        min_area = 1000
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[
            :num_pieces
        ]

        detected_pieces = []
        img_height, img_width = image.shape[:2]

        for cnt in contours:
            M = cv2.moments(cnt)
            hu_moments = cv2.HuMoments(M).flatten()
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            angle = get_robust_orientation(cnt, gray.shape)

            # Convert pixel coordinates to mm relative to pickup frame (bottom-left is (0,0))
            pickup_x_mm = (cx / img_width) * PICKUP_FRAME_WIDTH_MM
            pickup_y_mm = (
                (img_height - cy) / img_height
            ) * PICKUP_FRAME_HEIGHT_MM  # Flip Y axis
            pickup_pose = (pickup_x_mm, pickup_y_mm, 0, angle)  # (x, y, z, rotation)

            detected_pieces.append((cnt, hu_moments, (cx, cy), angle, pickup_pose))

        return detected_pieces

    def load_target_pieces(self, puzzle_size):
        try:
            with open(f"configs/Puzzle_{puzzle_size}.json", "r") as f:
                target_pieces = json.load(f)

            # Convert target centroids from pixels to mm relative to target frame
            for piece in target_pieces:
                centroid_px = piece["centroid"]
                target_x_mm = (
                    centroid_px[0] / TARGET_FRAME_RESOLUTION[0]
                ) * TARGET_FRAME_WIDTH_MM
                target_y_mm = (
                    (TARGET_FRAME_RESOLUTION[1] - centroid_px[1])
                    / TARGET_FRAME_RESOLUTION[1]
                ) * TARGET_FRAME_HEIGHT_MM  # Flip Y axis
                piece["target_pose"] = (
                    target_x_mm,
                    target_y_mm,
                    0,
                    piece["orientation"],
                )  # (x, y, z, rotation)

            return target_pieces
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def match_pieces(self, detected_pieces, target_pieces):
        import scipy.optimize

        # Create cost matrix
        cost_matrix = np.zeros((len(detected_pieces), len(target_pieces)))
        for i, (_, detected_hu, _, _, _) in enumerate(detected_pieces):
            for j, target in enumerate(target_pieces):
                target_hu = np.array(target["hu_moments"])
                cost_matrix[i, j] = np.linalg.norm(detected_hu - target_hu)

        # Solve assignment problem using Hungarian algorithm
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)

        solution_map = []
        for detected_idx, target_idx in zip(row_ind, col_ind):
            (
                detected_cnt,
                detected_hu,
                detected_centroid,
                detected_angle,
                pickup_pose,
            ) = detected_pieces[detected_idx]
            target = target_pieces[target_idx]

            target["centroid"]
            best_target_angle = target["orientation"]
            best_match = target["id"]
            target_pose = target["target_pose"]

            # Compute transformation in mm and degrees
            translation_mm = (
                target_pose[0] - pickup_pose[0],
                target_pose[1] - pickup_pose[1],
            )
            # Normalize angles to [0, 180) range for proper rotation calculation
            detected_angle_norm = detected_angle % 180
            best_target_angle_norm = best_target_angle % 180
            rotation_deg = -(
                best_target_angle_norm - detected_angle_norm
            )  # Invert sign for counter-clockwise positive

            solution_map.append(
                (
                    pickup_pose,
                    target_pose,
                    translation_mm,
                    rotation_deg,
                    best_match,
                )
            )

        # Sort solution_map to match the order of detected_pieces
        solution_map = [solution_map[i] for i in np.argsort(row_ind)]

        return solution_map

    def update_display(self):
        self.display_raw_image()
        self.display_warped_image()
        self.display_threshold_image()
        self.display_captured_image()
        self.display_solution_image()

    def display_raw_image(self):
        if self.captured_image is not None:
            # Resize to fit canvas (downscale 4K to fit screen if needed)
            canvas_width = self.raw_canvas.winfo_width()
            canvas_height = self.raw_canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                img_height, img_width = self.captured_image.shape[:2]

                # If image is 4K, resize to fit screen better while maintaining aspect ratio
                if img_width == 3840 and img_height == 2160:
                    scale = min(canvas_width / 1920, canvas_height / 1080)
                    display_width = int(1920 * scale)
                    display_height = int(1080 * scale)
                else:
                    scale = min(canvas_width / img_width, canvas_height / img_height)
                    display_width = int(img_width * scale)
                    display_height = int(img_height * scale)

                resized = cv2.resize(
                    self.captured_image, (display_width, display_height)
                )
                img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                self.raw_photo = ImageTk.PhotoImage(img)

                # Center the image
                x = (canvas_width - display_width) // 2
                y = (canvas_height - display_height) // 2

                self.raw_canvas.create_image(x, y, anchor="nw", image=self.raw_photo)

    def display_warped_image(self):
        if self.warped_image is not None:
            # Resize to fit canvas
            canvas_width = self.warped_canvas.winfo_width()
            canvas_height = self.warped_canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                img_height, img_width = self.warped_image.shape[:2]
                scale = min(canvas_width / img_width, canvas_height / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)

                resized = cv2.resize(self.warped_image, (new_width, new_height))
                img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                self.warped_photo = ImageTk.PhotoImage(img)

                # Center the image
                x = (canvas_width - new_width) // 2
                y = (canvas_height - new_height) // 2

                self.warped_canvas.create_image(
                    x, y, anchor="nw", image=self.warped_photo
                )

    def display_threshold_image(self):
        if self.threshold_image is not None:
            # Resize to fit canvas
            canvas_width = self.threshold_canvas.winfo_width()
            canvas_height = self.threshold_canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                img_height, img_width = self.threshold_image.shape[:2]
                scale = min(canvas_width / img_width, canvas_height / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)

                resized = cv2.resize(self.threshold_image, (new_width, new_height))
                # Convert grayscale to RGB for display
                if len(resized.shape) == 2:
                    resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
                img = Image.fromarray(resized)
                self.threshold_photo = ImageTk.PhotoImage(img)

                # Center the image
                x = (canvas_width - new_width) // 2
                y = (canvas_height - new_height) // 2

                self.threshold_canvas.create_image(
                    x, y, anchor="nw", image=self.threshold_photo
                )

    def display_captured_image(self):
        if self.stretched_image is not None and self.detected_pieces:
            # Create black background image for highlighted pieces only
            img_height, img_width = self.stretched_image.shape[:2]
            highlighted = np.zeros((img_height, img_width, 3), dtype=np.uint8)

            # Generate unique colors
            num_pieces = len(self.detected_pieces)
            self.piece_colors = []
            for i in range(num_pieces):
                hue = int(180 * i / num_pieces)
                hsv_color = np.uint8([[[hue, 255, 255]]])
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
                color = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))
                self.piece_colors.append(color)

            # Draw filled contours on black background
            for i, (cnt, _, _, _, _) in enumerate(self.detected_pieces):
                cv2.drawContours(highlighted, [cnt], -1, self.piece_colors[i], -1)

            # Draw PCA axes for detected pieces
            for i, (cnt, _, (cx, cy), angle, _) in enumerate(self.detected_pieces):
                length = 100  # pixels, longer for visibility
                end_x = int(cx + length * np.cos(np.radians(angle)))
                end_y = int(cy + length * np.sin(np.radians(angle)))
                cv2.arrowedLine(
                    highlighted,
                    (cx, cy),
                    (end_x, end_y),
                    (0, 0, 0),
                    5,
                    tipLength=0.1,
                )

            # Resize to fit canvas
            canvas_width = self.captured_canvas.winfo_width()
            canvas_height = self.captured_canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                scale = min(canvas_width / img_width, canvas_height / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)

                resized = cv2.resize(highlighted, (new_width, new_height))
                img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                self.captured_photo = ImageTk.PhotoImage(img)

                # Center the image
                x = (canvas_width - new_width) // 2
                y = (canvas_height - new_height) // 2

                self.captured_canvas.create_image(
                    x, y, anchor="nw", image=self.captured_photo
                )

    def on_mouse_move(self, event):
        if (
            not self.detected_pieces
            or not self.solution_map
            or self.stretched_image is None
            or not hasattr(self, "info_label")
        ):
            return

        # Determine which canvas triggered the event
        canvas = event.widget
        canvas_x = canvas.canvasx(event.x)
        canvas_y = canvas.canvasy(event.y)

        # Convert to image coordinates
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        img_height, img_width = self.stretched_image.shape[:2]
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        offset_x = (canvas_width - new_width) // 2
        offset_y = (canvas_height - new_height) // 2

        img_x = (canvas_x - offset_x) / scale
        img_y = (canvas_y - offset_y) / scale

        # Check if mouse is over a piece
        for i, (cnt, _, centroid, _, _) in enumerate(self.detected_pieces):
            if cv2.pointPolygonTest(cnt, (img_x, img_y), False) >= 0:
                if self.hovered_piece != i:
                    self.hovered_piece = i
                    self.highlight_solution_piece(i)
                    self.update_info_text(i)
                return

        # Mouse not over any piece
        if self.hovered_piece is not None:
            self.hovered_piece = None
            self.display_solution_image()
            self.info_label.configure(text="Hover over pieces to see target positions")

    def on_mouse_leave(self, event):
        if self.hovered_piece is not None and hasattr(self, "info_label"):
            self.hovered_piece = None
            self.display_solution_image()
            self.info_label.configure(text="Hover over pieces to see target positions")

    def highlight_solution_piece(self, piece_index):
        if piece_index >= len(self.solution_map) or not self.piece_colors:
            return

        pickup_pose, target_pose, _, _, piece_id = self.solution_map[piece_index]
        cnt, _, _, _, _ = self.detected_pieces[piece_index]

        if self.solution_image is not None:
            highlighted = self.solution_image.copy()

            # Overlay all pieces with detected orientations
            for i, (other_cnt, _, _, _, other_pickup_pose) in enumerate(
                self.detected_pieces
            ):
                if i < len(self.solution_map):
                    _, other_target_pose, _, _, _ = self.solution_map[i]
                    # For the hovered piece, use target orientation and apply scaling; for others, use detected orientation without scaling
                    use_target = i == piece_index
                    apply_scaling = i == piece_index
                    piece_id = (
                        self.solution_map[i][4] if i < len(self.solution_map) else None
                    )
                    transformed_cnt = self.transform_contour(
                        other_cnt,
                        other_pickup_pose,
                        other_target_pose,
                        use_target_orientation=use_target,
                        apply_scaling=apply_scaling,
                        target_piece_id=piece_id,
                    )
                    cv2.drawContours(
                        highlighted, [transformed_cnt], -1, self.piece_colors[i], -1
                    )

            # Draw PCA axes for target pieces
            if self.target_pieces:
                for piece in self.target_pieces:
                    centroid_px = piece["centroid"]
                    orientation = piece["orientation"]
                    cx, cy = centroid_px
                    length = 100  # pixels, longer for visibility
                    end_x = int(cx + length * np.cos(np.radians(orientation)))
                    end_y = int(cy + length * np.sin(np.radians(orientation)))
                    cv2.arrowedLine(
                        highlighted,
                        (cx, cy),
                        (end_x, end_y),
                        (0, 0, 0),
                        5,
                        tipLength=0.1,
                    )

            # Convert target pose back to pixels for additional highlighting
            target_x_px = int(
                (target_pose[0] / TARGET_FRAME_WIDTH_MM) * TARGET_FRAME_RESOLUTION[0]
            )
            target_y_px = int(
                TARGET_FRAME_RESOLUTION[1]
                - (target_pose[1] / TARGET_FRAME_HEIGHT_MM) * TARGET_FRAME_RESOLUTION[1]
            )

            # Update solution canvas
            canvas_width = self.solution_canvas.winfo_width()
            canvas_height = self.solution_canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                img_height, img_width = highlighted.shape[:2]
                scale = min(canvas_width / img_width, canvas_height / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)

                resized = cv2.resize(highlighted, (new_width, new_height))
                img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                self.solution_highlighted_photo = ImageTk.PhotoImage(img)

                x = (canvas_width - new_width) // 2
                y = (canvas_height - new_height) // 2

                self.solution_canvas.create_image(
                    x, y, anchor="nw", image=self.solution_highlighted_photo
                )

    def update_info_text(self, piece_index):
        if piece_index >= len(self.solution_map):
            return

        pickup_pose, target_pose, translation_mm, rotation_deg, piece_id = (
            self.solution_map[piece_index]
        )

        info_text = (
            f"Piece {piece_id}: Pickup ({pickup_pose[0]:.1f}, {pickup_pose[1]:.1f}, {pickup_pose[2]:.1f}, {pickup_pose[3]:.1f}°) -> "
            f"Target ({target_pose[0]:.1f}, {target_pose[1]:.1f}, {target_pose[2]:.1f}, {target_pose[3]:.1f}°) | "
            f"Translate ({translation_mm[0]:.1f}, {translation_mm[1]:.1f}) mm | Rotate {rotation_deg:.1f}°"
        )

        self.info_label.configure(text=info_text)

    def on_resize(self, event):
        # Rescale images when window is resized
        self.update_display()
        # Reapply highlight if hovering
        if self.hovered_piece is not None:
            self.highlight_solution_piece(self.hovered_piece)

    def toggle_live_mode(self):
        if self.is_live_mode:
            self.stop_live_mode()
        else:
            self.start_live_mode()

    def start_live_mode(self):
        if not self.solution_exists:
            self.status_label.configure(text="No solution available", text_color="red")
            return

        try:
            num_pieces = int(self.num_pieces.get())
            self.is_live_mode = True
            self.live_stop_event.clear()
            self.live_btn.configure(
                text="Stop Live Mode", fg_color="red", hover_color="dark red"
            )
            self.status_label.configure(text="Live mode started", text_color="green")

            # Start live processing thread
            self.live_thread = threading.Thread(
                target=self.live_processing_loop, args=(num_pieces,), daemon=True
            )
            self.live_thread.start()

        except ValueError:
            self.status_label.configure(
                text="Invalid number of pieces", text_color="red"
            )

    def stop_live_mode(self):
        self.is_live_mode = False
        self.live_stop_event.set()
        self.live_btn.configure(
            text="Start Live Mode", fg_color="blue", hover_color="dark blue"
        )
        self.status_label.configure(text="Live mode stopped", text_color="gray")

        # Close webcam if open
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def live_processing_loop(self, num_pieces):
        try:
            # Load calibration
            config = self.load_config()
            if config is None:
                self.root.after(
                    0,
                    lambda: self.status_label.configure(
                        text="No calibration found", text_color="red"
                    ),
                )
                return

            M = np.array(config["M"])
            output_size = tuple(config["output_size"])

            # Load target pieces
            target_pieces = self.load_target_pieces(num_pieces)

            # Open webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.root.after(
                    0,
                    lambda: self.status_label.configure(
                        text="Cannot open webcam", text_color="red"
                    ),
                )
                return

            # Set webcam settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)
            self.cap.set(cv2.CAP_PROP_CONTRAST, 100)
            self.cap.set(cv2.CAP_PROP_SATURATION, 0)

            while not self.live_stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    continue

                # Store current frame
                self.captured_image = frame.copy()

                # Apply perspective transform
                transformed = cv2.warpPerspective(frame, M, output_size)
                self.warped_image = transformed

                # Stretch to 16:10
                stretched = self.stretch_to_16_10(transformed)
                self.stretched_image = stretched

                # Detect pieces (this also sets self.threshold_image)
                self.detected_pieces = self.detect_pieces(stretched, num_pieces)

                # Match pieces
                self.solution_map = self.match_pieces(
                    self.detected_pieces, target_pieces
                )

                # Update GUI
                self.root.after(0, self.update_display)

                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)

        except Exception as e:
            error_msg = str(e)
            self.root.after(
                0,
                lambda: self.status_label.configure(
                    text=f"Live mode error: {error_msg}", text_color="red"
                ),
            )

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = PuzzleSolverGUI()
    app.run()
