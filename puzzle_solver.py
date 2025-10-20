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
        self.solution_image = None
        self.detected_pieces = []
        self.solution_map = []
        self.hovered_piece = None

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

        # Status label
        self.status_label = ctk.CTkLabel(control_frame, text="Ready", text_color="gray")
        self.status_label.pack(side="right", padx=(10, 0))

        # Image display area
        display_frame = ctk.CTkFrame(main_frame)
        display_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Left panel - Captured image and highlighted pieces
        left_panel = ctk.CTkFrame(display_frame)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 5))

        # Raw image section
        raw_frame = ctk.CTkFrame(left_panel)
        raw_frame.pack(fill="both", expand=True, pady=(10, 5))

        raw_label = ctk.CTkLabel(
            raw_frame, text="Raw Puzzle", font=ctk.CTkFont(size=14, weight="bold")
        )
        raw_label.pack(pady=(5, 5))

        self.raw_canvas = ctk.CTkCanvas(raw_frame, bg="gray20", highlightthickness=0)
        self.raw_canvas.pack(fill="both", expand=True, padx=10, pady=(0, 5))

        # Highlighted pieces section
        highlighted_frame = ctk.CTkFrame(left_panel)
        highlighted_frame.pack(fill="both", expand=True, pady=(0, 10))

        highlighted_label = ctk.CTkLabel(
            highlighted_frame,
            text="Highlighted Pieces",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        highlighted_label.pack(pady=(5, 5))

        self.captured_canvas = ctk.CTkCanvas(
            highlighted_frame, bg="gray20", highlightthickness=0
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

    def check_solution_exists(self):
        try:
            num_pieces = int(self.num_pieces.get())
            json_path = f"Puzzle_{num_pieces}.json"
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
        except Exception as e:
            print(f"Error loading solution image: {e}")

    def display_solution_image(self):
        if self.solution_image is not None:
            # Resize to fit canvas
            canvas_width = self.solution_canvas.winfo_width()
            canvas_height = self.solution_canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                img_height, img_width = self.solution_image.shape[:2]
                scale = min(canvas_width / img_width, canvas_height / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)

                resized = cv2.resize(self.solution_image, (new_width, new_height))
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
            subprocess.Popen([sys.executable, "puzzle_capture.py"])
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
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
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

            # Stretch to 16:10
            stretched = self.stretch_to_16_10(transformed)

            # Detect pieces
            self.detected_pieces = self.detect_pieces(stretched, num_pieces)

            # Load target pieces
            target_pieces = self.load_target_pieces(num_pieces)

            # Match pieces
            self.solution_map = self.match_pieces(self.detected_pieces, target_pieces)

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
            with open("config.json", "r") as f:
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

    def detect_pieces(self, image, num_pieces):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to smooth edges and reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold the blurred image
        _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        min_area = 1000
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

    def load_target_pieces(self, puzzle_size):
        try:
            with open(f"Puzzle_{puzzle_size}.json", "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def match_pieces(self, detected_pieces, target_pieces):
        import scipy.optimize

        # Create cost matrix
        cost_matrix = np.zeros((len(detected_pieces), len(target_pieces)))
        for i, (_, detected_hu, _, _) in enumerate(detected_pieces):
            for j, target in enumerate(target_pieces):
                target_hu = np.array(target["hu_moments"])
                cost_matrix[i, j] = np.linalg.norm(detected_hu - target_hu)

        # Solve assignment problem using Hungarian algorithm
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)

        solution_map = []
        for detected_idx, target_idx in zip(row_ind, col_ind):
            detected_cnt, detected_hu, detected_centroid, detected_angle = (
                detected_pieces[detected_idx]
            )
            target = target_pieces[target_idx]

            best_target_centroid = target["centroid"]
            best_target_angle = target["orientation"]
            best_match = target["id"]

            translation = (
                best_target_centroid[0] - detected_centroid[0],
                best_target_centroid[1] - detected_centroid[1],
            )
            rotation = best_target_angle - detected_angle

            # Debug logging
            # print(
            #     f"Piece {best_match}: detected_angle={detected_angle:.2f}, target_angle={best_target_angle:.2f}, rotation={rotation:.2f}"
            # )

            solution_map.append(
                (
                    detected_centroid,
                    best_target_centroid,
                    translation,
                    rotation,
                    best_match,
                )
            )

        # Sort solution_map to match the order of detected_pieces
        solution_map = [solution_map[i] for i in np.argsort(row_ind)]

        return solution_map

    def update_display(self):
        self.display_raw_image()
        self.display_captured_image()
        self.display_solution_image()

    def display_raw_image(self):
        if self.captured_image is not None:
            # Resize to fit canvas
            canvas_width = self.raw_canvas.winfo_width()
            canvas_height = self.raw_canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                img_height, img_width = self.captured_image.shape[:2]
                scale = min(canvas_width / img_width, canvas_height / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)

                resized = cv2.resize(self.captured_image, (new_width, new_height))
                img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                self.raw_photo = ImageTk.PhotoImage(img)

                # Center the image
                x = (canvas_width - new_width) // 2
                y = (canvas_height - new_height) // 2

                self.raw_canvas.create_image(x, y, anchor="nw", image=self.raw_photo)

    def display_captured_image(self):
        if self.captured_image is not None and self.detected_pieces:
            # Create black background image for highlighted pieces only
            img_height, img_width = self.captured_image.shape[:2]
            highlighted = np.zeros((img_height, img_width, 3), dtype=np.uint8)

            # Generate unique colors
            num_pieces = len(self.detected_pieces)
            colors = []
            for i in range(num_pieces):
                hue = int(180 * i / num_pieces)
                hsv_color = np.uint8([[[hue, 255, 255]]])
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
                colors.append((int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])))

            # Draw filled contours on black background
            for i, (cnt, _, _, _) in enumerate(self.detected_pieces):
                cv2.drawContours(highlighted, [cnt], -1, colors[i], -1)

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
        if not self.detected_pieces or not self.solution_map:
            return

        # Determine which canvas triggered the event
        canvas = event.widget
        canvas_x = canvas.canvasx(event.x)
        canvas_y = canvas.canvasy(event.y)

        # Convert to image coordinates
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if self.captured_image is not None:
            img_height, img_width = self.captured_image.shape[:2]
            scale = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)

            offset_x = (canvas_width - new_width) // 2
            offset_y = (canvas_height - new_height) // 2

            img_x = (canvas_x - offset_x) / scale
            img_y = (canvas_y - offset_y) / scale

            # Check if mouse is over a piece
            for i, (cnt, _, centroid, _) in enumerate(self.detected_pieces):
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
        if self.hovered_piece is not None:
            self.hovered_piece = None
            self.display_solution_image()
            self.info_label.configure(text="Hover over pieces to see target positions")

    def highlight_solution_piece(self, piece_index):
        if piece_index >= len(self.solution_map):
            return

        _, target_centroid, _, _, piece_id = self.solution_map[piece_index]

        if self.solution_image is not None:
            highlighted = self.solution_image.copy()

            # Draw a larger circle at the target position for better visibility
            cv2.circle(highlighted, tuple(target_centroid), 50, (0, 255, 0), 8)
            cv2.putText(
                highlighted,
                f"Piece {piece_id}",
                (target_centroid[0] - 60, target_centroid[1] - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3,
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

        current_centroid, target_centroid, translation, rotation, piece_id = (
            self.solution_map[piece_index]
        )

        info_text = (
            f"Piece {piece_id}: Current ({current_centroid[0]}, {current_centroid[1]}) -> "
            f"Target ({target_centroid[0]}, {target_centroid[1]}) | "
            f"Translate ({translation[0]}, {translation[1]}) | Rotate {rotation:.1f}°"
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
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
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

                # Stretch to 16:10
                stretched = self.stretch_to_16_10(transformed)

                # Detect pieces
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
