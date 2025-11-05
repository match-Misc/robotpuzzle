import argparse
import json
import os
import sys

import cv2
import numpy as np


def get_robust_orientation(contour, image_shape):
    """
    Calculate orientation using PCA (Principal Component Analysis) on contour points.
    This method finds the main axis of the shape without assuming symmetry.
    """
    # Convert contour to numpy array of points
    points = contour.reshape(-1, 2).astype(np.float64)

    # Calculate mean
    mean = np.mean(points, axis=0)

    # Center the points
    centered = points - mean

    # Calculate covariance matrix
    cov = np.cov(centered.T)

    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # The eigenvector corresponding to the largest eigenvalue is the main axis
    main_axis = eigenvectors[:, np.argmax(eigenvalues)]

    # Calculate angle from the horizontal
    angle_rad = np.arctan2(main_axis[1], main_axis[0])
    angle_deg = np.degrees(angle_rad)

    # Normalize to 0-180 range
    if angle_deg < 0:
        angle_deg += 180

    return angle_deg, main_axis


# --- 1. Parse Arguments ---
parser = argparse.ArgumentParser(description="Detect puzzle pieces in an image.")
parser.add_argument("image_path", help="Path to the input image")
parser.add_argument("num_pieces", type=int, help="Number of pieces to detect")
parser.add_argument("--show", action="store_true", help="Display the detection image")
parser.add_argument(
    "--savemasks", action="store_true", help="Save colored masks for each piece"
)

args = parser.parse_args()

image_path = args.image_path
num_pieces = args.num_pieces
show_image = args.show
save_masks = args.savemasks
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Error: Could not load image from path: {image_path}")
else:
    print("Image loaded successfully. Detecting individual shapes...")
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # --- 2. Preprocessing the Image ---
    # Invert the image so the white puzzle pieces become black objects
    # on a white background. This is what findContours expects.
    # A threshold of 240 ensures the anti-aliased gray lines become white.
    _, thresh = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY_INV)

    # --- 3. Find All Contours ---
    # THE CRITICAL CHANGE: Use cv2.RETR_LIST instead of cv2.RETR_EXTERNAL.
    # cv2.RETR_LIST finds all contours without establishing a hierarchy.
    # This ensures we get each individual puzzle piece.
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # --- 4. Filter and Sort the Contours ---
    # The above step will find the shapes AND the outer border.
    # We can isolate the shapes by sorting all found contours by their area
    # in descending order and taking the top num_pieces. This is a robust way to
    # discard the frame and any other potential noise.
    if len(contours) >= num_pieces:
        # Sort by area, largest first
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Select the largest contours, which will be our shapes
        shapes = sorted_contours[1 : num_pieces + 1]

        print(f"\nSuccessfully isolated {len(shapes)} shapes. Analyzing them now...\n")
        print("-" * 50)

        # --- 5. Analyze and Describe Each Shape ---
        pieces_data = []
        for i, cnt in enumerate(shapes):
            shape_number = i + 1

            # --- Description Properties ---
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            M = cv2.moments(cnt)

            # Calculate centroid
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            # Calculate Hu Moments for shape descriptor
            hu_moments = cv2.HuMoments(M).flatten().tolist()

            # Calculate orientation using robust moments method
            angle, main_axis = get_robust_orientation(cnt, thresh.shape)

            color = tuple(np.random.randint(50, 220, 3).tolist())

            # Collect piece data
            piece_data = {
                "id": shape_number,
                "centroid": [cx, cy],
                "area": area,
                "perimeter": perimeter,
                "hu_moments": hu_moments,
                "orientation": angle,
                "main_axis": main_axis.tolist(),
                "color": color,
            }
            pieces_data.append(piece_data)

            # Print the description
            print(f"Shape #{shape_number}")
            print(f"  - Area: {area:.2f} pixels²")
            print(f"  - Perimeter: {perimeter:.2f} pixels")
            print(f"  - Centroid (Center): ({cx}, {cy})")
            print("-" * 50)

            # --- 6. Visualize the Results ---
            # Fill the entire shape area with a unique color
            cv2.drawContours(output_image, [cnt], -1, color, -1)  # Fill the shape

            # Draw the centroid as a red circle
            cv2.circle(output_image, (cx, cy), 10, (0, 0, 255), -1)

            # Draw the PCA axis as a green line
            axis_length = 150  # Length of the axis line
            end_point = (
                int(cx + main_axis[0] * axis_length),
                int(cy + main_axis[1] * axis_length),
            )
            cv2.line(output_image, (cx, cy), end_point, (0, 255, 0), 3)

            # Put the shape number text
            cv2.putText(
                output_image,
                f"#{shape_number}",
                (cx - 25, cy + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
    else:
        print(
            f"Error: Expected to find at least {num_pieces} shapes, but only found {len(contours)}."
        )
        sys.exit(1)

    # --- 7. Output JSON ---
    json_filename = f"configs/Puzzle_{num_pieces}.json"
    with open(json_filename, "w") as f:
        json.dump(pieces_data, f, indent=4)
    print(f"\nJSON data saved to '{json_filename}'.")

    # --- 8. Save Colored Masks ---
    if save_masks:
        # Ensure configs directory exists
        os.makedirs("configs", exist_ok=True)

        for piece in pieces_data:
            piece_id = piece["id"]
            color = piece["color"]
            # Create a colored mask with transparent background
            rgba = np.zeros((thresh.shape[0], thresh.shape[1], 4), dtype=np.uint8)
            mask = np.zeros_like(thresh, dtype=np.uint8)
            cv2.drawContours(mask, [shapes[piece_id - 1]], -1, 255, -1)
            rgba[mask == 255] = [color[0], color[1], color[2], 255]

            # Save the mask as an image
            mask_filename = f"configs/piece_{piece_id}_mask.png"
            cv2.imwrite(mask_filename, rgba)
            print(f"Colored mask for piece {piece_id} saved to '{mask_filename}'.")

    # --- 9. Display Image if Requested ---
    if show_image:
        # Resize to half size for better screen fit
        display_image = cv2.resize(output_image, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Detected Pieces", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
