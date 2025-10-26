#!/usr/bin/env python3
"""
Dot Identifier for Custom 3D Calibration Plate

This script processes stereo camera calibration images containing an asymmetric circle grid
with 264 white dots on a black background. The calibration plate has:
- 23 columns total
- Grid A: Odd columns (1,3,5,7,9,11,13,15,17,19,21,23) with 12 rows each, except column 23 which has no row 2 dot (143 dots)
- Grid B: Even columns (2,4,6,8,10,12,14,16,18,20,22) with 11 rows each (121 dots)
- Dot spacing: 5mm within same grid
- Dot diameter: 1.2mm
- Grid A and B separated by 1mm depth

The script detects all dots, clusters them into columns, assigns them to grids,
and outputs ordered point arrays for stereo calibration.

Usage:
    python3 dot_identifier.py --image test_cam0_1.jpg

Author: Generated for Raspberry Pi 5 stereo camera calibration
"""

import cv2
import numpy as np
import argparse
import os
import sys
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from the specified path.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        np.ndarray: Loaded image or None if failed
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image '{image_path}'.")
        return None
    
    print(f"Loaded image: {image_path} with shape {image.shape}")
    return image


def preprocess_image(image: np.ndarray, output_prefix: str = None) -> np.ndarray:
    """
    Preprocess the image to isolate white dots on black background.
    
    Args:
        image (np.ndarray): Input BGR image
        output_prefix (str): Optional prefix for saving intermediate images
        
    Returns:
        np.ndarray: Binary image with white dots
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if output_prefix:
        cv2.imwrite(f"{output_prefix}_step1_grayscale.png", gray)
        print(f"Saved: {output_prefix}_step1_grayscale.png")
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if output_prefix:
        cv2.imwrite(f"{output_prefix}_step2_blurred.png", blurred)
        print(f"Saved: {output_prefix}_step2_blurred.png")
    
    # Apply threshold to get binary image (white dots on black background)
    # Using adaptive threshold to handle varying lighting conditions
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2
    )
    if output_prefix:
        cv2.imwrite(f"{output_prefix}_step3_binary.png", binary)
        print(f"Saved: {output_prefix}_step3_binary.png")
    
    # Morphological operations to clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if output_prefix:
        cv2.imwrite(f"{output_prefix}_step4_morphology_open.png", binary_open)
        print(f"Saved: {output_prefix}_step4_morphology_open.png")
    
    binary = cv2.morphologyEx(binary_open, cv2.MORPH_CLOSE, kernel)
    if output_prefix:
        cv2.imwrite(f"{output_prefix}_step5_morphology_close.png", binary)
        print(f"Saved: {output_prefix}_step5_morphology_close.png")
    
    return binary


def is_center_white(image: np.ndarray, x: int, y: int, threshold: int = 180) -> bool:
    """
    Check if the center pixel of a detected circle is white (bright).
    
    Args:
        image: Original image (grayscale or color)
        x, y: Center coordinates
        threshold: Minimum brightness value to consider white
        
    Returns:
        bool: True if center pixel is white enough
    """
    if len(image.shape) == 3:
        # Convert to grayscale if color image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    h, w = gray.shape
    if 0 <= x < w and 0 <= y < h:
        return gray[y, x] >= threshold
    return False


def detect_dots_hough(binary_image: np.ndarray, original_image: np.ndarray = None,
                      min_dist: int = 25, param1: int = 50, param2: int = 15, 
                      min_radius: int = 9, max_radius: int = 17, white_threshold: int = 180) -> List[Tuple[float, float]]:
    """
    Detect dots using HoughCircles with white center validation.
    
    Args:
        binary_image (np.ndarray): Binary image with white dots
        original_image (np.ndarray): Original image for center validation
        white_threshold (int): Threshold for white center detection
        
    Returns:
        List[Tuple[float, float]]: List of (x, y) coordinates of dot centers
    """
    # Apply HoughCircles to detect circular dots
    circles = cv2.HoughCircles(
        binary_image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,  # Minimum distance between circle centers
        param1=param1,     # Upper threshold for edge detection
        param2=param2,     # Accumulator threshold for center detection
        minRadius=min_radius, # Minimum circle radius
        maxRadius=max_radius  # Maximum circle radius
    )
    
    dots = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        total_detected = len(circles)
        
        # Filter circles by center color if original image provided
        if original_image is not None:
            filtered_circles = []
            for x, y, r in circles:
                if is_center_white(original_image, x, y, white_threshold):
                    filtered_circles.append((x, y, r))
            
            dots = [(float(x), float(y)) for x, y, r in filtered_circles]
            print(f"HoughCircles detected {total_detected} circles, {len(dots)} with white centers (threshold: {white_threshold})")
        else:
            dots = [(float(x), float(y)) for x, y, r in circles]
            print(f"HoughCircles detected {len(dots)} dots (no center validation)")
    
    return dots


def cluster_into_columns(dots: List[Tuple[float, float]], num_columns: int = 23) -> List[List[Tuple[float, float]]]:
    """
    Cluster dots into columns based on x-position with robust gradient-based refinement.
    
    Args:
        dots (List[Tuple[float, float]]): List of dot coordinates
        num_columns (int): Expected number of columns (23)
        
    Returns:
        List[List[Tuple[float, float]]]: List of columns, each containing dots
    """
    if len(dots) == 0:
        return [[] for _ in range(num_columns)]
    
    # Extract x-coordinates for clustering
    x_coords = np.array([dot[0] for dot in dots]).reshape(-1, 1)
    
    # Use K-means clustering to group dots into columns
    kmeans = KMeans(n_clusters=num_columns, random_state=42, n_init=10)
    column_labels = kmeans.fit_predict(x_coords)
    
    # Sort cluster centers by x-position to get column order
    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(cluster_centers)
    
    # Create mapping from cluster label to column index
    label_to_column = {old_label: new_idx for new_idx, old_label in enumerate(sorted_indices)}
    
    # Group dots by column
    initial_columns = [[] for _ in range(num_columns)]
    for i, dot in enumerate(dots):
        column_idx = label_to_column[column_labels[i]]
        initial_columns[column_idx].append(dot)
    
    # Calculate column centers and expected spacing pattern
    column_centers = []
    for column in initial_columns:
        if column:
            avg_x = np.mean([dot[0] for dot in column])
            column_centers.append(avg_x)
        else:
            column_centers.append(0)  # Will be interpolated
    
    # Robust column assignment using gradient analysis
    refined_columns = refine_column_assignment(dots, initial_columns, column_centers)
    
    # Sort dots within each column by y-coordinate (top to bottom)
    for column in refined_columns:
        column.sort(key=lambda dot: dot[1])
    
    print(f"Clustered dots into {num_columns} columns:")
    for i, column in enumerate(refined_columns):
        print(f"  Column {i+1}: {len(column)} dots")
    
    return refined_columns


def refine_column_assignment(dots: List[Tuple[float, float]], 
                           initial_columns: List[List[Tuple[float, float]]], 
                           column_centers: List[float]) -> List[List[Tuple[float, float]]]:
    """
    Refine column assignment using gradient analysis and expected spacing patterns.
    
    Args:
        dots: Original dot list
        initial_columns: Initial K-means clustering result
        column_centers: X-coordinates of column centers
        
    Returns:
        Refined column assignments
    """
    # Calculate average spacing between columns
    valid_centers = [c for c in column_centers if c > 0]
    if len(valid_centers) < 2:
        return initial_columns
    
    spacings = []
    for i in range(1, len(valid_centers)):
        spacings.append(valid_centers[i] - valid_centers[i-1])
    
    avg_spacing = np.median(spacings)
    spacing_tolerance = avg_spacing * 0.4  # 40% tolerance for spacing variation
    
    print(f"Average column spacing: {avg_spacing:.1f} pixels")
    print(f"Spacing tolerance: {spacing_tolerance:.1f} pixels")
    
    # Identify problematic assignments based on expected patterns
    refined_columns = [[] for _ in range(len(initial_columns))]
    
    # For each dot, find the best column assignment
    for dot in dots:
        x, y = dot
        best_column = 0
        min_distance = float('inf')
        
        # Find closest column center
        for col_idx, center_x in enumerate(column_centers):
            if center_x > 0:  # Valid center
                distance = abs(x - center_x)
                if distance < min_distance:
                    min_distance = distance
                    best_column = col_idx
        
        # Verify assignment makes sense based on spacing pattern
        expected_center = column_centers[best_column]
        if expected_center > 0 and min_distance < spacing_tolerance:
            refined_columns[best_column].append(dot)
        else:
            # Fall back to original assignment if refinement fails
            original_col = find_original_column(dot, initial_columns)
            if original_col is not None:
                refined_columns[original_col].append(dot)
    
    # Post-process: redistribute dots from over-populated to under-populated columns
    refined_columns = redistribute_boundary_dots(refined_columns, column_centers, avg_spacing)
    
    return refined_columns


def find_original_column(target_dot: Tuple[float, float], 
                        initial_columns: List[List[Tuple[float, float]]]) -> int:
    """Find which column a dot was originally assigned to."""
    for col_idx, column in enumerate(initial_columns):
        if target_dot in column:
            return col_idx
    return 0  # Default to first column


def redistribute_boundary_dots(columns: List[List[Tuple[float, float]]], 
                              column_centers: List[float], 
                              avg_spacing: float) -> List[List[Tuple[float, float]]]:
    """
    Redistribute dots that might be on column boundaries to improve balance.
    
    Args:
        columns: Current column assignments
        column_centers: X-coordinates of column centers  
        avg_spacing: Average spacing between columns
        
    Returns:
        Redistributed column assignments
    """
    # Expected dots per column based on grid pattern
    expected_counts = []
    for i in range(len(columns)):
        col_num = i + 1
        if col_num % 2 == 1:  # Odd columns (Grid A)
            if col_num == 23:  # Special case: column 23 missing row 2
                expected_counts.append(11)
            else:
                expected_counts.append(12)
        else:  # Even columns (Grid B)
            expected_counts.append(11)
    
    print("Expected vs Actual dot counts:")
    for i, (expected, actual) in enumerate(zip(expected_counts, [len(col) for col in columns])):
        if expected != actual:
            print(f"  Column {i+1}: Expected {expected}, Got {actual} ({'over' if actual > expected else 'under'} by {abs(actual - expected)})")
    
    # Apply targeted redistribution for problematic adjacent pairs
    # Dynamically find problem pairs instead of hard-coding them
    problem_pairs = []
    
    for i in range(len(columns) - 1):
        current_count = len(columns[i])
        next_count = len(columns[i + 1])
        current_expected = expected_counts[i]
        next_expected = expected_counts[i + 1]
        
        # Check if redistribution would help both columns
        if (current_count > current_expected and next_count < next_expected):
            problem_pairs.append((i, i + 1))
        elif (current_count < current_expected and next_count > next_expected):
            problem_pairs.append((i + 1, i))  # Reverse order
    
    print(f"Found {len(problem_pairs)} problematic column pairs for redistribution")
    
    for from_idx, to_idx in problem_pairs:
        if from_idx < len(columns) and to_idx < len(columns):
            from_count = len(columns[from_idx])
            to_count = len(columns[to_idx])
            from_expected = expected_counts[from_idx]
            to_expected = expected_counts[to_idx]
            
            print(f"Checking redistribution from column {from_idx + 1} to {to_idx + 1}")
            print(f"  From: {from_count}/{from_expected}, To: {to_count}/{to_expected}")
            
            # Check if redistribution would help both columns
            if from_count > from_expected and to_count < to_expected:
                # Find boundary dots between these columns
                boundary_dots = find_boundary_dots(columns[from_idx], columns[to_idx], 
                                                 column_centers[from_idx], column_centers[to_idx])
                
                print(f"  Found {len(boundary_dots)} boundary dot candidates")
                
                if boundary_dots:
                    # Move the best candidate
                    best_dot = boundary_dots[0]  # Closest to target column
                    columns[from_idx].remove(best_dot)
                    columns[to_idx].append(best_dot)
                    print(f"  SUCCESS: Redistributed dot {best_dot} from column {from_idx + 1} to column {to_idx + 1}")
                    
                    # Resort the target column
                    columns[to_idx].sort(key=lambda dot: dot[1])
                    
                    # Update counts for potential multiple redistributions
                    from_count -= 1
                    to_count += 1
                    
                    # Check if we can redistribute one more dot if needed
                    if from_count > from_expected and to_count < to_expected and len(boundary_dots) > 1:
                        second_dot = boundary_dots[1]
                        columns[from_idx].remove(second_dot)
                        columns[to_idx].append(second_dot)
                        print(f"  SUCCESS: Redistributed second dot {second_dot} from column {from_idx + 1} to column {to_idx + 1}")
                        columns[to_idx].sort(key=lambda dot: dot[1])
                else:
                    print(f"  No suitable boundary dots found for redistribution")
            else:
                print(f"  Redistribution not needed or wouldn't help")
    
    return columns


def find_boundary_dots(from_column: List[Tuple[float, float]], 
                      to_column: List[Tuple[float, float]],
                      from_center: float, 
                      to_center: float) -> List[Tuple[float, float]]:
    """
    Find dots in from_column that are closest to to_column center.
    
    Returns dots sorted by proximity to target column (closest first).
    """
    if not from_column:
        return []
    
    # Calculate which dots are closest to the boundary
    boundary_candidates = []
    
    # Find the midpoint between columns as boundary
    boundary_x = (from_center + to_center) / 2
    
    for dot in from_column:
        x, y = dot
        dist_to_target = abs(x - to_center)
        dist_to_boundary = abs(x - boundary_x)
        
        # Consider dots that are close to the boundary between columns
        # Use a more generous criterion for boundary detection
        if dist_to_boundary < abs(from_center - to_center) * 0.3:  # Within 30% of inter-column distance
            boundary_candidates.append((dot, dist_to_target))
    
    # If no boundary candidates, try the dots closest to target
    if not boundary_candidates:
        for dot in from_column:
            x, y = dot
            dist_to_target = abs(x - to_center)
            boundary_candidates.append((dot, dist_to_target))
    
    # Sort by distance to target column (closest first)
    boundary_candidates.sort(key=lambda x: x[1])
    
    print(f"    Boundary candidates: {len(boundary_candidates)}")
    if boundary_candidates:
        print(f"    Best candidate distance to target: {boundary_candidates[0][1]:.1f}px")
    
    return [dot for dot, _ in boundary_candidates[:2]]  # Return top 2 candidates


def assign_grids_and_sort(columns: List[List[Tuple[float, float]]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign columns to grids A and B, and create ordered point arrays.
    
    Grid A: Odd columns (1,3,5,7,9,11,13,15,17,19,21,23), 12 rows each, except column 23 missing row 2
    Grid B: Even columns (2,4,6,8,10,12,14,16,18,20,22), 11 rows each
    
    Args:
        columns (List[List[Tuple[float, float]]]): 23 columns of dots
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: gridA_points (143, 2) and gridB_points (121, 2)
    """
    gridA_points = []
    gridB_points = []
    
    for col_idx, column in enumerate(columns):
        column_num = col_idx + 1  # 1-indexed column numbers
        
        if column_num % 2 == 1:  # Odd columns go to Grid A
            if column_num == 23:
                # Special case: column 23 has no row 2 dot (missing one dot)
                # Expected: 11 dots instead of 12
                expected_dots = 11
            else:
                expected_dots = 12
            
            if len(column) != expected_dots:
                print(f"Warning: Column {column_num} (Grid A) has {len(column)} dots, expected {expected_dots}")
            
            for dot in column:
                gridA_points.append([dot[0], dot[1]])
                
        else:  # Even columns go to Grid B
            expected_dots = 11
            
            if len(column) != expected_dots:
                print(f"Warning: Column {column_num} (Grid B) has {len(column)} dots, expected {expected_dots}")
            
            for dot in column:
                gridB_points.append([dot[0], dot[1]])
    
    # Convert to numpy arrays
    gridA_points = np.array(gridA_points, dtype=np.float32)
    gridB_points = np.array(gridB_points, dtype=np.float32)
    
    print(f"Grid A points: {gridA_points.shape}")
    print(f"Grid B points: {gridB_points.shape}")
    
    return gridA_points, gridB_points


def save_results(gridA_points: np.ndarray, gridB_points: np.ndarray, 
                 columns: List[List[Tuple[float, float]]], output_prefix: str):
    """
    Save results as numpy arrays and CSV file.
    
    Args:
        gridA_points (np.ndarray): Grid A point coordinates
        gridB_points (np.ndarray): Grid B point coordinates
        columns (List[List[Tuple[float, float]]]): Original column data
        output_prefix (str): Prefix for output files
    """
    # Save as numpy arrays
    np.save(f"{output_prefix}_gridA_points.npy", gridA_points)
    np.save(f"{output_prefix}_gridB_points.npy", gridB_points)
    print(f"Saved numpy arrays: {output_prefix}_gridA_points.npy and {output_prefix}_gridB_points.npy")
    
    # Create CSV data
    csv_data = []
    
    for col_idx, column in enumerate(columns):
        column_num = col_idx + 1
        grid = "A" if column_num % 2 == 1 else "B"
        
        for row_idx, (x, y) in enumerate(column):
            csv_data.append({
                "grid": grid,
                "column": column_num,
                "row": row_idx + 1,
                "x": x,
                "y": y
            })
    
    # Save as CSV
    df = pd.DataFrame(csv_data)
    csv_filename = f"{output_prefix}_dots.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Saved CSV file: {csv_filename}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Grid A: {len(gridA_points)} points")
    print(f"  Grid B: {len(gridB_points)} points")
    print(f"  Total: {len(gridA_points) + len(gridB_points)} points")


def main():
    """
    Main function to process calibration images and detect dots.
    """
    parser = argparse.ArgumentParser(description="Detect and organize dots from 3D calibration plate images")
    parser.add_argument("--image", required=True, help="Path to the input image file")
    parser.add_argument("--output", help="Output prefix for saved files (default: based on input filename)")
    parser.add_argument("--debug", action="store_true", help="Enable additional debug output")
    
    # HoughCircles detection parameters
    parser.add_argument("--hough-min-dist", type=int, default=25,
                       help="Minimum distance between circle centers (default: 25)")
    parser.add_argument("--hough-param1", type=int, default=50,
                       help="HoughCircles param1 - edge threshold (default: 50, lower = more edges)")
    parser.add_argument("--hough-param2", type=int, default=15,
                       help="HoughCircles param2 - accumulator threshold (default: 15, lower = more circles)")
    parser.add_argument("--hough-min-radius", type=int, default=9,
                       help="Minimum circle radius in pixels (default: 9)")
    parser.add_argument("--hough-max-radius", type=int, default=17,
                       help="Maximum circle radius in pixels (default: 17)")
    parser.add_argument("--white-threshold", type=int, default=180,
                       help="White center threshold 0-255 (default: 180, higher = more strict)")
    
    args = parser.parse_args()
    
    # Set output prefix
    if args.output:
        output_prefix = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        output_prefix = base_name
    
    print(f"Processing image: {args.image}")
    print(f"Output prefix: {output_prefix}")
    
    # Step 1: Load image
    image = load_image(args.image)
    if image is None:
        sys.exit(1)
    
    # Step 2: Preprocess image
    print("Preprocessing image...")
    binary_image = preprocess_image(image, output_prefix)
    
    # Step 3: Detect dots using HoughCircles
    print("Detecting dots using HoughCircles...")
    dots = detect_dots_hough(binary_image, image,
                            min_dist=args.hough_min_dist,
                            param1=args.hough_param1, 
                            param2=args.hough_param2,
                            min_radius=args.hough_min_radius,
                            max_radius=args.hough_max_radius,
                            white_threshold=args.white_threshold)
    
    if len(dots) == 0:
        print("Error: No dots detected!")
        sys.exit(1)
    
    # Step 4: Cluster into columns
    print("Clustering dots into columns...")
    columns = cluster_into_columns(dots, num_columns=23)
    
    # Step 5: Assign to grids and sort
    print("Assigning dots to grids...")
    gridA_points, gridB_points = assign_grids_and_sort(columns)
    
    # Step 6: Save results
    print("Saving results...")
    save_results(gridA_points, gridB_points, columns, output_prefix)
    
    # Always create visualization and save to dot_identifier_tests folder
    print("Creating visualization...")
    
    # Calculate figure size to maintain original resolution
    height, width = image.shape[:2]
    fig_width = width / 100  # 100 DPI base
    fig_height = height / 100
    
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Plot Grid A points in red (larger dots for visibility)
    if len(gridA_points) > 0:
        plt.scatter(gridA_points[:, 0], gridA_points[:, 1], c='red', s=30, alpha=0.8, label='Grid A', edgecolors='white', linewidth=0.5)
    
    # Plot Grid B points in blue (larger dots for visibility)
    if len(gridB_points) > 0:
        plt.scatter(gridB_points[:, 0], gridB_points[:, 1], c='blue', s=30, alpha=0.8, label='Grid B', edgecolors='white', linewidth=0.5)
    
    plt.legend(fontsize=12)
    plt.title(f"Detected Dots: Grid A (Red) and Grid B (Blue) - Total: {len(gridA_points) + len(gridB_points)}", fontsize=14)
    plt.axis('off')
    
    # Save visualization using the same output prefix
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    if args.output:
        # Use the output prefix directory for visualization
        output_dir = os.path.dirname(args.output)
        if output_dir:  # If output has a directory component
            visualization_filename = f"{args.output}_visualization.png"
        else:  # If output is just a filename
            visualization_filename = f"{args.output}_visualization.png"
    else:
        # Fallback to dot_identifier_tests folder
        visualization_filename = f"/home/francodp/camera_app/dot_identifier_tests/{base_name}_tested.png"
    
    plt.savefig(visualization_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved high-resolution visualization: {visualization_filename}")
    plt.close()
    
    print("Processing complete!")


if __name__ == "__main__":
    main()