#!/usr/bin/env python3
"""
3D Dot Pattern Stereo Calibration - Method 2
============================================

This script performs stereo camera calibration using a 3D calibration plate
with white dots arranged in two grids at different depths.

Uses pre-processed .npy files with identified and sorted dot positions.

Grid Configuration:
- Grid A: 1mm behind Grid B (further from cameras) - 143 dots
- Grid B: Reference plane (0mm depth) - 121 dots  
- Missing dot in Grid A at column 12, row 2
- Total: 264 dots with known 3D positions

Author: Camera Calibration System
Date: 2025-10-01
"""

import cv2
import numpy as np
import glob
import os
import json
from pathlib import Path

# === CONFIGURATION ===
# 3D Calibration Plate Parameters
GRID_A_DEPTH = -1.0  # 1mm behind Grid B (negative = away from camera)
GRID_B_DEPTH = 0.0   # Reference plane
DOT_SPACING = 5.0    # 5mm spacing between dots

# Grid layout (actual plate geometry)
GRID_A_ROWS = 12     # Grid A: 12x12 grid
GRID_A_COLS = 12
GRID_A_MISSING_COL = 12  # Missing dot at column 12
GRID_A_MISSING_ROW = 2   # Missing dot at row 2

GRID_B_ROWS = 11     # Grid B: 11x11 grid  
GRID_B_COLS = 11     # (11x11 = 121 dots)

EXPECTED_DOTS_A = 143  # Grid A: 12x12 - 1 = 143 dots
EXPECTED_DOTS_B = 121  # Grid B: 11x11 = 121 dots
EXPECTED_TOTAL = 264   # Total: 143 + 121 = 264 dots

# === Paths ===
dot_data_dir = "/home/francodp/camera_app/1_Dot identification and sorting/Completed dot identification"
output_dir = "results"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

def create_3d_object_points():
    """
    Create 3D object points for the calibration plate using column-by-column ordering.
    
    Grid B: 121 dots (11×11) at reference plane (z=0)
    Grid A: 143 dots (12×12 - 1) at 1mm behind Grid B (z=-1)
    Missing dot in Grid A at column 12, row 2
    
    Returns:
        tuple: (grid_b_points, grid_a_points) - 3D coordinates in mm
    """
    print("Creating 3D object points for calibration plate...")
    
    # Grid B: 11×11 = 121 dots at z=0 (reference plane)
    grid_b_points = []
    for col in range(GRID_B_COLS):
        for row in range(GRID_B_ROWS):
            x = col * DOT_SPACING
            y = row * DOT_SPACING
            z = GRID_B_DEPTH
            grid_b_points.append([x, y, z])
    
    # Grid A: 12×12 - 1 = 143 dots at z=-1 (1mm behind)
    grid_a_points = []
    for col in range(GRID_A_COLS):
        for row in range(GRID_A_ROWS):
            # Skip the missing dot at column 12, row 2
            if col == (GRID_A_MISSING_COL - 1) and row == (GRID_A_MISSING_ROW - 1):
                continue
                
            x = col * DOT_SPACING
            y = row * DOT_SPACING
            z = GRID_A_DEPTH
            grid_a_points.append([x, y, z])
    
    grid_b_points = np.array(grid_b_points, dtype=np.float32)
    grid_a_points = np.array(grid_a_points, dtype=np.float32)
    
    print(f"✓ Created 3D object points:")
    print(f"  Grid B: {len(grid_b_points)} points (11×11) at z={GRID_B_DEPTH}mm")
    print(f"  Grid A: {len(grid_a_points)} points (12×12-1) at z={GRID_A_DEPTH}mm")
    print(f"  Total: {len(grid_b_points) + len(grid_a_points)} points")
    print(f"  Missing dot: column {GRID_A_MISSING_COL}, row {GRID_A_MISSING_ROW}")
    
    return grid_b_points, grid_a_points

def load_dot_data(camera_side, image_number):
    """
    Load dot data from .npy files for a specific camera and image.
    
    Args:
        camera_side (str): 'Left Camera' or 'Right Camera'
        image_number (str): Image number (e.g., '010')
    
    Returns:
        tuple: (grid_b_dots, grid_a_dots) - 2D pixel coordinates
    """
    base_path = Path(dot_data_dir) / camera_side
    
    # Find the correct directory (should match pattern like 'dot_identifier_test_left010')
    camera_prefix = 'left' if 'Left' in camera_side else 'right'
    pattern = f"dot_identifier_test_{camera_prefix}{image_number}"
    
    dot_dirs = list(base_path.glob(f"*{pattern}*"))
    
    if not dot_dirs:
        raise FileNotFoundError(f"No dot data directory found for {camera_side} image {image_number}")
    
    dot_dir = dot_dirs[0]  # Take the first match
    
    # Load grid files
    grid_a_file = dot_dir / f"{camera_prefix}{image_number}_gridA_points.npy"
    grid_b_file = dot_dir / f"{camera_prefix}{image_number}_gridB_points.npy"
    
    if not grid_a_file.exists() or not grid_b_file.exists():
        raise FileNotFoundError(f"Grid files not found in {dot_dir}")
    
    grid_a_dots = np.load(grid_a_file)
    grid_b_dots = np.load(grid_b_file)
    
    print(f"  {camera_side}: Grid A={len(grid_a_dots)} dots, Grid B={len(grid_b_dots)} dots")
    
    return grid_b_dots, grid_a_dots

def find_available_images():
    """
    Find available dot data files.
    
    Returns:
        list: Available image numbers
    """
    left_path = Path(dot_data_dir) / "Left Camera"
    
    if not left_path.exists():
        return []
    
    # Find all dot directories and extract image numbers
    image_numbers = []
    for dot_dir in left_path.iterdir():
        if dot_dir.is_dir() and "dot_identifier_test_left" in dot_dir.name:
            # Extract number from directory name like 'dot_identifier_test_left010'
            parts = dot_dir.name.split('left')
            if len(parts) > 1:
                image_numbers.append(parts[1])
    
    return sorted(image_numbers)

def main():
    """Main calibration routine using pre-processed .npy dot files for a single image pair"""
    print("=== 3D Dot Pattern Stereo Calibration ===")
    print(f"Expected dots: {EXPECTED_TOTAL}")
    print(f"Grid A: {GRID_A_ROWS}×{GRID_A_COLS}-1 = {EXPECTED_DOTS_A} dots")
    print(f"Grid B: {GRID_B_ROWS}×{GRID_B_COLS} = {EXPECTED_DOTS_B} dots")
    print(f"Depth separation: {abs(GRID_A_DEPTH - GRID_B_DEPTH)}mm")
    print(f"Missing dot: column {GRID_A_MISSING_COL}, row {GRID_A_MISSING_ROW}")
    
    # Find available dot data
    available_images = find_available_images()
    
    if not available_images:
        print(f"\n✗ No dot data found in {dot_data_dir}/")
        print("Please ensure you have processed dot identification files")
        return
    
    print(f"\nFound dot data for images: {available_images}")
    
    # Use the first available image (or ask user to specify)
    if len(available_images) == 1:
        img_num = available_images[0]
        print(f"Using image pair: {img_num}")
    else:
        print(f"Multiple images available: {available_images}")
        img_num = input(f"Enter image number to use (default: {available_images[0]}): ").strip()
        if not img_num:
            img_num = available_images[0]
        
        if img_num not in available_images:
            print(f"✗ Image {img_num} not found in available data")
            return
    
    print(f"\nProcessing image pair: {img_num}")
    
    # Create 3D object points
    grid_b_3d, grid_a_3d = create_3d_object_points()
    
    # Combine grids for calibration (Grid B first, then Grid A)
    combined_3d_points = np.vstack([grid_b_3d, grid_a_3d])
    
    try:
        # Load dot data for both cameras
        print(f"Loading dot data for image {img_num}...")
        left_grid_b, left_grid_a = load_dot_data("Left Camera", img_num)
        right_grid_b, right_grid_a = load_dot_data("Right Camera", img_num)
        
        # Validate dot counts
        if len(left_grid_a) != EXPECTED_DOTS_A or len(left_grid_b) != EXPECTED_DOTS_B:
            print(f"✗ Left camera: Invalid dot counts (A:{len(left_grid_a)}, B:{len(left_grid_b)})")
            print(f"  Expected: A={EXPECTED_DOTS_A}, B={EXPECTED_DOTS_B}")
            return
            
        if len(right_grid_a) != EXPECTED_DOTS_A or len(right_grid_b) != EXPECTED_DOTS_B:
            print(f"✗ Right camera: Invalid dot counts (A:{len(right_grid_a)}, B:{len(right_grid_b)})")
            print(f"  Expected: A={EXPECTED_DOTS_A}, B={EXPECTED_DOTS_B}")
            return
        
        # Combine grids (Grid B first, then Grid A to match 3D points order)
        left_combined = np.vstack([left_grid_b, left_grid_a])
        right_combined = np.vstack([right_grid_b, right_grid_a])
        
        print(f"✓ Loaded {len(left_combined)} dots per camera")
        print(f"  Left Grid B: {len(left_grid_b)} dots, Grid A: {len(left_grid_a)} dots")
        print(f"  Right Grid B: {len(right_grid_b)} dots, Grid A: {len(right_grid_a)} dots")
        
        # For single image calibration, we need to create arrays with one set of points
        # OpenCV expects lists of point arrays (one per image)
        objpoints = [combined_3d_points]  # 3D points in real world space
        imgpoints_left = [left_combined.astype(np.float32)]  # 2D points in left image
        imgpoints_right = [right_combined.astype(np.float32)]  # 2D points in right image
        
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
        return
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Get image size (assume standard Pi Camera Module 3 resolution)
    img_shape = (4608, 2592)  # (width, height)
    
    print(f"\nPerforming stereo calibration with single image pair...")
    print(f"Image size: {img_shape}")
    print(f"Calibration points per image: {len(combined_3d_points)}")
    
    # Note: Single image calibration has limitations but can work for initial testing
    print("\n⚠️  Note: Single image calibration has limitations:")
    print("   - May not capture full range of distortions")
    print("   - Less robust than multi-image calibration")
    print("   - Good for initial testing and validation")
    
    # Create initial intrinsic matrix estimates for non-planar calibration
    # Using proven values from checkerboard calibration as better initial estimates
    print("\nUsing checkerboard calibration results as initial estimates...")
    
    # From checkerboard calibration (V4_test at the new angle)
    initial_left_matrix = np.array([
        [3508.5, 0, 2160.7],
        [0, 3504.8, 1395.1],
        [0, 0, 1]
    ], dtype=np.float32)
    
    initial_right_matrix = np.array([
        [3511.1, 0, 2287.6],
        [0, 3522.8, 1312.4],
        [0, 0, 1]
    ], dtype=np.float32)
    
    print(f"Left camera initial estimates: fx={initial_left_matrix[0,0]:.1f}, fy={initial_left_matrix[1,1]:.1f}")
    print(f"Right camera initial estimates: fx={initial_right_matrix[0,0]:.1f}, fy={initial_right_matrix[1,1]:.1f}")
    print("These proven estimates should give much better 3D calibration results")
    
    # Perform stereo calibration
    try:
        # Individual camera calibrations first (with proven initial estimates)
        print("\nCalibrating individual cameras...")
        ret_left, mtx_left, dist_left, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints_left, img_shape, initial_left_matrix.copy(), None,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS
        )
        ret_right, mtx_right, dist_right, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints_right, img_shape, initial_right_matrix.copy(), None,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS
        )
        
        print(f"Left camera RMS error: {ret_left:.3f}")
        print(f"Right camera RMS error: {ret_right:.3f}")
        
        # Stereo calibration
        print("Performing stereo calibration...")
        ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right,
            mtx_left, dist_left, mtx_right, dist_right,
            img_shape,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        print(f"Stereo calibration RMS error: {ret_stereo:.3f}")
        
        # Calculate baseline
        baseline = np.linalg.norm(T) * 1000  # Convert to mm
        print(f"Baseline distance: {baseline:.1f}mm")
        
        # Rectification
        print("Computing rectification...")
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtx_left, dist_left, mtx_right, dist_right,
            img_shape, R, T, alpha=0
        )
        
        # Save results
        print("Saving calibration results...")
        
        # Save as NPZ file
        np.savez(os.path.join(output_dir, "3d_plate_stereo_calib.npz"),
                 cameraMatrix1=mtx_left, distCoeffs1=dist_left,
                 cameraMatrix2=mtx_right, distCoeffs2=dist_right,
                 R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
                 baseline_mm=baseline, rms_error=ret_stereo,
                 image_used=img_num, method="3D_dots_single_image")
        
        # Save as JSON file (human-readable)
        calib_dict = {
            "method": "3D_dots_single_image",
            "image_used": img_num,
            "rms_error": float(ret_stereo),
            "baseline_mm": float(baseline),
            "left_camera": {
                "camera_matrix": mtx_left.tolist(),
                "distortion_coeffs": dist_left.flatten().tolist(),
                "focal_lengths": {"fx": float(mtx_left[0,0]), "fy": float(mtx_left[1,1])},
                "principal_point": {"cx": float(mtx_left[0,2]), "cy": float(mtx_left[1,2])}
            },
            "right_camera": {
                "camera_matrix": mtx_right.tolist(),
                "distortion_coeffs": dist_right.flatten().tolist(),
                "focal_lengths": {"fx": float(mtx_right[0,0]), "fy": float(mtx_right[1,1])},
                "principal_point": {"cx": float(mtx_right[0,2]), "cy": float(mtx_right[1,2])}
            },
            "stereo": {
                "rotation_matrix": R.tolist(),
                "translation_vector": T.flatten().tolist(),
                "essential_matrix": E.tolist(),
                "fundamental_matrix": F.tolist()
            }
        }
        
        with open(os.path.join(output_dir, "3d_plate_stereo_calib.json"), "w") as f:
            json.dump(calib_dict, f, indent=4)
        
        # Calculate validation errors
        # Calculate mean reprojection error per camera
        left_errors = []
        right_errors = []
        
        for i, (obj_pts, left_pts, right_pts) in enumerate(zip(objpoints, imgpoints_left, imgpoints_right)):
            # Reproject for left camera
            rvec_l, tvec_l = cv2.solvePnP(obj_pts, left_pts, mtx_left, dist_left)[:2]
            proj_left, _ = cv2.projectPoints(obj_pts, rvec_l, tvec_l, mtx_left, dist_left)
            left_error = cv2.norm(left_pts, proj_left.reshape(-1, 2), cv2.NORM_L2) / len(obj_pts)
            left_errors.append(left_error)
            
            # Reproject for right camera
            rvec_r, tvec_r = cv2.solvePnP(obj_pts, right_pts, mtx_right, dist_right)[:2]
            proj_right, _ = cv2.projectPoints(obj_pts, rvec_r, tvec_r, mtx_right, dist_right)
            right_error = cv2.norm(right_pts, proj_right.reshape(-1, 2), cv2.NORM_L2) / len(obj_pts)
            right_errors.append(right_error)
        
        left_mean_error = np.mean(left_errors)
        right_mean_error = np.mean(right_errors)
        overall_mean_error = (left_mean_error + right_mean_error) / 2
        
        # Calculate baseline improvement (comparing to initial estimate if available)
        improvement = 0  # Single image doesn't have improvement metric
        
        # Save detailed report
        report = f"""=== 3D PLATE STEREO CALIBRATION RESULTS ===

CALIBRATION DATA:
- Total correspondences: {len(combined_3d_points)} points
- Grid A (Z={GRID_A_DEPTH}mm): {EXPECTED_DOTS_A} points
- Grid B (Z={GRID_B_DEPTH}mm): {EXPECTED_DOTS_B} points
- Dot spacing: {DOT_SPACING}mm

CALIBRATION QUALITY:
- RMS reprojection error: {ret_stereo:.4f} pixels
- Final baseline: {baseline:.2f}mm
- Baseline calibration RMS: {ret_stereo:.4f} pixels
- Improvement: {improvement:.4f} pixels

VALIDATION ERRORS:
- Left camera mean error: {left_mean_error:.10f}px
- Right camera mean error: {right_mean_error:.4f}px
- Overall mean error: {overall_mean_error:.10f}px

FINAL CAMERA PARAMETERS:

Left Camera:
- Focal lengths: fx={mtx_left[0,0]:.1f}, fy={mtx_left[1,1]:.1f}
- Principal point: cx={mtx_left[0,2]:.1f}, cy={mtx_left[1,2]:.1f}
- Distortion: k1={dist_left[0,0]:.4f}, k2={dist_left[0,1]:.4f}, p1={dist_left[0,2]:.4f}, p2={dist_left[0,3]:.4f}, k3={dist_left[0,4]:.4f}

Right Camera:
- Focal lengths: fx={mtx_right[0,0]:.1f}, fy={mtx_right[1,1]:.1f}
- Principal point: cx={mtx_right[0,2]:.1f}, cy={mtx_right[1,2]:.1f}
- Distortion: k1={dist_right[0,0]:.4f}, k2={dist_right[0,1]:.4f}, p1={dist_right[0,2]:.4f}, p2={dist_right[0,3]:.4f}, k3={dist_right[0,4]:.4f}

Stereo Parameters:
- Translation: [{T[0,0]:.6f}, {T[1,0]:.6f}, {T[2,0]:.6f}] meters
- Baseline: {baseline:.2f}mm

FILES GENERATED:
- 3d_plate_stereo_calib.npz: NumPy format
- 3d_plate_stereo_calib.json: JSON format (human-readable)
- 3d plate calibration report.txt: This report
"""
        
        with open(os.path.join(output_dir, "3d plate calibration report.txt"), "w") as f:
            f.write(report)
        
        print("\n✓ 3D Plate Stereo Calibration Complete!")
        print(f"📁 Results saved in: {output_dir}/")
        print(f"📊 RMS Error: {ret_stereo:.4f} pixels")
        print(f"📏 Baseline: {baseline:.2f}mm")
        print(f"🎯 Image used: {img_num}")
        print(f"📝 Total points: {len(combined_3d_points)}")
        
    except Exception as e:
        print(f"\n✗ Calibration failed: {e}")
        print("Check that dot identification files are valid and accessible")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
