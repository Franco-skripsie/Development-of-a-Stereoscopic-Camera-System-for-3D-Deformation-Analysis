import cv2
import numpy as np
import glob
import os
import json

# === CONFIGURATION ===
import numpy as np
import glob
import os

# === CONFIGURATION ===
chessboard_size = (9, 6)  # inner corners (width, height)
square_size = 0.0145  # in meters (14.5mm)

# === Paths ===
calib_dir = "calib"
left_images = sorted(glob.glob(os.path.join(calib_dir, "left*.png")))
right_images = sorted(glob.glob(os.path.join(calib_dir, "right*.png")))

# === Prepare object points ===
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D points in real world space
imgpoints_left = []  # 2D points in left images
imgpoints_right = []  # 2D points in right images

# === Detect corners ===
valid_pairs = 0
debug_dir = "debug_corner_grayscale_3"
os.makedirs(debug_dir, exist_ok=True)

for left_img_path, right_img_path in zip(left_images, right_images):
    imgL = cv2.imread(left_img_path)
    imgR = cv2.imread(right_img_path)
    
    if imgL is None or imgR is None:
        print(f"Could not load images: {left_img_path}, {right_img_path}")
        continue
        
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Enhanced chessboard detection with multiple flags
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
    retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, flags)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, flags)

    # Create debug images with corner visualization
    # Convert to grayscale but keep 3 channels (BGR) for colored corner drawing
    debug_imgL = cv2.cvtColor(grayL, cv2.COLOR_GRAY2BGR)
    debug_imgR = cv2.cvtColor(grayR, cv2.COLOR_GRAY2BGR)
    
    # Draw detected corners (even if detection failed, show what was found)
    if retL and cornersL is not None:
        cv2.drawChessboardCorners(debug_imgL, chessboard_size, cornersL, retL)
    if retR and cornersR is not None:
        cv2.drawChessboardCorners(debug_imgR, chessboard_size, cornersR, retR)
    
    # Add status text to debug images
    base_name = os.path.splitext(os.path.basename(left_img_path))[0]
    left_status = f"LEFT: {'SUCCESS' if retL else 'FAILED'} - Found {len(cornersL) if cornersL is not None else 0} corners"
    right_status = f"RIGHT: {'SUCCESS' if retR else 'FAILED'} - Found {len(cornersR) if cornersR is not None else 0} corners"
    
    # Add text overlay
    cv2.putText(debug_imgL, left_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if retL else (0, 0, 255), 2)
    cv2.putText(debug_imgR, right_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if retR else (0, 0, 255), 2)
    
    # Add expected corner count
    expected_text = f"Expected: {chessboard_size[0] * chessboard_size[1]} corners ({chessboard_size[0]}x{chessboard_size[1]})"
    cv2.putText(debug_imgL, expected_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(debug_imgR, expected_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save debug images
    debug_left_path = os.path.join(debug_dir, f"debug_left_{base_name}.png")
    debug_right_path = os.path.join(debug_dir, f"debug_right_{base_name}.png")
    cv2.imwrite(debug_left_path, debug_imgL)
    cv2.imwrite(debug_right_path, debug_imgR)

    if retL and retR:
        objpoints.append(objp)
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1),
                                     criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001))
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1),
                                     criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001))
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)
        valid_pairs += 1
        print(f"✓ Valid chessboard found in pair {valid_pairs}: {os.path.basename(left_img_path)}")
        
        # Save refined corners visualization
        debug_imgL_refined = cv2.cvtColor(grayL, cv2.COLOR_GRAY2BGR)
        debug_imgR_refined = cv2.cvtColor(grayR, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(debug_imgL_refined, chessboard_size, cornersL, True)
        cv2.drawChessboardCorners(debug_imgR_refined, chessboard_size, cornersR, True)
        
        # Add refined status
        cv2.putText(debug_imgL_refined, "REFINED CORNERS", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(debug_imgR_refined, "REFINED CORNERS", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Save refined debug images
        debug_left_refined_path = os.path.join(debug_dir, f"refined_left_{base_name}.png")
        debug_right_refined_path = os.path.join(debug_dir, f"refined_right_{base_name}.png")
        cv2.imwrite(debug_left_refined_path, debug_imgL_refined)
        cv2.imwrite(debug_right_refined_path, debug_imgR_refined)
        
    else:
        left_status = "✓" if retL else "✗"
        right_status = "✓" if retR else "✗"
        print(f"✗ Skipping pair (L:{left_status} R:{right_status}): {os.path.basename(left_img_path)}")
        
        # Additional diagnostic information
        if not retL:
            print(f"   Left camera: Could not detect {chessboard_size[0]}×{chessboard_size[1]} checkerboard pattern")
        if not retR:
            print(f"   Right camera: Could not detect {chessboard_size[0]}×{chessboard_size[1]} checkerboard pattern")

print(f"\nFound {valid_pairs} valid image pairs for calibration")
print(f"📂 Debug images saved to '{debug_dir}/' directory:")
print("   - debug_left_XXX.png: Initial corner detection results (left camera)")
print("   - debug_right_XXX.png: Initial corner detection results (right camera)")
print("   - refined_left_XXX.png: Sub-pixel refined corners (successful pairs only)")
print("   - refined_right_XXX.png: Sub-pixel refined corners (successful pairs only)")

if valid_pairs < 10:
    print("WARNING: Less than 10 valid pairs. Consider capturing more images for better calibration.")
    print("💡 Check debug images to see why pairs were rejected:")
    print("   - Look for incomplete checkerboard visibility")
    print("   - Check for blurry or poorly lit images")
    print("   - Verify checkerboard is flat and properly oriented")
elif valid_pairs < 15:
    print("WARNING: Less than 15 pairs. More images recommended for optimal calibration.")

# === Calibration ===
if len(objpoints) == 0:
    print("ERROR: No valid image pairs found for calibration!")
    exit(1)

print("\nPerforming individual camera calibrations...")
retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)

print(f"Left camera RMS error: {retL:.3f}")
print(f"Right camera RMS error: {retR:.3f}")

print("\nPerforming stereo calibration...")
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtxL, distL, mtxR, distR,
    grayL.shape[::-1],
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
    flags=flags
)

print(f"Stereo calibration RMS error: {retval:.3f}")

# Calculate baseline distance
baseline = np.linalg.norm(T) * 1000  # Convert to mm
print(f"Baseline distance: {baseline:.1f}mm")

# === Calculate percentage differences between cameras ===
print("\n=== INTRINSIC PARAMETER DIFFERENCES ===")

# Focal lengths
fx_diff = abs(cameraMatrix1[0,0] - cameraMatrix2[0,0]) / cameraMatrix1[0,0] * 100
fy_diff = abs(cameraMatrix1[1,1] - cameraMatrix2[1,1]) / cameraMatrix1[1,1] * 100
print(f"fx difference: {fx_diff:.2f}%")
print(f"fy difference: {fy_diff:.2f}%")

# Principal points
cx_diff = abs(cameraMatrix1[0,2] - cameraMatrix2[0,2]) / cameraMatrix1[0,2] * 100
cy_diff = abs(cameraMatrix1[1,2] - cameraMatrix2[1,2]) / cameraMatrix1[1,2] * 100
print(f"cx difference: {cx_diff:.2f}%")
print(f"cy difference: {cy_diff:.2f}%")

# Distortion coefficients
k1_left, k2_left, p1_left, p2_left, k3_left = distCoeffs1.flatten()
k1_right, k2_right, p1_right, p2_right, k3_right = distCoeffs2.flatten()

# Use average for percentage calculation to handle sign differences
k1_diff = abs(k1_left - k1_right) / (abs(k1_left) + abs(k1_right)) * 200 if (abs(k1_left) + abs(k1_right)) > 0 else 0
k2_diff = abs(k2_left - k2_right) / (abs(k2_left) + abs(k2_right)) * 200 if (abs(k2_left) + abs(k2_right)) > 0 else 0
k3_diff = abs(k3_left - k3_right) / (abs(k3_left) + abs(k3_right)) * 200 if (abs(k3_left) + abs(k3_right)) > 0 else 0

print(f"\nDistortion coefficient differences:")
print(f"k1 difference: {k1_diff:.2f}%")
print(f"k2 difference: {k2_diff:.2f}%")
print(f"k3 difference: {k3_diff:.2f}%")

# Quality indicators
print(f"\n{'✓' if fx_diff < 2 and fy_diff < 2 else '⚠️'} Focal length match: {'Excellent' if fx_diff < 2 and fy_diff < 2 else 'Acceptable' if fx_diff < 5 and fy_diff < 5 else 'Poor'}")
print(f"{'✓' if cx_diff < 2 and cy_diff < 2 else '⚠️'} Principal point match: {'Excellent' if cx_diff < 2 and cy_diff < 2 else 'Acceptable' if cx_diff < 5 and cy_diff < 5 else 'Poor'}")
print(f"{'✓' if k1_diff < 20 and k2_diff < 20 else '⚠️'} Distortion match: {'Good' if k1_diff < 20 and k2_diff < 20 else 'Different lenses'}")

# === Interpret Rotation Matrix ===
print("\n=== ROTATION ANALYSIS ===")

# Method 1: Rotation angle (axis-angle representation)
rotation_vector, _ = cv2.Rodrigues(R)
rotation_angle = np.linalg.norm(rotation_vector) * 180 / np.pi
rotation_axis = rotation_vector.flatten() / np.linalg.norm(rotation_vector)

print(f"Total rotation angle: {rotation_angle:.2f}°")
print(f"Rotation axis (unit vector): [{rotation_axis[0]:.3f}, {rotation_axis[1]:.3f}, {rotation_axis[2]:.3f}]")

# Method 2: Euler angles (more intuitive)
# Extract rotation around X, Y, Z axes
sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
singular = sy < 1e-6

if not singular:
    x_angle = np.arctan2(R[2,1], R[2,2]) * 180 / np.pi  # Roll
    y_angle = np.arctan2(-R[2,0], sy) * 180 / np.pi     # Pitch
    z_angle = np.arctan2(R[1,0], R[0,0]) * 180 / np.pi  # Yaw
else:
    x_angle = np.arctan2(-R[1,2], R[1,1]) * 180 / np.pi
    y_angle = np.arctan2(-R[2,0], sy) * 180 / np.pi
    z_angle = 0

print(f"\nEuler angles (rotation around each axis):")
print(f"  Roll  (X-axis): {x_angle:+.2f}° {'⚠️' if abs(x_angle) > 3 else '✓'}")
print(f"  Pitch (Y-axis): {y_angle:+.2f}° {'⚠️' if abs(y_angle) > 3 else '✓'}")
print(f"  Yaw   (Z-axis): {z_angle:+.2f}° {'⚠️' if abs(z_angle) > 3 else '✓'}")

print(f"\nCamera alignment quality:")
if rotation_angle < 2:
    print("  ✓ Excellent: Cameras are very well aligned")
elif rotation_angle < 5:
    print("  ✓ Good: Cameras are acceptably aligned")
elif rotation_angle < 10:
    print("  ⚠️ Moderate: Some misalignment present")
else:
    print("  ⚠️ Poor: Significant misalignment detected")

# === Rectification ===
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    cameraMatrix1, distCoeffs1,
    cameraMatrix2, distCoeffs2,
    grayL.shape[::-1], R, T, alpha=0
)

# === Save Calibration Data in Multiple Formats ===

# 1. NumPy format (compact, preserves precision)
np.savez("stereo_calib.npz",
         cameraMatrix1=cameraMatrix1, distCoeffs1=distCoeffs1,
         cameraMatrix2=cameraMatrix2, distCoeffs2=distCoeffs2,
         R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
         baseline_mm=baseline, rms_error=retval)

# 2. OpenCV YAML format (standard format)
fs = cv2.FileStorage("stereo_calib.yaml", cv2.FILE_STORAGE_WRITE)
fs.write("cameraMatrix1", cameraMatrix1)
fs.write("distCoeffs1", distCoeffs1)
fs.write("cameraMatrix2", cameraMatrix2)
fs.write("distCoeffs2", distCoeffs2)
fs.write("R", R)
fs.write("T", T)
fs.write("E", E)
fs.write("F", F)
fs.write("R1", R1)
fs.write("R2", R2)
fs.write("P1", P1)
fs.write("P2", P2)
fs.write("Q", Q)
fs.write("baseline_mm", baseline)
fs.write("rms_error", retval)
fs.release()

# 3. JSON format (human-readable)
calib_data = {
    "calibration_info": {
        "checkerboard_size": list(chessboard_size),
        "square_size_mm": square_size * 1000,
        "image_pairs_used": len(objpoints),
        "rms_error": float(retval),
        "baseline_mm": float(baseline)
    },
    "left_camera": {
        "camera_matrix": cameraMatrix1.tolist(),
        "distortion_coefficients": distCoeffs1.flatten().tolist(),
        "rms_error": float(retL)
    },
    "right_camera": {
        "camera_matrix": cameraMatrix2.tolist(),
        "distortion_coefficients": distCoeffs2.flatten().tolist(),
        "rms_error": float(retR)
    },
    "stereo_parameters": {
        "rotation_matrix": R.tolist(),
        "translation_vector": T.flatten().tolist(),
        "essential_matrix": E.tolist(),
        "fundamental_matrix": F.tolist()
    },
    "rectification": {
        "R1": R1.tolist(),
        "R2": R2.tolist(),
        "P1": P1.tolist(),
        "P2": P2.tolist(),
        "Q": Q.tolist(),
        "roi1": list(roi1),
        "roi2": list(roi2)
    }
}

with open("stereo_calib.json", "w") as f:
    json.dump(calib_data, f, indent=2)

# 4. Create a calibration report
report = f"""
=== STEREO CALIBRATION REPORT ===

Configuration:
- Checkerboard size: {chessboard_size[0]}×{chessboard_size[1]} inner corners
- Square size: {square_size*1000:.1f}mm
- Image pairs used: {len(objpoints)}

Calibration Results:
- Left camera RMS error: {retL:.3f} pixels
- Right camera RMS error: {retR:.3f} pixels
- Stereo calibration RMS error: {retval:.3f} pixels
- Baseline distance: {baseline:.2f}mm

Left Camera Parameters:
- Focal lengths (fx, fy): {cameraMatrix1[0,0]:.1f}, {cameraMatrix1[1,1]:.1f}
- Principal point (cx, cy): {cameraMatrix1[0,2]:.1f}, {cameraMatrix1[1,2]:.1f}
- Distortion coefficients: [{', '.join([f'{d:.4f}' for d in distCoeffs1.flatten()])}]

Right Camera Parameters:
- Focal lengths (fx, fy): {cameraMatrix2[0,0]:.1f}, {cameraMatrix2[1,1]:.1f}
- Principal point (cx, cy): {cameraMatrix2[0,2]:.1f}, {cameraMatrix2[1,2]:.1f}
- Distortion coefficients: [{', '.join([f'{d:.4f}' for d in distCoeffs2.flatten()])}]

Percentage Differences (Left vs Right):
- fx: {fx_diff:.2f}%
- fy: {fy_diff:.2f}%
- cx: {cx_diff:.2f}%
- cy: {cy_diff:.2f}%
- k1 (radial distortion): {k1_diff:.2f}%
- k2 (radial distortion): {k2_diff:.2f}%
- k3 (radial distortion): {k3_diff:.2f}%

Stereo Geometry:
- Translation: [{', '.join([f'{t:.3f}' for t in T.flatten()])}] (meters)
- Total rotation angle: {rotation_angle:.2f} degrees
- Rotation breakdown:
  * Roll  (X-axis): {x_angle:+.2f}°
  * Pitch (Y-axis): {y_angle:+.2f}°
  * Yaw   (Z-axis): {z_angle:+.2f}°

Quality Assessment:
- RMS Error < 0.5 pixels: {'✓' if retval < 0.5 else '✗'}
- Reasonable focal lengths: {'✓' if 1000 < cameraMatrix1[0,0] < 5000 else '✗'}
- Principal point near center: {'✓' if abs(cameraMatrix1[0,2] - grayL.shape[1]/2) < 200 else '✗'}
- Focal length match (<2%): {'✓' if fx_diff < 2 and fy_diff < 2 else '✗'}
- Principal point match (<2%): {'✓' if cx_diff < 2 and cy_diff < 2 else '✗'}
- Camera alignment < 5°: {'✓' if rotation_angle < 5 else '✗'}

Files Generated:
- stereo_calib.npz: NumPy format (recommended for Python)
- stereo_calib.yaml: OpenCV format (standard)
- stereo_calib.json: JSON format (human-readable)
- calibration_report.txt: This report
"""

with open("calibration_report.txt", "w") as f:
    f.write(report)

print("\n=== CALIBRATION COMPLETE ===")
print(f"✓ Stereo calibration RMS error: {retval:.3f} pixels")
print(f"✓ Baseline distance: {baseline:.1f}mm")
print("✓ Files saved:")
print("  - stereo_calib.npz (NumPy format)")
print("  - stereo_calib.yaml (OpenCV format)")
print("  - stereo_calib.json (JSON format)")
print("  - calibration_report.txt (detailed report)")


