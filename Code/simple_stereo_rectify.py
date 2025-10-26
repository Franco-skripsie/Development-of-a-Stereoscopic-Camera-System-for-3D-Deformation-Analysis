import cv2
import numpy as np
import os

# === Configuration ===
CALIB_FILE = "stereo_calib.npz"
LEFT_IMAGE = "capture_depth_4/left001.png"   # Path to left image
RIGHT_IMAGE = "capture_depth_4/right001.png" # Path to right image
OUTPUT_DIR = "stereorectified_disparity_map"

# === Setup ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load calibration data ===
print("Loading calibration data...")
data = np.load(CALIB_FILE)
K1 = data['cameraMatrix1']
D1 = data['distCoeffs1'] 
K2 = data['cameraMatrix2']
D2 = data['distCoeffs2']
R1 = data['R1']
R2 = data['R2'] 
P1 = data['P1']
P2 = data['P2']
Q = data['Q']

print(f"✓ Calibration loaded - Baseline: {data.get('baseline_mm', 'Unknown')}mm")

# === Load images ===
print("Loading stereo images...")
imgL = cv2.imread(LEFT_IMAGE)
imgR = cv2.imread(RIGHT_IMAGE)

if imgL is None or imgR is None:
    print("Error: Could not load images!")
    exit(1)

print(f"✓ Images loaded - Size: {imgL.shape[1]}x{imgL.shape[0]}")

# === Create rectification maps ===
print("Creating rectification maps...")
img_shape = (imgL.shape[1], imgL.shape[0])  # (width, height)
map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_shape, cv2.CV_16SC2)
map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_shape, cv2.CV_16SC2)

# === Rectify images ===
print("Rectifying images...")
rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

# === Draw epipolar lines ===
print("Drawing epipolar lines...")
rectL_lines = rectL.copy()
rectR_lines = rectR.copy()

h, w = rectL.shape[:2]

# Horizontal epipolar lines (green)
for y in range(0, h, 40):
    cv2.line(rectL_lines, (0, y), (w-1, y), (0, 255, 0), 1)
    cv2.line(rectR_lines, (0, y), (w-1, y), (0, 255, 0), 1)

# Vertical reference lines (blue) 
for x in [w//4, w//2, 3*w//4]:
    cv2.line(rectL_lines, (x, 0), (x, h-1), (255, 0, 0), 1)
    cv2.line(rectR_lines, (x, 0), (x, h-1), (255, 0, 0), 1)

# === Save rectified images ===
print("Saving rectified images...")
cv2.imwrite(os.path.join(OUTPUT_DIR, "rect_left.png"), rectL)
cv2.imwrite(os.path.join(OUTPUT_DIR, "rect_right.png"), rectR)
cv2.imwrite(os.path.join(OUTPUT_DIR, "rect_left_lines.png"), rectL_lines)
cv2.imwrite(os.path.join(OUTPUT_DIR, "rect_right_lines.png"), rectR_lines)

# Side-by-side comparison
epipolar_view = np.hstack((rectL_lines, rectR_lines))
cv2.imwrite(os.path.join(OUTPUT_DIR, "epipolar_comparison.png"), epipolar_view)

# === SGBM Disparity Computation ===
print("Computing disparity with SGBM...")

# Convert to grayscale
grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

# Create SGBM matcher
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=512,      # Must be divisible by 16 - 512
    blockSize=5,           # Odd number - 5
    P1=8 * 3 * 5**2,      # Penalty for small disparity changes - 8
    P2=24 * 3 * 5**2,     # Penalty for large disparity changes - 24
    disp12MaxDiff=1,        # Left-right consistency check
    uniquenessRatio=5,     # Margin by which best match must win - 5
    mode=cv2.STEREO_SGBM_MODE_SGBM
)

# Compute disparity
disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

# === Save disparity maps ===
print("Saving disparity maps...")

# Normalize for visualization
disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Apply colormap
disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)


# Save disparity maps
cv2.imwrite(os.path.join(OUTPUT_DIR, "disparity_gray.png"), disp_vis)
cv2.imwrite(os.path.join(OUTPUT_DIR, "disparity_color.png"), disp_color)

print(f"✓ Processing complete!")
print(f"📁 Results saved in: {OUTPUT_DIR}/")
print("📄 Files generated:")
print("  - rect_left.png: Rectified left image")
print("  - rect_right.png: Rectified right image") 
print("  - rect_left_lines.png: Left image with epipolar lines")
print("  - rect_right_lines.png: Right image with epipolar lines")
print("  - epipolar_comparison.png: Side-by-side view with lines")
print("  - disparity_gray.png: Grayscale disparity map")
print("  - disparity_color.png: Colored disparity map")
print(f"📊 Disparity range: {disparity.min():.1f} to {disparity.max():.1f} pixels")