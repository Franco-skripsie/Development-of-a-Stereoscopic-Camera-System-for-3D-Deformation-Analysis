# Development of a Stereoscopic Camera System for 3D Deformation Analysis

This repository contains the hardware designs (CAD) and software for a custom stereoscopic camera system designed for 3D deformation analysis. The system uses two synchronized Raspberry Pi Camera Module 3 units to capture stereo image pairs, which can be used to compute depth information and analyze 3D deformation.

## Project Report
This repository supports the Mechatronic Project 478 final report. For a complete understanding of the project's background, design methodology, hardware selection, software algorithms, and full results, please read the complete PDF report.

View the Full Project Report `25866095-DuPlessis.pdf`

## Table of Contents

- [Overview](#overview)
- [Hardware Components](#hardware-components)
- [Software Components](#software-components)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
  - [1. Capturing Calibration Images](#1-capturing-calibration-images)
  - [2. Stereo Calibration](#2-stereo-calibration)
  - [3. 3D Dot Pattern Calibration](#3-3d-dot-pattern-calibration)
  - [4. Stereo Rectification and Disparity Mapping](#4-stereo-rectification-and-disparity-mapping)
- [Repository Structure](#repository-structure)
- [Calibration Patterns](#calibration-patterns)
- [Contributing](#contributing)
- [License](#license)

## Overview

This stereoscopic camera system is designed to:
- Capture synchronized stereo image pairs using dual Raspberry Pi Camera Module 3
- Perform accurate stereo calibration using both chessboard patterns and custom 3D dot patterns
- Generate disparity maps for depth estimation
- Enable 3D deformation analysis through precise depth measurements

The system features custom 3D-printed enclosures for camera protection, adjustable mounting, and optimized camera placement for stereoscopic imaging.

## Hardware Components

### Camera System
- **2x Raspberry Pi Camera Module 3** - High-resolution cameras with autofocus
- **Raspberry Pi 5** - Main controller for dual camera operation
- **Custom 3D-Printed Enclosures** (CAD files included):
  - `Camera Enclosure/` - Left and right camera housings with backplates
  - `RPi5 Enclosure/` - Protective enclosure for Raspberry Pi 5
  - `Base to Mounting Rail Connector/` - Hardware mounting system

### Calibration Equipment
- **Chessboard Pattern** - 9×6 inner corners, 14.5mm squares (for standard calibration)
- **Custom 3D Calibration Plate** - 264 white dots on black background:
  - Grid A: 143 dots (12×12 grid minus one) at -1mm depth
  - Grid B: 121 dots (11×11 grid) at reference plane (0mm)
  - 1.2mm dot diameter, 5mm spacing within grids
  - 1mm depth separation between grids for 3D calibration

## Software Components

The `Code/` directory contains Python scripts for the complete stereo vision pipeline:

### Calibration Scripts

1. **`stereo_calibration_collector.py`**
   - Interactive stereo image capture tool
   - Synchronized capture from both cameras
   - Live preview with crosshair alignment
   - Anti-flicker settings for LED lighting
   - Manual and timer capture modes
   - Quality validation and autofocus control

2. **`stereo_calibrate.py`**
   - Standard stereo calibration using chessboard pattern
   - Individual camera intrinsic calibration
   - Stereo extrinsic calibration (rotation & translation)
   - Outputs multiple formats: NPZ & JSON
   - Generates detailed calibration reports

3. **`stereo_calibrate_3d_dots.py`**
   - Advanced 3D calibration using custom dot pattern
   - Non-planar calibration for improved accuracy
   - Uses two grids at different depths (1mm separation)
   - Leverages proven chessboard calibration as initial estimates

### Dot Detection and Processing

4. **`dot_identifier.py`**
   - Automated white dot detection on black background
   - Processes custom 3D calibration plate images
   - Uses HoughCircles detection with white center validation
   - K-means clustering for column identification
   - Outputs sorted point arrays for Grid A and Grid B
   - Generates visualization and CSV export

5. **`live_dot_tuner.py`**
   - Interactive GUI for tuning dot detection parameters
   - Real-time parameter adjustment with visual feedback
   - HoughCircles parameter optimization
   - White center threshold tuning
   - Export optimized parameters for batch processing

### Stereo Processing

6. **`simple_stereo_rectify.py`**
   - Stereo image rectification
   - Epipolar line visualization
   - SGBM (Semi-Global Block Matching) disparity computation
   - Generates disparity maps (grayscale and color-coded)
   - Outputs rectified stereo pairs

## Prerequisites

### Hardware Requirements
- Raspberry Pi 5 (or compatible)
- 2× Raspberry Pi Camera Module 3
- Raspberry Pi SSD Kit (Recommended)
- Calibration patterns (printed or displayed)

### Software Requirements
- Python 3.7+
- OpenCV (cv2) with Python bindings
- NumPy
- Picamera2 (for Raspberry Pi Camera Module 3)
- Matplotlib
- Pandas
- scikit-learn
- Tkinter (for GUI tools)

## Installation

### 1. System Setup

```bash
# Update system packages
sudo apt-get update
sudo apt-get upgrade

# Install system dependencies
sudo apt-get install python3-pip python3-opencv
sudo apt-get install python3-picamera2
```

### 2. Python Dependencies

```bash
# Install required Python packages
pip3 install numpy opencv-python matplotlib pandas scikit-learn

# For GUI tools
sudo apt-get install python3-tk
```

### 3. Clone Repository

```bash
git clone https://github.com/Franco-skripsie/Development-of-a-Stereoscopic-Camera-System-for-3D-Deformation-Analysis.git
cd Development-of-a-Stereoscopic-Camera-System-for-3D-Deformation-Analysis/Code
```

## Usage Guide

### 1. Capturing Calibration Images

Use `stereo_calibration_collector.py` to capture synchronized stereo image pairs:

```bash
python3 stereo_calibration_collector.py
```

**Key Controls:**
- `c` - Capture stereo image pair
- `t` - Toggle between manual and timer modes
- `a` - Trigger autofocus on both cameras
- `l` - Lock white balance and autofocus
- `r` - Restore proven camera settings
- `q` - Quit

**Tips:**
- Capture 20-30 image pairs at different angles and distances
- Ensure the calibration pattern is clearly visible in both cameras
- Use the crosshair for alignment
- Vary the position and orientation of the pattern

### 2. Stereo Calibration

After capturing calibration images, perform stereo calibration:

```bash
python3 stereo_calibrate.py
```

This will:
- Detect chessboard corners in all image pairs
- Calibrate individual camera intrinsics
- Compute stereo extrinsic parameters (R, T)
- Generate rectification matrices
- Save calibration data: `stereo_calib.npz`, `stereo_calib.json`
- Create `calibration_report.txt` with detailed results

### 3. 3D Dot Pattern Calibration

For advanced calibration using the custom 3D dot pattern:

#### Step 3a: Detect and Sort Dots

```bash
# Basic usage
python3 dot_identifier.py --image path/to/image.png

# With custom parameters
python3 dot_identifier.py \
  --image calib/left001.png \
  --output results/left001 \
  --hough-min-dist 25 \
  --hough-param1 50 \
  --hough-param2 15 \
  --white-threshold 180
```

**Output Files:**
- `*_gridA_points.npy` - Grid A dot coordinates (143 points)
- `*_gridB_points.npy` - Grid B dot coordinates (121 points)
- `*_dots.csv` - All detected dots with grid/column/row info
- `*_visualization.png` - Visual verification of detection

#### Step 3b: Tune Detection Parameters (Optional)

Use the interactive tuner to optimize detection:

```bash
python3 live_dot_tuner.py --image path/to/image.png
```

The GUI allows real-time adjustment of:
- Minimum distance between circles
- Edge detection threshold
- Circle detection threshold
- Radius range
- White center threshold

#### Step 3c: 3D Calibration

After processing dot identification for stereo pairs:

```bash
python3 stereo_calibrate_3d_dots.py
```

This performs:
- Non-planar calibration using two grids at different depths
- Uses chessboard calibration as initial estimates
- Outputs: `3d_plate_stereo_calib.npz`, `3d_plate_stereo_calib.json`

### 4. Stereo Rectification and Disparity Mapping

Process stereo images to generate depth information:

```bash
python3 simple_stereo_rectify.py
```

**Configuration** (edit script):
```python
CALIB_FILE = "stereo_calib.npz"           # Calibration data
LEFT_IMAGE = "capture/left001.png"         # Input left image
RIGHT_IMAGE = "capture/right001.png"       # Input right image
OUTPUT_DIR = "results"                     # Output directory
```

**Output Files:**
- `rect_left.png` / `rect_right.png` - Rectified images
- `rect_left_lines.png` / `rect_right_lines.png` - Images with epipolar lines
- `epipolar_comparison.png` - Side-by-side view
- `disparity_gray.png` - Grayscale disparity map
- `disparity_color.png` - Color-coded disparity map

## Repository Structure

```
Development-of-a-Stereoscopic-Camera-System-for-3D-Deformation-Analysis/
│
├── Code/                                    # Python software
│   ├── stereo_calibration_collector.py      # Image capture tool
│   ├── stereo_calibrate.py                  # Chessboard calibration
│   ├── stereo_calibrate_3d_dots.py          # 3D dot pattern calibration
│   ├── dot_identifier.py                    # Dot detection and sorting
│   ├── live_dot_tuner.py                    # Interactive parameter tuning
│   └── simple_stereo_rectify.py             # Rectification and disparity
│
├── Camera Enclosure/                        # 3D printable camera housings
│   ├── Camera_Backplate.ipt                 # Camera mounting backplate
│   ├── Camera_Enclosure_Left.ipt            # Left camera enclosure
│   └── Camera_Enclosure_Right.ipt           # Right camera enclosure
│
├── RPi5 Enclosure/                          # Raspberry Pi 5 enclosure
│   ├── RPi5_Enclosure_Bottom.ipt            # Bottom enclosure part
│   └── RPi5_Enclosure_Top.ipt               # Top enclosure part
│
├── Base to Mounting Rail Connector/         # Mounting hardware
│   └── Base_Mounting_Rail_Connector.ipt     # Rail mounting connector
│
└── README.md                                # This file
```

**Note:** `.ipt` files are Autodesk Inventor part files. Open them with Autodesk Inventor or convert to STL for 3D printing.

## Calibration Patterns

### Chessboard Pattern
- **Size:** 9×6 inner corners
- **Square Size:** 15mm
- **Purpose:** Standard stereo calibration
- **Usage:** Print on flat, rigid surface or display on screen

### Custom 3D Dot Pattern
- **Total Dots:** 264 (143 + 121)
- **Grid A:** 12×12 grid (minus one dot at column 12, row 2)
  - Depth: -1mm (behind Grid B)
  - 143 dots total
- **Grid B:** 11×11 grid
  - Depth: 0mm (reference plane)
  - 121 dots total
- **Dot Properties:**
  - Diameter: 1.2mm
  - Color: White dots on black background
  - Spacing: 5mm within same grid
  - Depth separation: 1mm between grids
- **Purpose:** Advanced 3D calibration with non-planar target

## Workflow Summary

1. **Setup Hardware**
   - Assemble cameras in 3D-printed enclosures
   - Mount on stable base with appropriate baseline distance
   - Connect to Raspberry Pi 5

2. **Capture Calibration Images**
   - Use `stereo_calibration_collector.py`
   - Capture 20-30 chessboard image pairs
   - Optionally capture custom 3D dot pattern images

3. **Perform Calibration**
   - Run `stereo_calibrate.py` for standard calibration
   - Or process dots and run `stereo_calibrate_3d_dots.py` for 3D calibration

4. **Process Stereo Images**
   - Use `simple_stereo_rectify.py` to:
     - Rectify stereo pairs
     - Compute disparity maps
     - Analyze depth information

5. **Analyze Results**
   - Review calibration reports for accuracy
   - Examine disparity maps
   - Use depth data for deformation analysis

## Calibration Quality Indicators

### Good Calibration:
- RMS reprojection error < 0.5 pixels
- Horizontal epipolar lines in rectified images
- Smooth, continuous disparity maps
- Baseline distance matches physical measurement

### Signs of Poor Calibration:
- RMS error > 1.0 pixel
- Misaligned epipolar lines
- Noisy or discontinuous disparity maps
- Need more image pairs at varied angles

## Troubleshooting

### Camera Issues
- **Blank/dark images:** Check lighting, adjust exposure in `stereo_calibration_collector.py`
- **Sync issues:** Ensure both cameras are properly initialized
- **Focus problems:** Use autofocus (`a` key) or restore proven settings (`r` key)

### Calibration Issues
- **Low accuracy:** Capture more image pairs with better coverage
- **Pattern not detected:** Improve lighting, ensure pattern is flat and clearly visible
- **High RMS error:** Check for motion blur, improve pattern quality

### Dot Detection Issues
- **Too few/many dots detected:** Use `live_dot_tuner.py` to optimize parameters
- **Incorrect clustering:** Adjust `--hough-min-dist` parameter
- **False positives:** Increase `--white-threshold` value

## Contributing

Contributions are welcome. This is an open-source academic project, and improvements to the calibration algorithms, hardware designs, or documentation are appreciated. Please feel free to fork the repository, make changes, and open a pull request.

### How to Contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Test thoroughly with actual hardware
5. Commit with clear messages (`git commit -am 'Add feature'`)
6. Push to branch (`git push origin feature/improvement`)
7. Open a Pull Request

### Areas for Improvement:
- Additional calibration pattern support
- Real-time disparity computation optimisation
- Enhanced GUI for calibration workflow
- Automated quality assessment
- Integration with 3D reconstruction pipelines

## License

This project is licensed under the MIT License. See the (`LICENSE`) file for details.

## Acknowledgments

This stereoscopic camera system was developed for 3D deformation analysis research. The system combines:
- Standard stereo calibration techniques (Zhang's method)
- Custom 3D calibration patterns for improved accuracy
- Raspberry Pi Camera Module 3 for high-resolution imaging
- OpenCV for computer vision processing

---

**Last Updated:** 2025

**Hardware:** Raspberry Pi 5 + 2× Pi Camera Module 3  
**Software:** Python 3 + OpenCV + Picamera2
