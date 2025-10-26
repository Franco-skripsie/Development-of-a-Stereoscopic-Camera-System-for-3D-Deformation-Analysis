#!/usr/bin/env python3
"""
Live Dot Detection Tuner

A real-time GUI application for tuning blob detection parameters on calibration images.
Allows interactive adjustment of detection parameters with immediate visual feedback.

Required packages:
pip install opencv-python numpy matplotlib tkinter

Usage:
    cd "/home/francodp/camera_app/1_Dot identification and sorting/Route to completion_DotIdentifier" && python3 live_dot_tuner.py --image "/home/francodp/camera_app/calib/right008.png"

Features:
- Real-time parameter adjustment with sliders
- Live preview of detected dots
- Parameter value display
- Export final parameters
- Save detection results

Author: Generated for Raspberry Pi 5 stereo camera calibration
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import argparse
import os
import sys
import time
from typing import Tuple, List, Optional


class LiveDotTuner:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.original_image = None
        self.binary_image = None
        self.detected_dots = []
        
        # Load and preprocess image
        self.load_image()
        
        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Live Dot Detection Tuner")
        self.root.geometry("1400x800")
        
        # Parameter variables for HoughCircles
        self.hough_min_dist = tk.IntVar(value=25)
        self.hough_param1 = tk.IntVar(value=50)
        self.hough_param2 = tk.IntVar(value=15)
        self.hough_min_radius = tk.IntVar(value=9)
        self.hough_max_radius = tk.IntVar(value=17)
        
        # White center filtering threshold
        self.white_threshold = tk.IntVar(value=180)

        # GUI elements
        self.setup_gui()
        
        # Initial detection
        self.update_detection()
    
    def load_image(self):
        """Load and preprocess the calibration image."""
        if not os.path.exists(self.image_path):
            messagebox.showerror("Error", f"Image file not found: {self.image_path}")
            sys.exit(1)
        
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            messagebox.showerror("Error", f"Could not load image: {self.image_path}")
            sys.exit(1)
        
        # Preprocess image (same as main script)
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        self.binary_image = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2
        )
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_OPEN, kernel)
        self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_CLOSE, kernel)
        
        print(f"Loaded image: {self.image_path} with shape {self.original_image.shape}")
    
    def setup_gui(self):
        """Setup the GUI layout with controls and image display."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        control_frame = ttk.LabelFrame(main_frame, text="Detection Parameters", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Parameter controls
        self.create_parameter_controls(control_frame)
        
        # Right panel for image display
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Canvas for matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_parameter_controls(self, parent):
        """Create parameter control buttons for HoughCircles detection."""
        # Minimum Distance
        self.create_parameter_group(parent, "Min Distance Between Centers (px)", 
                                   self.hough_min_dist, 5, 100, 1, 5)
        
        # Param1 (Edge Detection Threshold)
        self.create_parameter_group(parent, "Param1 - Edge Threshold", 
                                   self.hough_param1, 10, 200, 1, 10)
        
        # Param2 (Accumulator Threshold)
        self.create_parameter_group(parent, "Param2 - Circle Threshold", 
                                   self.hough_param2, 5, 100, 1, 5)
        
        # Minimum Radius
        self.create_parameter_group(parent, "Minimum Circle Radius (px)", 
                                   self.hough_min_radius, 1, 50, 1, 5)
        
        # Maximum Radius
        self.create_parameter_group(parent, "Maximum Circle Radius (px)", 
                                   self.hough_max_radius, 5, 100, 1, 5)
        
        # White Center Threshold
        self.create_parameter_group(parent, "White Center Threshold (0-255)", 
                                   self.white_threshold, 100, 255, 5, 20)
        
        # Detection info
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        ttk.Label(parent, text="Detection Results", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        self.dots_count_label = ttk.Label(parent, text="Dots detected: 0", font=("Arial", 11))
        self.dots_count_label.pack(anchor=tk.W, pady=(5, 0))
        
        self.detection_info_label = ttk.Label(parent, text="Total circles: 0, White centers: 0", 
                                             font=("Arial", 9), foreground="gray")
        self.detection_info_label.pack(anchor=tk.W, pady=(2, 0))
        
        self.target_label = ttk.Label(parent, text="Target: 264 white dots", foreground="blue")
        self.target_label.pack(anchor=tk.W)
        
        # Quick preset buttons
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(parent, text="Quick Presets", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        preset_frame = ttk.Frame(parent)
        preset_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(preset_frame, text="Default", command=self.preset_default).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(preset_frame, text="Strict", command=self.preset_strict).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Lenient", command=self.preset_lenient).pack(side=tk.LEFT, padx=(5, 0))
        
        # Parameter explanation
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        info_frame = ttk.LabelFrame(parent, text="Parameter Guide", padding=5)
        info_frame.pack(fill=tk.X, pady=5)
        
        info_text = (
            "• Min Distance: Prevents multiple detections per dot\n"
            "• Param1: Lower = more edges detected\n" 
            "• Param2: Lower = more circles accepted\n"
            "• Radius: Range of expected dot sizes\n"
            "• White Threshold: Only circles with white centers\n"
            "  (255=pure white, 180=light gray, 100=medium gray)"
        )
        ttk.Label(info_frame, text=info_text, font=("Arial", 8), justify=tk.LEFT).pack(anchor=tk.W)
        
        # Buttons
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Export Parameters", 
                   command=self.export_parameters).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Save Results", 
                   command=self.save_results).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Save Debug Image", 
                   command=self.save_debug_image).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Load New Image", 
                   command=self.load_new_image).pack(fill=tk.X)

    def create_parameter_group(self, parent, label_text, variable, min_val, max_val, 
                              small_step, large_step):
        """Create a parameter control group with increment/decrement buttons."""
        # Label
        ttk.Label(parent, text=label_text, font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(10, 2))
        
        # Control frame
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Large decrement button
        ttk.Button(control_frame, text="--", width=3,
                   command=lambda: self.adjust_parameter(variable, -large_step, min_val, max_val)
                   ).pack(side=tk.LEFT, padx=(0, 2))
        
        # Small decrement button  
        ttk.Button(control_frame, text="-", width=2,
                   command=lambda: self.adjust_parameter(variable, -small_step, min_val, max_val)
                   ).pack(side=tk.LEFT, padx=2)
        
        # Value display
        value_frame = ttk.Frame(control_frame)
        value_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Create a label that will show the current value
        value_label = ttk.Label(value_frame, text="", font=("Arial", 10, "bold"), 
                               anchor=tk.CENTER, relief=tk.SUNKEN, padding=3)
        value_label.pack(fill=tk.X)
        
        # Store reference for updates
        setattr(self, f"{variable._name}_label", value_label)
        
        # Small increment button
        ttk.Button(control_frame, text="+", width=2,
                   command=lambda: self.adjust_parameter(variable, small_step, min_val, max_val)
                   ).pack(side=tk.RIGHT, padx=2)
        
        # Large increment button
        ttk.Button(control_frame, text="++", width=3,
                   command=lambda: self.adjust_parameter(variable, large_step, min_val, max_val)
                   ).pack(side=tk.RIGHT, padx=(2, 0))
        
        # Update initial value display
        self.update_parameter_display(variable)

    def adjust_parameter(self, variable, delta, min_val, max_val):
        """Adjust a parameter by the given delta, respecting bounds."""
        current_value = variable.get()
        new_value = current_value + delta
        
        # Clamp to bounds
        new_value = max(min_val, min(max_val, new_value))
        
        # Handle floating point precision for double variables
        if isinstance(variable, tk.DoubleVar):
            new_value = round(new_value, 3)
        
        variable.set(new_value)
        self.update_parameter_display(variable)
        self.on_parameter_change()

    def update_parameter_display(self, variable):
        """Update the display label for a parameter."""
        value = variable.get()
        label_name = f"{variable._name}_label"
        
        if hasattr(self, label_name):
            label = getattr(self, label_name)
            if isinstance(variable, tk.DoubleVar):
                label.config(text=f"{value:.3f}")
            else:
                label.config(text=str(int(value)))

    def preset_default(self):
        """Apply default HoughCircles parameters."""
        self.hough_min_dist.set(45)
        self.hough_param1.set(50)
        self.hough_param2.set(15)
        self.hough_min_radius.set(11)
        self.hough_max_radius.set(19)
        self.white_threshold.set(180)
        self.update_all_displays()
        self.on_parameter_change()

    def preset_strict(self):
        """Apply strict parameters for fewer, higher-quality circles."""
        self.hough_min_dist.set(50)
        self.hough_param1.set(60)
        self.hough_param2.set(25)
        self.hough_min_radius.set(11)
        self.hough_max_radius.set(19)
        self.white_threshold.set(200)
        self.update_all_displays()
        self.on_parameter_change()

    def preset_lenient(self):
        """Apply lenient parameters for more circles."""
        self.hough_min_dist.set(35)
        self.hough_param1.set(40)
        self.hough_param2.set(10)
        self.hough_min_radius.set(8)
        self.hough_max_radius.set(25)
        self.white_threshold.set(160)
        self.update_all_displays()
        self.on_parameter_change()

    def update_all_displays(self):
        """Update all parameter displays."""
        for var in [self.hough_min_dist, self.hough_param1, self.hough_param2, 
                   self.hough_min_radius, self.hough_max_radius, self.white_threshold]:
            self.update_parameter_display(var)
    
    def on_parameter_change(self, event=None):
        """Called when any parameter changes."""
        # Update detection
        self.update_detection()
    
    def is_center_white(self, x: int, y: int) -> bool:
        """
        Check if the center pixel of a detected circle is white (bright).
        
        Args:
            x, y: Center coordinates
            
        Returns:
            bool: True if center pixel is white enough
        """
        threshold = self.white_threshold.get()
        
        if len(self.original_image.shape) == 3:
            # Convert to grayscale if color image
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.original_image
        
        h, w = gray.shape
        if 0 <= x < w and 0 <= y < h:
            return gray[y, x] >= threshold
        return False

    def detect_blobs(self) -> List[Tuple[float, float, float]]:
        """Detect circles using HoughCircles with current parameters and white center filtering."""
        # Use HoughCircles detection
        circles = cv2.HoughCircles(
            self.binary_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.hough_min_dist.get(),
            param1=self.hough_param1.get(),
            param2=self.hough_param2.get(),
            minRadius=self.hough_min_radius.get(),
            maxRadius=self.hough_max_radius.get()
        )
        
        dots = []
        total_detected = 0
        white_center_count = 0
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            total_detected = len(circles)
            
            # Filter circles by center color - only keep white centers
            for x, y, r in circles:
                if self.is_center_white(x, y):
                    dots.append((float(x), float(y), float(r)))
                    white_center_count += 1
        
        # Update status with filtering info
        if hasattr(self, 'detection_info_label'):
            self.detection_info_label.config(
                text=f"Total circles: {total_detected}, White centers: {white_center_count}"
            )
        
        return dots
    
    def update_detection(self):
        """Update the detection and display."""
        # Detect dots with current parameters
        self.detected_dots = self.detect_blobs()
        
        # Update dot count
        count = len(self.detected_dots)
        self.dots_count_label.config(text=f"White dots detected: {count}")
        
        # Color code the count based on target (264 expected)
        if 250 <= count <= 270:
            color = "green"
        elif 200 <= count <= 290:
            color = "orange"
        else:
            color = "red"
        self.dots_count_label.config(foreground=color)
        
        # Update display
        self.update_display()
    
    def update_display(self):
        """Update the image display with detected dots."""
        self.ax.clear()
        
        # Display original image
        rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.ax.imshow(rgb_image)
        
        # Plot detected dots (now with radius information)
        if self.detected_dots:
            for i, (x, y, r) in enumerate(self.detected_dots):
                # Plot center point
                self.ax.plot(x, y, 'r+', markersize=8, markeredgewidth=2)
                # Plot circle outline
                circle = plt.Circle((x, y), r, fill=False, color='red', linewidth=1.5, alpha=0.7)
                self.ax.add_patch(circle)
                # Add dot number for identification
                self.ax.text(x + r + 3, y, str(i+1), color='yellow', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
        
        self.ax.set_title(f"Detected White Dots: {len(self.detected_dots)} (Target: ~264)")
        self.ax.axis('off')
        
        # Refresh canvas
        self.canvas.draw()
    
    def export_parameters(self):
        """Export current HoughCircles parameters to command line format."""
        params = (
            f"--hough-min-dist {self.hough_min_dist.get()} "
            f"--hough-param1 {self.hough_param1.get()} "
            f"--hough-param2 {self.hough_param2.get()} "
            f"--hough-min-radius {self.hough_min_radius.get()} "
            f"--hough-max-radius {self.hough_max_radius.get()} "
            f"--white-threshold {self.white_threshold.get()}"
        )
        
        command = f"python3 dot_identifier.py --image {self.image_path} {params}"
        
        # Show in dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Export HoughCircles Parameters")
        dialog.geometry("900x200")
        
        ttk.Label(dialog, text="Command line parameters:", font=("Arial", 10, "bold")).pack(pady=10)
        
        text_widget = tk.Text(dialog, height=4, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, command)
        text_widget.config(state=tk.DISABLED)
        
        # Copy to clipboard button
        def copy_to_clipboard():
            dialog.clipboard_clear()
            dialog.clipboard_append(command)
            messagebox.showinfo("Copied", "HoughCircles command copied to clipboard!")
        
        ttk.Button(dialog, text="Copy to Clipboard", command=copy_to_clipboard).pack(pady=10)
    
    def save_debug_image(self):
        """Save individual debug images for each detected dot."""
        if not self.detected_dots:
            messagebox.showwarning("No Results", "No dots detected to save!")
            return
        
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        timestamp = int(time.time()) % 10000  # Last 4 digits of timestamp for uniqueness
        
        # Ensure directory exists
        debug_dir = f"/home/francodp/camera_app/dot_identifier_tests/{base_name}_debug_{timestamp}"
        os.makedirs(debug_dir, exist_ok=True)
        
        saved_files = []
        
        # Create individual image for each detected dot
        for i, (x, y, r) in enumerate(self.detected_dots):
            # Create a copy of the original image for each dot
            debug_image = self.original_image.copy()
            
            # Convert to integer coordinates
            center_x, center_y, radius = int(x), int(y), int(r)
            
            # Draw only this specific dot with enhanced visibility
            # Draw circle outline (bright green, thicker)
            cv2.circle(debug_image, (center_x, center_y), radius, (0, 255, 0), 3)
            
            # Draw center point (bright red, larger)
            cv2.circle(debug_image, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Draw crosshair at center (bright blue, thicker)
            cv2.line(debug_image, (center_x - 15, center_y), (center_x + 15, center_y), (255, 0, 0), 3)
            cv2.line(debug_image, (center_x, center_y - 15), (center_x, center_y + 15), (255, 0, 0), 3)
            
            # Add comprehensive dot information
            info_lines = [
                f"DOT #{i+1} of {len(self.detected_dots)}",
                f"Center: ({center_x}, {center_y})",
                f"Radius: {radius}px",
                f"Parameters: MinDist={self.hough_min_dist.get()}, P1={self.hough_param1.get()}, P2={self.hough_param2.get()}"
            ]
            
            # Add info text with black background for visibility
            for j, text in enumerate(info_lines):
                y_pos = 30 + (j * 30)
                font_scale = 0.8 if j == 0 else 0.6
                thickness = 2 if j == 0 else 1
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness + 1
                )
                
                # Draw black background rectangle
                cv2.rectangle(debug_image, 
                             (10, y_pos - text_height - 8), 
                             (text_width + 25, y_pos + baseline + 8),
                             (0, 0, 0), -1)
                
                # Draw white text
                color = (0, 255, 255) if j == 0 else (255, 255, 255)  # Yellow for title, white for others
                cv2.putText(debug_image, text, (15, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness + 1)
            
                        
            # Save only full image
            full_filename = f"{debug_dir}/dot_{i+1:03d}_full.jpg"
            
            # Save full image
            cv2.imwrite(full_filename, debug_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
                                   
            saved_files.append(f"dot_{i+1:03d}_full.jpg")
        
        # Also save a master overview image with all dots numbered
        master_image = self.original_image.copy()
        for i, (x, y, r) in enumerate(self.detected_dots):
            center_x, center_y, radius = int(x), int(y), int(r)
            
            # Draw all dots with numbers
            cv2.circle(master_image, (center_x, center_y), radius, (0, 255, 0), 2)
            cv2.circle(master_image, (center_x, center_y), 3, (0, 0, 255), -1)
            
            # Add number label with background
            label = f"{i+1}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_pos = (center_x + radius + 8, center_y + 5)
            
            cv2.rectangle(master_image, 
                         (label_pos[0] - 3, label_pos[1] - text_height - 3),
                         (label_pos[0] + text_width + 3, label_pos[1] + baseline + 3),
                         (0, 0, 0), -1)
            cv2.putText(master_image, label, label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Add summary info to master image
        summary_lines = [
            f"DETECTION SUMMARY - Total: {len(self.detected_dots)} dots",
            f"Params: MinDist={self.hough_min_dist.get()}, Param1={self.hough_param1.get()}, Param2={self.hough_param2.get()}",
            f"Radius Range: {self.hough_min_radius.get()}-{self.hough_max_radius.get()}px"
        ]
        
        for j, text in enumerate(summary_lines):
            y_pos = 30 + (j * 25)
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(master_image, (10, y_pos - text_height - 5), 
                         (text_width + 20, y_pos + baseline + 5), (0, 0, 0), -1)
            cv2.putText(master_image, text, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        master_filename = f"{debug_dir}/000_master_overview.jpg"
        cv2.imwrite(master_filename, master_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Show success message
        messagebox.showinfo("Individual Debug Images Saved", 
                           f"Saved {len(self.detected_dots)} individual dot images to:\n{debug_dir}\n\n"
                           f"Files created:\n"
                           f"• 000_master_overview.jpg (all dots numbered)\n"
                           f"• dot_001_full.jpg\n"
                           f"• dot_002_full.jpg\n"
                           f"• ... (up to dot_{len(self.detected_dots):03d}_full.jpg)\n\n"
                           f"Each image shows:\n"
                           f"• Green circle = detected boundary\n"
                           f"• Red dot = center point\n"
                           f"• Blue crosshair = precise center\n"
                           f"• Info overlay = dot details")
        
        print(f"Debug images saved to directory: {debug_dir}")
        print(f"Total files created: {len(saved_files) + 1}")

    def save_results(self):
        """Save detection results."""
        if not self.detected_dots:
            messagebox.showwarning("No Results", "No dots detected to save!")
            return
        
        # Save visualization
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        output_path = f"/home/francodp/camera_app/dot_identifier_tests/{base_name}_tuned.png"
        
        # Create high-res figure for saving
        save_fig = plt.figure(figsize=(12, 10), dpi=300)
        save_ax = save_fig.add_subplot(111)
        
        rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        save_ax.imshow(rgb_image)
        
        if self.detected_dots:
            for x, y, r in self.detected_dots:
                save_ax.plot(x, y, 'r+', markersize=6, markeredgewidth=1.5)
                circle = plt.Circle((x, y), r, fill=False, color='red', linewidth=1, alpha=0.8)
                save_ax.add_patch(circle)
        
        save_ax.set_title(f"White Dot Detection: {len(self.detected_dots)} dots (Threshold: {self.white_threshold.get()})")
        save_ax.axis('off')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(save_fig)
        
        messagebox.showinfo("Saved", f"Results saved to {output_path}")
    
    def load_new_image(self):
        """Load a new image for tuning."""
        file_path = filedialog.askopenfilename(
            title="Select Calibration Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            self.image_path = file_path
            self.load_image()
            self.update_detection()
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


def main():
    """Main function to start the live tuner."""
    parser = argparse.ArgumentParser(description="Live dot detection parameter tuner")
    parser.add_argument("--image", required=True, help="Path to calibration image")
    
    args = parser.parse_args()
    
    # Create and run the tuner
    tuner = LiveDotTuner(args.image)
    tuner.run()


if __name__ == "__main__":
    main()