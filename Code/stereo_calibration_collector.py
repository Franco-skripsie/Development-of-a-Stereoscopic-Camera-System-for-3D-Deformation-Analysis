import cv2
from picamera2 import Picamera2
import os
import time
import numpy as np
import csv

# === Setup output ===

# Create output directory for Calibration images:
output_dir = "calib"

# Create output directory for Disparity Capture images:
#output_dir = "capture_disparity_1"

os.makedirs(output_dir, exist_ok=True)

# Ask user for number of image pairs to capture
while True:
    try:
        target_pairs = int(input("Enter number of image pairs to capture: ").strip())
        if target_pairs > 0:
            break
        else:
            print("Please enter a positive number.")
    except ValueError:
        print("Please enter a valid number.")

# === CSV Logging Setup ===
csv_path = os.path.join(output_dir, "capture_log.csv")
new_csv = not os.path.exists(csv_path)
csv_file = open(csv_path, mode='a', newline='')
csv_writer = csv.writer(csv_file)
if new_csv:
    csv_writer.writerow(["left_image", "right_image"])

# === Initialize cameras with proper settings for Pi Camera Module 3 ===
print("Initializing cameras...")
cam_left = Picamera2(0)
cam_right = Picamera2(1)

# Create still configuration with proper settings to prevent blank images
config_left = cam_left.create_still_configuration(
    main={"size": (4608, 2592)}                                                 # Full resolution for accurate calibration
)

config_right = cam_right.create_still_configuration(
    main={"size": (4608, 2592)}                                                 # Full resolution for accurate calibration
)

cam_left.configure(config_left)
cam_right.configure(config_right)

print("Starting cameras...")
cam_left.start()
cam_right.start()

# Wait for cameras to stabilize and auto-exposure to settle
print("Waiting for cameras to stabilize...")
time.sleep(2)

# Configure anti-flicker settings for 50Hz LED lighting
print("Configuring anti-flicker settings for 50Hz LED lighting...")
try:
    
    anti_flicker_controls_left = {
        "AeEnable": False,                                                      # Disable auto-exposure for consistent results
        "ExposureTime": 10000,                                                  # Multiple of 1/50s exposure (10ms)            
        "AnalogueGain": 2.0,                                                    # Fixed gain to prevent auto-adjustment
        "AwbEnable": True,                                                      # Keep auto white balance for color consistency
        "AeFlickerMode": 1,                                                     # Enable flicker detection/reduction (if supported)
        "AeFlickerPeriod": 5000                                                 # 5ms flicker period
    }
    
    anti_flicker_controls_right = {
        "AeEnable": False,
        "ExposureTime": 10000,                                                  # Match left camera exactly
        "AnalogueGain": 2.0,
        "AwbEnable": True,
        "AeFlickerMode": 1,
        "AeFlickerPeriod": 5000                                                 # 5ms flicker period
    }
    
    # Apply anti-flicker settings
    cam_left.set_controls(anti_flicker_controls_left)
    cam_right.set_controls(anti_flicker_controls_right)
    
    print("✓ Anti-flicker settings applied:")
    print(f"  - Exposure time: 10ms")
    print(f"  - Fixed analog gain: 2.0")
    print(f"  - Flicker period: 5ms (50Hz)")
    
except Exception as e:
    print(f"⚠️ Some anti-flicker settings not supported: {e}")
    print("📝 Using fallback anti-flicker configuration...")
    
    # Fallback: Basic anti-flicker settings
    try:
        fallback_controls = {
            "AeEnable": False,
            "ExposureTime": 10000,                                              # 10ms exposure
            "AnalogueGain": 2.0
        }
        
        cam_left.set_controls(fallback_controls)
        cam_right.set_controls(fallback_controls)
        print("✓ Basic anti-flicker settings applied")
        
    except Exception as fallback_e:
        print(f"❌ Anti-flicker configuration failed: {fallback_e}")
        print("⚠️ Images may show LED flicker - try manual exposure adjustment")

# Additional stabilization time after setting controls
time.sleep(2)

print("\nLive preview started.")
print("Press 'c' to CAPTURE a stereo image pair.")
print("Press 't' to TOGGLE between manual and timer capture modes.")
print("Press 'a' to trigger AUTOFOCUS on both cameras.")
print("Press 'l' to LOCK white balance and autofocus values.")
print("Press 'r' to RESTORE proven camera settings from successful capture.")
print("Press 'q' to STOP and quit.")
print(f"Target: {target_pairs} image pairs for calibration.\n")

# === Draw crosshair ===
def draw_crosshair(frame):
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    
    # Draw center crosshair
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)
    
    return frame

# === Get next index for file naming ===
def get_next_index(output_dir):
    indices = []
    for fname in os.listdir(output_dir):
        if fname.startswith("left") and fname.endswith(".png"):
            digits = ''.join(filter(str.isdigit, fname))
            if digits:
                indices.append(int(digits))
    return max(indices) + 1 if indices else 1

# === Image quality validation function ===
def validate_image_quality(image_array, camera_name):
    """Validate that the image is not blank/overexposed"""
    if image_array is None or image_array.size == 0:
        print(f"Warning: {camera_name} image is empty")
        return False
    
    # Check if image is grayscale or color
    if len(image_array.shape) == 3:
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    
    # Check for blank/overexposed images
    if mean_val < 10:
        print(f"Warning: {camera_name} image too dark (mean={mean_val:.1f})")
        return False
    elif mean_val > 240:
        print(f"Warning: {camera_name} image overexposed (mean={mean_val:.1f})")
        return False
    elif std_val < 5:
        print(f"Warning: {camera_name} image lacks detail (std={std_val:.1f})")
        return False
    
    return True

# === Autofocus function ===
def trigger_autofocus(cam_left, cam_right):
    """Trigger center-area autofocus on both cameras"""
    print("🎯 Triggering center-area autofocus...")
    
    try:
        # Set autofocus window to center area (around crosshairs)
        # Camera resolution: 4608x2592
        center_x, center_y = 4608 // 2, 2592 // 2
        
        # Define focus window around center (20% of image width/height)
        window_width = int(4608 * 0.2)                                          # ~920 pixels
        window_height = int(2592 * 0.2)                                         # ~518 pixels
        
        # Calculate window boundaries
        left_x = max(0, center_x - window_width // 2)
        right_x = min(4608, center_x + window_width // 2)
        top_y = max(0, center_y - window_height // 2)
        bottom_y = min(2592, center_y + window_height // 2)
        
        print(f"🔍 Focus window: center area {window_width}x{window_height} pixels")
        print(f"   Pixel coords: ({left_x}, {top_y}, {right_x - left_x}, {bottom_y - top_y})")
        
        # Set focus window controls for both cameras using pixel coordinates
        center_focus_controls = {
            "AfMode": 2,                                                        # Auto focus mode
            "AfMetering": 1,                                                    # Spot metering (focus on specific area)
            "AfWindows": [(left_x, top_y, right_x - left_x, bottom_y - top_y)], # Focus window in pixels
            "AfTrigger": 0                                                      # Start autofocus
        }
        
        # Apply center focus settings
        cam_left.set_controls(center_focus_controls)
        cam_right.set_controls(center_focus_controls)
        
        print("✓ Center-area autofocus triggered on both cameras")
        
        # Wait longer for focus to complete on specific area
        time.sleep(3)
        
        # Check autofocus status
        try:
            left_af_state = cam_left.capture_metadata().get("AfState", "Unknown")
            right_af_state = cam_right.capture_metadata().get("AfState", "Unknown")
            print(f"📷 Left camera AF state: {left_af_state}")
            print(f"📷 Right camera AF state: {right_af_state}")
            
            # Get lens positions for verification
            left_lens_pos = cam_left.capture_metadata().get("LensPosition", "Unknown")
            right_lens_pos = cam_right.capture_metadata().get("LensPosition", "Unknown")
            print(f"🔧 Left lens position: {left_lens_pos}")
            print(f"🔧 Right lens position: {right_lens_pos}")
            
        except Exception as status_e:
            print("✓ Center autofocus completed (status check unavailable)")
        
        print("🎯 Center-area autofocus complete - cameras focused on crosshair region")
        
    except Exception as e:
        print(f"⚠️ Center autofocus error: {e}")
        print("📝 Note: Center autofocus may not be available on this camera model")
        
        # Try alternative autofocus method
        try:
            print("🔄 Trying standard autofocus method...")
            cam_left.set_controls({"AfMode": 2, "AfTrigger": 0})
            cam_right.set_controls({"AfMode": 2, "AfTrigger": 0})
            time.sleep(2)
            print("✓ Standard autofocus method completed")
        except:
            print("❌ Autofocus not supported on this hardware")

# === Restore proven camera settings ===
def restore_proven_settings(cam_left, cam_right):
    print("🔧 Restoring proven camera settings from successful capture...")
    
    try:
        
        proven_settings_left = {
            "AeEnable": False,                                          # Disable auto-exposure
            "ExposureTime": 10000,                                      # 10ms anti-flicker
            "AnalogueGain": 2.0,                                        # Fixed gain
            "AwbEnable": False,                                         # Disable auto white balance
            "ColourGains": (2.205003261566162, 1.7105246782302856),     # Proven AWB gains
            "AfMode": 0,                                                # Manual focus mode
            "LensPosition": 1.9790470600128174,                         # Proven focus position
            "AeFlickerMode": 1,                                         # Anti-flicker enabled
            "AeFlickerPeriod": 5000                                     # 5ms flicker period
        }
        
        proven_settings_right = {
            "AeEnable": False,                                          # Disable auto-exposure
            "ExposureTime": 10000,                                      # 10ms anti-flicker
            "AnalogueGain": 2.0,                                        # Fixed gain
            "AwbEnable": False,                                         # Disable auto white balance
            "ColourGains": (2.2194361686706543, 1.70102059841156),      # Proven AWB gains
            "AfMode": 0,                                                # Manual focus mode
            "LensPosition": 1.6614744663238525,                         # Proven focus position
            "AeFlickerMode": 1,                                         # Anti-flicker enabled
            "AeFlickerPeriod": 5000                                     # 5ms flicker period
        }
        
        # Apply proven settings
        cam_left.set_controls(proven_settings_left)
        cam_right.set_controls(proven_settings_right)
        
        print("✓ Proven settings applied successfully:")
        print(f"  Left camera:")
        print(f"    - Exposure: 20ms, Gain: 2.0")
        print(f"    - AWB gains: {proven_settings_left['ColourGains']}")
        print(f"    - Lens position: {proven_settings_left['LensPosition']}")
        print(f"  Right camera:")
        print(f"    - Exposure: 20ms, Gain: 2.0")
        print(f"    - AWB gains: {proven_settings_right['ColourGains']}")
        print(f"    - Lens position: {proven_settings_right['LensPosition']}")
        print("🎯 Cameras configured for optimal calibration image quality")
                
        # Brief pause for settings to take effect
        time.sleep(1)
        
    except Exception as e:
        print(f"❌ Failed to apply proven settings: {e}")
        print("📝 Note: Some proven settings may not be supported")
        
        # Fallback to basic proven settings
        try:
            print("🔄 Applying basic proven settings...")
            basic_proven = {
                "AeEnable": False,
                "ExposureTime": 10000,
                "AnalogueGain": 2.0
            }
            
            cam_left.set_controls(basic_proven)
            cam_right.set_controls(basic_proven)
            print("✓ Basic proven settings applied (exposure + gain)")
            
        except Exception as fallback_e:
            print(f"❌ Fallback settings failed: {fallback_e}")

# === Lock white balance and autofocus ===
def lock_wb_and_focus(cam_left, cam_right):
    """Lock white balance and autofocus on current values for both cameras"""
    print("🔒 Locking white balance and autofocus...")
    
    try:
        # Get current white balance and focus values
        left_metadata = cam_left.capture_metadata()
        right_metadata = cam_right.capture_metadata()
        
        # Extract current values
        left_awb_gains = left_metadata.get("ColourGains", None)
        right_awb_gains = right_metadata.get("ColourGains", None)
        left_lens_pos = left_metadata.get("LensPosition", None)
        right_lens_pos = right_metadata.get("LensPosition", None)
        
        print(f"📊 Current values:")
        print(f"   Left AWB gains: {left_awb_gains}")
        print(f"   Right AWB gains: {right_awb_gains}")
        print(f"   Left lens position: {left_lens_pos}")
        print(f"   Right lens position: {right_lens_pos}")
        
        # Lock settings on both cameras
        lock_controls_left = {}
        lock_controls_right = {}
        
        # Lock white balance if available
        if left_awb_gains is not None and len(left_awb_gains) >= 2:
            lock_controls_left.update({
                "AwbEnable": False,                                             # Disable auto white balance
                "ColourGains": left_awb_gains                                   # Set fixed gains
            })
            print(f"✓ Left camera AWB locked to gains: {left_awb_gains}")
        else:
            print("⚠️ Could not get left camera AWB gains, keeping auto WB")
            
        if right_awb_gains is not None and len(right_awb_gains) >= 2:
            lock_controls_right.update({
                "AwbEnable": False,                                             # Disable auto white balance
                "ColourGains": right_awb_gains                                  # Set fixed gains
            })
            print(f"✓ Right camera AWB locked to gains: {right_awb_gains}")
        else:
            print("⚠️ Could not get right camera AWB gains, keeping auto WB")
        
        # Lock autofocus if available
        if left_lens_pos is not None:
            lock_controls_left.update({
                "AfMode": 0,                                                    # Manual focus mode
                "LensPosition": left_lens_pos                                   # Set fixed position
            })
            print(f"✓ Left camera focus locked to position: {left_lens_pos}")
        else:
            print("⚠️ Could not get left camera lens position")
            
        if right_lens_pos is not None:
            lock_controls_right.update({
                "AfMode": 0,                                                    # Manual focus mode
                "LensPosition": right_lens_pos                                  # Set fixed position
            })
            print(f"✓ Right camera focus locked to position: {right_lens_pos}")
        else:
            print("⚠️ Could not get right camera lens position")
        
        # Apply lock settings
        if lock_controls_left:
            cam_left.set_controls(lock_controls_left)
        if lock_controls_right:
            cam_right.set_controls(lock_controls_right)
            
        print("🔒 White balance and autofocus locked on both cameras")
        print("💡 Settings will remain fixed for consistent calibration images")
        
        # Brief pause for settings to take effect
        time.sleep(0.5)
        
    except Exception as e:
        print(f"❌ Failed to lock settings: {e}")
        print("📝 Note: Some lock features may not be supported on this camera model")



# === Main loop ===
captured_pairs = 0
img_index = get_next_index(output_dir)

# Capture mode settings
capture_mode = "manual"  # Can be "manual" or "timer"
timer_duration = 3  # seconds
last_capture_time = time.time()

print("Camera initialization complete!")
print(f"Starting image index: {img_index}")
print(f"Target: {target_pairs} checkerboard pairs")
print("Preview display: BGR colorspace")
print(f"Capture mode: {capture_mode.upper()}")

try:
    while captured_pairs < target_pairs:
        # Capture frames with error handling
        try:
            frame_left = cam_left.capture_array()
            frame_right = cam_right.capture_array()
        except Exception as e:
            print(f"Error capturing frames: {e}")
            time.sleep(0.1)
            continue

        # Validate image quality for preview
        if not validate_image_quality(frame_left, "Left"):
            print("Left camera quality issue - check lighting/settings")
        if not validate_image_quality(frame_right, "Right"):
            print("Right camera quality issue - check lighting/settings")

        # Create preview (resize for performance, but maintain aspect ratio)
        scale_factor = 0.2  # 20% of full resolution for preview
        preview_w = int(4608 * scale_factor)
        preview_h = int(2592 * scale_factor)
        
        preview_left = cv2.resize(frame_left, (preview_w, preview_h))
        preview_right = cv2.resize(frame_right, (preview_w, preview_h))
        
        # Convert RGB to BGR for proper OpenCV display
        preview_left = cv2.cvtColor(preview_left, cv2.COLOR_RGB2BGR)
        preview_right = cv2.cvtColor(preview_right, cv2.COLOR_RGB2BGR)
        
        preview_left = draw_crosshair(preview_left.copy())
        preview_right = draw_crosshair(preview_right.copy())

        combined_preview = np.hstack((preview_left, preview_right))
        
        # Add progress info with key commands and capture mode
        mode_display = f"Mode: {capture_mode.upper()}"
        if capture_mode == "timer":
            time_since_last = time.time() - last_capture_time
            time_remaining = max(0, timer_duration - time_since_last)
            mode_display += f" ({time_remaining:.1f}s)"
        
        progress_text = f"Captured: {captured_pairs}/{target_pairs} | {mode_display} | 'c': capture, 't': toggle, 'a': AF, 'l': lock, 'r': restore, 'q': quit"
        cv2.putText(combined_preview, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.imshow("Stereo Checkerboard Calibration (Left | Right)", combined_preview)

        # Check if timer capture should trigger
        should_capture = False
        if capture_mode == "timer" and (time.time() - last_capture_time) >= timer_duration:
            should_capture = True
            last_capture_time = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Manual capture when 'c' is pressed
            should_capture = True
            
        elif key == ord('t'):
            # Toggle capture mode
            if capture_mode == "manual":
                capture_mode = "timer"
                last_capture_time = time.time()  # Reset timer
                print(f"📸 Switched to TIMER mode - auto-capture every {timer_duration} seconds")
            else:
                capture_mode = "manual"
                print("📸 Switched to MANUAL mode - press 'c' to capture")
        
        # Perform capture if triggered (by manual 'c' or timer)
        if should_capture:
            if validate_image_quality(frame_left, "Left") and validate_image_quality(frame_right, "Right"):
                filename_left = f"left{img_index:03d}.png"
                filename_right = f"right{img_index:03d}.png"
                path_left = os.path.join(output_dir, filename_left)
                path_right = os.path.join(output_dir, filename_right)

                # Convert RGB to BGR for OpenCV saving (better quality than PIL)
                frame_left_bgr = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
                frame_right_bgr = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
                
                # Save images with OpenCV for better quality control
                success_left = cv2.imwrite(path_left, frame_left_bgr)
                success_right = cv2.imwrite(path_right, frame_right_bgr)
                
                if success_left and success_right:
                    # Verify saved file sizes
                    size_left = os.path.getsize(path_left)
                    size_right = os.path.getsize(path_right)
                    
                    if size_left > 500000 and size_right > 500000:  # > 500KB for full resolution
                        # Log to CSV
                        csv_writer.writerow([filename_left, filename_right])
                        csv_file.flush()

                        captured_pairs += 1
                        print(f"✓ Captured pair {captured_pairs}/{target_pairs}: {filename_left} ({size_left//1024}KB), {filename_right} ({size_right//1024}KB)")
                        img_index += 1
                        
                        if captured_pairs >= target_pairs:
                            print(f"\n🎉 Successfully captured all {target_pairs} pairs!")
                            break
                    else:
                        print(f"✗ Saved files too small - L:{size_left}B, R:{size_right}B")
                        # Remove small files
                        try:
                            os.remove(path_left)
                            os.remove(path_right)
                        except:
                            pass
                else:
                    print("✗ Failed to save images")
            else:
                print("✗ Cannot capture - poor image quality. Check lighting and focus.")
                
        if key == ord('a'):
            # Trigger center-area autofocus when 'a' is pressed
            trigger_autofocus(cam_left, cam_right)
            
        elif key == ord('l'):
            # Lock white balance and autofocus when 'l' is pressed
            lock_wb_and_focus(cam_left, cam_right)
            
        elif key == ord('r'):
            # Restore proven camera settings when 'r' is pressed
            restore_proven_settings(cam_left, cam_right)
            
        elif key == ord('q'):
            print("Quitting early.")
            break

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    cam_left.stop()
    cam_right.stop()
    csv_file.close()
    cv2.destroyAllWindows()
