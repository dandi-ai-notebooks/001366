"""
Script to demonstrate Radon Transform for vessel diameter analysis.
This script will apply the Radon Transform to a vessel image to determine
vessel orientation and diameter.
"""
import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import radon, iradon
from skimage.measure import profile_line

# Load the NWB file (using the first file)
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001366/assets/2f12bce3-f841-46ca-b928-044269122a59/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
movies = nwb.acquisition["Movies"]

# Extract a frame with a clear vessel
frame_idx = 3000  # Middle frame
frame = movies.data[frame_idx]

# Select region of interest (ROI) with a vessel
# Based on our previous analysis
roi_x1, roi_y1 = 200, 150
roi_x2, roi_y2 = 300, 250
roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

# Function to enhance contrast for better visualization
def enhance_contrast(image, percentile_low=1, percentile_high=99):
    low = np.percentile(image, percentile_low)
    high = np.percentile(image, percentile_high)
    image_scaled = np.clip((image - low) / (high - low), 0, 1)
    return image_scaled

# Preprocess the ROI for analysis
# 1. Enhance contrast
roi_enhanced = enhance_contrast(roi)
# 2. Apply Gaussian filter to reduce noise (sigma=1.5)
roi_smoothed = gaussian_filter(roi_enhanced, sigma=1.5)

# Display the ROI
plt.figure(figsize=(8, 8))
plt.imshow(roi_enhanced, cmap='gray')
plt.title('Region of Interest (ROI) with Vessel')
plt.colorbar(label='Normalized Intensity')
plt.savefig('tmp_scripts/radon_roi.png', dpi=150)
plt.close()

# Apply Radon Transform to the ROI
# The radon transform takes projections of an image along specified angles
# For vessels, the projection perpendicular to the vessel orientation will have
# the sharpest peak, which can be used to determine vessel diameter
theta = np.linspace(0., 180., 180)  # Angles in degrees
sinogram = radon(roi_smoothed, theta=theta)

# Plot the Radon Transform (sinogram)
plt.figure(figsize=(10, 8))
plt.imshow(sinogram, cmap='inferno', aspect='auto', 
           extent=(0, 180, 0, sinogram.shape[0]))
plt.title('Radon Transform (Sinogram)')
plt.xlabel('Angle (degrees)')
plt.ylabel('Distance (pixels)')
plt.colorbar(label='Projection Intensity')
plt.savefig('tmp_scripts/radon_sinogram.png', dpi=150)
plt.close()

# Find the angle with the highest variance in the projections
# This angle is perpendicular to the vessel orientation
projection_variance = np.var(sinogram, axis=0)
max_var_angle_idx = np.argmax(projection_variance)
vessel_angle = theta[max_var_angle_idx]
perpendicular_angle = (vessel_angle + 90) % 180
print(f"Detected vessel angle: {vessel_angle:.2f} degrees")
print(f"Perpendicular angle: {perpendicular_angle:.2f} degrees")

# Plot the variance of projections at different angles
plt.figure(figsize=(10, 6))
plt.plot(theta, projection_variance)
plt.axvline(x=vessel_angle, color='r', linestyle='-', 
            label=f'Vessel angle: {vessel_angle:.2f}°')
plt.axvline(x=perpendicular_angle, color='g', linestyle='--', 
            label=f'Perpendicular: {perpendicular_angle:.2f}°')
plt.title('Projection Variance vs. Angle')
plt.xlabel('Angle (degrees)')
plt.ylabel('Variance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('tmp_scripts/radon_variance.png', dpi=150)
plt.close()

# Extract the projection perpendicular to the vessel
perpendicular_idx = (max_var_angle_idx + 45) % 180  # +45 because we need perpendicular
perpendicular_projection = sinogram[:, perpendicular_idx]

# Plot the projection perpendicular to the vessel
plt.figure(figsize=(10, 6))
plt.plot(perpendicular_projection)
plt.title(f'Projection Perpendicular to Vessel (Angle: {perpendicular_angle:.2f}°)')
plt.xlabel('Position (pixels)')
plt.ylabel('Projection Intensity')
plt.grid(True, alpha=0.3)
plt.savefig('tmp_scripts/radon_perpendicular_projection.png', dpi=150)
plt.close()

# Determine vessel diameter using Full Width at Half Maximum (FWHM) on the projection
# Find the peak in the perpendicular projection
peak_idx = np.argmax(perpendicular_projection)
peak_value = perpendicular_projection[peak_idx]
# Determine the baseline (minimum value)
baseline = np.min(perpendicular_projection)
# Calculate half-max value
half_max = baseline + (peak_value - baseline) / 2

# Find the indices where the projection crosses the half-max value
above_half_max = perpendicular_projection > half_max
# Find transitions
transitions = np.diff(above_half_max.astype(int))
rising_indices = np.where(transitions == 1)[0]
falling_indices = np.where(transitions == -1)[0]

if len(rising_indices) > 0 and len(falling_indices) > 0:
    # Make sure we're looking at the right pair of rising/falling edges
    if falling_indices[0] < rising_indices[0]:
        # The first transition is falling, so take the first falling and the next rising
        left_idx = falling_indices[0]
        if len(rising_indices) > 0:
            right_idx = rising_indices[0]
        else:
            right_idx = None
    else:
        # The first transition is rising, so take the first rising and falling
        left_idx = rising_indices[0]
        if len(falling_indices) > 0:
            right_idx = falling_indices[0]
        else:
            right_idx = None
            
    if right_idx is not None:
        # Calculate diameter using FWHM
        vessel_diameter_pixels = right_idx - left_idx
        print(f"Vessel diameter: {vessel_diameter_pixels} pixels")
        
        # Visualize the FWHM measurement
        plt.figure(figsize=(10, 6))
        plt.plot(perpendicular_projection, 'b-', linewidth=1.5)
        plt.axhline(y=half_max, color='r', linestyle='--', alpha=0.7, label='Half Maximum')
        plt.axvline(x=left_idx, color='g', linestyle='-', alpha=0.7, label='FWHM Edges')
        plt.axvline(x=right_idx, color='g', linestyle='-', alpha=0.7)
        plt.title(f'Vessel Diameter: {vessel_diameter_pixels} pixels (using FWHM)')
        plt.xlabel('Position (pixels)')
        plt.ylabel('Projection Intensity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('tmp_scripts/radon_diameter_fwhm.png', dpi=150)
        plt.close()
else:
    print("Could not determine vessel diameter: no clear edges detected.")

# Calculate and display the inverse Radon transform (reconstruction)
# This is useful to verify if the Radon transform captured the vessel structure correctly
reconstruction = iradon(sinogram, theta=theta, filter_name='ramp')
plt.figure(figsize=(8, 8))
plt.imshow(reconstruction, cmap='gray')
plt.title('Reconstructed Image from Radon Transform')
plt.colorbar(label='Intensity')
plt.savefig('tmp_scripts/radon_reconstruction.png', dpi=150)
plt.close()

# Visualize the detected vessel orientation on the original ROI
plt.figure(figsize=(8, 8))
plt.imshow(roi_enhanced, cmap='gray')

# Calculate the center of the ROI
center_y, center_x = roi_enhanced.shape[0] // 2, roi_enhanced.shape[1] // 2

# Calculate endpoints for a line along the vessel orientation (red line)
length = 50  # Length of the line in pixels
angle_rad = np.deg2rad(vessel_angle)
dx = length * np.cos(angle_rad)
dy = length * np.sin(angle_rad)
x1_vessel, y1_vessel = center_x - dx, center_y - dy
x2_vessel, y2_vessel = center_x + dx, center_y + dy

# Calculate endpoints for a line perpendicular to the vessel (green line)
perp_angle_rad = np.deg2rad(perpendicular_angle)
dx_perp = length * np.cos(perp_angle_rad)
dy_perp = length * np.sin(perp_angle_rad)
x1_perp, y1_perp = center_x - dx_perp, center_y - dy_perp
x2_perp, y2_perp = center_x + dx_perp, center_y + dy_perp

# Draw the lines on the plot
plt.plot([x1_vessel, x2_vessel], [y1_vessel, y2_vessel], 'r-', linewidth=2, 
         label=f'Vessel Orientation ({vessel_angle:.1f}°)')
plt.plot([x1_perp, x2_perp], [y1_perp, y2_perp], 'g-', linewidth=2, 
         label=f'Measurement Line ({perpendicular_angle:.1f}°)')

plt.title('Detected Vessel Orientation')
plt.legend(loc='upper right')
plt.savefig('tmp_scripts/radon_vessel_orientation.png', dpi=150)
plt.close()

# Extract intensity profile along the perpendicular direction for comparison
# This is similar to our earlier approach but guided by Radon transform orientation
profile = profile_line(roi_enhanced, 
                      (center_y - dy_perp, center_x - dx_perp), 
                      (center_y + dy_perp, center_x + dx_perp), 
                      linewidth=1, mode='reflect')

plt.figure(figsize=(10, 6))
plt.plot(profile, 'b-', linewidth=1.5)
plt.title('Intensity Profile Along Radon-Detected Perpendicular Line')
plt.xlabel('Position (pixels)')
plt.ylabel('Intensity')
plt.grid(True, alpha=0.3)
plt.savefig('tmp_scripts/radon_intensity_profile.png', dpi=150)
plt.close()

print("Radon transform analysis completed.")