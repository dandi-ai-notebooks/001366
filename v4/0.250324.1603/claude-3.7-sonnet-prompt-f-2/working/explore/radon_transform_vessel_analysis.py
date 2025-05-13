"""
Analyze vessel diameter using Radon transform method.
This script will:
1. Load a frame from the NWB file
2. Apply the Radon transform to detect vessel orientation
3. Compare the Radon transform approach with threshold-based detection
4. Visualize the results for vessel diameter measurement
"""

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import ndimage
from skimage.transform import radon, rescale

# Set up the plot styling
sns.set_theme()

# Define function to detect vessel orientation using Radon transform
def detect_vessel_orientation(image, theta_range=np.arange(0, 180, 1)):
    """
    Use Radon transform to detect the main orientation of a vessel in the image
    """
    sinogram = radon(image, theta=theta_range)
    
    # Find the angle with the highest variance in the transform
    variance = np.var(sinogram, axis=0)
    orientation = theta_range[np.argmax(variance)]
    
    return orientation, sinogram, variance

# Define function to measure vessel diameter using Radon transform
def measure_vessel_diameter_radon(image, orientation):
    """
    Measure vessel diameter using Radon transform perpendicular to vessel orientation
    """
    # Calculate perpendicular angle to vessel orientation
    perp_angle = (orientation + 90) % 180
    
    # Apply Radon transform at the perpendicular angle
    sinogram_perp = radon(image, theta=[perp_angle])
    
    # Vessel diameter can be estimated from the width of the peak in the sinogram
    profile = sinogram_perp[:, 0]
    
    # Normalize profile
    normalized_profile = profile / np.max(profile)
    
    # Use full width at half maximum (FWHM) to estimate diameter
    # Find points where the profile crosses the half-maximum line
    half_max = np.max(normalized_profile) / 2.0
    above_half_max = normalized_profile > half_max
    
    # Find edges
    edges = np.where(np.diff(above_half_max.astype(int)))[0]
    if len(edges) >= 2:
        # Calculate FWHM
        fwhm = edges[-1] - edges[0]
        return fwhm
    else:
        return np.nan

# Load the NWB file (using the smaller F15 dataset for faster processing)
url = "https://api.dandiarchive.org/api/assets/71fa07fc-4309-4013-8edd-13213a86a67d/download/"
print(f"Loading NWB file from {url}")

remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the Movies dataset
movies = nwb.acquisition["Movies"]
print(f"Dataset shape: {movies.data.shape}, rate: {movies.rate} fps")

# Load a frame
frame_idx = 1000  # Select a frame from the middle of the dataset
print(f"Loading frame {frame_idx}...")
frame = movies.data[frame_idx, :, :]

# Visualize the original frame
plt.figure(figsize=(10, 8))
plt.imshow(frame, cmap='gray')
plt.colorbar(label='Pixel Value')
plt.title(f"Original Frame (Index: {frame_idx})")
plt.tight_layout()
plt.savefig("explore/radon_original_frame.png")
print("Saved original frame")

# Apply Radon transform to detect vessel orientation
print("\nApplying Radon transform to detect vessel orientation...")
orientation, sinogram, variance = detect_vessel_orientation(frame)
print(f"Detected vessel orientation: {orientation:.1f} degrees")

# Visualize the Radon transform sinogram
plt.figure(figsize=(10, 8))
plt.imshow(sinogram, aspect='auto', cmap='viridis',
           extent=(0, 180, 0, sinogram.shape[0]))
plt.colorbar(label='Intensity')
plt.title("Radon Transform Sinogram")
plt.xlabel("Angle (degrees)")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("explore/radon_sinogram.png")
print("Saved Radon transform sinogram")

# Visualize the variance across angles to show orientation detection
plt.figure(figsize=(12, 6))
plt.plot(np.arange(0, 180, 1), variance)
plt.axvline(x=orientation, color='r', linestyle='--', 
            label=f"Detected Orientation: {orientation:.1f} degrees")
plt.title("Variance Across Different Angles")
plt.xlabel("Angle (degrees)")
plt.ylabel("Variance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("explore/radon_orientation_variance.png")
print("Saved orientation variance plot")

# Visualize the original frame with vessel orientation line
plt.figure(figsize=(10, 8))
plt.imshow(frame, cmap='gray')

# Draw a line showing the detected vessel orientation
center_y, center_x = frame.shape[0] // 2, frame.shape[1] // 2
length = max(frame.shape[0], frame.shape[1]) // 2
angle_rad = np.deg2rad(orientation)
delta_x = length * np.cos(angle_rad)
delta_y = length * np.sin(angle_rad)
plt.plot([center_x - delta_x, center_x + delta_x], 
         [center_y - delta_y, center_y + delta_y], 
         'r-', linewidth=2, label=f"Vessel Orientation: {orientation:.1f}°")

# Draw a perpendicular line for diameter measurement
perp_angle_rad = np.deg2rad((orientation + 90) % 180)
perp_delta_x = length * np.cos(perp_angle_rad)
perp_delta_y = length * np.sin(perp_angle_rad)
plt.plot([center_x - perp_delta_x, center_x + perp_delta_x], 
         [center_y - perp_delta_y, center_y + perp_delta_y], 
         'g-', linewidth=2, label="Diameter Measurement Line")

plt.colorbar(label='Pixel Value')
plt.title("Frame with Detected Vessel Orientation")
plt.legend()
plt.tight_layout()
plt.savefig("explore/radon_vessel_orientation.png")
print("Saved frame with vessel orientation")

# Measure vessel diameter using Radon transform
vessel_diameter = measure_vessel_diameter_radon(frame, orientation)
print(f"\nVessel diameter using Radon transform: {vessel_diameter:.2f} pixels")

# Apply a threshold-based method for comparison
# Invert and normalize the frame (vessel is dark in F15 dataset)
inverted_frame = np.max(frame) - frame
normalized_frame = inverted_frame / np.max(inverted_frame)

# Apply threshold
threshold = 0.4
binary_vessel = normalized_frame > threshold

# Visualize the binary vessel mask
plt.figure(figsize=(10, 8))
plt.imshow(binary_vessel, cmap='gray')
plt.title("Binary Vessel Mask (Threshold Method)")
plt.colorbar(label='Binary Value')
plt.tight_layout()
plt.savefig("explore/vessel_binary_mask.png")
print("Saved binary vessel mask")

# Count the number of pixels in a cross-section perpendicular to vessel orientation
# Use the same cross-section line as in the Radon transform
y_indices = np.arange(frame.shape[0])
x_indices = np.round(center_x + (y_indices - center_y) *
                     np.tan(perp_angle_rad)).astype(int)

# Keep only points that are inside the image
valid_indices = (x_indices >= 0) & (x_indices < frame.shape[1])
y_valid = y_indices[valid_indices]
x_valid = x_indices[valid_indices]

# Extract the binary profile along the cross-section
binary_profile = [binary_vessel[y, x] for y, x in zip(y_valid, x_valid)]
binary_profile = np.array(binary_profile)

# Estimate vessel diameter as the number of 'True' pixels in the profile
threshold_diameter = np.sum(binary_profile)
print(f"Vessel diameter using threshold method: {threshold_diameter} pixels")

# Visualize the binary profile along the measurement line
plt.figure(figsize=(12, 6))
plt.plot(binary_profile.astype(int))
plt.axhline(y=0.5, color='r', linestyle='--', label="Threshold")
plt.title("Binary Profile Along Measurement Line")
plt.xlabel("Position Along Line")
plt.ylabel("Binary Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("explore/binary_profile.png")
print("Saved binary profile plot")

# Compare the results of the two methods
print("\nComparison of Vessel Diameter Measurement Methods:")
print(f"Radon Transform Method: {vessel_diameter:.2f} pixels")
print(f"Threshold Method: {threshold_diameter} pixels")
print(f"Difference: {abs(vessel_diameter - threshold_diameter):.2f} pixels")

# Visualize a comparison of both techniques on the original frame
plt.figure(figsize=(10, 8))
plt.imshow(frame, cmap='gray')

# Draw the perpendicular line
plt.plot([center_x - perp_delta_x, center_x + perp_delta_x], 
         [center_y - perp_delta_y, center_y + perp_delta_y], 
         'g-', linewidth=2, label="Measurement Line")

# Mark the Radon transform diameter
radon_start_x = center_x - (vessel_diameter/2) * np.cos(perp_angle_rad)
radon_start_y = center_y - (vessel_diameter/2) * np.sin(perp_angle_rad)
radon_end_x = center_x + (vessel_diameter/2) * np.cos(perp_angle_rad)
radon_end_y = center_y + (vessel_diameter/2) * np.sin(perp_angle_rad)
plt.plot([radon_start_x, radon_end_x], [radon_start_y, radon_end_y], 
         'r-', linewidth=4, label=f"Radon: {vessel_diameter:.1f} pixels")

# Mark the threshold-based diameter
threshold_start_x = center_x - (threshold_diameter/2) * np.cos(perp_angle_rad)
threshold_start_y = center_y - (threshold_diameter/2) * np.sin(perp_angle_rad)
threshold_end_x = center_x + (threshold_diameter/2) * np.cos(perp_angle_rad)
threshold_end_y = center_y + (threshold_diameter/2) * np.sin(perp_angle_rad)
plt.plot([threshold_start_x, threshold_end_x], [threshold_start_y, threshold_end_y], 
         'b--', linewidth=3, label=f"Threshold: {threshold_diameter} pixels")

plt.colorbar(label='Pixel Value')
plt.title("Comparison of Vessel Diameter Measurement Methods")
plt.legend()
plt.tight_layout()
plt.savefig("explore/vessel_diameter_method_comparison.png")
print("Saved method comparison visualization")