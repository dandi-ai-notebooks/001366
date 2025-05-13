"""
Analyze vessel orientation and diameter without using external libraries like skimage.
This script will:
1. Load a frame from the NWB file
2. Use Sobel filters to detect vessel edges and orientation
3. Measure vessel diameter using intensity profiles
4. Compare different approaches to vessel diameter measurement
"""

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel

# Set up the plot styling
sns.set_theme()

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
plt.savefig("explore/simple_original_frame.png")
print("Saved original frame")

# Preprocess the frame - normalize and smooth to reduce noise
normalized_frame = frame.astype(float) / np.max(frame)
smoothed_frame = gaussian_filter(normalized_frame, sigma=1.0)

# Apply edge detection to highlight vessel boundaries
# The vessel in F15 dataset is dark, so we'll invert it first
inverted_frame = 1.0 - smoothed_frame
edges_x = sobel(inverted_frame, axis=1)
edges_y = sobel(inverted_frame, axis=0)
edges = np.sqrt(edges_x**2 + edges_y**2)

# Visualize the edge detection
plt.figure(figsize=(10, 8))
plt.imshow(edges, cmap='viridis')
plt.colorbar(label='Edge Magnitude')
plt.title("Edge Detection")
plt.tight_layout()
plt.savefig("explore/simple_edge_detection.png")
print("Saved edge detection result")

# Calculate gradient direction to estimate vessel orientation
# We'll use a histogram of gradient orientations (similar to Hough transform)
gradient_orientation = np.arctan2(edges_y, edges_x) * 180 / np.pi
gradient_orientation = (gradient_orientation + 180) % 180  # Convert to 0-180 range

# Create a histogram of gradient orientations, weighted by edge magnitude
hist_bins = np.arange(0, 181, 1)
hist_weights = edges.flatten()
hist, _ = np.histogram(gradient_orientation.flatten(), bins=hist_bins, weights=hist_weights)

# Find the dominant orientation
dominant_angle = hist_bins[:-1][np.argmax(hist)]
print(f"Detected vessel orientation: {dominant_angle:.1f} degrees")

# Visualize the orientation histogram
plt.figure(figsize=(12, 6))
plt.bar(hist_bins[:-1], hist, width=1)
plt.axvline(x=dominant_angle, color='r', linestyle='--', 
            label=f"Dominant Orientation: {dominant_angle:.1f} degrees")
plt.title("Histogram of Gradient Orientations")
plt.xlabel("Angle (degrees)")
plt.ylabel("Weighted Count")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("explore/simple_orientation_histogram.png")
print("Saved orientation histogram")

# Visualize the detected orientation on the original frame
plt.figure(figsize=(10, 8))
plt.imshow(frame, cmap='gray')

# Draw a line showing the detected vessel orientation
center_y, center_x = frame.shape[0] // 2, frame.shape[1] // 2
length = max(frame.shape[0], frame.shape[1]) // 2
angle_rad = np.deg2rad(dominant_angle)
delta_x = length * np.cos(angle_rad)
delta_y = length * np.sin(angle_rad)
plt.plot([center_x - delta_x, center_x + delta_x], 
         [center_y - delta_y, center_y + delta_y], 
         'r-', linewidth=2, label=f"Vessel Orientation: {dominant_angle:.1f}°")

# Draw a perpendicular line for diameter measurement
perp_angle = (dominant_angle + 90) % 180
perp_angle_rad = np.deg2rad(perp_angle)
perp_delta_x = length * np.cos(perp_angle_rad)
perp_delta_y = length * np.sin(perp_angle_rad)
plt.plot([center_x - perp_delta_x, center_x + perp_delta_x], 
         [center_y - perp_delta_y, center_y + perp_delta_y], 
         'g-', linewidth=2, label="Diameter Measurement Line")

plt.colorbar(label='Pixel Value')
plt.title("Frame with Detected Vessel Orientation")
plt.legend()
plt.tight_layout()
plt.savefig("explore/simple_vessel_orientation.png")
print("Saved frame with vessel orientation")

# Extract intensity profile along the perpendicular line for vessel diameter measurement
# We'll create a line of coordinates from the center to the edge of the image
line_length = min(frame.shape) // 2
line_y = np.round(center_y + np.arange(-line_length, line_length+1) * np.sin(perp_angle_rad)).astype(int)
line_x = np.round(center_x + np.arange(-line_length, line_length+1) * np.cos(perp_angle_rad)).astype(int)

# Keep only points inside the image bounds
valid_indices = (line_y >= 0) & (line_y < frame.shape[0]) & (line_x >= 0) & (line_x < frame.shape[1])
line_y = line_y[valid_indices]
line_x = line_x[valid_indices]

# Extract the intensity profile
intensity_profile = np.array([inverted_frame[y, x] for y, x in zip(line_y, line_x)])

# Visualize the intensity profile
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(intensity_profile)), intensity_profile)
plt.title("Intensity Profile Across Vessel")
plt.xlabel("Position Along Line")
plt.ylabel("Inverted Intensity")
plt.grid(True)
plt.tight_layout()
plt.savefig("explore/simple_intensity_profile.png")
print("Saved intensity profile")

# Measure vessel diameter using the Full Width at Half Maximum (FWHM) method
def measure_fwhm(profile):
    # Smooth the profile to reduce noise
    smoothed_profile = gaussian_filter(profile, sigma=2)
    
    # Find the baseline (minimum) and peak (maximum) values
    baseline = np.min(smoothed_profile)
    peak = np.max(smoothed_profile)
    
    # Calculate the half-maximum value
    half_max = baseline + (peak - baseline) / 2
    
    # Find points where the profile crosses the half-maximum line
    above_half_max = smoothed_profile > half_max
    
    # Find edges
    edges = np.where(np.diff(above_half_max.astype(int)))[0]
    if len(edges) >= 2:
        # Calculate FWHM
        fwhm = edges[-1] - edges[0]
        return fwhm, half_max, smoothed_profile
    else:
        return np.nan, half_max, smoothed_profile

fwhm_diameter, half_max, smoothed_profile = measure_fwhm(intensity_profile)
print(f"Vessel diameter using FWHM: {fwhm_diameter:.2f} pixels")

# Visualize the FWHM measurement
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(intensity_profile)), intensity_profile, 'b-', alpha=0.6, label="Original")
plt.plot(np.arange(len(smoothed_profile)), smoothed_profile, 'r-', label="Smoothed")
plt.axhline(y=half_max, color='g', linestyle='--', label=f"Half Maximum")

if not np.isnan(fwhm_diameter):
    # Find the edges for FWHM visualization
    above_half_max = smoothed_profile > half_max
    edges = np.where(np.diff(above_half_max.astype(int)))[0]
    if len(edges) >= 2:
        left_edge = edges[0]
        right_edge = edges[-1]
        plt.axvline(x=left_edge, color='k', linestyle='-', label=f"FWHM: {fwhm_diameter:.1f} pixels")
        plt.axvline(x=right_edge, color='k', linestyle='-')
        plt.axhspan(ymin=half_max, ymax=np.max(smoothed_profile), 
                    xmin=left_edge/len(smoothed_profile), xmax=right_edge/len(smoothed_profile),
                    alpha=0.2, color='g')

plt.title("Vessel Diameter Measurement using Full Width at Half Maximum (FWHM)")
plt.xlabel("Position Along Line")
plt.ylabel("Intensity")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("explore/simple_fwhm_measurement.png")
print("Saved FWHM measurement visualization")

# Apply a threshold-based method for comparison
def measure_with_threshold(profile, threshold_percentile=75):
    # Normalize the profile
    normalized_profile = (profile - np.min(profile)) / (np.max(profile) - np.min(profile))
    
    # Calculate threshold
    threshold = np.percentile(normalized_profile, threshold_percentile)
    
    # Find regions above threshold
    above_threshold = normalized_profile > threshold
    
    # Find vessel edges
    edges = np.where(np.diff(above_threshold.astype(int)))[0]
    
    if len(edges) >= 2:
        # Use first and last edges
        diameter = edges[-1] - edges[0]
        return diameter, threshold, normalized_profile
    else:
        return np.nan, threshold, normalized_profile

threshold_diameter, threshold, normalized_profile = measure_with_threshold(intensity_profile)
print(f"Vessel diameter using threshold method: {threshold_diameter:.2f} pixels")

# Visualize the threshold-based measurement
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(normalized_profile)), normalized_profile, 'b-', label="Normalized Profile")
plt.axhline(y=threshold, color='r', linestyle='--', label=f"Threshold")

if not np.isnan(threshold_diameter):
    # Find the edges for threshold visualization
    above_threshold = normalized_profile > threshold
    edges = np.where(np.diff(above_threshold.astype(int)))[0]
    if len(edges) >= 2:
        left_edge = edges[0]
        right_edge = edges[-1]
        plt.axvline(x=left_edge, color='k', linestyle='-', label=f"Threshold Diam: {threshold_diameter:.1f} pixels")
        plt.axvline(x=right_edge, color='k', linestyle='-')
        plt.axhspan(ymin=threshold, ymax=np.max(normalized_profile), 
                    xmin=left_edge/len(normalized_profile), xmax=right_edge/len(normalized_profile),
                    alpha=0.2, color='r')

plt.title("Vessel Diameter Measurement using Threshold Method")
plt.xlabel("Position Along Line")
plt.ylabel("Normalized Intensity")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("explore/simple_threshold_measurement.png")
print("Saved threshold measurement visualization")

# Compare the results of both methods
print("\nComparison of Vessel Diameter Measurement Methods:")
print(f"FWHM Method: {fwhm_diameter:.2f} pixels")
print(f"Threshold Method: {threshold_diameter:.2f} pixels")
if not (np.isnan(fwhm_diameter) or np.isnan(threshold_diameter)):
    print(f"Difference: {abs(fwhm_diameter - threshold_diameter):.2f} pixels")
    print(f"Relative Difference: {abs(fwhm_diameter - threshold_diameter)/fwhm_diameter*100:.2f}%")

# Visualize the results on the original frame
plt.figure(figsize=(10, 8))
plt.imshow(frame, cmap='gray')

# Mark the perpendicular measurement line
plt.plot([center_x - perp_delta_x, center_x + perp_delta_x], 
         [center_y - perp_delta_y, center_y + perp_delta_y], 
         'g-', linewidth=1, alpha=0.7, label="Measurement Line")

# Mark the vessel with both measurement methods
if not np.isnan(fwhm_diameter):
    fwhm_start_x = center_x - (fwhm_diameter/2) * np.cos(perp_angle_rad)
    fwhm_start_y = center_y - (fwhm_diameter/2) * np.sin(perp_angle_rad)
    fwhm_end_x = center_x + (fwhm_diameter/2) * np.cos(perp_angle_rad)
    fwhm_end_y = center_y + (fwhm_diameter/2) * np.sin(perp_angle_rad)
    plt.plot([fwhm_start_x, fwhm_end_x], [fwhm_start_y, fwhm_end_y], 
             'r-', linewidth=3, label=f"FWHM: {fwhm_diameter:.1f} pixels")

if not np.isnan(threshold_diameter):
    threshold_start_x = center_x - (threshold_diameter/2) * np.cos(perp_angle_rad)
    threshold_start_y = center_y - (threshold_diameter/2) * np.sin(perp_angle_rad)
    threshold_end_x = center_x + (threshold_diameter/2) * np.cos(perp_angle_rad)
    threshold_end_y = center_y + (threshold_diameter/2) * np.sin(perp_angle_rad)
    plt.plot([threshold_start_x, threshold_end_x], [threshold_start_y, threshold_end_y], 
             'b--', linewidth=2, label=f"Threshold: {threshold_diameter:.1f} pixels")

plt.colorbar(label='Pixel Value')
plt.title("Comparison of Vessel Diameter Measurement Methods")
plt.legend()
plt.tight_layout()
plt.savefig("explore/simple_method_comparison.png")
print("Saved method comparison visualization")