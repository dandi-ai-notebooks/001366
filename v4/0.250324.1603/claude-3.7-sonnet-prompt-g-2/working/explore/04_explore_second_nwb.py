"""
This script explores the image data in the second (larger) NWB file.
It loads sample frames and creates visualizations to understand the differences 
between the two NWB files in the dataset.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# URL for the second NWB file (larger one)
url = "https://api.dandiarchive.org/api/assets/2f12bce3-f841-46ca-b928-044269122a59/download/"

# Load the file
print("Loading the second NWB file...")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the Movies dataset
movies = nwb.acquisition['Movies']
print(f"Movie data shape: {movies.data.shape}")
print(f"Movie frame rate: {movies.rate} Hz")
print(f"Data type: {movies.data.dtype}")

# Load a few frames to explore (a small sample due to the large image size)
print("Loading sample frames...")
frame_indices = [0]
frames = [movies.data[i] for i in frame_indices]

# Get the minimum and maximum values
min_val = min(frame.min() for frame in frames)
max_val = max(frame.max() for frame in frames)
print(f"Min value: {min_val}")
print(f"Max value: {max_val}")

# Create a figure to display the first frame
plt.figure(figsize=(10, 10))
plt.imshow(frames[0], cmap='gray', norm=Normalize(vmin=min_val, vmax=max_val))
plt.title(f"Second NWB File - Frame 0")
plt.colorbar(label='Pixel Value')
plt.axis('off')
plt.tight_layout()
plt.savefig('explore/second_nwb_frame.png', dpi=150)
plt.close()

# Find potential vessel locations by looking for dark regions
# For large images, we'll use downsampling for efficiency
def find_potential_vessels(frame, threshold_factor=0.3):
    """Find dark regions that might be vessels."""
    # Calculate threshold as a factor of the range
    value_range = np.max(frame) - np.min(frame)
    threshold = np.min(frame) + threshold_factor * value_range
    
    # Find dark regions
    dark_regions = frame < threshold
    
    return dark_regions

# Apply vessel detection
print("Detecting potential vessels...")
vessel_mask = find_potential_vessels(frames[0])

# Visualize the detected vessels
plt.figure(figsize=(10, 10))
plt.imshow(frames[0], cmap='gray', norm=Normalize(vmin=min_val, vmax=max_val))
plt.imshow(vessel_mask, cmap='hot', alpha=0.3)
plt.title(f"Potential Vessel Locations")
plt.axis('off')
plt.tight_layout()
plt.savefig('explore/second_nwb_vessels.png', dpi=150)
plt.close()

# Extract a region of interest containing a vessel
# First, let's try to detect a promising region of interest
def find_roi_with_vessel(vessel_mask, size=100):
    """Find a region of interest containing a vessel."""
    # Find the center of mass of all vessel pixels
    vessel_pixels = np.column_stack(np.where(vessel_mask))
    if len(vessel_pixels) == 0:
        # Fallback: use center of image
        center = np.array(vessel_mask.shape) // 2
    else:
        center = np.mean(vessel_pixels, axis=0).astype(int)
    
    # Define ROI boundaries
    half_size = size // 2
    y_start = max(0, center[0] - half_size)
    y_end = min(vessel_mask.shape[0], center[0] + half_size)
    x_start = max(0, center[1] - half_size)
    x_end = min(vessel_mask.shape[1], center[1] + half_size)
    
    return (y_start, y_end, x_start, x_end)

# Find and extract a ROI
roi = find_roi_with_vessel(vessel_mask)
roi_image = frames[0][roi[0]:roi[1], roi[2]:roi[3]]

# Visualize the ROI
plt.figure(figsize=(8, 8))
plt.imshow(roi_image, cmap='gray', norm=Normalize(vmin=min_val, vmax=max_val))
plt.title(f"Region of Interest with Vessel")
plt.colorbar(label='Pixel Value')
plt.axis('off')
plt.tight_layout()
plt.savefig('explore/second_nwb_roi.png', dpi=150)
plt.close()

# Extract an intensity profile across the ROI
roi_middle_row = roi_image.shape[0] // 2
profile = roi_image[roi_middle_row, :]

# Plot the intensity profile
plt.figure(figsize=(10, 6))
plt.plot(profile)
plt.title('Intensity Profile Across Vessel (ROI Middle Row)')
plt.xlabel('Column Index')
plt.ylabel('Pixel Value')
plt.grid(True)
plt.tight_layout()
plt.savefig('explore/second_nwb_profile.png', dpi=150)
plt.close()

# Close the file
h5_file.close()
print("Analysis complete.")