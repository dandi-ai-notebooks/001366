"""
Script to visualize frames from the second NWB file in Dandiset 001366.
This script will generate visualizations of selected frames and save them as PNG files.
"""
import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Load the NWB file
print("Loading second NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001366/assets/71fa07fc-4309-4013-8edd-13213a86a67d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the movies data
movies = nwb.acquisition["Movies"]
print(f"Image dimensions: {movies.data.shape}")
print(f"Sampling rate: {movies.rate} Hz")

# Function to enhance image contrast for better visualization
def enhance_contrast(image, percentile_low=1, percentile_high=99):
    low = np.percentile(image, percentile_low)
    high = np.percentile(image, percentile_high)
    image_scaled = np.clip((image - low) / (high - low), 0, 1)
    return image_scaled

# Visualize first frame
print("Visualizing first frame...")
first_frame = movies.data[0]
plt.figure(figsize=(8, 6))
plt.imshow(first_frame, cmap='gray')
plt.colorbar(label='Intensity')
plt.title(f'First Frame (Original Values) - Second NWB File')
plt.savefig('tmp_scripts/second_nwb_first_frame_original.png', dpi=150)
plt.close()

# Visualize first frame with enhanced contrast
print("Visualizing first frame with enhanced contrast...")
first_frame_enhanced = enhance_contrast(first_frame)
plt.figure(figsize=(8, 6))
plt.imshow(first_frame_enhanced, cmap='gray')
plt.colorbar(label='Normalized Intensity')
plt.title(f'First Frame (Enhanced Contrast) - Second NWB File')
plt.savefig('tmp_scripts/second_nwb_first_frame_enhanced.png', dpi=150)
plt.close()

# Visualize a frame from the middle of the recording
print("Visualizing middle frame...")
middle_idx = movies.data.shape[0] // 2
middle_frame = movies.data[middle_idx]
middle_frame_enhanced = enhance_contrast(middle_frame)
plt.figure(figsize=(8, 6))
plt.imshow(middle_frame_enhanced, cmap='gray')
plt.colorbar(label='Normalized Intensity')
plt.title(f'Middle Frame (Enhanced Contrast, Frame #{middle_idx}) - Second NWB File')
plt.savefig('tmp_scripts/second_nwb_middle_frame_enhanced.png', dpi=150)
plt.close()

# Create a difference image between two consecutive frames to highlight motion
print("Creating difference image...")
frame1 = movies.data[middle_idx]
frame2 = movies.data[middle_idx + 1]
diff_image = np.abs(frame2.astype(np.float32) - frame1.astype(np.float32))
diff_image_enhanced = enhance_contrast(diff_image)
plt.figure(figsize=(8, 6))
plt.imshow(diff_image_enhanced, cmap='inferno')
plt.colorbar(label='Normalized Difference')
plt.title(f'Difference Between Consecutive Frames (Frames #{middle_idx} and #{middle_idx+1})')
plt.savefig('tmp_scripts/second_nwb_difference_image.png', dpi=150)
plt.close()

# Create a standard deviation projection of a small segment of frames
# This can help visualize areas with high temporal variance (like pulsating vessels)
print("Creating standard deviation projection...")
n_frames = 30  # Use 30 frames
start_idx = middle_idx - n_frames // 2
frames_subset = np.array([movies.data[i] for i in range(start_idx, start_idx + n_frames)])
std_projection = np.std(frames_subset, axis=0)
std_projection_enhanced = enhance_contrast(std_projection)
plt.figure(figsize=(8, 6))
plt.imshow(std_projection_enhanced, cmap='viridis')
plt.colorbar(label='Normalized StdDev')
plt.title(f'Standard Deviation Across {n_frames} Frames - Second NWB File')
plt.savefig('tmp_scripts/second_nwb_std_projection.png', dpi=150)
plt.close()

# Create a maximum intensity projection
print("Creating maximum intensity projection...")
max_projection = np.max(frames_subset, axis=0)
max_projection_enhanced = enhance_contrast(max_projection)
plt.figure(figsize=(8, 6))
plt.imshow(max_projection_enhanced, cmap='gray')
plt.colorbar(label='Normalized Max Intensity')
plt.title(f'Maximum Intensity Projection Across {n_frames} Frames - Second NWB File')
plt.savefig('tmp_scripts/second_nwb_max_projection.png', dpi=150)
plt.close()

print("All visualizations completed and saved to tmp_scripts directory.")