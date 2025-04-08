"""
This script loads the NWB file containing vessel imaging data and creates visualizations:
1. A sample frame from the movie
2. Time series analysis of pixel intensities

The goal is to understand the structure and quality of the vessel imaging data.
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001366/assets/71fa07fc-4309-4013-8edd-13213a86a67d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the Movies data
movies = nwb.acquisition["Movies"]

# Get dimensions and sampling rate
frame_rate = movies.rate
n_frames = movies.data.shape[0]  # Total number of frames
timepoints = np.arange(n_frames) / frame_rate  # Generate timestamps based on frame rate
print(f"Frame rate: {frame_rate} Hz")
print(f"Number of frames: {n_frames}")
print(f"Recording duration: {timepoints[-1]:.2f} seconds")

# Load a sample frame (frame 0)
sample_frame = movies.data[0, :, :]
print(f"Frame dimensions: {sample_frame.shape}")

# Create figure with subplots
plt.figure(figsize=(15, 7))
gs = GridSpec(1, 2, width_ratios=[1, 1.5])

# Plot sample frame
ax1 = plt.subplot(gs[0])
im = ax1.imshow(sample_frame, cmap='gray')
ax1.set_title('Sample Frame (t=0)')
plt.colorbar(im, ax=ax1, label='Pixel Intensity')
ax1.set_xlabel('X Position (pixels)')
ax1.set_ylabel('Y Position (pixels)')

# Select central region for time series analysis
# Using a smaller subset of frames for the time series to avoid memory issues
n_frames_analyze = 100  # Analyze first 100 frames
center_y, center_x = np.array(sample_frame.shape) // 2
roi_size = 10
roi = movies.data[:n_frames_analyze, center_y-roi_size:center_y+roi_size, center_x-roi_size:center_x+roi_size]
mean_intensity = np.mean(roi, axis=(1,2))

# Plot time series
ax2 = plt.subplot(gs[1])
ax2.plot(timepoints[:n_frames_analyze], mean_intensity, 'b-', label='Mean Intensity')
ax2.set_title('ROI Intensity Over Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Mean Pixel Intensity')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig('tmp_scripts/vessel_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Also save just the sample frame separately
plt.figure(figsize=(8, 8))
plt.imshow(sample_frame, cmap='gray')
plt.colorbar(label='Pixel Intensity')
plt.title('Sample Frame of Vessel Imaging')
plt.xlabel('X Position (pixels)')
plt.ylabel('Y Position (pixels)')
plt.savefig('tmp_scripts/sample_frame.png', dpi=300, bbox_inches='tight')
plt.close()

# Print some statistics about the sample frame
print(f"\nSample frame statistics:")
print(f"Min intensity: {np.min(sample_frame)}")
print(f"Max intensity: {np.max(sample_frame)}")
print(f"Mean intensity: {np.mean(sample_frame):.2f}")
print(f"Standard deviation: {np.std(sample_frame):.2f}")