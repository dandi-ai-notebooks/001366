"""
This script explores the image data in the first NWB file.
It loads a few frames from the Movies dataset and creates visualizations
to understand what the vessel images look like.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# URL for the first NWB file (smaller one)
url = "https://api.dandiarchive.org/api/assets/71fa07fc-4309-4013-8edd-13213a86a67d/download/"

# Load the file
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the Movies dataset
movies = nwb.acquisition['Movies']
print(f"Movie data shape: {movies.data.shape}")
print(f"Movie frame rate: {movies.rate} Hz")
print(f"Data type: {movies.data.dtype}")

# Load a few frames to explore
# Let's sample frames at different time points
frame_indices = [0, 1000, 2000, 3000]
frames = [movies.data[i] for i in frame_indices]

# Get the minimum and maximum values across all frames for consistent visualization
min_val = min(frame.min() for frame in frames)
max_val = max(frame.max() for frame in frames)
print(f"Min value: {min_val}")
print(f"Max value: {max_val}")

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

# Plot each frame
for i, (frame, ax) in enumerate(zip(frames, axes)):
    # Normalize and display the image
    im = ax.imshow(frame, cmap='gray', norm=Normalize(vmin=min_val, vmax=max_val))
    ax.set_title(f"Frame {frame_indices[i]} (t={frame_indices[i]/movies.rate:.2f}s)")
    ax.axis('off')

plt.colorbar(im, ax=axes, label='Pixel Value', shrink=0.8)
plt.tight_layout()
plt.savefig('explore/vessel_frames.png', dpi=150)
plt.close()

# Now let's look at how the image changes over time at a specific point
# First, let's find a point that's likely part of the vessel
# We'll look at the first frame and find a point with high intensity
first_frame = frames[0]
middle_row = first_frame.shape[0] // 2
middle_col = first_frame.shape[1] // 2

# Sample 100 frames over the duration of the recording to see how a pixel changes over time
num_samples = 100
sampled_indices = np.linspace(0, movies.data.shape[0]-1, num_samples, dtype=int)
pixel_values = []

for i in sampled_indices:
    # Get the value at the middle point
    pixel_values.append(movies.data[i, middle_row, middle_col])

# Plot the pixel values over time
plt.figure(figsize=(12, 6))
plt.plot(sampled_indices / movies.rate, pixel_values, marker='o', linestyle='-')
plt.title('Pixel Value Over Time at Center Point')
plt.xlabel('Time (seconds)')
plt.ylabel('Pixel Value')
plt.grid(True)
plt.savefig('explore/pixel_over_time.png', dpi=150)
plt.close()

# Let's also create a time-lapse animation by showing multiple frames in a single image
# We'll take 9 frames evenly spaced from the start
timeline_indices = np.linspace(0, min(999, movies.data.shape[0]-1), 9, dtype=int)
timeline_frames = [movies.data[i] for i in timeline_indices]

# Create a figure with subplots
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
axes = axes.flatten()

# Plot each frame
for i, (frame, ax) in enumerate(zip(timeline_frames, axes)):
    # Normalize and display the image
    im = ax.imshow(frame, cmap='gray', norm=Normalize(vmin=min_val, vmax=max_val))
    ax.set_title(f"t={timeline_indices[i]/movies.rate:.2f}s")
    ax.axis('off')

plt.colorbar(im, ax=axes, label='Pixel Value', shrink=0.8)
plt.tight_layout()
plt.savefig('explore/vessel_timeline.png', dpi=150)
plt.close()

# Let's also create a profile across the vessel
# We'll use the middle row of the first frame
profile = first_frame[middle_row, :]
plt.figure(figsize=(10, 6))
plt.plot(profile)
plt.title('Intensity Profile Across Vessel (Middle Row)')
plt.xlabel('Column Index')
plt.ylabel('Pixel Value')
plt.grid(True)
plt.savefig('explore/vessel_profile.png', dpi=150)
plt.close()

# Close the file
h5_file.close()