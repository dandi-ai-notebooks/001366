# This script loads a remote NWB file containing pial surface vessel movies,
# extracts a small subset of frames, and generates exploratory plots:
# - a single sample frame,
# - mean intensity projection of first 100 frames,
# - intensity time series from selected vessel pixels (to assess pulsatility).
# Plots are saved as PNG files (no interactive display).

import matplotlib.pyplot as plt
import numpy as np
import h5py
import remfile
import pynwb

# Load remote NWB using provided approach
url = "https://api.dandiarchive.org/api/assets/71fa07fc-4309-4013-8edd-13213a86a67d/download/"
file_rf = remfile.File(url)
f = h5py.File(file_rf, 'r')
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

# Access movie data
img_series = nwb.acquisition['Movies']
data = img_series.data  # this is an h5py.Dataset, shape (9553, 214, 132)

# Safe subset of ~100 frames for quick exploration
n_frames = data.shape[0]
subset_stop = min(100, n_frames)
frames = data[0:subset_stop, :, :]  # shape (subset_stop, height, width)

# Single middle frame
middle_frame_idx = subset_stop // 2
middle_frame = frames[middle_frame_idx, :, :]

plt.figure(figsize=(6, 6))
plt.imshow(middle_frame, cmap='gray')
plt.title('Middle Frame of Initial 100 Frames')
plt.axis('off')
plt.savefig('tmp_scripts/middle_frame.png', bbox_inches='tight')
plt.close()

# Temporal mean projection over first 100 frames
mean_proj = np.mean(frames, axis=0)

plt.figure(figsize=(6, 6))
plt.imshow(mean_proj, cmap='gray')
plt.title('Mean Intensity Projection (First 100 Frames)')
plt.axis('off')
plt.savefig('tmp_scripts/mean_projection.png', bbox_inches='tight')
plt.close()

# Plot pixel intensity over time for a few pixels
pixel_coords = [(107, 66), (60, 30), (150, 100)]  # (x,y) example pixels, customize if needed
plt.figure()
for x, y in pixel_coords:
    if y < frames.shape[1] and x < frames.shape[2]:
        plt.plot(np.arange(subset_stop) / img_series.rate, frames[:, y, x], label=f'Pixel ({x},{y})')

plt.xlabel('Time (s)')
plt.ylabel('Intensity')
plt.legend()
plt.title('Intensity Time Series at Selected Pixels (First 100 frames)')
plt.savefig('tmp_scripts/pixel_traces.png', bbox_inches='tight')
plt.close()

# Cleanup
io.close()
file_rf.close()