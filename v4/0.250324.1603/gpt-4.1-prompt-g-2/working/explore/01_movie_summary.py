# Script to summarize 'Movies' ImageSeries data from the given NWB file using remfile streaming.
# Outputs: prints stats and saves plots for only the first 100 frames for speed.
import matplotlib.pyplot as plt
import numpy as np
import pynwb
import h5py
import remfile

# Parameters
nwb_url = "https://api.dandiarchive.org/api/assets/71fa07fc-4309-4013-8edd-13213a86a67d/download/"
output_mean_png = "explore/movies_mean_image.png"
output_frame_png = "explore/movies_sample_frame.png"
N_FRAMES = 100  # number of frames for summary

# Load NWB file
remote_file = remfile.File(nwb_url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access Movies ImageSeries
movies = nwb.acquisition["Movies"]
data = movies.data
num_frames = data.shape[0]
height = data.shape[1]
width = data.shape[2]
n_subset = min(N_FRAMES, num_frames)
print(f"Movies data shape: {data.shape}, dtype: {data.dtype}")
print(f"Total frames: {num_frames}, Frame size: {height}x{width}")
print(f"Using only the first {n_subset} frames for summary and plots.")

# Compute and plot mean image (across time) for first N_FRAMES
mean_image = np.array(data[0:n_subset]).mean(axis=0)
fig, ax = plt.subplots()
im = ax.imshow(mean_image, cmap="gray")
ax.set_title(f"Movies Mean Image (First {n_subset} Frames)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(output_mean_png)
plt.close(fig)

# Plot sample frame (first frame)
sample_frame = np.array(data[0])
fig, ax = plt.subplots()
im = ax.imshow(sample_frame, cmap="gray")
ax.set_title("Sample Frame 0 from Movies")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(output_frame_png)
plt.close(fig)

# Print frame stats
print("Mean pixel value (of mean image):", mean_image.mean())
print("Std pixel value (of mean image):", mean_image.std())
print("Sample frame 0 min/max:", sample_frame.min(), sample_frame.max())