# This script computes a mean projection over the *first 100 frames* of the Movies ImageSeries and plots it,
# and also creates a histogram of pixel intensities for frame 0.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

url = "https://api.dandiarchive.org/api/assets/71fa07fc-4309-4013-8edd-13213a86a67d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()
movies = nwb.acquisition['Movies']

# Compute a mean projection over the first 100 frames only
mean_img = np.mean(movies.data[0:100, :, :], axis=0)
plt.figure(figsize=(6,4))
plt.imshow(mean_img, cmap='gray')
plt.title("Movies ImageSeries - Mean Projection (first 100 frames)")
plt.xlabel("X pixels")
plt.ylabel("Y pixels")
plt.colorbar(label="Mean pixel intensity")
plt.tight_layout()
plt.savefig("explore/movies_mean_projection.png")
plt.close()

# Histogram of pixel intensities for frame 0
frame_0 = movies.data[0, :, :]
plt.figure(figsize=(5,3))
plt.hist(frame_0.flatten(), bins=64, color='slategray')
plt.title("Pixel Intensity Histogram (Frame 0)")
plt.xlabel("Pixel intensity")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("explore/frame0_histogram.png")
plt.close()