# This script loads a region of interest from the NWB movie data and plots the average intensity over time.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import os

# Load
url = "https://api.dandiarchive.org/api/assets/2f12bce3-f841-46ca-b928-044269122a59/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the 'Movies' ImageSeries
acquisition = nwb.acquisition
Movies = acquisition["Movies"]
movie_data = Movies.data # This is a h5py.Dataset

# Define the region of interest (ROI) [y_start:y_end, x_start:x_end]
roi = (slice(250, 350), slice(250, 350))

# Extract data for the ROI across a subset of frames
# Load only the data for the specified ROI for a limited number of time points
num_frames_to_load = 1000
roi_data_over_time = movie_data[:num_frames_to_load, roi[0], roi[1]]

# Calculate the average intensity for each frame within the ROI
average_intensity = np.mean(roi_data_over_time, axis=(1, 2))

# Get timestamps for the subset of frames (optional, using frame index as time for simplicity here)
# timestamps = Movies.starting_time + np.arange(num_frames_to_load) * (1.0 / Movies.rate)

# Plot average intensity over time
plt.figure(figsize=(12, 6))
plt.plot(average_intensity)
plt.xlabel('Frame Index')
plt.ylabel('Average Pixel Intensity (uint16)')
plt.title(f'Average Pixel Intensity in ROI Over Time (First {num_frames_to_load} Frames)')
plt.grid(True)
plt.savefig('explore/average_intensity_over_time.png')
plt.close()

io.close() # Close the NWB file after reading