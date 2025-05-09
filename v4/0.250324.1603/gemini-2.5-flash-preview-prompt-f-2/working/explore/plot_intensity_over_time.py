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

# Define the pixel coordinate to track (row, column)
pixel_coord = (300, 300)

# Extract pixel intensity for a subset of frames
# Load only the data for the specified pixel for a limited number of time points
num_frames_to_load = 1000
pixel_intensity_over_time = movie_data[:num_frames_to_load, pixel_coord[0], pixel_coord[1]]

# Plot pixel intensity over time
plt.figure(figsize=(12, 6))
plt.plot(pixel_intensity_over_time)
plt.xlabel('Frame Index')
plt.ylabel('Pixel Intensity (uint16)')
plt.title(f'Pixel Intensity at ({pixel_coord[0]}, {pixel_coord[1]}) Over Time (First {num_frames_to_load} Frames)')
plt.grid(True)
plt.savefig('explore/pixel_intensity_over_time.png')
plt.close()

io.close() # Close the NWB file after reading