# Script to plot the intensity of a single pixel over time

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load
url = "https://api.dandiarchive.org/api/assets/2f12bce3-f841-46ca-b928-044269122a59/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the movie data
movie_data = nwb.acquisition['Movies'].data

# Choose a pixel coordinate and number of frames
pixel_y, pixel_x = 256, 256
num_frames = 100

# Extract intensity values for the pixel over time
intensity_values = movie_data[:num_frames, pixel_y, pixel_x]

# Get the timestamps for these frames
timestamps = (np.arange(num_frames) / nwb.acquisition['Movies'].rate) + nwb.acquisition['Movies'].starting_time

# Plot the intensity over time
plt.figure()
plt.plot(timestamps, intensity_values)
plt.xlabel('Time (s)')
plt.ylabel('Pixel Intensity')
plt.title(f'Pixel Intensity over Time at ({pixel_y}, {pixel_x}) for first {num_frames} frames')
plt.grid(True)
plt.savefig('explore/pixel_intensity.png')
plt.close()

print("Saved explore/pixel_intensity.png")