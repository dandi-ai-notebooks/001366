# This script loads and plots a few frames from the NWB file.

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

# Select and plot a few frames
frame_indices = [0, 1000, 2000, 3000, 4000, 5000]

if not os.path.exists('explore'):
    os.makedirs('explore')

for i, frame_index in enumerate(frame_indices):
    if frame_index < movie_data.shape[0]:
        frame = movie_data[frame_index, :, :]
        plt.figure()
        plt.imshow(frame, cmap='gray')
        plt.title(f'Frame {frame_index}')
        plt.axis('off')
        plt.savefig(f'explore/frame_{frame_index}.png')
        plt.close()
    else:
        print(f"Frame index {frame_index} is out of bounds.")

io.close() # Close the NWB file after reading