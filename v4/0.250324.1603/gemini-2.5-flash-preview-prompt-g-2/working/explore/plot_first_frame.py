# Script to load and plot the first frame of the movie data

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

# Load the first frame
frame = movie_data[0, :, :]

# Plot the frame
plt.figure()
plt.imshow(frame, cmap='gray')
plt.title('First frame of the movie')
plt.axis('off')
plt.savefig('explore/first_frame.png')
plt.close()

print("Saved explore/first_frame.png")