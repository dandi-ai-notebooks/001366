# Goal: Load the first frame of the 'Movies' ImageSeries and save it as a PNG.
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

# Access the movie data
movies_data = nwb.acquisition['Movies'].data

# Get the first frame
first_frame = movies_data[0, :, :]

# Plot and save the first frame
plt.figure(figsize=(8, 8))
plt.imshow(first_frame, cmap='gray')
plt.title('First Frame of Movie')
plt.xlabel('X pixels')
plt.ylabel('Y pixels')
plt.colorbar(label='Intensity')
plt.savefig('explore/first_frame.png')
# plt.show() # Do not show, as it will hang

print("Saved first_frame.png")

io.close()