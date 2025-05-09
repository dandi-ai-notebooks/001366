# This script investigates the structure and metadata of the NWB file
# and plots frame 0 of the "Movies" ImageSeries as an image.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# File URL and details (hard-coded based on asset info)
url = "https://api.dandiarchive.org/api/assets/71fa07fc-4309-4013-8edd-13213a86a67d/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print general metadata
print("Session description:", nwb.session_description)
print("Experimenter:", nwb.experimenter)
print("Experiment description:", nwb.experiment_description)
print("Institution:", nwb.institution)
print("Subject description:", nwb.subject.description)
print("Subject ID:", nwb.subject.subject_id)
print("Subject age:", nwb.subject.age)
print("Subject sex:", nwb.subject.sex)
print("Subject strain:", nwb.subject.strain)
print("Session start time:", nwb.session_start_time)

# "Movies" ImageSeries
movies = nwb.acquisition['Movies']
print("Movies description:", movies.description)
print("Movies comments:", movies.comments)
print("Movies rate (Hz):", movies.rate)
print("Movies data shape:", movies.data.shape)
print("Movies data dtype:", movies.data.dtype)

# Plot the first image frame (frame 0)
frame_0 = movies.data[0, :, :]
plt.figure(figsize=(6,4))
plt.imshow(frame_0, cmap='gray')
plt.title("Movies ImageSeries - Frame 0")
plt.xlabel("X pixels")
plt.ylabel("Y pixels")
plt.colorbar(label="Pixel intensity")
plt.tight_layout()
plt.savefig("explore/movies_frame0.png")
plt.close()