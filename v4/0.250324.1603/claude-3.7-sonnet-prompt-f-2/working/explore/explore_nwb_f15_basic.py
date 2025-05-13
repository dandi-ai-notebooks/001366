"""
Explore basic information and visualize a sample frame from the F15 NWB file.
This script will:
1. Load the F15 NWB file
2. Print basic information about the file
3. Load a sample frame from the Movies dataset
4. Visualize and save the sample frame as an image
"""

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set up the plot styling
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/71fa07fc-4309-4013-8edd-13213a86a67d/download/"
print(f"Loading NWB file from {url}")

remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information
print("\nBasic Information:")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Age: {nwb.subject.age}")
print(f"Sex: {nwb.subject.sex}")
print(f"Strain: {nwb.subject.strain}")
print(f"Species: {nwb.subject.species}")
print(f"Session ID: {nwb.session_id}")
print(f"Session Description: {nwb.session_description[:100]}...")
print(f"Experiment Description: {nwb.experiment_description}")
print(f"Keywords: {nwb.keywords[:]}")
print(f"Institution: {nwb.institution}")

# Get information about the Movies dataset
movies = nwb.acquisition["Movies"]
print("\nMovies Dataset Information:")
print(f"Shape: {movies.data.shape}")
print(f"Data Type: {movies.data.dtype}")
print(f"Frame Rate: {movies.rate} frames/second")
print(f"Description: {movies.description}")
print(f"Unit: {movies.unit}")

# Load a single frame (the first frame)
print("\nLoading the first frame...")
frame_idx = 0
first_frame = movies.data[frame_idx, :, :]
print(f"Frame shape: {first_frame.shape}")
print(f"Min value: {np.min(first_frame)}")
print(f"Max value: {np.max(first_frame)}")
print(f"Mean value: {np.mean(first_frame)}")
print(f"Std dev: {np.std(first_frame)}")

# Visualize the frame
plt.figure(figsize=(10, 8))
plt.imshow(first_frame, cmap='gray')
plt.colorbar(label='Pixel Value')
plt.title(f"First Frame from F15 Dataset (Subject ID: {nwb.subject.subject_id})")
plt.xlabel("X Position (pixels)")
plt.ylabel("Y Position (pixels)")
plt.tight_layout()
plt.savefig("explore/f15_first_frame.png")
print("First frame image saved to 'explore/f15_first_frame.png'")

# Load a frame from the middle of the dataset
middle_idx = movies.data.shape[0] // 2
print(f"\nLoading frame from middle of dataset (index {middle_idx})...")
middle_frame = movies.data[middle_idx, :, :]

plt.figure(figsize=(10, 8))
plt.imshow(middle_frame, cmap='gray')
plt.colorbar(label='Pixel Value')
plt.title(f"Middle Frame from F15 Dataset (Frame {middle_idx})")
plt.xlabel("X Position (pixels)")
plt.ylabel("Y Position (pixels)")
plt.tight_layout()
plt.savefig("explore/f15_middle_frame.png")
print(f"Middle frame image saved to 'explore/f15_middle_frame.png'")