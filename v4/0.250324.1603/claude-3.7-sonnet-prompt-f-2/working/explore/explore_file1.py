# Script to explore basic information about the second NWB file (smaller file)
# and save a sample frame to see what the data looks like

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Set up figure parameters
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] = 100

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/71fa07fc-4309-4013-8edd-13213a86a67d/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information
print(f"NWB file identifier: {nwb.identifier}")
print(f"Session description: {nwb.session_description[:100]}...")
print(f"Subject: {nwb.subject.subject_id}, Sex: {nwb.subject.sex}, Age: {nwb.subject.age}")
print(f"Subject description: {nwb.subject.description}")
print(f"Institution: {nwb.institution}")

# Get movie data
movies = nwb.acquisition["Movies"]
print("\nMovie information:")
print(f"Frame rate: {movies.rate} fps")
print(f"Movie dimensions: {movies.data.shape}")
print(f"Data type: {movies.data.dtype}")
print(f"Units: {movies.unit}")
print(f"Description: {movies.description}")

# Get a sample frame (first frame)
print("\nExtracting first frame...")
first_frame = movies.data[0, :, :]
print(f"Frame shape: {first_frame.shape}")
print(f"Min pixel value: {np.min(first_frame)}")
print(f"Max pixel value: {np.max(first_frame)}")
print(f"Mean pixel value: {np.mean(first_frame)}")

# Plot and save the first frame
plt.figure()
plt.imshow(first_frame, cmap='gray')
plt.colorbar(label='Pixel Value')
plt.title('First Frame of Vessel Movie')
plt.savefig('explore/frame1_file1.png')
print("Saved first frame to explore/frame1_file1.png")

# Get a sample frame from the middle of the movie
middle_idx = movies.data.shape[0] // 2
print(f"\nExtracting middle frame (index {middle_idx})...")
middle_frame = movies.data[middle_idx, :, :]
print(f"Frame shape: {middle_frame.shape}")
print(f"Min pixel value: {np.min(middle_frame)}")
print(f"Max pixel value: {np.max(middle_frame)}")
print(f"Mean pixel value: {np.mean(middle_frame)}")

# Plot and save the middle frame
plt.figure()
plt.imshow(middle_frame, cmap='gray')
plt.colorbar(label='Pixel Value')
plt.title(f'Middle Frame (#{middle_idx}) of Vessel Movie')
plt.savefig('explore/middle_frame_file1.png')
print(f"Saved middle frame to explore/middle_frame_file1.png")

# Get keywords from the NWB file
keywords = nwb.keywords[:]
print("\nKeywords:")
for kw in keywords:
    print(f"- {kw}")

print("\nExploration complete!")