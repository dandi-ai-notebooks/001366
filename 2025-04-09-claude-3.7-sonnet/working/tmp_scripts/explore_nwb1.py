"""
This script explores the first NWB file in Dandiset 001366, which contains a 
time-series of images showing a pial vessel in a mouse. The script:
1. Loads the NWB file
2. Examines the structure and metadata
3. Visualizes sample frames from the image series
4. Creates plots to analyze basic properties of the image data
"""

import matplotlib.pyplot as plt
import numpy as np
import pynwb
import h5py
import remfile
import os

# Ensure output directory exists
os.makedirs("tmp_scripts", exist_ok=True)

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/2f12bce3-f841-46ca-b928-044269122a59/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

# Print basic metadata
print(f"Dataset identifier: {nwb.identifier}")
print(f"Session description: {nwb.session_description[:100]}...")
print(f"Session start time: {nwb.session_start_time}")
print(f"Experiment description: {nwb.experiment_description}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject species: {nwb.subject.species}")
print(f"Subject sex: {nwb.subject.sex}")
print(f"Subject age: {nwb.subject.age}")
print(f"Subject strain: {nwb.subject.strain}")

# Get image data information
movies = nwb.acquisition["Movies"]
print(f"\nMovie data shape: {movies.data.shape}")
print(f"Movie frame rate: {movies.rate} frames/second")
print(f"Movie description: {movies.description}")

# Sample a few frames from different parts of the movie
n_frames = movies.data.shape[0]
sample_indices = [0, n_frames//4, n_frames//2, (3*n_frames)//4, n_frames-1]

# Load the sample frames
sample_frames = []
for idx in sample_indices:
    sample_frames.append(movies.data[idx, :, :])
    
# Plot the sample frames
plt.figure(figsize=(15, 10))
for i, (idx, frame) in enumerate(zip(sample_indices, sample_frames)):
    plt.subplot(2, 3, i+1)
    plt.imshow(frame, cmap='gray')
    plt.title(f"Frame {idx}")
    plt.colorbar()
    plt.axis('off')
plt.tight_layout()
plt.savefig("tmp_scripts/sample_frames_nwb1.png")
plt.close()

# Analyze pixel intensity distribution in a sample frame
middle_frame = sample_frames[2]  # Middle frame
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(middle_frame, cmap='gray')
plt.title(f"Middle Frame (Frame {sample_indices[2]})")
plt.colorbar()
plt.axis('off')

plt.subplot(1, 2, 2)
plt.hist(middle_frame.flatten(), bins=50)
plt.title("Pixel Intensity Distribution")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.savefig("tmp_scripts/intensity_distribution_nwb1.png")
plt.close()

# Create a temporal profile by taking a line across the vessel
# For this, we'll extract a line from the middle of each sample frame
line_profiles = []
for frame in sample_frames:
    # Extract a horizontal line from the middle of the frame
    middle_row = frame.shape[0] // 2
    line_profile = frame[middle_row, :]
    line_profiles.append(line_profile)

# Plot the line profiles
plt.figure(figsize=(10, 6))
for i, (idx, profile) in enumerate(zip(sample_indices, line_profiles)):
    plt.plot(profile, label=f"Frame {idx}")
plt.title("Horizontal Line Profiles Across Frames")
plt.xlabel("Pixel Position")
plt.ylabel("Pixel Intensity")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("tmp_scripts/line_profiles_nwb1.png")
plt.close()

# Create a temporal profile for a specific pixel over time
# Sample 100 evenly spaced frames from the movie
time_indices = np.linspace(0, n_frames-1, 100, dtype=int)
center_x, center_y = movies.data.shape[1] // 2, movies.data.shape[2] // 2

# Get values for a pixel near the center
time_series_values = []
for idx in time_indices:
    time_series_values.append(movies.data[idx, center_x, center_y])

# Plot the time series
time_points = time_indices / movies.rate  # Convert to seconds
plt.figure(figsize=(12, 6))
plt.plot(time_points, time_series_values)
plt.title(f"Intensity of Central Pixel ({center_x}, {center_y}) Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Pixel Intensity")
plt.grid(alpha=0.3)
plt.savefig("tmp_scripts/central_pixel_time_series_nwb1.png")
plt.close()

print("Script execution completed successfully")