"""
Explore basic information and visualize sample frames from the larger M4 NWB file.
This script will:
1. Load the NWB file
2. Print basic information about the file
3. Load sample frames from different parts of the dataset
4. Visualize the sample frames
"""

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set up the plot styling
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/2f12bce3-f841-46ca-b928-044269122a59/download/"
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
plt.title(f"First Frame from M4 Dataset (Subject ID: {nwb.subject.subject_id})")
plt.xlabel("X Position (pixels)")
plt.ylabel("Y Position (pixels)")
plt.tight_layout()
plt.savefig("explore/m4_first_frame.png")
print("First frame image saved to 'explore/m4_first_frame.png'")

# Load a frame from the middle of the dataset
middle_idx = movies.data.shape[0] // 2
print(f"\nLoading frame from middle of dataset (index {middle_idx})...")
middle_frame = movies.data[middle_idx, :, :]

plt.figure(figsize=(10, 8))
plt.imshow(middle_frame, cmap='gray')
plt.colorbar(label='Pixel Value')
plt.title(f"Middle Frame from M4 Dataset (Frame {middle_idx})")
plt.xlabel("X Position (pixels)")
plt.ylabel("Y Position (pixels)")
plt.tight_layout()
plt.savefig("explore/m4_middle_frame.png")
print(f"Middle frame image saved to 'explore/m4_middle_frame.png'")

# Load a sequence of frames to check for differences over time (every 500 frames)
print("\nLoading a sequence of frames to visualize changes over time...")
sample_frames = []
sample_indices = [0, 500, 1000, 1500, 2000]

for idx in sample_indices:
    sample_frames.append(movies.data[idx, :, :])

# Visualize the sequence of frames
fig, axes = plt.subplots(1, len(sample_indices), figsize=(20, 6))
for i, (idx, frame) in enumerate(zip(sample_indices, sample_frames)):
    im = axes[i].imshow(frame, cmap='gray')
    axes[i].set_title(f"Frame {idx}")
    axes[i].set_xlabel("X Position (pixels)")
    if i == 0:
        axes[i].set_ylabel("Y Position (pixels)")
    else:
        axes[i].set_yticks([])
    
fig.suptitle(f"Sequence of Frames from M4 Dataset", fontsize=16)
fig.tight_layout()
plt.savefig("explore/m4_frame_sequence.png")
print("Frame sequence saved to 'explore/m4_frame_sequence.png'")

# Create a zoomed-in version of the first frame to better see details
print("\nCreating a zoomed view of the first frame...")

# Try to identify a region with a vessel by looking at pixel intensity differences
frame_diff = np.abs(first_frame - np.mean(first_frame))
max_diff_y, max_diff_x = np.unravel_index(np.argmax(frame_diff), frame_diff.shape)

# Define the zoom region, ensuring it's within the image boundaries
zoom_size = 100  # Size of the zoomed region
zoom_y_min = max(0, max_diff_y - zoom_size//2)
zoom_y_max = min(first_frame.shape[0], max_diff_y + zoom_size//2)
zoom_x_min = max(0, max_diff_x - zoom_size//2)
zoom_x_max = min(first_frame.shape[1], max_diff_x + zoom_size//2)

# Extract and visualize the zoomed region
zoomed_frame = first_frame[zoom_y_min:zoom_y_max, zoom_x_min:zoom_x_max]

plt.figure(figsize=(10, 8))
plt.imshow(zoomed_frame, cmap='gray')
plt.colorbar(label='Pixel Value')
plt.title(f"Zoomed Region from First Frame (centered at y={max_diff_y}, x={max_diff_x})")
plt.xlabel("X Position (pixels)")
plt.ylabel("Y Position (pixels)")
plt.tight_layout()
plt.savefig("explore/m4_first_frame_zoomed.png")
print("Zoomed first frame saved to 'explore/m4_first_frame_zoomed.png'")