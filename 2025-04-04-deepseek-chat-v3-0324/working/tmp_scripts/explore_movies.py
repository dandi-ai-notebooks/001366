# Script to explore the Movies acquisition in the NWB file
# Goal: Understand the structure and visualize sample frames

import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file(
    "https://lindi.neurosift.org/dandi/dandisets/001366/assets/71fa07fc-4309-4013-8edd-13213a86a67d/nwb.lindi.json"
)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get the Movies acquisition
movies = nwb.acquisition["Movies"]
print(f"Movie shape: {movies.data.shape}")
print(f"Frame rate: {movies.rate} Hz")
print(f"Duration: {movies.data.shape[0]/movies.rate:.2f} seconds")

# Plot sample frames
sample_indices = np.linspace(0, movies.data.shape[0]-1, 5, dtype=int)
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, idx in enumerate(sample_indices):
    frame = movies.data[idx]
    axes[i].imshow(frame, cmap='gray')
    axes[i].set_title(f"Frame {idx}")
    axes[i].axis('off')
plt.tight_layout()
plt.savefig("tmp_scripts/movie_frames.png")