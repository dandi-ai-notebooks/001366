# Purpose: Visualize frames from the Movies dataset and save them as PNG images.

import pynwb
import lindi
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load https://api.dandiarchive.org/api/assets/2f12bce3-f841-46ca-b928-044269122a59/download/
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001366/assets/2f12bce3-f841-46ca-b928-044269122a59/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

movies = nwb.acquisition["Movies"]
data = movies.data

# Plot three frames from the movie
frames_to_plot = [0, 2000, 4000]

for i, frame_index in enumerate(frames_to_plot):
    frame = data[frame_index, :, :]
    plt.figure(figsize=(6, 6))
    plt.imshow(frame, cmap='gray')
    plt.title(f"Frame {frame_index}")
    plt.axis('off')
    plt.savefig(f"tmp_scripts/frame_{frame_index}.png")
    plt.close()