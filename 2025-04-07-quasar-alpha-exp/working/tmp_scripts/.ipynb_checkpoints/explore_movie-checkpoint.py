# This script loads the NWB imaging data from a Lindi stream and explores basic properties.
# It saves plots of an example frame and the mean projection across a subset of frames.
# Since the data is streamed, we limit to the first 100 frames for speed.
# The plots will be saved as PNG files for later review.

import numpy as np
import matplotlib.pyplot as plt
import lindi
import pynwb

# Lindi URL from dandiset info
lindi_url = "https://lindi.neurosift.org/dandi/dandisets/001366/assets/71fa07fc-4309-4013-8edd-13213a86a67d/nwb.lindi.json"

f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

movies = nwb.acquisition['Movies']
data = movies.data  # HDF5 dataset, not yet loaded

print('Shape of Movies data:', data.shape)
print('Dtype of Movies data:', data.dtype)
print('Num frames:', data.shape[0])
print('Height, width:', data.shape[1], data.shape[2])

num_frames_to_load = min(100, data.shape[0])

subset = data[:num_frames_to_load]  # load first frames as numpy array
print('Loaded subset shape:', subset.shape)

mean_img = np.mean(subset, axis=0)
single_frame = subset[0]

print('Mean image stats:')
print('  min:', mean_img.min(), 'max:', mean_img.max(), 'mean:', mean_img.mean())
print('Single frame stats:')
print('  min:', single_frame.min(), 'max:', single_frame.max(), 'mean:', single_frame.mean())

plt.figure()
plt.imshow(mean_img, cmap='gray')
plt.title(f'Mean projection (first {num_frames_to_load} frames)')
plt.axis('off')
plt.savefig('tmp_scripts/movie_mean_projection.png', bbox_inches='tight')

plt.figure()
plt.imshow(single_frame, cmap='gray')
plt.title('Example frame (frame 0)')
plt.axis('off')
plt.savefig('tmp_scripts/movie_example_frame.png', bbox_inches='tight')