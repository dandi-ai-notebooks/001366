# Goal: Plot the average intensity over time for a defined ROI in the movie.
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()

# Load
url = "https://api.dandiarchive.org/api/assets/2f12bce3-f841-46ca-b928-044269122a59/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the movie data and acquisition rate
movies_data = nwb.acquisition['Movies'].data
frame_rate = nwb.acquisition['Movies'].rate # Hz

# Define an ROI (e.g., a small box within a vessel)
# Based on first_frame.png, a vessel is clearly visible.
# Let's pick a region in the center of the main vessel.
# The vessel seems to be around y=250, x from 150 to 350
roi_y_start, roi_y_end = 240, 260
roi_x_start, roi_x_end = 200, 220

# Number of frames to process (e.g., first 10 seconds = 300 frames at 30Hz)
num_frames_to_process = 300
if num_frames_to_process > movies_data.shape[0]:
    num_frames_to_process = movies_data.shape[0]

# Calculate mean intensity in ROI for each frame
mean_roi_intensity = []
for i in range(num_frames_to_process):
    frame_roi = movies_data[i, roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    mean_roi_intensity.append(np.mean(frame_roi))

mean_roi_intensity = np.array(mean_roi_intensity)

# Create a time vector
time_vector = np.arange(num_frames_to_process) / frame_rate

# Plot the time series
plt.figure(figsize=(10, 4))
plt.plot(time_vector, mean_roi_intensity)
plt.title(f'Mean Intensity in ROI ({roi_y_start}-{roi_y_end}, {roi_x_start}-{roi_x_end}) over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Mean ROI Intensity')
plt.grid(True)
plt.savefig('explore/roi_intensity_timeseries.png')
# plt.show() # Do not show

print(f"Saved explore/roi_intensity_timeseries.png. Processed {num_frames_to_process} frames.")

io.close()