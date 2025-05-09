# explore/script_02_roi_intensity.py
# Purpose: Calculate and plot the average pixel intensity over time 
# for a central region of interest (ROI) in the 'Movies' data.
# This can help visualize temporal dynamics, such as vessel pulsatility.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# DANDI URL for the NWB file
url = "https://api.dandiarchive.org/api/assets/2f12bce3-f841-46ca-b928-044269122a59/download/"

remote_file = None
h5_file = None
io = None

try:
    print(f"Attempting to load NWB file from: {url}")
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file, 'r')
    io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
    nwb = io.read()
    print("NWB file loaded successfully.")

    if "Movies" in nwb.acquisition:
        movies_series = nwb.acquisition["Movies"]
        print(f"Movies data shape: {movies_series.data.shape}")
        
        num_frames_to_process = 300  # Approx 10 seconds at 30 Hz
        if movies_series.data.shape[0] < num_frames_to_process:
            num_frames_to_process = movies_series.data.shape[0]
        
        if num_frames_to_process > 0:
            # Define a central ROI (e.g., 20x20 pixels)
            h, w = movies_series.data.shape[1], movies_series.data.shape[2]
            roi_size = 20
            roi_y_start = h // 2 - roi_size // 2
            roi_y_end = h // 2 + roi_size // 2
            roi_x_start = w // 2 - roi_size // 2
            roi_x_end = w // 2 + roi_size // 2
            
            print(f"Defined ROI: y=[{roi_y_start}:{roi_y_end}], x=[{roi_x_start}:{roi_x_end}]")

            avg_intensities = []
            timestamps = []
            
            # Calculate timestamps for the selected frames
            # Movies.rate is in Hz (frames per second)
            # Movies.starting_time is the time of the first frame in seconds
            frame_interval = 1.0 / movies_series.rate if movies_series.rate > 0 else 0
            
            for i in range(num_frames_to_process):
                frame_data = movies_series.data[i, roi_y_start:roi_y_end, roi_x_start:roi_x_end]
                avg_intensities.append(np.mean(frame_data))
                # Note: movies_series.timestamps is often not populated for ImageSeries.
                # We calculate time based on frame rate and starting_time.
                current_time = movies_series.starting_time + i * frame_interval
                timestamps.append(current_time)
                if (i+1) % 50 == 0:
                    print(f"Processed frame {i+1}/{num_frames_to_process}")

            avg_intensities = np.array(avg_intensities)
            timestamps = np.array(timestamps)

            print(f"Calculated average intensities for {len(avg_intensities)} frames.")

            # Plot the average intensity over time
            sns.set_theme()
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, avg_intensities)
            plt.title(f"Average Intensity in Central ROI ({roi_size}x{roi_size}px) over Time (First {num_frames_to_process} Frames)")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Average Pixel Intensity")
            plt.grid(True)
            
            plot_filename = "explore/roi_intensity_timeseries.png"
            plt.savefig(plot_filename)
            plt.close()
            print(f"Plot saved to {plot_filename}")

        else:
            print("Movies data is empty or too short to process.")
    else:
        print("'Movies' data not found in acquisition.")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    if io:
        try:
            io.close()
            print("NWBHDF5IO closed.")
        except Exception as e_close_io:
            print(f"Error closing NWBHDF5IO: {e_close_io}")
    if h5_file:
        try:
            h5_file.close()
            print("H5 file closed.")
        except Exception as e_close_h5:
            print(f"Error closing H5 file: {e_close_h5}")
    print("Script finished.")