# explore/script_01_load_frame.py
# Purpose: Load and display a single frame from the 'Movies' ImageSeries
# to understand the basic structure of the imaging data.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# DANDI URL for the NWB file
url = "https://api.dandiarchive.org/api/assets/2f12bce3-f841-46ca-b928-044269122a59/download/"

remote_file = None
h5_file = None
io = None

try:
    print(f"Attempting to load NWB file from: {url}")
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file, 'r') # Read-only mode
    io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Read-only mode for NWBHDF5IO
    nwb = io.read()
    print("NWB file loaded successfully.")

    # Access the 'Movies' ImageSeries
    if "Movies" in nwb.acquisition:
        movies_series = nwb.acquisition["Movies"]
        print(f"Movies data shape: {movies_series.data.shape}")
        print(f"Movies data dtype: {movies_series.data.dtype}")

        # Load the first frame
        # The data is (frames, height, width)
        if movies_series.data.shape[0] > 0:
            first_frame = movies_series.data[0, :, :]
            print(f"Loaded first frame with shape: {first_frame.shape}")

            # Plot the first frame
            plt.figure(figsize=(8, 8))
            plt.imshow(first_frame, cmap='gray')
            plt.title("First Frame from 'Movies' Data (031224-M4)")
            plt.xlabel("Pixel X")
            plt.ylabel("Pixel Y")
            plt.colorbar(label="Pixel Intensity (uint16)")
            
            # Save the plot
            plot_filename = "explore/first_frame.png"
            plt.savefig(plot_filename)
            plt.close() # Close the plot to free memory
            print(f"Plot saved to {plot_filename}")
        else:
            print("Movies data is empty (0 frames).")

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
    # remfile.File objects do not need to be explicitly closed.
    print("Script finished.")