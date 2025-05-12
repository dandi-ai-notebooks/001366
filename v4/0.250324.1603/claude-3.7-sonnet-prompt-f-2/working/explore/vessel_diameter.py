# Script to explore vessel diameter measurement and pulsatility visualization
# from the NWB file

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Set up figure parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# Function to estimate vessel diameter using a simple threshold-based approach
def measure_vessel_diameter(frame, row_idx, threshold_factor=0.5):
    """
    Measure vessel diameter at a specific row using a threshold-based approach.
    
    Parameters:
    - frame: 2D numpy array of the image
    - row_idx: Row index where to measure diameter
    - threshold_factor: Factor to determine threshold (0-1)
    
    Returns:
    - diameter: Estimated diameter in pixels
    - profile: Intensity profile at the selected row
    - vessel_center: Estimated center position of vessel
    """
    # Extract intensity profile at the specified row
    profile = frame[row_idx, :]
    
    # Invert profile (vessels are darker than background)
    profile_inv = np.max(profile) - profile
    
    # Find the vessel center (maximum of inverted profile)
    vessel_center = np.argmax(profile_inv)
    
    # Calculate threshold based on min and max values
    min_val = np.min(profile)
    max_val = np.max(profile)
    threshold = min_val + threshold_factor * (max_val - min_val)
    
    # Find points where intensity crosses threshold (FWHM-like approach)
    vessel_points = np.where(profile <= threshold)[0]
    
    # If no points found below threshold, return 0 diameter
    if len(vessel_points) == 0:
        return 0, profile, vessel_center
    
    # Find leftmost and rightmost points
    left_edge = np.min(vessel_points)
    right_edge = np.max(vessel_points)
    
    # Calculate diameter
    diameter = right_edge - left_edge
    
    return diameter, profile, vessel_center

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/71fa07fc-4309-4013-8edd-13213a86a67d/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get movie data
movies = nwb.acquisition["Movies"]
frame_rate = movies.rate
print(f"Frame rate: {frame_rate} fps")

# Sample frames at regular intervals rather than loading all frames
# We'll sample 100 frames across the movie's duration
num_frames = movies.data.shape[0]
sample_indices = np.linspace(0, num_frames-1, 100, dtype=int)

# Choose a row near the middle of the image to measure vessel diameter
row_idx = movies.data.shape[1] // 2  # middle row

# Process sampled frames to get diameter measurements
diameters = []
timestamps = []

print(f"Processing {len(sample_indices)} frames...")
for i, idx in enumerate(sample_indices):
    # Load the frame
    frame = movies.data[idx, :, :]
    
    # Apply Gaussian blur to reduce noise
    frame_smooth = ndimage.gaussian_filter(frame, sigma=1)
    
    # Measure vessel diameter
    diameter, profile, vessel_center = measure_vessel_diameter(frame_smooth, row_idx)
    
    # Store results
    diameters.append(diameter)
    timestamps.append(idx / frame_rate)  # Convert frame index to time in seconds
    
    # Print progress
    if (i + 1) % 10 == 0:
        print(f"Processed {i+1}/{len(sample_indices)} frames")

# Convert lists to numpy arrays
diameters = np.array(diameters)
timestamps = np.array(timestamps)

# Plot diameter vs time
plt.figure()
plt.plot(timestamps, diameters)
plt.xlabel('Time (seconds)')
plt.ylabel('Vessel Diameter (pixels)')
plt.title('Vessel Diameter Over Time')
plt.grid(True)
plt.savefig('explore/vessel_diameter_time.png')
print("Saved diameter vs time plot to explore/vessel_diameter_time.png")

# Calculate and display basic statistics
mean_diameter = np.mean(diameters)
std_diameter = np.std(diameters)
min_diameter = np.min(diameters)
max_diameter = np.max(diameters)
pulsatility_index = (max_diameter - min_diameter) / mean_diameter

print("\nVessel Diameter Statistics:")
print(f"Mean Diameter: {mean_diameter:.2f} pixels")
print(f"Standard Deviation: {std_diameter:.2f} pixels")
print(f"Coefficient of Variation: {(std_diameter/mean_diameter):.3f}")
print(f"Min Diameter: {min_diameter:.2f} pixels")
print(f"Max Diameter: {max_diameter:.2f} pixels")
print(f"Pulsatility Index: {pulsatility_index:.3f}")

# Also plot the histogram of diameters
plt.figure()
plt.hist(diameters, bins=15, edgecolor='black')
plt.xlabel('Vessel Diameter (pixels)')
plt.ylabel('Frequency')
plt.title('Histogram of Vessel Diameters')
plt.grid(True)
plt.savefig('explore/vessel_diameter_histogram.png')
print("Saved diameter histogram to explore/vessel_diameter_histogram.png")

# Visualize vessel profile and threshold for a sample frame
sample_frame_idx = sample_indices[len(sample_indices) // 2]  # middle of sampled frames
sample_frame = movies.data[sample_frame_idx, :, :]
sample_frame_smooth = ndimage.gaussian_filter(sample_frame, sigma=1)

# Measure vessel diameter and get profile
sample_diameter, sample_profile, sample_center = measure_vessel_diameter(sample_frame_smooth, row_idx)

# Calculate threshold
min_val = np.min(sample_profile)
max_val = np.max(sample_profile)
threshold = min_val + 0.5 * (max_val - min_val)

# Plot frame with measurement line
plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
plt.imshow(sample_frame_smooth, cmap='gray')
plt.axhline(y=row_idx, color='r', linestyle='-')
plt.title(f'Frame #{sample_frame_idx} with Measurement Line (row {row_idx})')
plt.colorbar(label='Pixel Value')

# Plot intensity profile with threshold
plt.subplot(2, 1, 2)
plt.plot(sample_profile, 'b-', label='Intensity Profile')
plt.axhline(y=threshold, color='r', linestyle='-', label='Threshold')
plt.axvline(x=sample_center, color='g', linestyle='--', label='Vessel Center')
plt.xlabel('Column Index')
plt.ylabel('Pixel Value')
plt.title(f'Intensity Profile at Row {row_idx} (Diameter: {sample_diameter:.2f} pixels)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('explore/vessel_profile.png')
print("Saved vessel profile visualization to explore/vessel_profile.png")

print("\nExploration complete!")