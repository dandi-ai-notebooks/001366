# Script to explore the larger NWB file and compare with the first file

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Set up figure parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# Load the larger NWB file
url = "https://api.dandiarchive.org/api/assets/2f12bce3-f841-46ca-b928-044269122a59/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information
print(f"NWB file identifier: {nwb.identifier}")
print(f"Session description: {nwb.session_description[:100]}...")
print(f"Subject: {nwb.subject.subject_id}, Sex: {nwb.subject.sex}, Age: {nwb.subject.age}")
print(f"Subject description: {nwb.subject.description[:100]}...")
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

# Plot and save the first frame - use only a region of interest (not all 512x512 pixels)
# Create a region of interest around the vessel
roi_row_start = 200
roi_row_end = 400
roi_col_start = 100
roi_col_end = 300
roi = first_frame[roi_row_start:roi_row_end, roi_col_start:roi_col_end]

plt.figure()
plt.imshow(roi, cmap='gray')
plt.colorbar(label='Pixel Value')
plt.title('First Frame of Vessel Movie (ROI)')
plt.savefig('explore/frame1_file2_roi.png')
print("Saved first frame ROI to explore/frame1_file2_roi.png")

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

# Get a sample of frames to analyze vessel diameter over time
num_frames = min(100, movies.data.shape[0])  # Use up to 100 frames to keep it manageable
sample_indices = np.linspace(0, movies.data.shape[0]-1, num_frames, dtype=int)

# Find a good row to measure vessel diameter
middle_frame_idx = sample_indices[len(sample_indices) // 2]
middle_frame = movies.data[middle_frame_idx, :, :]

# Apply Gaussian blur to reduce noise
middle_frame_smooth = ndimage.gaussian_filter(middle_frame, sigma=1)

# Create ROI for the middle frame
roi_middle = middle_frame_smooth[roi_row_start:roi_row_end, roi_col_start:roi_col_end]

# Plot the ROI with a red line at a row where we'll measure diameter
row_to_measure = (roi_row_end - roi_row_start) // 2  # Middle row of ROI
plt.figure()
plt.imshow(roi_middle, cmap='gray')
plt.axhline(y=row_to_measure, color='r', linestyle='-')
plt.title(f'Middle Frame with Measurement Line (row {row_to_measure})')
plt.colorbar(label='Pixel Value')
plt.savefig('explore/middle_frame_file2_roi.png')
print(f"Saved middle frame ROI to explore/middle_frame_file2_roi.png")

# Measure vessel diameter over time
diameters = []
timestamps = []

print(f"\nProcessing {len(sample_indices)} frames...")
for i, idx in enumerate(sample_indices):
    # Load the frame
    frame = movies.data[idx, roi_row_start:roi_row_end, roi_col_start:roi_col_end]
    
    # Apply Gaussian blur to reduce noise
    frame_smooth = ndimage.gaussian_filter(frame, sigma=1)
    
    # Measure vessel diameter
    diameter, profile, vessel_center = measure_vessel_diameter(frame_smooth, row_to_measure)
    
    # Store results
    diameters.append(diameter)
    timestamps.append(idx / movies.rate)  # Convert frame index to time in seconds
    
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
plt.title('Vessel Diameter Over Time (Second NWB File)')
plt.grid(True)
plt.savefig('explore/vessel_diameter_time_file2.png')
print("Saved diameter vs time plot to explore/vessel_diameter_time_file2.png")

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
plt.title('Histogram of Vessel Diameters (Second NWB File)')
plt.grid(True)
plt.savefig('explore/vessel_diameter_histogram_file2.png')
print("Saved diameter histogram to explore/vessel_diameter_histogram_file2.png")

# Visualize vessel profile for a sample frame
sample_frame_idx = sample_indices[len(sample_indices) // 2]  # middle of sampled frames
sample_frame = movies.data[sample_frame_idx, roi_row_start:roi_row_end, roi_col_start:roi_col_end]
sample_frame_smooth = ndimage.gaussian_filter(sample_frame, sigma=1)

# Measure vessel diameter and get profile
sample_diameter, sample_profile, sample_center = measure_vessel_diameter(sample_frame_smooth, row_to_measure)

# Calculate threshold
min_val = np.min(sample_profile)
max_val = np.max(sample_profile)
threshold = min_val + 0.5 * (max_val - min_val)

# Plot frame with measurement line
plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
plt.imshow(sample_frame_smooth, cmap='gray')
plt.axhline(y=row_to_measure, color='r', linestyle='-')
plt.title(f'Frame #{sample_frame_idx} with Measurement Line (row {row_to_measure})')
plt.colorbar(label='Pixel Value')

# Plot intensity profile with threshold
plt.subplot(2, 1, 2)
plt.plot(sample_profile, 'b-', label='Intensity Profile')
plt.axhline(y=threshold, color='r', linestyle='-', label='Threshold')
plt.axvline(x=sample_center, color='g', linestyle='--', label='Vessel Center')
plt.xlabel('Column Index')
plt.ylabel('Pixel Value')
plt.title(f'Intensity Profile at Row {row_to_measure} (Diameter: {sample_diameter:.2f} pixels)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('explore/vessel_profile_file2.png')
print("Saved vessel profile visualization to explore/vessel_profile_file2.png")

print("\nExploration complete!")