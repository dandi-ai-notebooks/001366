"""
This script demonstrates different approaches for measuring vessel diameter 
from the NWB files in Dandiset 001366. The approaches include:
1. Full Width at Half Maximum (FWHM) method
2. Radon transform method
3. Basic intensity thresholding

The script examines a few sample frames and compares the results of different methods.
"""

import matplotlib.pyplot as plt
import numpy as np
import pynwb
import h5py
import remfile
from scipy import ndimage
from skimage.transform import radon
import os
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter

# Ensure output directory exists
os.makedirs("tmp_scripts", exist_ok=True)

# Load the second NWB file (which has a clearer vessel)
url = "https://api.dandiarchive.org/api/assets/71fa07fc-4309-4013-8edd-13213a86a67d/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

# Get movie data
movies = nwb.acquisition["Movies"]
print(f"Movie data shape: {movies.data.shape}")
print(f"Movie frame rate: {movies.rate} frames/second")

# Get a single frame for analysis
middle_frame_idx = movies.data.shape[0] // 2
middle_frame = movies.data[middle_frame_idx, :, :]

# Helper function to find vessel diameter using FWHM method
def measure_diameter_fwhm(intensity_profile):
    # Invert the profile since our vessel is dark on bright background
    inverted_profile = np.max(intensity_profile) - intensity_profile
    
    # Smooth the profile to reduce noise
    smoothed_profile = gaussian_filter(inverted_profile, sigma=1)
    
    # Find the peak
    peaks, _ = find_peaks(smoothed_profile, height=np.max(smoothed_profile) * 0.5)
    
    if len(peaks) == 0:
        return None, None, None
    
    # Use the highest peak
    peak_idx = peaks[np.argmax(smoothed_profile[peaks])]
    
    # Calculate FWHM
    half_max = smoothed_profile[peak_idx] / 2
    
    # Find indices where the profile crosses half max
    above_half_max = smoothed_profile > half_max
    
    # Find left and right crossing points
    try:
        region_indices = np.where(above_half_max)[0]
        left_idx = region_indices[0]
        right_idx = region_indices[-1]
        diameter = right_idx - left_idx
        
        return diameter, left_idx, right_idx
    except:
        return None, None, None

# Helper function to find vessel angle using Radon transform
def find_vessel_angle(frame):
    # Smooth the frame to reduce noise
    smoothed_frame = gaussian_filter(frame, sigma=2)
    
    # Use Radon transform to find the vessel angle
    theta = np.linspace(0., 180., max(frame.shape), endpoint=False)
    sinogram = radon(smoothed_frame, theta=theta)
    
    # Find the angle that maximizes the variance in the projection
    projection_variance = np.var(sinogram, axis=0)
    max_var_angle_idx = np.argmax(projection_variance)
    angle = theta[max_var_angle_idx]
    
    return angle, sinogram, theta

# Apply Gaussian smoothing to reduce noise
smoothed_frame = gaussian_filter(middle_frame, sigma=1)

# 1. Find the vessel orientation using Radon transform
angle, sinogram, theta = find_vessel_angle(middle_frame)
print(f"Estimated vessel angle: {angle:.2f} degrees")

# 2. Get a line profile perpendicular to the vessel
# First rotate the image so the vessel is horizontal
rotated_frame = ndimage.rotate(middle_frame, angle - 90, reshape=False)

# Get a horizontal line profile across the middle of the rotated image
middle_row = rotated_frame.shape[0] // 2
line_profile = rotated_frame[middle_row, :]

# 3. Apply FWHM method to measure vessel diameter
diameter_fwhm, left_idx, right_idx = measure_diameter_fwhm(line_profile)

if diameter_fwhm is not None:
    print(f"Vessel diameter using FWHM method: {diameter_fwhm:.2f} pixels")
else:
    print("Could not determine vessel diameter using FWHM method")

# 4. Apply basic thresholding method
# Assuming the vessel is darker than the background
threshold = np.mean(line_profile) - 0.5 * np.std(line_profile)
vessel_pixels = line_profile < threshold
vessel_regions = np.where(vessel_pixels)[0]

if len(vessel_regions) > 0:
    diameter_threshold = vessel_regions[-1] - vessel_regions[0]
    print(f"Vessel diameter using thresholding method: {diameter_threshold:.2f} pixels")
else:
    diameter_threshold = None
    print("Could not determine vessel diameter using thresholding method")

# Plot the results
plt.figure(figsize=(16, 12))

# Plot 1: Original middle frame
plt.subplot(2, 3, 1)
plt.imshow(middle_frame, cmap='gray')
plt.title(f"Middle Frame (Frame {middle_frame_idx})")
plt.colorbar()
plt.axis('off')

# Plot 2: Rotated frame with horizontal marker
plt.subplot(2, 3, 2)
plt.imshow(rotated_frame, cmap='gray')
plt.axhline(y=middle_row, color='r', linestyle='-')
plt.title(f"Rotated Frame (Angle: {angle:.2f}°)")
plt.colorbar()
plt.axis('off')

# Plot 3: Line profile with diameter measurements
plt.subplot(2, 3, 3)
plt.plot(line_profile, label="Intensity Profile")
if diameter_fwhm is not None:
    plt.axvline(x=left_idx, color='g', linestyle='--', label="FWHM Left")
    plt.axvline(x=right_idx, color='g', linestyle='--', label="FWHM Right")
if diameter_threshold is not None and len(vessel_regions) > 0:
    plt.axvline(x=vessel_regions[0], color='r', linestyle=':', label="Threshold Left")
    plt.axvline(x=vessel_regions[-1], color='r', linestyle=':', label="Threshold Right")
plt.title("Line Profile with Diameter Measurements")
plt.xlabel("Pixel Position")
plt.ylabel("Intensity")
plt.legend()
plt.grid(alpha=0.3)

# Plot 4: FWHM visualization
plt.subplot(2, 3, 4)
inverted_profile = np.max(line_profile) - line_profile
smoothed_profile = gaussian_filter(inverted_profile, sigma=1)
plt.plot(smoothed_profile, label="Inverted & Smoothed Profile")
if diameter_fwhm is not None:
    # Calculate half max from the FWHM points
    max_val = np.max(smoothed_profile[left_idx:right_idx+1])
    half_max = max_val / 2
    plt.axhline(y=half_max, color='g', linestyle='--', label="Half Maximum")
    plt.axvline(x=left_idx, color='g', linestyle='--')
    plt.axvline(x=right_idx, color='g', linestyle='--')
plt.title("FWHM Method Visualization")
plt.xlabel("Pixel Position")
plt.ylabel("Inverted Intensity")
plt.legend()
plt.grid(alpha=0.3)

# Plot 5: Radon transform sinogram
plt.subplot(2, 3, 5)
plt.imshow(sinogram, cmap='gray', aspect='auto', 
           extent=(0., 180., 0, sinogram.shape[0]))
plt.axvline(x=angle, color='r', linestyle='--', label=f"Vessel Angle: {angle:.2f}°")
plt.title("Radon Transform Sinogram")
plt.xlabel("Angle (degrees)")
plt.ylabel("Distance")
plt.legend()

# Plot 6: Variance of projections
plt.subplot(2, 3, 6)
projection_variance = np.var(sinogram, axis=0)
plt.plot(theta, projection_variance)
plt.axvline(x=angle, color='r', linestyle='--', label=f"Max Variance Angle: {angle:.2f}°")
plt.title("Variance of Projections")
plt.xlabel("Angle (degrees)")
plt.ylabel("Variance")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("tmp_scripts/vessel_diameter_methods.png")
plt.close()

# Analyze diameter across multiple frames to detect pulsatility
# Sample frames across the time series
n_samples = 20
sample_indices = np.linspace(0, movies.data.shape[0]-1, n_samples, dtype=int)

fwhm_diameters = []
threshold_diameters = []

for idx in sample_indices:
    frame = movies.data[idx, :, :]
    
    # Rotate frame based on vessel angle
    rotated_frame = ndimage.rotate(frame, angle - 90, reshape=False)
    
    # Get line profile
    line_profile = rotated_frame[middle_row, :]
    
    # FWHM method
    diameter, _, _ = measure_diameter_fwhm(line_profile)
    if diameter is not None:
        fwhm_diameters.append(diameter)
    else:
        fwhm_diameters.append(np.nan)
    
    # Thresholding method
    threshold = np.mean(line_profile) - 0.5 * np.std(line_profile)
    vessel_pixels = line_profile < threshold
    vessel_regions = np.where(vessel_pixels)[0]
    
    if len(vessel_regions) > 0:
        threshold_diameters.append(vessel_regions[-1] - vessel_regions[0])
    else:
        threshold_diameters.append(np.nan)

# Plot diameter variations over time
plt.figure(figsize=(12, 6))
time_points = sample_indices / movies.rate  # Convert to seconds

plt.plot(time_points, fwhm_diameters, 'g-o', label="FWHM Method")
plt.plot(time_points, threshold_diameters, 'r-^', label="Threshold Method")

plt.title("Vessel Diameter Variation Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Diameter (pixels)")
plt.legend()
plt.grid(alpha=0.3)

# Add smoothed trend lines
valid_fwhm = ~np.isnan(fwhm_diameters)
valid_threshold = ~np.isnan(threshold_diameters)

if np.sum(valid_fwhm) > 2:
    smooth_fwhm = gaussian_filter(np.array(fwhm_diameters)[valid_fwhm], sigma=1)
    plt.plot(np.array(time_points)[valid_fwhm], smooth_fwhm, 'g--', alpha=0.7, label="Smoothed FWHM")

if np.sum(valid_threshold) > 2:
    smooth_threshold = gaussian_filter(np.array(threshold_diameters)[valid_threshold], sigma=1)
    plt.plot(np.array(time_points)[valid_threshold], smooth_threshold, 'r--', alpha=0.7, label="Smoothed Threshold")

plt.legend()
plt.tight_layout()
plt.savefig("tmp_scripts/vessel_diameter_pulsatility.png")
plt.close()

print("Script execution completed successfully")