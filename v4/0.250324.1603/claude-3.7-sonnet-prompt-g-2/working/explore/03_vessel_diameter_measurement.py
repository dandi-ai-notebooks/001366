"""
This script explores methods for measuring vessel diameter and detecting pulsatility.
It implements and compares different approaches mentioned in the dataset title,
including full width at half maximum (FWHM) and potentially Radon transform.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter

# URL for the first NWB file (smaller one)
url = "https://api.dandiarchive.org/api/assets/71fa07fc-4309-4013-8edd-13213a86a67d/download/"

# Load the file
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the Movies dataset and basic info
movies = nwb.acquisition['Movies']
print(f"Movie data shape: {movies.data.shape}")
print(f"Movie frame rate: {movies.rate} Hz")

# Function to measure vessel diameter using FWHM method
def measure_diameter_fwhm(profile):
    """
    Measure vessel diameter using Full Width at Half Maximum method.
    
    Args:
        profile: 1D array representing intensity across the vessel
    
    Returns:
        diameter: FWHM diameter in pixels
        vessel_center: estimated center position of the vessel
    """
    # Find background intensity level (using average of the first and last 10 pixels)
    background = np.mean(np.concatenate([profile[:10], profile[-10:]]))
    
    # Find minimum intensity (vessel is darker)
    min_val = np.min(profile)
    min_idx = np.argmin(profile)
    
    # Calculate half maximum value
    half_max = (background + min_val) / 2
    
    # Interpolate profile for more precise calculation
    x = np.arange(len(profile))
    f = interp1d(x, profile, kind='cubic', fill_value="extrapolate")
    x_interp = np.linspace(0, len(profile) - 1, num=1000)
    profile_interp = f(x_interp)
    
    # Find where profile crosses half-max value
    above_half_max = profile_interp > half_max
    transitions = np.where(np.diff(above_half_max))[0]
    
    # Need at least 2 transitions for a valid measurement
    if len(transitions) >= 2:
        # Convert indices back to original scale
        left_idx = x_interp[transitions[0]]
        right_idx = x_interp[transitions[-1]]
        diameter = right_idx - left_idx
        vessel_center = (left_idx + right_idx) / 2
        return diameter, vessel_center
    
    return None, min_idx

# Function to measure vessel diameter using derivative method
def measure_diameter_derivative(profile, sigma=1.0):
    """
    Measure vessel diameter using first derivative to detect edges.
    
    Args:
        profile: 1D array representing intensity across the vessel
        sigma: Gaussian smoothing factor
    
    Returns:
        diameter: Edge-to-edge diameter in pixels
        vessel_center: estimated center position of the vessel
    """
    # Smooth the profile to reduce noise
    smoothed = ndimage.gaussian_filter1d(profile, sigma)
    
    # Calculate first derivative
    derivative = np.gradient(smoothed)
    
    # Find peaks in the absolute derivative (edge detection)
    peaks, _ = find_peaks(np.abs(derivative), height=np.std(derivative)*1.0)
    
    # We need at least two peaks to get a diameter
    if len(peaks) >= 2:
        # For a vessel profile, we expect first a negative peak (left edge)
        # then a positive peak (right edge)
        neg_peaks = peaks[derivative[peaks] < 0]
        pos_peaks = peaks[derivative[peaks] > 0]
        
        if len(neg_peaks) > 0 and len(pos_peaks) > 0:
            # Find the strongest negative and positive peaks
            left_edge = neg_peaks[np.argmin(derivative[neg_peaks])]
            right_edge = pos_peaks[np.argmax(derivative[pos_peaks])]
            
            # Make sure left edge is actually left of right edge
            if left_edge < right_edge:
                diameter = right_edge - left_edge
                vessel_center = (left_edge + right_edge) / 2
                return diameter, vessel_center
    
    # Fallback: find the darkest point as center
    min_idx = np.argmin(profile)
    return None, min_idx

# Function to visualize diameter measurement on a profile
def visualize_diameter_measurement(profile, diameter_fwhm, center_fwhm, 
                                  diameter_derivative, center_derivative):
    """Visualize the diameter measurements on the intensity profile."""
    plt.figure(figsize=(12, 6))
    
    # Plot original profile
    x = np.arange(len(profile))
    plt.plot(x, profile, 'k-', label='Intensity Profile')
    
    # Plot FWHM measurement
    if diameter_fwhm is not None:
        half_width = diameter_fwhm / 2
        background = np.mean(np.concatenate([profile[:10], profile[-10:]]))
        min_val = np.min(profile)
        half_max = (background + min_val) / 2
        
        plt.plot([center_fwhm - half_width, center_fwhm + half_width], 
                [half_max, half_max], 'r-', linewidth=2, label=f'FWHM: {diameter_fwhm:.1f} px')
        plt.axvline(x=center_fwhm, color='r', linestyle='--', alpha=0.5)
    
    # Plot derivative-based measurement
    if diameter_derivative is not None:
        half_width = diameter_derivative / 2
        plt.plot([center_derivative - half_width, center_derivative + half_width],
                [profile[int(center_derivative)], profile[int(center_derivative)]],
                'g-', linewidth=2, label=f'Derivative: {diameter_derivative:.1f} px')
        plt.axvline(x=center_derivative, color='g', linestyle='--', alpha=0.5)
    
    plt.grid(True)
    plt.legend()
    plt.title('Vessel Diameter Measurement Comparison')
    plt.xlabel('Position (pixels)')
    plt.ylabel('Intensity')
    plt.tight_layout()
    plt.savefig('explore/diameter_measurement.png', dpi=150)
    plt.close()

# Function to track vessel diameter over time
def track_vessel_diameter_over_time(data, rate, num_frames=100):
    """
    Track vessel diameter over time for a subset of frames.
    
    Args:
        data: 3D array of image data (frames, height, width)
        rate: frame rate in Hz
        num_frames: number of frames to analyze
    
    Returns:
        times: array of time points
        diameters_fwhm: array of FWHM diameter measurements
        diameters_deriv: array of derivative-based diameter measurements
    """
    # Sample frames evenly across the dataset
    frame_indices = np.linspace(0, min(data.shape[0]-1, 1000), num_frames, dtype=int)
    
    # Calculate time points
    times = frame_indices / rate
    
    # Arrays to store diameter measurements
    diameters_fwhm = np.zeros(num_frames)
    diameters_deriv = np.zeros(num_frames)
    
    # Use the middle row for measurements
    row_idx = data.shape[1] // 2
    
    for i, frame_idx in enumerate(frame_indices):
        # Get intensity profile across the middle row
        profile = data[frame_idx, row_idx, :]
        
        # Measure diameter using both methods
        diam_fwhm, _ = measure_diameter_fwhm(profile)
        diam_deriv, _ = measure_diameter_derivative(profile)
        
        # Store measurements
        diameters_fwhm[i] = diam_fwhm if diam_fwhm is not None else np.nan
        diameters_deriv[i] = diam_deriv if diam_deriv is not None else np.nan
    
    return times, diameters_fwhm, diameters_deriv

# Choose a single frame for detailed analysis
frame_idx = 0
frame = movies.data[frame_idx]

# Use the middle row for profile analysis
middle_row = frame.shape[0] // 2
profile = frame[middle_row, :]

# Measure diameter using both methods
diam_fwhm, center_fwhm = measure_diameter_fwhm(profile)
diam_deriv, center_deriv = measure_diameter_derivative(profile)

print(f"FWHM diameter: {diam_fwhm:.2f} pixels")
print(f"Derivative-based diameter: {diam_deriv:.2f} pixels")

# Visualize the diameter measurements
visualize_diameter_measurement(profile, diam_fwhm, center_fwhm, diam_deriv, center_deriv)

# Track vessel diameter over time
print("Tracking vessel diameter over time...")
times, diameters_fwhm, diameters_deriv = track_vessel_diameter_over_time(movies.data, movies.rate, num_frames=50)

# Plot diameter over time
plt.figure(figsize=(12, 6))
plt.plot(times, diameters_fwhm, 'r-o', label='FWHM Method')
plt.plot(times, diameters_deriv, 'g-o', label='Derivative Method')

# Calculate and plot trend lines
if not np.all(np.isnan(diameters_fwhm)):
    valid_fwhm = ~np.isnan(diameters_fwhm)
    trend_fwhm = savgol_filter(diameters_fwhm[valid_fwhm], 
                             min(11, sum(valid_fwhm) - (sum(valid_fwhm) % 2) - 1), 3)
    plt.plot(times[valid_fwhm], trend_fwhm, 'r--', linewidth=2, label='FWHM Trend')

if not np.all(np.isnan(diameters_deriv)):
    valid_deriv = ~np.isnan(diameters_deriv)
    trend_deriv = savgol_filter(diameters_deriv[valid_deriv], 
                              min(11, sum(valid_deriv) - (sum(valid_deriv) % 2) - 1), 3)
    plt.plot(times[valid_deriv], trend_deriv, 'g--', linewidth=2, label='Derivative Trend')

plt.title('Vessel Diameter Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Diameter (pixels)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('explore/diameter_over_time.png', dpi=150)
plt.close()

# Calculate pulsatility index if possible
if not np.all(np.isnan(diameters_fwhm)):
    valid_fwhm = ~np.isnan(diameters_fwhm)
    if sum(valid_fwhm) > 10:
        diam_max = np.nanmax(diameters_fwhm)
        diam_min = np.nanmin(diameters_fwhm)
        diam_mean = np.nanmean(diameters_fwhm)
        pulsatility_index = (diam_max - diam_min) / diam_mean
        print(f"FWHM Pulsatility Index: {pulsatility_index:.4f}")

if not np.all(np.isnan(diameters_deriv)):
    valid_deriv = ~np.isnan(diameters_deriv)
    if sum(valid_deriv) > 10:
        diam_max = np.nanmax(diameters_deriv)
        diam_min = np.nanmin(diameters_deriv)
        diam_mean = np.nanmean(diameters_deriv)
        pulsatility_index = (diam_max - diam_min) / diam_mean
        print(f"Derivative Pulsatility Index: {pulsatility_index:.4f}")

# Close the file
h5_file.close()