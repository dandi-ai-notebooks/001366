"""
Script to analyze vessel diameter over time for a specific vessel segment.
This script will extract a region of interest containing a vessel,
use intensity profiles to estimate vessel diameter over a short time window,
and plot the results.
"""
import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def enhance_contrast(image, percentile_low=1, percentile_high=99):
    """Enhance image contrast for better visualization."""
    low = np.percentile(image, percentile_low)
    high = np.percentile(image, percentile_high)
    image_scaled = np.clip((image - low) / (high - low), 0, 1)
    return image_scaled

def estimate_vessel_diameter(profile, pixel_size_um=1.0):
    """Estimate vessel diameter using full width at half maximum (FWHM) method."""
    # Apply slight smoothing to reduce noise
    profile_smooth = gaussian_filter1d(profile, sigma=1.0)
    
    # Find the approximate background
    background = np.percentile(profile_smooth, 20)
    
    # Find the peak (vessel center)
    peak_idx, _ = find_peaks(profile_smooth, height=np.max(profile_smooth)*0.7)
    
    if len(peak_idx) == 0:
        return None  # No clear peak found
    
    peak_idx = peak_idx[0]  # Take the first peak if multiple found
    peak_value = profile_smooth[peak_idx]
    
    # Calculate half-max value (assumes vessel is brighter than background)
    half_max = background + (peak_value - background) / 2
    
    # Find indices where the profile crosses the half-max
    above_half_max = profile_smooth > half_max
    transitions = np.diff(above_half_max.astype(int))
    rising_indices = np.where(transitions == 1)[0]
    falling_indices = np.where(transitions == -1)[0]
    
    if len(rising_indices) == 0 or len(falling_indices) == 0:
        return None  # Cannot find clear vessel boundaries
    
    # Get the first rising and the first falling edge after the rising edge
    rising_idx = rising_indices[0]
    falling_candidates = falling_indices[falling_indices > rising_idx]
    
    if len(falling_candidates) == 0:
        return None  # No falling edge found after the rising edge
    
    falling_idx = falling_candidates[0]
    
    # Calculate diameter in pixels, then convert to µm if pixel_size is provided
    diameter_pixels = falling_idx - rising_idx
    diameter_um = diameter_pixels * pixel_size_um
    
    return {
        'diameter_pixels': diameter_pixels,
        'diameter_um': diameter_um,
        'peak_idx': peak_idx,
        'rising_idx': rising_idx,
        'falling_idx': falling_idx,
        'half_max': half_max
    }

# Load the NWB file
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001366/assets/2f12bce3-f841-46ca-b928-044269122a59/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
movies = nwb.acquisition["Movies"]

# Define the time window and region of interest (ROI)
# We'll analyze a segment of frames
start_frame = 3000  # Starting from the middle frame we visualized earlier
frame_count = 100   # Analyze 100 frames (3.33 seconds at 30 Hz)

# Define ROI coordinates based on our visual inspection of the previous visualizations
# The main vessel runs diagonally; we'll select a straight segment
roi_x1, roi_y1 = 200, 150
roi_x2, roi_y2 = 300, 250
roi_width = roi_x2 - roi_x1
roi_height = roi_y2 - roi_y1

print(f"Extracting region of interest from frames {start_frame} to {start_frame + frame_count - 1}...")
# Extract the ROI for each frame
roi_frames = []
for i in range(start_frame, start_frame + frame_count):
    frame = movies.data[i]
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_frames.append(roi)

roi_frames = np.array(roi_frames)

# Visualize the ROI from the first frame in our sequence
plt.figure(figsize=(8, 8))
plt.imshow(enhance_contrast(roi_frames[0]), cmap='gray')
plt.title(f'Region of Interest (Frame #{start_frame})')
plt.savefig('tmp_scripts/roi_first_frame.png', dpi=150)
plt.close()

# The vessel runs diagonally in the ROI, so we'll take a line perpendicular to the vessel
# for measuring diameter
# Define perpendicular line:
perp_x1, perp_y1 = 20, 20
perp_x2, perp_y2 = 80, 80
# These coordinates define a line perpendicular to the main vessel

# Function to extract an intensity profile along a line
def extract_profile(image, x1, y1, x2, y2, num_points=100):
    """Extract intensity profile along a line from (x1, y1) to (x2, y2)."""
    x = np.linspace(x1, x2, num_points).astype(int)
    y = np.linspace(y1, y2, num_points).astype(int)
    return image[y, x]

# Extract intensity profile for the first frame and visualize
profile = extract_profile(roi_frames[0], perp_x1, perp_y1, perp_x2, perp_y2)
plt.figure(figsize=(10, 5))
plt.plot(profile)
plt.title(f'Intensity Profile Across Vessel (Frame #{start_frame})')
plt.xlabel('Position along profile (pixels)')
plt.ylabel('Intensity')
plt.grid(True, alpha=0.3)
plt.savefig('tmp_scripts/intensity_profile.png', dpi=150)
plt.close()

# Visualize the ROI with the perpendicular line used for measurement
plt.figure(figsize=(8, 8))
plt.imshow(enhance_contrast(roi_frames[0]), cmap='gray')
plt.plot([perp_x1, perp_x2], [perp_y1, perp_y2], 'r-', linewidth=1)
plt.title(f'Measurement Line Across Vessel (Frame #{start_frame})')
plt.savefig('tmp_scripts/measurement_line.png', dpi=150)
plt.close()

# Analyze vessel diameter over time
print("Analyzing vessel diameter over time...")
diameters = []
timestamps = []
sampling_rate = movies.rate  # Hz

for i, roi in enumerate(roi_frames):
    profile = extract_profile(roi, perp_x1, perp_y1, perp_x2, perp_y2)
    diameter_info = estimate_vessel_diameter(profile)
    
    if diameter_info:
        diameters.append(diameter_info['diameter_pixels'])
        # Calculate timestamp in seconds
        time_sec = (start_frame + i) / sampling_rate
        timestamps.append(time_sec)
        
        # Visualize every 25th frame with diameter measurement
        if i % 25 == 0:
            plt.figure(figsize=(12, 6))
            
            # Plot the ROI
            plt.subplot(1, 2, 1)
            plt.imshow(enhance_contrast(roi), cmap='gray')
            plt.plot([perp_x1, perp_x2], [perp_y1, perp_y2], 'r-', linewidth=1)
            plt.title(f'Frame #{start_frame + i}')
            
            # Plot the intensity profile with FWHM markers
            plt.subplot(1, 2, 2)
            plt.plot(profile, 'b-', linewidth=1.5)
            
            # Mark the half-max line and crossing points
            plt.axhline(y=diameter_info['half_max'], color='r', linestyle='--', alpha=0.7)
            plt.axvline(x=diameter_info['rising_idx'], color='g', linestyle='-', alpha=0.7)
            plt.axvline(x=diameter_info['falling_idx'], color='g', linestyle='-', alpha=0.7)
            
            plt.title(f'Diameter: {diameter_info["diameter_pixels"]:.2f} pixels')
            plt.xlabel('Position along profile (pixels)')
            plt.ylabel('Intensity')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'tmp_scripts/diameter_frame_{i}.png', dpi=150)
            plt.close()

# Plot diameter over time
plt.figure(figsize=(10, 6))
plt.plot(timestamps, diameters, 'b-')
plt.title('Vessel Diameter Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Diameter (pixels)')
plt.grid(True, alpha=0.3)
plt.savefig('tmp_scripts/diameter_timeseries.png', dpi=150)
plt.close()

# Calculate some basic statistics
mean_diameter = np.mean(diameters)
std_diameter = np.std(diameters)
pulsatility = (np.max(diameters) - np.min(diameters)) / mean_diameter * 100  # in percent

print(f"Mean vessel diameter: {mean_diameter:.2f} pixels")
print(f"Standard deviation: {std_diameter:.2f} pixels")
print(f"Pulsatility (max-min/avg): {pulsatility:.2f}%")

# Perform frequency analysis to identify pulsation rate
if len(diameters) > 20:  # Ensure we have enough data points
    from scipy import signal
    # Detrend the diameter data to remove any linear trend
    diameters_detrended = signal.detrend(diameters)
    
    # Compute power spectral density
    fs = sampling_rate  # sampling frequency (same as movie frame rate)
    f, Pxx = signal.welch(diameters_detrended, fs=fs, nperseg=min(len(diameters_detrended), 64))
    
    # Plot the power spectral density
    plt.figure(figsize=(10, 6))
    plt.semilogy(f, Pxx)
    plt.title('Power Spectral Density of Vessel Diameter Variations')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.grid(True, alpha=0.3)
    plt.savefig('tmp_scripts/diameter_psd.png', dpi=150)
    plt.close()
    
    # Find the peak frequency (excluding DC component at 0 Hz)
    peak_freq_idx = np.argmax(Pxx[1:]) + 1  # Skip the first point (0 Hz)
    peak_freq = f[peak_freq_idx]
    
    print(f"Dominant pulsation frequency: {peak_freq:.2f} Hz")
    print(f"Corresponding to a period of {1/peak_freq:.2f} seconds")