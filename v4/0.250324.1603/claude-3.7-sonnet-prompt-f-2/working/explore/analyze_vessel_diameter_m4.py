"""
Analyze vessel diameter changes over time in the M4 NWB file.
This script will:
1. Load a sequence of frames from the M4 NWB file
2. Extract vessel diameter at a specific cross-section
3. Plot the vessel diameter changes over time
4. Analyze the vessel pulsatility
"""

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import signal
from scipy.ndimage import gaussian_filter1d

# Set up the plot styling
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/2f12bce3-f841-46ca-b928-044269122a59/download/"
print(f"Loading NWB file from {url}")

remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information about the dataset
movies = nwb.acquisition["Movies"]
print(f"Dataset shape: {movies.data.shape}, rate: {movies.rate} fps")

# Define the parameters for analysis
# We'll analyze only a subset of frames to keep processing manageable
n_frames = 600  # Approximately 20 seconds of data at 30 fps
start_frame = 2000  # Skip the first portion of the recording
y_position = 220  # The y-coordinate for measuring vessel diameter (based on where vessel is clearly visible)

# Load the frames
frames = []
timestamps = []
print(f"Loading {n_frames} frames starting from frame {start_frame}...")
for i in range(start_frame, start_frame + n_frames):
    frames.append(movies.data[i, :, :])
    timestamps.append(i / movies.rate)  # Convert frame index to time in seconds

print(f"Loaded {len(frames)} frames")

# Visualize the vessel cross-section for the first frame
plt.figure(figsize=(12, 10))
plt.imshow(frames[0], cmap='gray')
plt.axhline(y=y_position, color='r', linestyle='-', label=f'Measurement line (y={y_position})')
plt.title("Vessel Diameter Measurement Line (M4 Dataset)")
plt.xlabel("X Position (pixels)")
plt.ylabel("Y Position (pixels)")
plt.colorbar(label='Pixel Value')
plt.legend()
plt.tight_layout()
plt.savefig("explore/m4_vessel_measurement_line.png")
print("Saved measurement line visualization")

# Plot the intensity profile along the measurement line for the first frame
intensity_profile_first = frames[0][y_position, :]
plt.figure(figsize=(12, 6))
plt.plot(intensity_profile_first)
plt.title(f"Intensity Profile at y={y_position} (First Frame)")
plt.xlabel("X Position (pixels)")
plt.ylabel("Pixel Intensity")
plt.grid(True)
plt.tight_layout()
plt.savefig("explore/m4_intensity_profile_first_frame.png")
print("Saved intensity profile for first frame")

# Extract intensity profiles along the measurement line for all frames
intensity_profiles = np.array([frame[y_position, :] for frame in frames])

# Plot the intensity profiles as a heatmap
plt.figure(figsize=(12, 8))
plt.imshow(intensity_profiles, aspect='auto', cmap='viridis', 
           extent=[0, intensity_profiles.shape[1], timestamps[-1], timestamps[0]])
plt.colorbar(label='Pixel Intensity')
plt.title("Vessel Intensity Profiles Over Time (M4 Dataset)")
plt.xlabel("X Position (pixels)")
plt.ylabel("Time (seconds)")
plt.tight_layout()
plt.savefig("explore/m4_vessel_intensity_profiles_heatmap.png")
print("Saved intensity profiles heatmap")

# Define a function to estimate vessel diameter
# For this dataset, the vessel appears as a bright structure (higher intensity)
def estimate_vessel_diameter(intensity_profile, threshold_percentile=80):
    # We don't need to invert the profile since vessel is bright in this dataset
    
    # Smooth the profile to reduce noise
    smoothed_profile = gaussian_filter1d(intensity_profile, sigma=2)
    
    # Calculate a threshold based on percentile
    threshold = np.percentile(smoothed_profile, threshold_percentile)
    
    # Find regions above threshold
    above_threshold = smoothed_profile > threshold
    
    # Find the vessel edges
    edges = np.where(np.diff(above_threshold.astype(int)))[0]
    
    # If we found the vessel (at least two edges)
    if len(edges) >= 2:
        # Calculate diameter as distance between first and last edge
        # We'll take the first and last edges to account for possible multiple crossings
        diameter = edges[-1] - edges[0]
        return diameter
    else:
        return np.nan

# Calculate vessel diameter for each frame
print("Calculating vessel diameters...")
vessel_diameters = []
for profile in intensity_profiles:
    diameter = estimate_vessel_diameter(profile)
    vessel_diameters.append(diameter)

vessel_diameters = np.array(vessel_diameters)

# Plot the vessel diameter over time
plt.figure(figsize=(12, 6))
plt.plot(timestamps, vessel_diameters, 'b-')
plt.title("Vessel Diameter Over Time (M4 Dataset)")
plt.xlabel("Time (seconds)")
plt.ylabel("Vessel Diameter (pixels)")
plt.grid(True)
plt.tight_layout()
plt.savefig("explore/m4_vessel_diameter_time_series.png")
print("Saved vessel diameter time series")

# Calculate and plot the power spectrum to assess pulsatility
print("Analyzing vessel pulsatility...")

# Remove NaN values if any
valid_diameters = vessel_diameters[~np.isnan(vessel_diameters)]
if len(valid_diameters) < 10:
    print("Not enough valid diameter measurements for frequency analysis")
else:
    # Detrend the diameter time series
    detrended_diameters = signal.detrend(valid_diameters)
    
    # Calculate the power spectrum
    fps = movies.rate  # frames per second
    freqs, psd = signal.welch(detrended_diameters, fps, nperseg=min(256, len(detrended_diameters)//2))
    
    # Plot the power spectrum
    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs, psd)
    plt.title('Power Spectrum of Vessel Diameter Variations (M4 Dataset)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("explore/m4_vessel_diameter_power_spectrum.png")
    print("Saved vessel diameter power spectrum")
    
    # Identify dominant frequencies
    peak_indices = signal.find_peaks(psd, height=0)[0]
    if len(peak_indices) > 0:
        dominant_freq_idx = peak_indices[np.argmax(psd[peak_indices])]
        dominant_freq = freqs[dominant_freq_idx]
        print(f"Dominant frequency: {dominant_freq:.2f} Hz (approximately {dominant_freq*60:.1f} cycles per minute)")
    else:
        print("No clear dominant frequency found")

# Calculate basic statistics
mean_diameter = np.nanmean(vessel_diameters)
std_diameter = np.nanstd(vessel_diameters)
min_diameter = np.nanmin(vessel_diameters)
max_diameter = np.nanmax(vessel_diameters)
cv = std_diameter / mean_diameter * 100  # Coefficient of variation

print("\nVessel Diameter Statistics:")
print(f"Mean: {mean_diameter:.2f} pixels")
print(f"Standard Deviation: {std_diameter:.2f} pixels")
print(f"Coefficient of Variation: {cv:.2f}%")
print(f"Min: {min_diameter:.2f} pixels")
print(f"Max: {max_diameter:.2f} pixels")
print(f"Range: {max_diameter - min_diameter:.2f} pixels")