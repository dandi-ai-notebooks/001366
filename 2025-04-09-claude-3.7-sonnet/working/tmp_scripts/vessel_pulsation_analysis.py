"""
This script analyzes vessel pulsatility in the NWB files from Dandiset 001366.
It focuses on:
1. Temporal analysis of vessel diameter or intensity changes
2. Frequency analysis to identify pulsation frequencies
3. Visualization of pulsatility patterns
"""

import matplotlib.pyplot as plt
import numpy as np
import pynwb
import h5py
import remfile
from scipy import ndimage, signal
from skimage.transform import radon
import os
from scipy.signal import find_peaks, peak_widths, welch
from scipy.ndimage import gaussian_filter
from scipy.fftpack import fft

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

# Number of frames to sample (use a subset for efficiency)
num_frames = 1000
sample_indices = np.linspace(0, movies.data.shape[0]-1, num_frames, dtype=int)

# Find vessel orientation for the first frame
middle_frame_idx = movies.data.shape[0] // 2
middle_frame = movies.data[middle_frame_idx, :, :]

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
    
    return angle

# Get vessel angle
angle = find_vessel_angle(middle_frame)
print(f"Estimated vessel angle: {angle:.2f} degrees")

# Extract a time series of vessel profiles
middle_row = middle_frame.shape[0] // 2  # Use middle row
intensity_profiles = []
rotated_frames = []

for idx in sample_indices:
    frame = movies.data[idx, :, :]
    rotated_frame = ndimage.rotate(frame, angle - 90, reshape=False)
    rotated_frames.append(rotated_frame)
    intensity_profiles.append(rotated_frame[middle_row, :])

# Convert to numpy array for easier manipulation
intensity_profiles = np.array(intensity_profiles)

# Create a spatiotemporal map (kymograph)
plt.figure(figsize=(10, 8))
plt.imshow(intensity_profiles, aspect='auto', cmap='viridis',
           extent=[0, intensity_profiles.shape[1], 
                   sample_indices[-1]/movies.rate, sample_indices[0]/movies.rate])
plt.title("Vessel Kymograph (Spatiotemporal Map)")
plt.xlabel("Position (pixels)")
plt.ylabel("Time (seconds)")
plt.colorbar(label="Intensity")
plt.savefig("tmp_scripts/vessel_kymograph.png")
plt.close()

# Analyze the time series at a specific position
# Find the position where the vessel is located (dark region)
mean_profile = np.mean(intensity_profiles, axis=0)
vessel_pos = np.argmin(mean_profile)  # Position with minimum intensity (vessel center)

# Extract intensity time series at vessel center
vessel_center_ts = intensity_profiles[:, vessel_pos]
time_points = sample_indices / movies.rate  # Convert to seconds

# Plot vessel center intensity time series
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time_points, vessel_center_ts)
plt.title(f"Vessel Center Intensity Over Time (Position {vessel_pos})")
plt.xlabel("Time (seconds)")
plt.ylabel("Intensity")
plt.grid(alpha=0.3)

# Frequency analysis using FFT
# Remove mean (detrend)
ts_detrended = vessel_center_ts - np.mean(vessel_center_ts)
n = len(ts_detrended)
dt = (sample_indices[-1] - sample_indices[0]) / (movies.rate * (len(sample_indices) - 1))
freq = np.fft.fftfreq(n, d=dt)
freq_half = freq[:n//2]
magnitude = np.abs(np.fft.fft(ts_detrended))
magnitude_half = magnitude[:n//2]

plt.subplot(2, 1, 2)
plt.plot(freq_half, magnitude_half)
plt.title("Frequency Spectrum of Vessel Center Intensity")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(alpha=0.3)
plt.xlim(0, 2)  # Focus on frequencies up to 2 Hz (physiologically relevant)
plt.tight_layout()
plt.savefig("tmp_scripts/vessel_pulsation_frequency.png")
plt.close()

# Use Welch's method for better frequency resolution
f, Pxx = welch(ts_detrended, fs=1/dt, nperseg=256)

plt.figure(figsize=(10, 6))
plt.semilogy(f, Pxx)
plt.title("Power Spectral Density (Welch's Method)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power/Frequency (dB/Hz)")
plt.grid(alpha=0.3)
plt.xlim(0, 2)  # Focus on physiologically relevant frequencies
plt.axvline(f[np.argmax(Pxx[f < 2])], color='r', linestyle='--', 
            label=f"Peak frequency: {f[np.argmax(Pxx[f < 2])]*60:.1f} cycles/minute")
plt.legend()
plt.tight_layout()
plt.savefig("tmp_scripts/vessel_psd.png")

# Create heatmap of frequency content along the vessel
# Use a sliding window approach across positions
position_freq_maps = []
peak_freqs = []
positions = range(0, intensity_profiles.shape[1], 5)  # Sample positions at regular intervals

for pos in positions:
    ts = intensity_profiles[:, pos]
    ts_detrended = ts - np.mean(ts)
    f, Pxx = welch(ts_detrended, fs=1/dt, nperseg=256)
    position_freq_maps.append(Pxx)
    
    # Find peak frequency (if exists)
    if np.max(Pxx[f < 2]) > 0:
        peak_freq = f[np.argmax(Pxx[f < 2])] * 60  # Convert to cycles/minute
        peak_freqs.append(peak_freq)
    else:
        peak_freqs.append(0)

# Convert to numpy arrays
position_freq_maps = np.array(position_freq_maps)
peak_freqs = np.array(peak_freqs)

# Plot frequency content heatmap
plt.figure(figsize=(12, 8))

# Plot mean intensity profile to identify vessel location
plt.subplot(3, 1, 1)
plt.plot(mean_profile)
plt.title("Mean Intensity Profile")
plt.xlabel("Position (pixels)")
plt.ylabel("Intensity")
plt.axvline(vessel_pos, color='r', linestyle='--', label=f"Vessel Center (pos {vessel_pos})")
plt.legend()
plt.grid(alpha=0.3)

# Plot peak frequency at each position
plt.subplot(3, 1, 2)
plt.plot(positions, peak_freqs)
plt.title("Peak Frequency at Different Positions")
plt.xlabel("Position (pixels)")
plt.ylabel("Frequency (cycles/minute)")
plt.grid(alpha=0.3)
plt.axvline(vessel_pos, color='r', linestyle='--')

# Plot frequency content heatmap
plt.subplot(3, 1, 3)
extent = [0, intensity_profiles.shape[1], f[-1], f[0]]
plt.imshow(position_freq_maps.T, aspect='auto', origin='lower', 
           extent=extent, cmap='viridis')
plt.title("Frequency Content Along Vessel Profile")
plt.xlabel("Position (pixels)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Power")
plt.axvline(vessel_pos, color='r', linestyle='--')

plt.tight_layout()
plt.savefig("tmp_scripts/vessel_frequency_analysis.png")
plt.close()

print("Script execution completed successfully")