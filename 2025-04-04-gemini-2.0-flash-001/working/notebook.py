# %% [markdown]
# # AI-generated notebook for exploring Dandiset 001366
#
# This notebook was AI-generated using dandi-notebook-gen and has not been fully verified.
# Users should be cautious when interpreting the code or results.
#
# This notebook provides an introduction to Dandiset 001366 and demonstrates how to access and visualize sample data from NWB files.
# It includes explanatory markdown cells that guide the user through the analysis process and provides examples of common analyses that might be relevant to the dataset's content.
#
# Before using this notebook, please make sure you have the following packages installed:
# ```bash
# pip install pynwb lindi matplotlib seaborn
# ```

# %% [markdown]
# ## Introduction to Dandiset 001366
#
# Dandiset 001366, named "Comparison of Approaches for Surface Vessel Diameter and Pulsatility Quantification", contains movies of a pial vessel of mice used in the experiments.
#
# Key metadata:
# - Description: movies of a pial vessel of mice used in the experiments.
# - Keywords: full width at half maximum, vessel pulsation, image analysis, vessel diameter, radon transform
# - Contributor: Zhao, Yue, Ranjan, Aditya, Wong, Devin T., Huang, Qinwen, Ghanizada, Hashmat, Nedergaard, Maiken , Kelley, Douglas H. , Boster, Kimberly A. S., National Institute of Health, National Institute of Health, United States Army
# - License: spdx:CC-BY-4.0
#
# More information can be found at https://dandiarchive.org/dandiset/001366

# %%
from dandi.dandiapi import DandiAPIClient

client = DandiAPIClient()
dandiset = client.get_dandiset("001366")
assets = list(dandiset.get_assets())
print(f"Number of assets in the Dandiset: {len(assets)}")

# %% [markdown]
# ## Exploring the dataset's structure
#
# The Dandiset contains the following assets:

# %%
for asset in assets:
    print(f"- {asset.path}")

# %% [markdown]
# ## Accessing and visualizing sample data from NWB files
#
# We will now access and visualize sample data from one of the NWB files in the Dandiset: `sub-031224-M4/sub-031224-M4_ses-03122024-m4-baseline_image.nwb`.

# %%
import pynwb
import lindi
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load https://api.dandiarchive.org/api/assets/2f12bce3-f841-46ca-b928-044269122a59/download/
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001366/assets/2f12bce3-f841-46ca-b928-044269122a59/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")

movies = nwb.acquisition["Movies"]
print(f"Movies starting time: {movies.starting_time}")
print(f"Movies rate: {movies.rate}")
print(f"Movies shape: {movies.data.shape}")

# %% [markdown]
# ## Visualizing frames from the movie
#
# The following code visualizes three frames from the movie. Although the images are somewhat blurry, the vascular structure is clearly visible.

# %%
data = movies.data

# Plot three frames from the movie
frames_to_plot = [0, 2000, 4000]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, frame_index in enumerate(frames_to_plot):
    frame = data[frame_index, :, :]
    axes[i].imshow(frame, cmap='gray')
    axes[i].set_title(f"Frame {frame_index}")
    axes[i].axis('off')

plt.show()

# %% [markdown]
# ## Examples of common analyses
#
# This notebook provides a basic introduction to the dataset and demonstrates how to load and visualize the data.
# The user can perform further analyses, such as:
# - Measuring vessel diameter and pulsatility using image analysis techniques.
# - Applying Radon transform to analyze vessel orientation.
# - Analyzing the full width at half maximum of the vessel signal.
#
# It is important to remember that this notebook was AI-generated and has not been fully verified.
# Users should be cautious when interpreting the code or results and should consult with experts in the field for further analysis.