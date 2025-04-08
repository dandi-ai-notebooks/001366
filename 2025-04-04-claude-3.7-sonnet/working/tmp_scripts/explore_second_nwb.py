"""
Script to explore the basic structure of the second NWB file from Dandiset 001366.
This script will extract information about the image data structure, dimensions,
and other key properties to help inform our notebook development.
"""
import pynwb
import lindi
import numpy as np

# Load the NWB file
print("Loading second NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001366/assets/71fa07fc-4309-4013-8edd-13213a86a67d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print basic information about the file
print("\nBasic information about the file:")
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Experimenter: {nwb.experimenter}")
print(f"Experiment description: {nwb.experiment_description}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject species: {nwb.subject.species}")
print(f"Subject sex: {nwb.subject.sex}")
print(f"Subject age: {nwb.subject.age}")

# Get the movies
print("\nImage data information:")
movies = nwb.acquisition["Movies"]
print(f"Movies name: {movies.name}")
print(f"Movies description: {movies.description if hasattr(movies, 'description') else 'No description'}")
print(f"Movies starting time: {movies.starting_time} sec")
print(f"Movies rate: {movies.rate} Hz")

# Get information about the image dimensions and data type
print("\nImage dimensions and type:")
print(f"Image data type: {movies.data.dtype}")
print(f"Image dimensions: {movies.data.shape}")

# Get information about the first frame
print("\nFirst frame information:")
first_frame = movies.data[0]
print(f"First frame shape: {first_frame.shape}")
print(f"First frame min value: {np.min(first_frame)}")
print(f"First frame max value: {np.max(first_frame)}")
print(f"First frame mean value: {np.mean(first_frame)}")

# Print other available fields in the NWB file
print("\nAvailable acquisition fields:")
for field in nwb.acquisition:
    print(f"  - {field}")

print("\nAvailable processing modules:")
if hasattr(nwb, 'processing') and nwb.processing:
    for module_name in nwb.processing:
        module = nwb.processing[module_name]
        print(f"  - {module_name}")
        for data_interface_name in module.data_interfaces:
            print(f"    - {data_interface_name}")
else:
    print("  No processing modules found")

print("\nAvailable analysis fields:")
if hasattr(nwb, 'analysis') and nwb.analysis:
    for field in nwb.analysis:
        print(f"  - {field}")
else:
    print("  No analysis fields found")