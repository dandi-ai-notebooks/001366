"""
This script explores the basic metadata of the NWB files in the Dandiset.
It will print out key information about both NWB files to help understand 
what they contain and how they're structured.
"""

import pynwb
import h5py
import remfile
import numpy as np

# URLs for both NWB files
url1 = "https://api.dandiarchive.org/api/assets/71fa07fc-4309-4013-8edd-13213a86a67d/download/"
url2 = "https://api.dandiarchive.org/api/assets/2f12bce3-f841-46ca-b928-044269122a59/download/"

def print_nwb_info(url, file_name):
    print(f"\n==== NWB File: {file_name} ====")
    print(f"URL: {url}")
    
    # Load the file
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()
    
    # Print basic metadata
    print(f"Session Description: {nwb.session_description}")
    print(f"Identifier: {nwb.identifier}")
    print(f"Session Start Time: {nwb.session_start_time}")
    print(f"Experimenter: {', '.join(nwb.experimenter)}")
    print(f"Institution: {nwb.institution}")
    print(f"Keywords: {nwb.keywords[:]}")
    print(f"Experiment Description: {nwb.experiment_description}")
    
    # Print subject information
    print("\nSubject Information:")
    print(f"  Subject ID: {nwb.subject.subject_id}")
    print(f"  Age: {nwb.subject.age}")
    print(f"  Sex: {nwb.subject.sex}")
    print(f"  Species: {nwb.subject.species}")
    print(f"  Strain: {nwb.subject.strain}")
    
    # Print acquisition information
    print("\nAcquisition Data:")
    for name, data in nwb.acquisition.items():
        print(f"  {name}:")
        print(f"    Type: {type(data).__name__}")
        if hasattr(data, 'description'):
            print(f"    Description: {data.description}")
        if hasattr(data, 'data'):
            print(f"    Data Shape: {data.data.shape}")
            print(f"    Data Type: {data.data.dtype}")
        if hasattr(data, 'rate'):
            print(f"    Rate: {data.rate} Hz")
        
    # Close the file
    h5_file.close()

# Process both files
print_nwb_info(url1, "sub-F15_ses-F15BC-19102023_image.nwb")
print_nwb_info(url2, "sub-031224-M4_ses-03122024-m4-baseline_image.nwb")