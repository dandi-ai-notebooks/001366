# Purpose: Load an NWB file and print basic information.

import pynwb
import lindi

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