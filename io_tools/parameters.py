import json
#import numpy as np

# Function to read microscope.json parameters
def _read_microscope_json(_microscope_file:str,
    ) -> dict:
    return json.load(open(_microscope_file, 'r'))

# Function to read color_usage file