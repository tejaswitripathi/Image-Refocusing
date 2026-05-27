from coc_map import getMetadata, generate_coc_map
import os
import json
import numpy as np

datadir = "bedroom/dataset/"
contents = os.listdir(datadir)
print(contents)

for folder in contents:
    if folder == '.DS_Store':
        continue
    filepath = datadir + folder + "/"
    metadata = getMetadata(filepath)
    coc_px = generate_coc_map(metadata)
    coc_px = np.nan_to_num(
        coc_px,
        nan=0.0,
        posinf=25.0,
        neginf=0.0
    )
    with open(filepath + "coc.json", 'w') as f:
        json.dump(coc_px.tolist(), f)