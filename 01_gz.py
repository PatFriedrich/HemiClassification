import os
from os import system as oss

### dir
mdir = "../symm_mwp1_maps/"

### loop
for e in sorted(os.listdir(mdir)):
    if e.endswith(".nii"):
        epath = mdir+"/"+e
        # gunzip
        oss('gzip {inp}'.format(inp=epath))


    
