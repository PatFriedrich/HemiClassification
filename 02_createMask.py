### modules
import glob
import nilearn as nl
from nilearn import image
import numpy as np
import pandas as pd
import os
from os import system as oss

### input paths (all images)
datapaths = sorted(glob.glob('../symm_mwp1_maps/mwp*.nii.gz'))

### generate code
code = "fslmaths"
for e in datapaths:
    code = code+" "+e+" -thr 0.1 -add"
code = code[:-4]+"mask.nii.gz"
print(code)
### run code to generate mask
oss(code)

### binarize & threshold mask
oss('fslmaths mask.nii.gz -thr 1 -bin mask_bin.nii.gz -odt char')
