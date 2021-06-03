### modules
import os
import glob
import nilearn as nl
from nilearn import image
import numpy as np
import pandas as pd

### directories
datapaths = sorted(glob.glob('../symm_mwp1_maps/mwp*.nii.gz'))

### vessel
nl_img = nl.image.threshold_img('mask.nii.gz', 1)
nl_img = nl_img.get_fdata()
all_img = np.zeros([len(datapaths), nl_img.shape[0], nl_img.shape[1], nl_img.shape[2]])
all_img[1,:,:,:].shape

### read in all images
for subj in range(len(datapaths)):
    img = nl.image.threshold_img(datapaths[subj], 0.1)
    imgX = img.get_fdata()
    all_img[subj,:,:,:] = imgX
all_img.shape

### split hemispheres
all_rh = all_img[:,:71, :, :]
all_lh = all_img[:,72:, :, :]
all_lh_flipped = np.flip(all_lh, 1)

print(all_rh.shape)
print(all_lh_flipped.shape)
### looki looki
from matplotlib import pyplot as plt

z=59
fig=plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax1.imshow(all_rh[0,:,:,z], cmap='hot')
ax2 = fig.add_subplot(3,1,2)
ax2.imshow(all_lh[0,:,:,z], cmap='hot')
ax3 = fig.add_subplot(3,1,3)
ax3.imshow(all_lh_flipped[0,:,:,z], cmap='hot')

### Reshape 4D matrix (all subj 3D images) to 2D matrix (all subj 1D vector)
# right hemispheres
all_rh_2d = np.reshape(all_rh, (all_rh.shape[0], all_rh.shape[1]*all_rh.shape[2]*all_rh.shape[3]))
print(all_rh_2d.shape)
# left_flipped hemispheres
all_lh_flipped_2d = np.reshape(all_lh_flipped, (all_lh_flipped.shape[0], all_lh_flipped.shape[1]*all_lh_flipped.shape[2]*all_lh_flipped.shape[3]))
print(all_lh_flipped_2d.shape)
# concatenate the two matrices --> RHs top X rows and LHs bottom X rows
X = np.concatenate((all_rh_2d, all_lh_flipped_2d), axis=0); print(X.shape)

### save as np
np.savetxt('RHsLHs_2D.csv', X, delimiter=";")

### save as hdf
df = pd.DataFrame(X)
df.to_hdf('RHsLhs.hdf', key='df', mode='w')

### outro
print('done')