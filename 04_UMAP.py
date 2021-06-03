##### import modules
# standard
import umap
import glob
import nilearn as nl
from nilearn import image
import numpy as np
import pandas as pd
from inspect import getmembers, isfunction
# figures
import seaborn as sns
from matplotlib import pyplot as plt
# ML
from julearn import run_cross_validation
from julearn.utils import configure_logging
# linear SVC for getting the feature weights
from sklearn.svm import LinearSVC
# saving dictionaries
import pickle
def save_obj(obj, name ):
    with open(str(name) + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name ):
    with open(str(name) + '.pkl', 'rb') as f:
        return pickle.load(f)
# z-standardizing
from scipy.stats import zscore


##### i/o and z-transform
df = pd.read_hdf('RHsLhs.hdf')
### z-transform rows
dfz = df.apply(zscore, axis=1, result_type='expand')
df

##### UMAP - first 2 dimensions
emb_2d = umap.UMAP(random_state=7, n_neighbors=30, min_dist=0.0,
                   n_components=2).fit_transform(dfz)
# save results
save_obj(emb_2d, './04z_results/emb_2d')


##### visualize results :)

### create pd.DataFrame
# empty DataFrame
index = np.array(range(0, 205))
index.shape
df_ = pd.DataFrame(index=index, columns=['rh1', 'rh2', 'lh1', 'lh2'])
# fill empty DataFrame
df_['rh1']=emb_2d[:int(emb_2d.shape[0]/2), 0]
df_['rh2']=emb_2d[:int(emb_2d.shape[0]/2), 1]
df_['lh1']=emb_2d[int(emb_2d.shape[0]/2):, 0]
df_['lh2']=emb_2d[int(emb_2d.shape[0]/2):, 1]


df_


#### visualization
fig, ax = plt.subplots(2,1, figsize=(10, 10), sharey=False,)
# kdeplot
sns.kdeplot(x=df_.lh1, y=df_.lh2, ax=ax[0], cmap='Blues', shade=False, thresh=0.1)
sns.kdeplot(x=df_.rh1, y=df_.rh2, ax=ax[0], cmap='Oranges', shade=False, thresh=0.1)
# scatterplot
sns.scatterplot(x='lh1', y='lh2', data=df_, ax=ax[1], cmap='Blues')
sns.scatterplot(x='rh1', y='rh2', data=df_, ax=ax[1], cmap='Blues')
# show
plt.show()
