"""
Compute the correlation of conjuction clusters of FA and SA 

1. loop through cluster table 
2. get the cluster mask of fa and sa 
3. get the conjuction mask
4. use the conjunction mask to get beta weights of FA and SA 
5. compute the correlation of beta weights of FA and SA
"""
import numpy as np
import pandas as pd 
import os 

from algo import *
from nilearn import image
import scipy.stats as stats
from sklearn.preprocessing import scale

root_path = '/data/home/attsig/attention_signature'
nii_path = os.path.join(root_path, 
                        'svc_analysis', 
                        'nii_file', 
                        'cluster_map_small_version')
df_path = os.path.join(root_path, 
                       'svc_analysis', 
                       'cluster_table_small_version')

# load dataframe
df_data_fa = pd.read_csv(os.path.join(df_path,
                                    'fa_integrated.csv'))
df_data_sa = pd.read_csv(os.path.join(df_path,
                                    'sa_integrated.csv'))

# NaN mask 
mask_NaN = np.load(os.path.join(root_path, 'train_test_data',
                                        'mask_NaN.npy'))

# load beta maps 
beta_path = os.path.join(root_path, 
                         'svc_analysis', 
                         'prediction_results')
beta_fa = np.load(os.path.join(beta_path, 
                               'fa',
                               'avg_beta.npy'))
beta_sa = np.load(os.path.join(beta_path,
                                'sa',
                                'avg_beta.npy'))
beta_fa, beta_sa = scale(beta_fa), scale(beta_sa)

def load_mask(series, data_type):
    cluster_idx = series['Unnamed: 0']
    num_voxel = series['number_of_voxels']

    nii_arr = image.get_data(os.path.join(nii_path, 
                                          f'{data_type}_screened',
                f'{cluster_idx}_{num_voxel}_{data_type}.nii.gz')) \
                .flatten()     
    nii_arr = nii_arr[mask_NaN]

    return nii_arr

# data holder in numpy format
corr_data = np.empty((df_data_sa.shape[0], 3))
names = np.array(df_data_sa['cluster_name'])
corr_data = np.concatenate((names.reshape(-1,1), corr_data), axis=1)

# four columns:
# 1. cluster name 
# 2. conjunction number of voxels
# 3. correlation of FA and SA
# 4. p-value of correlation of FA and SA
# 5. Cohen's d
for index, series_sa in df_data_sa.iterrows(): 
    #series_sa = df_data_sa.iloc[5]
    # find the reference row in fa 
    series_fa = df_data_fa[df_data_fa['Unnamed: 0'] == series_sa['Unnamed: 0']] \
                        .iloc[0]

    # load the cluster mask of fa and sa
    fa_mask = ~np.isnan(load_mask(series_fa, 'fa'))
    sa_mask = ~np.isnan(load_mask(series_sa, 'sa'))

    # compute the conjunction boolean mask of FA and SA
    conj_mask = np.logical_and(fa_mask, sa_mask)
    num_conj = np.sum(conj_mask)
    corr_data[index, 1] = num_conj
    if np.sum(conj_mask) <=1:
        continue

    # mask the beta maps of FA and SA
    beta_fa_masked = beta_fa[conj_mask]
    beta_sa_masked = beta_sa[conj_mask]

    # compute correlation 
    print(f"computing {series_fa['cluster_name']}")
    r, p = stats.pearsonr(beta_fa_masked, beta_sa_masked)
    #d = r * np.sqrt((num_conj-2) / (1-r**2))
    corr_data[index, 2] = r
    corr_data[index, 3] = np.round(p, 2)
    #corr_data[index, 4] = d

# convert to dataframe
df_corr = pd.DataFrame(corr_data,
                          columns=['cluster_name',
                                      'num_voxel',
                                      'r',
                                        'p_value',
                                        #'Cohen\'s d'
                                        ])
df_corr.to_csv(os.path.join(root_path,
                            'svc_analysis',
                            'corr_table',
                            'conjunction_corr.csv'))
