"""
Univariate Analysis of Beta Maps 
1. use both train and test data 
2. matched-sampled t-test 
3. activation is demostrated as t-value 
4. masked out non-significant voxels in two levels:
    1. FDR correction at q < 0.05
    2. FWE with bonfferoni correction at p < 0.05
5. bask brain based on cohen's d 
"""
import numpy as np
import os
from algo import *

from nilearn import image
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import fdrcorrection, multipletests

data_path = '/data/home/attsig/attention_signature/attention_from_liu/\
attention_data_complete/train_test_data'

mask_nan = np.load('/data/home/attsig/attention_signature/\
train_test_data/mask_NaN.npy')

# FA Analysis 
## load data  
train_arr = np.load(os.path.join(data_path, 
                                'train_X_fa.npy'))
test_arr = np.load(os.path.join(data_path, 
                                'test_X_fa.npy'))
## concate subject data 
sub_arr = np.vstack((train_arr, test_arr))
sub_arr_0 = sub_arr[::2]
sub_arr_1 = sub_arr[1::2]

t_val, p_val = ttest_rel(sub_arr_0, sub_arr_1, axis=0)

## two levels of p_val 
### FDR Correction 
mask_fdr, _ = fdrcorrection(p_val, alpha=0.05)

### FWE Correction 
mask_fwe = multipletests(p_val, alpha=0.05, method='bonferroni')[0]

# gen nii file 
def gen_nii_file(mask, activation, mask_nan, save_path):
    masked_back_beta = mask_back(mask,
                            activation,
                            mask_type='mni')
    masked_back_mni = mask_back(mask_nan,
                            masked_back_beta,
                            mask_type='mni')
    nii_file = to_mni(masked_back_mni)

    nib.save(nii_file, save_path)
 

save_path_r = os.path.join('/data/home/attsig/attention_signature/svc_analysis',
                        'nii_file',
                        'univariate_analysis')
fdr_path = os.path.join(save_path_r,
                        'fa_fdr_corrected.nii.gz')
fwe_path = os.path.join(save_path_r,
                        'fa_fwe_corrected.nii.gz')


gen_nii_file(mask_fdr, t_val, mask_nan, fdr_path)
gen_nii_file(mask_fwe, t_val, mask_nan, fwe_path)

### cohen'd 
cohen_arr = compare_cohen_d(sub_arr_0, sub_arr_1)
masked_back_mni = mask_back(mask_nan,
                            cohen_arr,
                            mask_type='mni')
nii_file = to_mni(masked_back_mni)
nib.save(nii_file, os.path.join(save_path_r,
                                'fa_cohen.nii.gz'))

# SA Analysis 
## load data  
train_arr = np.load(os.path.join(data_path, 
                                'train_X_sa.npy'))
test_arr = np.load(os.path.join(data_path, 
                                'test_X_sa.npy'))
## concate subject data 
sub_arr = np.vstack((train_arr, test_arr))
sub_arr_0 = sub_arr[::2]
sub_arr_1 = sub_arr[1::2]

t_val, p_val = ttest_rel(sub_arr_0, sub_arr_1, axis=0)

## two levels of p_val 
### FDR Correction 
mask_fdr, _ = fdrcorrection(p_val, alpha=0.05)

### FWE Correction 
mask_fwe = multipletests(p_val, alpha=0.05, method='bonferroni')[0]

# gen nii file 
fdr_path = os.path.join(save_path_r,
                        'sa_fdr_corrected.nii.gz')
fwe_path = os.path.join(save_path_r,
                        'sa_fwe_corrected.nii.gz')


gen_nii_file(mask_fdr, t_val, mask_nan, fdr_path)
gen_nii_file(mask_fwe, t_val, mask_nan, fwe_path)

### cohen'd 
cohen_arr = compare_cohen_d(sub_arr_0, sub_arr_1)
masked_back_mni = mask_back(mask_nan,
                            cohen_arr,
                            mask_type='mni')
nii_file = to_mni(masked_back_mni)
nib.save(nii_file, os.path.join(save_path_r,
                                'sa_cohen.nii.gz'))