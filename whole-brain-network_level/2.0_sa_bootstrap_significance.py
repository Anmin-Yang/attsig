"""
Use bootstrapped data to generate Boolean Masks 
at different significance level 

1. concate bootstrapped beta weights 
2. statistical inferernce at different levels with boolean masks (saved)
3. nii file after masked 
    a. the brain beta weights is first z-scored
    b. masked out none-significant voxels 
"""
import numpy as np
import os
from algo import *

from sklearn.preprocessing import scale 
from statsmodels.stats.multitest import fdrcorrection, multipletests

# define path 
root_path = '/data/home/attsig/attention_signature/svc_analysis'
boot_path = os.path.join(root_path, 'beta_bootstrap') 
weight_path = os.path.join(root_path, 'prediction_results')
mask_path = os.path.join(root_path, 'masks')

# load data 
weights = np.load(os.path.join(weight_path, 'sa', 'beta.npy'), 
                    allow_pickle=True).item() 
## average weights  
avg_weights = np.zeros(weights[0].size)
for weight in weights.values():
    avg_weights = np.vstack((avg_weights, weight))     
avg_weights = avg_weights[1:, :]
avg_weights = np.mean(avg_weights, axis=0)
## concate bootstrap data 
boot_time = 5
boot_arr = np.zeros(avg_weights.size)
for i in range(boot_time):
    temp_dic = np.load(os.path.join(boot_path, f'sa_boot_{i}.npy'),
                        allow_pickle=True).item()
    for val in temp_dic.values():
        boot_arr = np.vstack((boot_arr, val))
boot_arr = boot_arr[1:,:]

# statistical inference 
## arbitary alpha value
alpha_pool = [0.05,0.01,0.005,0.001]
for alpha_level in alpha_pool:
    voxel_mask = stat_infer(boot_arr,alpha=alpha_level)  
    num_passed = np.sum(voxel_mask != 0)
    np.save(os.path.join(mask_path, f'sa_{alpha_level}.npy'),
            voxel_mask)
    print(f'At alpha level {alpha_level},\
         the number of passed voxels is {num_passed} out of {voxel_mask.size}.')

## FDR correction 
p_val = p_value_continue(boot_arr)

mask_fdr, _ = fdrcorrection(p_val,alpha=0.05)
print(f'passed voxel num: {np.sum(mask_fdr)}')
np.save(os.path.join(mask_path, 'sa_fdr.npy'), mask_fdr)

## FWE correction 
# Bonferroni 
mask_fwe = multipletests(p_val,alpha=0.05,method='bonferroni')[0]
print(f'passed voxel num: {np.sum(mask_fwe)}')
np.save(os.path.join(mask_path, 'sa_bonferroni.npy'), mask_fwe)


# save masked brain image 
## I only save mask with fdr correction 
## z_score 
avg_weights = scale(avg_weights)

## mask out 
masked_activation = avg_weights[mask_fdr]
masked_back_beta = mask_back(mask_fdr,
                            masked_activation,
                            mask_type='beta')

mask_nan = np.load('/data/home/attsig/attention_signature/\
train_test_data/mask_NaN.npy')
masked_back_mni = mask_back(mask_nan,
                            masked_back_beta,
                            mask_type='mni')
masked_back_mni = np.nan_to_num(masked_back_mni)

# save nii file 
nii_file = to_mni(masked_back_mni)
nib.save(nii_file,
        os.path.join(root_path,
                    'brain_img',
                    'main_map',
                    'sa_fdr_corrected.nii.gz'))