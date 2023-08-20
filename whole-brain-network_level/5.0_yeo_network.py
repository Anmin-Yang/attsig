"""
Summarize significant voxels in using Yeo Atlas 
"""
import numpy as np
import os
import argparse
import nibabel as nib
from nilearn.image import resample_to_img

parser = argparse.ArgumentParser()
parser.add_argument("type",
                    help="{'fa', 'sa}",
                    type=str)

if __name__ == '__main__':
    args = parser.parse_args()


    root_path = '/Users/anmin/Documents/attsig_data'
    atlas_path = os.path.join(root_path, 'Yeo_JNeurophysiol11_MNI152')
    data_path = os.path.join(root_path, 'nii_file')

    # load region mask 
    mask = nib.load(os.path.join(atlas_path, 
            'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz')) 

    beta = nib.load(os.path.join(data_path, 
                                    f'{args.type}_fdr_corrected.nii.gz'))

    resampled_beta = resample_to_img(beta, mask)

    mask_arr = np.array(mask.dataobj).astype('int32')
    mask_arr = np.squeeze(mask_arr).flatten()
    beta_arr = np.nan_to_num(np.array(resampled_beta.dataobj).flatten())


    beta_to_atlas = {}
    counter = 0
    for i, val in enumerate(mask_arr):
        base = beta_to_atlas.get(val, 0)
        beta_to_atlas[val] = base + abs(beta_arr[i])
    
    save_path = os.path.join(root_path, 'yoe_dic',
                            f'{args.type}_sum.npy')
    
    np.save(save_path, beta_to_atlas)









