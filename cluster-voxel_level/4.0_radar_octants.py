"""
Plot Radar plot of Octants
1. the outer circle is defined by 7 cortical regions
2. 8 colors define the 8 octants
3. the value at the vertex is the count of significant voxels against the number of voxels in this cortical area
"""
import numpy as np
import os
import argparse
import nibabel as nib
from nilearn.image import resample_to_img
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns 

root_path = '/Users/anmin/Documents/attsig_data'
atlas_path = os.path.join(root_path, 'Yeo_JNeurophysiol11_MNI152')
data_path = os.path.join(root_path, 'nii_file')

mask = nib.load(os.path.join(atlas_path, 
            'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz')) 
nan_mask = nib.load(os.path.join(root_path,
                                'nii_file',
                                'mask_nan.nii.gz')).get_data().flatten()
nan_mask = np.nan_to_num(nan_mask).astype(bool)
                                 

conjunction_map = nib.load(os.path.join(data_path, 
                                'conjunction_map_ml.nii.gz')) 
resampled_mask = resample_to_img(mask, conjunction_map).get_data().squeeze()\
                    .flatten()[nan_mask]

conjunction_map = conjunction_map.get_data().flatten()[nan_mask]
# 1: conjunction; 2: fa; 3: sa

# load octant mask 
octant_mask = np.load(os.path.join(root_path, 'octant_mask.npy')) 

holder = {}
junction_idxes = np.where(conjunction_map==1)[0]
for octant in range(1,9):
    if octant==3 or octant==7:
        continue
    holder[octant] = {}
    octant_idxes = np.where(octant_mask == octant)[0]
    for network in range(1, 8):
        network_idxes = np.where(resampled_mask==network)[0]
        junction_num = np.intersect1d(junction_idxes,
                                    np.intersect1d(octant_idxes,network_idxes)).size
        juction_ratio = junction_num / network_idxes.size
        holder[octant][network] = juction_ratio
    
# compute the proportion to the highest value
for octant in range(1,9):
    if octant==3 or octant==7:
        continue
    max_value = max(holder[octant].values())

    for network in range(1, 8):
        holder[octant][network] /= max_value

# plot
save_path = os.path.join(root_path, 
                         'figure',
                         'octant_radar')
net_name = ['Visual',
            'Somatomotor',
            'Dorsal Attention',
            'Ventral Attention',
            'Limbic',
            'Frontoparietal',
            'Default']
pal = sns.color_palette('hls',8)
idx_color_mapper = {1 : pal[0],
                    2 : pal[1],
                    4 : pal[2],
                    5 : pal[4],
                    6 : pal[-1],
                    8 : pal[5]}

angles = np.linspace(0, 2*np.pi, len(net_name), endpoint=False)
angles=np.concatenate((angles,[angles[0]]))
net_name.append(net_name[0])

for octant in holder.keys():
    plt.style.use('seaborn-muted')
    fig=plt.figure()
    ax=fig.add_subplot(polar=True)

    vals = list(holder[octant].values())
    vals.append(vals[0])

    ax.plot(angles, vals, color=idx_color_mapper[octant], 
            linewidth=3, 
            label=f'Octant {octant}')

    ax.set_thetagrids(angles * 180/np.pi, net_name)
    plt.grid(True)
    plt.tight_layout()
    ax.axes.yaxis.set_ticks([0.5])
    plt.ylim(0,1)
    plt.tick_params(labelsize=2)

    plt.savefig(os.path.join(save_path, 
                             f'octant_{octant}.png'), 
                dpi=500)
    plt.close()


