"""
1. get the full list of the subjects remotely 

The following computed locally:
2. Subject Information:
    a. gender
    b. age
2. search for the file location of specific subject 
3. read .mat file and interpreate accuracy based on history code(refer to jinhua code if possible)
4. list of accurcy of each conditon in either experiment(4 lists in total)
5. compute mean and std 
"""
import numpy as np 
import pandas as pd 
import os 
from algo import cohen_d

from collections import Counter
from scipy.stats import ttest_rel

# Get sub names 
root_path = '/Users/anmin/Documents/attsig_data'
s_lst = np.load(os.path.join(root_path,
                             'sub_info',
                             'sub_names_used.npy'))
# Get sub info 
df_info = pd.read_excel(os.path.join(root_path,
                                'sub_info',
                                'subj_info.xlsx'),
                    engine='openpyxl')
def get_sub_info(key):
    sub_arr = []
    for _, row in df_info.iterrows():
        if row.NSPID.strip() in s_lst:
            sub_arr.append(row[key])
 
    return sub_arr

age_arr = np.array(get_sub_info('AGE'))
# delete outliers
age_arr = age_arr[age_arr < 30]
print(np.mean(age_arr), np.std(age_arr))

hand_arr = get_sub_info('HAND')
print(Counter(hand_arr))

sex_arr = get_sub_info('SEX')
print(Counter(sex_arr))

### Behavioral Performance 
## locate data 
# test if all the data could be found in Neo_2011_MRI_Round2
data_path = os.path.join(root_path,
                         'mri_test',
                         'Neo_2011_MRI_Round2',
                         'data')
file_names = []
for file in os.listdir(data_path):
    if file == '.DS_Store':
        continue
    path_1 = os.path.join(data_path, file, 'fa')
    for ma_file in os.listdir(path_1):
        if ma_file == '.DS_Store':
            continue
        file_names.append(ma_file)
    
names_i = [file[:5] for file in file_names]
is_in = []
for s in s_lst:
    if s in names_i:
        is_in.append(True)
    else:
        is_in.append(False)

# Directly read out Hit Rate from Jinhua Data 
def read_hit_rate(type):
    df = pd.read_excel(os.path.join(root_path,
                                  'behavioral_data',
                                  'behavioral.xls'),
                    sheet_name=type,
                    )
    hit_rate = {df.columns[5]: [],
                df.columns[7]: [],
                df.columns[8]: [],
                df.columns[10]: [],}
    keys = list(hit_rate.keys())
    for _, row in df.iterrows():
        if row.id in s_lst:
            for i in range(4):
                hit_rate[keys[i]].append(row[keys[i]])
    
    return hit_rate

fa = read_hit_rate('FA')
sa = read_hit_rate('SA')
for key, val in sa.items():
    print(f'------------{key}------------')
    print(f'{np.nanmean(val)}±{np.nanstd(val)}')

# ttest
print('-------------FA, d-prime--------------')
nan_mask = np.isnan(fa['fea_dp']) + np.isnan(fa['con_dp'])
print(ttest_rel(np.array(fa['fea_dp'])[~nan_mask], 
                np.array(fa['con_dp'])[~nan_mask]))
d = abs(cohen_d(np.array(fa['fea_dp'])[~nan_mask],
                np.array(fa['con_dp'])[~nan_mask]))
print(f'cohend: {d}')
 

print('-------------FA, RT--------------')
nan_mask = np.isnan(fa['fea_RT']) + np.isnan(fa['con_RT'])
print(ttest_rel(np.array(fa['fea_RT'])[~nan_mask], 
                np.array(fa['con_RT'])[~nan_mask]))
d = abs(cohen_d(np.array(fa['fea_RT'])[~nan_mask],
                np.array(fa['con_RT'])[~nan_mask]))
print(f'cohend: {d}')

print('-------------SA, d-prime--------------')
nan_mask = np.isnan(sa['fov_dp']) + np.isnan(sa['per_dp'])
print(ttest_rel(np.array(sa['fov_dp'])[~nan_mask], 
                np.array(sa['per_dp'])[~nan_mask]))
d = abs(cohen_d(np.array(sa['fov_dp'])[~nan_mask],
                np.array(sa['per_dp'])[~nan_mask]))
print(f'cohend: {d}')

print('-------------SA, RT--------------')
nan_mask = np.isnan(sa['fov_RT']) + np.isnan(sa['per_RT'])
print(ttest_rel(np.array(sa['fov_RT'])[~nan_mask], 
                np.array(sa['per_RT'])[~nan_mask]))
d = abs(cohen_d(np.array(sa['fov_RT'])[~nan_mask],
                np.array(sa['per_RT'])[~nan_mask]))
print(f'cohend: {d}')