"""
Statistical Evlaluation of Cluster Analysis 
1. Cancate bootstrap data 
2. one-sample t-test with one direction 
    1. for [prediction case], null result against 0.5
    2. for [lesion case]
        - null result (the drop of preformance) against 0
        - null result (model performance) against 0.5
"""
import numpy as np
import pandas as pd
import os

from algo import * 
from scipy.stats import ttest_1samp

root_path = '/data/home/attsig/attention_signature/svc_analysis'
boot_path = os.path.join(root_path, 'cluster_bootstrap_small_version_retrain')

BOOT_TIME = 10 
data_type = ['fa', 'sa']

# concate bootstrapped data
# the 0.npy file as base to be concatnated 
for dt in data_type:
    base_dic = np.load(os.path.join(boot_path,
                                dt,
                                f'0.npy'),
                        allow_pickle=True).item()
    
    for i in range(1, BOOT_TIME):
        file_path = os.path.join(boot_path,
                                dt,
                                f'{i}.npy')
        temp_dic = np.load(file_path,
                            allow_pickle=True).item()
        
        for k, val in temp_dic.items():
            for k_sub, val_sub in val.items():
                base_dic[k][k_sub][0:0] = val_sub
    
    # print(len(base_dic))
    # print(len(base_dic[k][k_sub]))
    np.save(os.path.join(boot_path,
                        f'{dt}_concatnated.npy'),
            base_dic)

PREDICTION_NULL = 0.5
LESION_NULL = 0
# statistical analysis for cluster of FA 
LEISION_TEST_FA = 0.99
LEISION_TRANSFER_FA = 0.66
fa_boot = np.load(os.path.join(boot_path, 
                            'fa_concatnated.npy'),
                allow_pickle=True).item()
fa_df = pd.read_csv(os.path.join(root_path,
                                'cluster_table_small_version',
                                'fa_evaluation_without_CI.csv'))

for k, val in fa_boot.items():
    df_row = fa_df[fa_df['Unnamed: 0.1'] == k]

    for sub_k in val.keys():
        boot_val = val[sub_k]
        point_val = df_row[sub_k].values[0]
        
        if 'lesion' in sub_k:
            if 'test' in sub_k:
                boot_val_l = [LEISION_TEST_FA - i for i in boot_val]
            else:
                boot_val_l = [LEISION_TRANSFER_FA - i for i in boot_val]
            
            _, p_val_drop = ttest_1samp(boot_val_l, 
                                LESION_NULL,
                                alternative='greater')
            effect_size_drop = - cohen_d(boot_val_l, [LESION_NULL]*len(boot_val_l))

            _, p_val_model = ttest_1samp(boot_val,
                                PREDICTION_NULL,
                                alternative='greater')
            effect_size_model = - cohen_d(boot_val, [PREDICTION_NULL]*len(boot_val))
        else:
            _, p_val = ttest_1samp(boot_val, 
                                PREDICTION_NULL,
                                alternative='greater')
            effect_size = - cohen_d(boot_val, [PREDICTION_NULL]*len(boot_val))
        
        ci = CI(np.array(boot_val)) #(lb, hb, mean)
        lb, hb, avg = "%.2f" % ci[0][0], "%.2f" % ci[0][1], "%.2f" % ci[0][2]
        
        if 'lesion' not in sub_k:
            if p_val <= 0.001:
                asterisks = '***'
            elif 0.001 < p_val <= 0.01:
                asterisks = '**'
            elif 0.01 < p_val <=0.05:
                asterisks = '*'
            else:
                asterisks = 'NS: ' + "%.2f" % p_val
            
            if effect_size < 0.2:
                taos = 'NP: ' + "%.2f" % effect_size
            elif 0.2 <= effect_size < 0.5:
                taos = '†'
            elif 0.5 <= effect_size < 0.8:
                taos = '††'
            elif effect_size >= 0.8:
                taos = '†††' 
        else:
            if p_val_model <= 0.001:
                asterisks = '***'
            elif 0.001 < p_val_model <= 0.01:
                asterisks = '**'
            elif 0.01 < p_val_model <=0.05:
                asterisks = '*'
            else:
                asterisks = 'NS: ' + "%.2f" % p_val_model
            
            if effect_size_model < 0.2:
                taos = 'NP: ' + "%.2f" % effect_size_model
            elif 0.2 <= effect_size_model < 0.5:
                taos = '†'
            elif 0.5 <= effect_size_model < 0.8:
                taos = '††'
            elif effect_size_model >= 0.8:
                taos = '†††' 
 
        fill_val = avg \
                    + f'[{lb}, {hb}]; '\
                    + f'({asterisks})'\
                    + f'({taos})'
        if 'lesion' in sub_k:
            if p_val_drop <= 0.001:
                double_daggers = '‡‡‡'
            elif 0.001 < p_val_drop <= 0.01:
                double_daggers = '‡‡'
            elif 0.01 < p_val_drop <=0.05:
                double_daggers = '‡'
            else:
                double_daggers = 'NS: ' + "%.2f" % p_val_drop
            
            if effect_size_drop < 0.2:
                sections = 'NP: ' + "%.2f" % effect_size_drop
            elif 0.2 <= effect_size_drop < 0.5:
                sections = '§'
            elif 0.5 <= effect_size_drop < 0.8:
                sections = '§§'
            elif effect_size_drop >= 0.8:
                sections = '§§§'

            fill_val += '; ' + f'({double_daggers})' + f'({sections})'
        fa_df.loc[fa_df['Unnamed: 0.1']==k, sub_k] = fill_val

    # modify mni coordinates
    mni_val = df_row['mni_coordinates'].values[0][1:-1]
    mni_l = mni_val.split(',')
    mni_l_int = [int(float(i)) for i in mni_l]
    mni = f'({mni_l_int[0]}, {mni_l_int[1]}, {mni_l_int[2]})'
    fa_df.loc[fa_df['Unnamed: 0.1']==k, 'mni_coordinates'] = mni

    col_lst = ['x', 'y', 'z']
    for i in range(len(col_lst)):
         fa_df.loc[fa_df['Unnamed: 0.1']==k, col_lst[i]] = mni_l_int[i]

    # modify z-score 
    z_val = df_row['z-score'].values[0][1:-1]
    z_l = z_val.split(',')
    z_l_round = ["%.2f" % float(i) for i in z_l]
    z = f'{z_l_round[0]}({z_l_round[1]})'
    fa_df.loc[fa_df['Unnamed: 0.1']==k, 'z-score'] = z

fa_df.to_csv(os.path.join(root_path,
                        'cluster_table_small_version_retrain',
                        'fa_evaluation_with_CI_add_statistics.csv')) 

# statistical analysis for cluster of SA 
LEISION_TEST_SA = 0.67
LEISION_TRANSFER_SA = 0.60
sa_boot = np.load(os.path.join(boot_path, 
                            'sa_concatnated.npy'),
                allow_pickle=True).item()
sa_df = pd.read_csv(os.path.join(root_path,
                                'cluster_table_small_version',
                                'sa_evaluation_without_CI.csv'))      

for k, val in sa_boot.items():
    df_row = sa_df[sa_df['Unnamed: 0.1'] == k]

    for sub_k in val.keys():
        boot_val = val[sub_k]
        point_val = df_row[sub_k].values[0]
        
        if 'lesion' in sub_k:
            if 'test' in sub_k:
                boot_val_l = [LEISION_TEST_SA - i for i in boot_val]
            else:
                boot_val_l = [LEISION_TRANSFER_SA - i for i in boot_val]
            
            _, p_val_drop = ttest_1samp(boot_val_l, 
                                LESION_NULL,
                                alternative='greater')
            effect_size_drop = - cohen_d(boot_val_l, [LESION_NULL]*len(boot_val_l))

            _, p_val_model = ttest_1samp(boot_val,
                                PREDICTION_NULL,
                                alternative='greater')
            effect_size_model = - cohen_d(boot_val, [PREDICTION_NULL]*len(boot_val))
        else:
            _, p_val = ttest_1samp(boot_val, 
                                PREDICTION_NULL,
                                alternative='greater')
            effect_size = - cohen_d(boot_val, [PREDICTION_NULL]*len(boot_val))
        
        ci = CI(np.array(boot_val)) #(lb, hb, mean)
        lb, hb, avg = "%.2f" % ci[0][0], "%.2f" % ci[0][1], "%.2f" % ci[0][2]
        
        if 'lesion' not in sub_k:
            if p_val <= 0.001:
                asterisks = '***'
            elif 0.001 < p_val <= 0.01:
                asterisks = '**'
            elif 0.01 < p_val <=0.05:
                asterisks = '*'
            else:
                asterisks = 'NS: ' + "%.2f" % p_val
            
            if effect_size < 0.2:
                taos = 'NP: ' + "%.2f" % effect_size
            elif 0.2 <= effect_size < 0.5:
                taos = '†'
            elif 0.5 <= effect_size < 0.8:
                taos = '††'
            elif effect_size >= 0.8:
                taos = '†††' 
        else:
            if p_val_model <= 0.001:
                asterisks = '***'
            elif 0.001 < p_val_model <= 0.01:
                asterisks = '**'
            elif 0.01 < p_val_model <=0.05:
                asterisks = '*'
            else:
                asterisks = 'NS: ' + "%.2f" % p_val_model
            
            if effect_size_model < 0.2:
                taos = 'NP: ' + "%.2f" % effect_size_model
            elif 0.2 <= effect_size_model < 0.5:
                taos = '†'
            elif 0.5 <= effect_size_model < 0.8:
                taos = '††'
            elif effect_size_model >= 0.8:
                taos = '†††' 
 
        fill_val = avg \
                    + f'[{lb}, {hb}]; '\
                    + f'({asterisks})'\
                    + f'({taos})'
        if 'lesion' in sub_k:
            if p_val_drop <= 0.001:
                double_daggers = '‡‡‡'
            elif 0.001 < p_val_drop <= 0.01:
                double_daggers = '‡‡'
            elif 0.01 < p_val_drop <=0.05:
                double_daggers = '‡'
            else:
                double_daggers = 'NS: ' + "%.2f" % p_val_drop
            
            if effect_size_drop < 0.2:
                sections = 'NP: ' + "%.2f" % effect_size_drop
            elif 0.2 <= effect_size_drop < 0.5:
                sections = '§'
            elif 0.5 <= effect_size_drop < 0.8:
                sections = '§§'
            elif effect_size_drop >= 0.8:
                sections = '§§§'

            fill_val += '; ' + f'({double_daggers})' + f'({sections})'
        sa_df.loc[sa_df['Unnamed: 0.1']==k, sub_k] = fill_val

    # modify mni coordinates
    mni_val = df_row['mni_coordinates'].values[0][1:-1]
    mni_l = mni_val.split(',')
    mni_l_int = [int(float(i)) for i in mni_l]
    mni = f'({mni_l_int[0]}, {mni_l_int[1]}, {mni_l_int[2]})'
    sa_df.loc[sa_df['Unnamed: 0.1']==k, 'mni_coordinates'] = mni

    col_lst = ['x', 'y', 'z']
    for i in range(len(col_lst)):
         sa_df.loc[sa_df['Unnamed: 0.1']==k, col_lst[i]] = mni_l_int[i]

    # modify z-score 
    z_val = df_row['z-score'].values[0][1:-1]
    z_l = z_val.split(',')
    z_l_round = ["%.2f" % float(i) for i in z_l]
    z = f'{z_l_round[0]}({z_l_round[1]})'
    sa_df.loc[sa_df['Unnamed: 0.1']==k, 'z-score'] = z

sa_df.to_csv(os.path.join(root_path,
                        'cluster_table_small_version_retrain',
                        'sa_evaluation_with_CI_add_statistics.csv')) 