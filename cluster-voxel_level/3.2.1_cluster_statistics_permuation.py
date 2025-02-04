"""
Statistical Evlaluation of Cluster Analysis based on permutation
1. Cancate bootstrap data 
2. Calculate the p-value based on permutation distribution 
"""
import numpy as np
import pandas as pd
import os
import pickle  

from algo import * 
from scipy.stats import ttest_1samp

root_path = '/data/home/attsig/attention_signature/svc_analysis'
boot_path = os.path.join(root_path, 'cluster_bootstrap_small_version_retrain')

# statistical analysis for cluster of FA 
LEISION_TEST_FA = 0.99
LEISION_TRANSFER_FA = 0.66

# load conacated bootstrap data 
fa_boot = np.load(os.path.join(boot_path, 
                            'fa_concatnated.npy'),
                allow_pickle=True).item()
fa_df = pd.read_csv(os.path.join(root_path,
                                'cluster_table_small_version',
                                'fa_evaluation_without_CI.csv'))

# load permutation dictionary 
with open(os.path.join(root_path, 
                       'permutation_dictionary', 
                       'permutation_dict_fa.pkl'), 
        'rb') as f:
    fa_permutation = pickle.load(f)

for k, val in fa_boot.items():
    df_row = fa_df[fa_df['Unnamed: 0.1'] == k]

    for sub_k in val.keys():
        boot_val = val[sub_k]
        ci = CI(np.array(boot_val)) #(lb, hb, mean)
        lb, hb, avg = "%.2f" % ci[0][0], "%.2f" % ci[0][1], "%.2f" % ci[0][2]
        avg = float(avg)

        perm_dic = fa_permutation[k]
        
        if 'lesion' in sub_k:
            if 'test' in sub_k:
                perm_val_l = [LEISION_TEST_FA - i for i in perm_dic[sub_k]]
                p_val_drop = p_value_arr(perm_val_l, 
                                        LEISION_TEST_FA - avg, 
                                        'right')
            else:
                perm_val_l = [LEISION_TRANSFER_FA - i for i in perm_dic[sub_k]]
                p_val_drop = p_value_arr(perm_val_l, 
                                        LEISION_TRANSFER_FA - avg, 
                                        'right')
            p_val_model =  p_value_arr(perm_dic[sub_k], 
                                    avg, 
                                    'right')
        else:      
            p_val =  p_value_arr(perm_dic[sub_k], 
                                avg, 
                                'right')

        if 'lesion' not in sub_k:
            if p_val <= 0.001:
                asterisks = '***'
            elif 0.001 < p_val <= 0.01:
                asterisks = '**'
            elif 0.01 < p_val <=0.05:
                asterisks = '*'
            else:
                asterisks = 'NS: ' + "%.2f" % p_val
        else:
            if p_val_model <= 0.001:
                asterisks = '***'
            elif 0.001 < p_val_model <= 0.01:
                asterisks = '**'
            elif 0.01 < p_val_model <=0.05:
                asterisks = '*'
            else:
                asterisks = 'NS: ' + "%.2f" % p_val_model

        fill_val = str(avg) \
                    + f'[{lb}, {hb}]; '\
                    + f'({asterisks})'
        if 'lesion' in sub_k:
            if p_val_drop <= 0.001:
                double_daggers = '†††'
            elif 0.001 < p_val_drop <= 0.01:
                double_daggers = '††'
            elif 0.01 < p_val_drop <=0.05:
                double_daggers = '†'
            else:
                double_daggers = 'NS: ' + "%.2f" % p_val_drop

            fill_val += '; ' + f'({double_daggers})'  
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
                        'fa_evaluation_with_CI_add_statistics_permutation.csv')) 

# statistical analysis for cluster of SA 
LEISION_TEST_SA = 0.67
LEISION_TRANSFER_SA = 0.60
sa_boot = np.load(os.path.join(boot_path, 
                            'sa_concatnated.npy'),
                allow_pickle=True).item()
sa_df = pd.read_csv(os.path.join(root_path,
                                'cluster_table_small_version',
                                'sa_evaluation_without_CI.csv'))    
sa_df_ref = pd.read_csv(os.path.join(root_path,
                                'cluster_table_small_version',
                                'sa_evaluation_without_CI.csv'))

# load permutation dictionary 
with open(os.path.join(root_path, 
                       'permutation_dictionary', 
                       'permutation_dict_sa.pkl'), 
        'rb') as f:
    sa_permutation = pickle.load(f)

for k, val in sa_boot.items():
    df_row = sa_df_ref[sa_df_ref['Unnamed: 0.1'] == k]

    for sub_k in val.keys():
        boot_val = val[sub_k]
        ci = CI(np.array(boot_val)) #(lb, hb, mean)
        lb, hb, avg = "%.2f" % ci[0][0], "%.2f" % ci[0][1], "%.2f" % ci[0][2]
        avg = float(avg)

        perm_dic = sa_permutation[k]
        if 'lesion' in sub_k:
            if 'test' in sub_k:
                perm_val_l = [LEISION_TEST_SA - i for i in perm_dic[sub_k]]
                p_val_drop = p_value_arr(perm_val_l, 
                                        LEISION_TEST_SA - avg, 
                                        'right')
            else:
                perm_val_l = [LEISION_TRANSFER_SA - i for i in perm_dic[sub_k]]
                p_val_drop = p_value_arr(perm_val_l, 
                                        LEISION_TRANSFER_SA - avg, 
                                        'right')
            p_val_model =  p_value_arr(perm_dic[sub_k], 
                                    avg, 
                                    'right')
        else:   
            p_val =  p_value_arr(perm_dic[sub_k], 
                                avg, 
                                'right')   
        if 'lesion' not in sub_k:
            if p_val <= 0.001:
                asterisks = '***'
            elif 0.001 < p_val <= 0.01:
                asterisks = '**'
            elif 0.01 < p_val <=0.05:
                asterisks = '*'
            else:
                asterisks = 'NS: ' + "%.2f" % p_val
        else:
            if p_val_model <= 0.001:
                asterisks = '***'
            elif 0.001 < p_val_model <= 0.01:
                asterisks = '**'
            elif 0.01 < p_val_model <=0.05:
                asterisks = '*'
            else:
                asterisks = 'NS: ' + "%.2f" % p_val_model

        fill_val = str(avg) \
                    + f'[{lb}, {hb}]; '\
                    + f'({asterisks})'
        if 'lesion' in sub_k:
            if p_val_drop <= 0.001:
                double_daggers = '†††'
            elif 0.001 < p_val_drop <= 0.01:
                double_daggers = '††'
            elif 0.01 < p_val_drop <=0.05:
                double_daggers = '†'
            else:
                double_daggers = 'NS: ' + "%.2f" % p_val_drop

            fill_val += '; ' + f'({double_daggers})'  
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
                        'sa_evaluation_with_CI_add_statistics_permutation.csv')) 