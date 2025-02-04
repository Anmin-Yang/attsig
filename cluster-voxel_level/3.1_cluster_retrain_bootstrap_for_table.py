"""
Bootstrap prediction and leision result
Parallel Computing and storing bootstrapped auc scores 

First loop is the row of DataFrame, aka the cluster
"""
import numpy as np
import pandas as pd
import os
import time 
import argparse
from algo import * 

from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("it_idx",
                    help="{0, 1, 2, 3, 4}",
                    type=int)
parser.add_argument("data_type",
                    help="{'fa', 'sa'}",
                    type=str)

def gen_customed_training_data(series, data_type, nan_mask, significant_mask,
                                train_X):
    """
    Generate training data based on cluster

    Parameters
    ----------
    series : pandas Series
        the series of the cluster
    data_type : str
        {'fa', 'sa'}
    nan_mask : ndarray  
        the mask from whole brain to used voxels
    significant_mask : ndarray 
        the mask from used voxels to significant voxels
        
    train_X : ndarray
        the training data
    
    Returns
    -------
    train_X_single_cluster : ndarray
        the training data 
    train_X_lesion : ndarray
        the training data with lesion  
    """
    cat = series['Unnamed: 0']
    num_voxel = series['number_of_voxels']
    # load mask 
    nii_arr = image.get_data(os.path.join(nii_path, 
                f'{cat}_{num_voxel}_{data_type}.nii.gz')).flatten()     
    nii_arr = nii_arr[nan_mask]
    mask_cat = nii_arr == cat

    train_X_single_cluster = train_X[:, mask_cat]

    lesion_mask = np.logical_and(~mask_cat, significant_mask)
    train_X_lesion = train_X[:, lesion_mask]
    return train_X_single_cluster, train_X_lesion

def resample_data(concated_data):
    boot_matrix = bootstrap_by_subject(concated_data)

    X = boot_matrix[:,:-1]
    y = boot_matrix[:,-1]

    return X, y

def get_auc(model, X, y):
    """
    Return auc score of the model
    """
    y_pred = model.decision_function(X) 
    fpr, tpr, _ = roc_curve(y, y_pred)
    return auc(fpr, tpr)

def boot_result(row, data_type, boot_size):
    """
    Return bootstrap results
    
    Returns
    -------
    boot_result: dic 
        dictionary of bootstrapped auc score
        key: {'prediction_test', 'prediction_transfer',
            'lesion_test', 'lesion_transfer'}
    """
    cat = row['Unnamed: 0']
    num_voxel = row['number_of_voxels']
    # load models 
    model_cluster = joblib.load(os.path.join(model_path,
                                                data_type,
                                                f'single_cluster_{cat}_{num_voxel}.pkl')
    )
    model_lesion = joblib.load(os.path.join(model_path,
                                            data_type,
                                            f'lesion_{cat}_{num_voxel}.pkl')
    )
    
    boot_result = {'prediction_test' : [],
                'prediction_transfer' : [],
                'lesion_test' : [],
                'lesion_transfer' : []}
    for i in range(boot_size):
        if i % 100 == 0:
            start = time.time()
        
        # resample data 
        cluster_test_X, test_Y = resample_data(concat_data_test_cluster)
        lesion_test_X, test_Y = resample_data(concat_data_test_lesion)
        cluster_transfer_X, transfer_Y = resample_data(concat_data_transfer_cluster)
        lesion_transfer_X, transfer_Y = resample_data(concat_data_transfer_lesion)
        
        # get AUC 
        boot_result['prediction_test'].append(get_auc(model_cluster, 
                                                      cluster_test_X, test_Y))
        boot_result['prediction_transfer'].append(get_auc(model_cluster, 
                                                          cluster_transfer_X, transfer_Y))
        boot_result['lesion_test'].append(get_auc(model_lesion, 
                                                  lesion_test_X, test_Y))
        boot_result['lesion_transfer'].append(get_auc(model_lesion,
                                                    lesion_transfer_X, transfer_Y))
        if i % 100 == 0:
            end = time.time()
            print(f"{i}th iteration is completed, using time {end-start}s")
        
    return boot_result

if __name__ == '__main__':
    args = parser.parse_args()
    boot_size = 1000
    # define path 
    root_path = '/data/home/attsig/attention_signature'
    model_path = os.path.join(root_path, 'svc_analysis', 'cluster_models')
    data_path = os.path.join(root_path, 'train_test_data')
    df_path = os.path.join(root_path, 'svc_analysis', 'cluster_table_small_version',
                    f'{args.data_type}_integrated.csv')
    nii_path = os.path.join(root_path, 'svc_analysis', 'nii_file', 'cluster_map_small_version',
                            f'{args.data_type}_screened')

    # load data of prediction 
    mask_NaN = np.load(os.path.join(root_path, 'train_test_data',
                                    'mask_NaN.npy'))

    significant_arr = image.get_data(os.path.join(root_path,
                                                'svc_analysis',
                                                'nii_file',
                                                f'{args.data_type}_fdr_corrected.nii.gz')).flatten()
    significant_arr = significant_arr[mask_NaN]
    significant_mask = ~np.isnan(significant_arr)


    # test and transfer data
    test_X = np.load(os.path.join(data_path, f'test_X_{args.data_type}.npy'))
    test_Y = np.load(os.path.join(data_path, f'test_Y_{args.data_type}.npy'))

    names = ['fa', 'sa'] 
    names.remove(args.data_type)

    transfer_X = np.load(os.path.join(data_path, f'test_X_{names[0]}.npy'))
    transfer_Y = np.load(os.path.join(data_path, f'test_Y_{names[0]}.npy'))


    test_X, transfer_X = scale(test_X), scale(transfer_X)

    # dataframe 
    df_data = pd.read_csv(df_path)

    df_dic = {}
    counter = 0 
    for _, row in df_data.iterrows():
        print('---------------------------------------------------------------')
        print(f'Starting {counter}th row of DataFrame, totaling {df_data.shape[0]} rows.')
        df_start = time.time()

        cat = row['Unnamed: 0']

        # reshape test and transfer data 
        cluster_test_X, lesion_test_X = gen_customed_training_data(row, 
                                                                    args.data_type, 
                                                                    mask_NaN, 
                                                                    significant_mask, 
                                                                    test_X)
        cluster_transfer_X, lesion_transfer_X = gen_customed_training_data(row, 
                                                                    args.data_type,
                                                                    mask_NaN,
                                                                    significant_mask,
                                                                    transfer_X)

        # concate data 
        concat_data_test_cluster = np.c_[cluster_test_X, test_Y]
        concat_data_test_lesion = np.c_[lesion_test_X, test_Y]
        concat_data_transfer_cluster = np.c_[cluster_transfer_X, transfer_Y]
        concat_data_transfer_lesion = np.c_[lesion_transfer_X, transfer_Y]

        result = boot_result(row, args.data_type, boot_size)

        df_dic[cat] = result
                
        df_end = time.time()
        print(f'{counter}th row of DataFrame is completed, using time {df_end - df_start}s.')
        counter += 1

    # save 
    save_path = os.path.join(root_path, 'svc_analysis',
                            'cluster_bootstrap_small_version_retrain', args.data_type,
                            f'{args.it_idx}.npy')
    np.save(save_path, df_dic)