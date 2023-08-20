"""
Retrain predictive model within the cluster 

Save beta maps for evaluation
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

def loop_train():
    """
    Loop through all the clusters, store weights and intercepts
    """

    for index, series in df_data.iterrows():
        start = time.time()
        print('------------------------------------')
        cat = series['Unnamed: 0']
        voxel_num = series['number_of_voxels']
        train_X_single_cluster, train_X_lesion = gen_customed_training_data(series, 
                                                                            args.data_type, 
                                                                            mask_NaN,
                                                                            significant_mask,
                                                                            train_X)
        svc = SVC(kernel='linear')
        svc.fit(train_X_single_cluster, train_y)
        print('Single Cluster')
        print(svc.score(train_X_single_cluster, train_y))

        # save model 
        joblib.dump(svc, os.path.join(model_path,
                                    args.data_type,
                                    f'single_cluster_{cat}_{voxel_num}.pkl'))
        
        svc = SVC(kernel='linear')
        svc.fit(train_X_lesion, train_y)
        print('Lesion')
        print(svc.score(train_X_lesion, train_y))

        # save model 
        joblib.dump(svc, os.path.join(model_path,
                                    args.data_type,
                                    f'lesion_{cat}_{voxel_num}.pkl'))



if __name__ == '__main__':
    args = parser.parse_args()

    # load data 
    root_path = '/data/home/attsig/attention_signature'
    weight_path = os.path.join(root_path,'svc_analysis/prediction_results')  
    data_path = os.path.join(root_path, 'train_test_data')
    df_path = os.path.join(root_path, 'svc_analysis', 'cluster_table_small_version',
                            f'{args.data_type}_integrated.csv')
    nii_path = os.path.join(root_path, 'svc_analysis', 'nii_file', 
                            'cluster_map_small_version',
                            f'{args.data_type}_screened')
    model_path = os.path.join(root_path, 'svc_analysis', 'cluster_models')

    # load data of prediction 
    mask_NaN = np.load(os.path.join(root_path, 'train_test_data',
                                        'mask_NaN.npy'))
    significant_arr = image.get_data(os.path.join(root_path,
                                                'svc_analysis',
                                                'nii_file',
                                                f'{args.data_type}_fdr_corrected.nii.gz')).flatten()
    significant_arr = significant_arr[mask_NaN]
    significant_mask = ~np.isnan(significant_arr)

    train_X = np.load(os.path.join(data_path, f'train_X_{args.data_type}.npy'))
    train_y = np.load(os.path.join(data_path, f'train_Y_{args.data_type}.npy'))
    train_X = scale(train_X)

    # dataframe 
    df_data = pd.read_csv(df_path)

    # main loop 
    loop_train()

    
