"""
Statistical infrerence for Single Cluster and virtural leision analysis
Permutaion test
- Generate the p value based on retrained model 
- save p value as dictionary
- to be integreated into the main table 
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
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("data_type",
                    help="{'fa', 'sa'}",
                    type=str)

# load test data 
def gen_customed_test_data(series, data_type, nan_mask, significant_mask,
                                test_X):
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

    test_X_single_cluster = test_X[:, mask_cat]

    lesion_mask = np.logical_and(~mask_cat, significant_mask)
    test_X_lesion = test_X[:, lesion_mask]
    return test_X_single_cluster, test_X_lesion

# compute auc 
def get_auc(model, X, y):
    """
    Return auc score of the model
    """
    y_pred = model.decision_function(X) 
    fpr, tpr, _ = roc_curve(y, y_pred)
    return auc(fpr, tpr)

if __name__ == '__main__':
    args = parser.parse_args()
    permute_size = 10000

    data_type_set = set(['fa', 'sa'])
    set_test = set([args.data_type])
    transfer_data_type = list(data_type_set - set_test)[0]

    # define path 
    root_path = '/data/home/attsig/attention_signature'
    data_path = os.path.join(root_path, 'train_test_data')
    df_path = os.path.join(root_path, 'svc_analysis', 'cluster_table_small_version',
                            f'{args.data_type}_integrated.csv')
    nii_path = os.path.join(root_path, 'svc_analysis', 'nii_file', 
                            'cluster_map_small_version',
                            f'{args.data_type}_screened')
    model_path = os.path.join(root_path, 'svc_analysis', 'cluster_models')
    save_path = os.path.join(root_path,
                             'svc_analysis',
                            'permutation_dictionary')
    
    # load data of prediction 
    mask_NaN = np.load(os.path.join(root_path, 'train_test_data',
                                        'mask_NaN.npy'))
    significant_arr = image.get_data(os.path.join(root_path,
                                                'svc_analysis',
                                                'nii_file',
                                                f'{args.data_type}_fdr_corrected.nii.gz')).flatten()
    significant_arr = significant_arr[mask_NaN]
    significant_mask = ~np.isnan(significant_arr)

    test_X = np.load(os.path.join(data_path, f'test_X_{args.data_type}.npy'))
    test_y = np.load(os.path.join(data_path, f'test_Y_{args.data_type}.npy'))
    test_X = scale(test_X)

    transfer_X = np.load(os.path.join(data_path, f'test_X_{transfer_data_type}.npy'))
    transfer_y = np.load(os.path.join(data_path, f'test_Y_{transfer_data_type}.npy'))
    transfer_X = scale(transfer_X)

    # load model parameters 
    df_data = pd.read_csv(df_path)
    permutaiton_dict = {}
    total_num = len(df_data)

    for index, series in df_data.iterrows():
        start = time.time()
        cat = series['Unnamed: 0']
        voxel_num = series['number_of_voxels']
        print(f"----------Processing Cluster {cat}----------")

        model_cluster = joblib.load(os.path.join(model_path,
                                        args.data_type,
                                        f'single_cluster_{cat}_{voxel_num}.pkl'))
        model_lesion = joblib.load(os.path.join(model_path,
                                        args.data_type,
                                        f'lesion_{cat}_{voxel_num}.pkl'))

        test_X_single_cluster, test_X_lesion = gen_customed_test_data(series, 
                                                                    args.data_type, 
                                                                    mask_NaN,
                                                                    significant_mask,
                                                                    test_X)

        transfer_X_single_cluster, transfer_X_lesion = gen_customed_test_data(series, 
                                                                    args.data_type, 
                                                                    mask_NaN,
                                                                    significant_mask,
                                                                    transfer_X)
        prediction_test_lst, prediciton_transfer_lst = [], []
        lesion_test_lst, lesion_transfer_lst = [], []
        for i in range(permute_size):
            # shuffle the data 
            np.random.shuffle(test_y)
            np.random.shuffle(transfer_y)
            # calculate the score
            prediction_test_lst.append(get_auc(model_cluster, test_X_single_cluster, test_y))
            prediciton_transfer_lst.append(get_auc(model_cluster, transfer_X_single_cluster, transfer_y))

            lesion_test_lst.append(get_auc(model_lesion, test_X_lesion, test_y))
            lesion_transfer_lst.append(get_auc(model_lesion, transfer_X_lesion, transfer_y))

        permutaiton_dict[cat] = {'prediction_test': prediction_test_lst, 
                                 'prediction_transfer': prediciton_transfer_lst,
                                 'lesion_test': lesion_test_lst,
                                 'lesion_transfer': lesion_transfer_lst}
        

        end = time.time()
        total_num -= 1
        print(f"Cluster {cat} is completed, using time {end - start}s, {total_num} left.")


    # save dictionary ads pickle 
    with open(os.path.join(save_path, 
                        f'permutation_dict_{args.data_type}.pkl'), 'wb') as f:
        pickle.dump(permutaiton_dict, f)

