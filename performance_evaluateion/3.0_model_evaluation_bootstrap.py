"""
Bootstrap prediction indexes for model evaluation 
"""

import numpy as np
import pandas as pd
import os
import time 
import random
import argparse
from algo import * 

from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score, precision_score, \
                            recall_score, f1_score, roc_curve, auc

parser = argparse.ArgumentParser()
parser.add_argument("data_type",
                    help={'fa_test',
                            'fa_transfer',
                            'sa_test',
                            'sa_transfer'},
                    type=str)

# use mean beta to evaluate model performance 
def decision_value(weight, intercept, feature_val):
    """
    Bypass the SVC framework, equivalent to decision function
    
    Parameters 
    ----------
    weight: ndarray with shape (n, )
        mean weights 
    intercept: float
        mean intercept 
    feature_val: ndarray with shape (m, n)
        scaled feature matrix
    
    Returns
    -------
    predict_cat : ndarray with shape (m, )
        the predicted category, {0, 1}
        
    decision_val : ndarray with shape (m, )

    """
    decision_val = np.dot(feature_val, weight) + intercept

    predict_cat = []
    for i in decision_val:
        if i < 0:
            predict_cat.append(0)
        else:
            predict_cat.append(1)
    predict_cat = np.array(predict_cat)

    return predict_cat, decision_val 

def boot_to_ci(data, boot_size, weights, intercepts):
    """
    Return bootstrap results 
    """
    boot_result = {'acc': [],
                'precision': [],
                'recall': [],
                'f-score': [],
                'auc': []}

    for i in range(boot_size):
        start = time.time()

        boot_matrix = bootstrap_by_subject(data)
        X = boot_matrix[:,:-1]
        y = boot_matrix[:,-1]

        predict_cat, decision_val = decision_value(weights, 
                                                intercepts, 
                                                X)
        fpr, tpr, _ = roc_curve(y, decision_val)
        
        # prediction matrices
        boot_result['acc'].append(accuracy_score(y, predict_cat))
        boot_result['precision'].append(precision_score(y, predict_cat))
        boot_result['recall'].append(recall_score(y, predict_cat))
        boot_result['f-score'].append(f1_score(y, predict_cat))
        boot_result['auc'].append(auc(fpr, tpr))

        end = time.time()
        print(f'{i}th iteration completed, using time {end - start}s.')

    return boot_result

if __name__ == '__main__':
    args = parser.parse_args()
    # load data 
    root_path = '/data/home/attsig/attention_signature'
    weight_path = os.path.join(root_path,'svc_analysis/prediction_results')  
    save_path = os.path.join(weight_path, 'model_evaluation','bootstrapped')
    data_path = os.path.join(root_path, 'train_test_data')

    if 'fa' in args.data_type:
        weights = np.load(os.path.join(weight_path, 'fa', 'beta.npy'), 
                            allow_pickle=True).item()
        intercepts = np.load(os.path.join(weight_path, 'fa', 'intercept.npy'), 
                            allow_pickle=True).item()
        test_X = np.load(os.path.join(data_path, 'test_X_fa.npy'))
        test_Y = np.load(os.path.join(data_path, 'test_Y_fa.npy'))
        transfer_X = np.load(os.path.join(data_path, 'test_X_sa.npy'))
        transfer_Y = np.load(os.path.join(data_path, 'test_Y_sa.npy'))
    else:
        weights = np.load(os.path.join(weight_path, 'sa', 'beta.npy'), 
                            allow_pickle=True).item()
        intercepts = np.load(os.path.join(weight_path, 'sa', 'intercept.npy'), 
                            allow_pickle=True).item()
        test_X = np.load(os.path.join(data_path, 'test_X_sa.npy'))
        test_Y = np.load(os.path.join(data_path, 'test_Y_sa.npy'))
        transfer_X = np.load(os.path.join(data_path, 'test_X_fa.npy'))
        transfer_Y = np.load(os.path.join(data_path, 'test_Y_fa.npy'))

    test_X, transfer_X = scale(test_X), scale(transfer_X)

    # mean weights and intercepts
    avg_weights = np.zeros(weights[0].size)
    for weight in weights.values():
        avg_weights = np.vstack((avg_weights, weight))     
    avg_weights = avg_weights[1:, :]
    avg_weights = np.mean(avg_weights, axis=0)

    avg_intercepts = np.mean([i for i in intercepts.values()])

    ## gen data matrix for bootstrapping 
    test_data_matrix = np.column_stack((test_X, test_Y))
    transfer_data_matrix = np.column_stack((transfer_X, transfer_Y))

    boot_size = 10000

    if 'test' in args.data_type:
        boot_dic = boot_to_ci(test_data_matrix, 
                              boot_size, 
                              avg_weights, avg_intercepts)
    else:
        boot_dic = boot_to_ci(transfer_data_matrix, 
                              boot_size, 
                              avg_weights, avg_intercepts)

    np.save(os.path.join(save_path, f'{args.data_type}.npy'), boot_dic)