"""
Permutation test on model performance 
"""
import numpy as np
import os
import time 
import random
import argparse
from algo import * 

from sklearn.metrics import accuracy_score, precision_score, \
                            recall_score, f1_score, roc_curve, auc

parser = argparse.ArgumentParser()
parser.add_argument("data_type",
                    help={'fa', 'sa', 'fa_inter_to_sa_intra'},
                    type=str)

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

def permute_on_auc(X, y, weights, intercepts, permute_size=10000, verbose=True):
    """
    Generate a null distribution of auc by permuting labels

    Parameters
    ----------
    X: ndarray with shape (m, n)
        feature matrix
        m: number of samples
        n: number of features
    y: ndarray with shape (m, )
        label vector
    weights: ndarray with shape (n, )
        mean weights
    intercepts: float
        mean intercept
    permute_size: int
        number of permutation
    verbose: bool
        whether to print out progress
    
    Returns
    -------
    auc_null: list  
        null distribution of auc
    """
    auc_null = []
    predict_cat, decision_val = decision_value(weights, 
                                                   intercepts, 
                                                   X)
    for i in range(permute_size):
        fpr, tpr, _ = roc_curve(y, np.random.permutation(predict_cat))  
        auc_null.append(auc(fpr, tpr))
        if verbose:
            if i % 100 == 0:
                print(f'{i}th iteration completed')
    return auc_null

def perm_p_val(val, null_lst, sign='greater'):
    """
    Calculate p value based on null distribution

    Parameters
    ----------
    val: float
        observed value
    null_lst: list
        null distribution
    sign: str
        'greater' or 'less'
    
    Returns
    -------
    p: float
        p value
    """
    if sign == 'greater':
        p = sum(null_lst > val) / len(null_lst)
    elif sign == 'less':
        p = sum(null_lst < val) / len(null_lst)
    else:
        raise ValueError('sign should be greater or less')
    return p

def load_dic(name):
    arr = np.load(os.path.join(weight_path, 
                               'model_evaluation',
                                 'bootstrapped',
                               f'{name}.npy'),
                allow_pickle=True).item()['auc']
    return np.mean(arr)

def write_result(result_dic, X, y, weight, intercept, val_auc, 
                 job, is_diff=False):
    """
    write results to result_dic

    Parameters
    ----------
    result_dic: dict
        result dictionary
    X: ndarray with shape (m, n)
        feature matrix
        m: number of samples
        n: number of features
    y: ndarray with shape (m, )
        label vector
    weight: ndarray with shape (n, )
        mean weights
    intercept: float
        mean intercept
    val_auc: float
        observed auc
    job: str
        'test' or 'transfer' or 'diff'
        if compute 'diff', X, y, weight, intercept are arbitrary 
    is_diff: bool
        whether to permute on difference of auc
    
    Returns
    -------
    result_dic: dict
        result dictionary
    """
    start = time.time()
    print('-----------------------------------')
    print(f'Permutation on {job}')
    
    if not is_diff:
        perm_lst = permute_on_auc(X, y, 
                                    weight, intercept)
        p_val = perm_p_val(val_auc, perm_lst)
    else:
        # permute on difference of auc
        perm_lst = np.array(result_dic['null_test']) \
                    - np.array(result_dic['null_transfer'])
        perm_lst = perm_lst.tolist()
        val_auc = result_dic['auc_test'] - result_dic['auc_transfer']
        p_val = perm_p_val(val_auc, perm_lst)
    
    result_dic[f'pval_{job}'] = p_val
    result_dic[f'auc_{job}'] = val_auc
    result_dic[f'null_{job}'] = perm_lst

    end = time.time()
    print(f'Permutation on {job} completed, using time {end - start}s.')
    
    return result_dic


if __name__ == '__main__':
    args = parser.parse_args()
    
    root_path = '/data/home/attsig/attention_signature'
    weight_path = os.path.join(root_path,
                            'svc_analysis',
                            'prediction_results')  
    data_path = os.path.join(root_path, 'train_test_data')
    save_path = os.path.join(weight_path, 'model_evaluation','permutation_modified')
    # load data 
    if args.data_type != 'fa_inter_to_sa_intra':
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

        # mean weights and intercepts
        avg_weights = np.zeros(weights[0].size)
        for weight in weights.values():
            avg_weights = np.vstack((avg_weights, weight))     
        avg_weights = avg_weights[1:, :]
        avg_weights = np.mean(avg_weights, axis=0)

        avg_intercepts = np.mean([i for i in intercepts.values()])

        # permuation step
        result_dic = {}
        # test data above chance
        result_dic = write_result(result_dic, test_X, test_Y,
                                avg_weights, avg_intercepts, 
                                load_dic(f'{args.data_type}_test'),
                                'test')
        # transfer data above chance
        result_dic = write_result(result_dic, transfer_X, transfer_Y,
                                avg_weights, avg_intercepts, 
                                load_dic(f'{args.data_type}_transfer'),
                                'transfer')
        # test data above transfer data
        result_dic = write_result(result_dic, test_X, test_Y,
                                avg_weights, avg_intercepts, 
                                load_dic(f'{args.data_type}_test'),
                                'diff',
                                is_diff=True)
        # save results
        np.save(os.path.join(save_path, f'{args.data_type}.npy'), result_dic)
    else: # test if sa_test is greater than fa_transfer
        result_dic = {}
        fa_dic = np.load(os.path.join(save_path, 'fa.npy'), 
                         allow_pickle=True).item() 
        sa_dic = np.load(os.path.join(save_path, 'sa.npy'),
                            allow_pickle=True).item()
        
        perm_lst = np.array(sa_dic['null_test']) \
                    - np.array(fa_dic['null_transfer'])
        perm_lst = perm_lst.tolist()

        val_auc = sa_dic['auc_test'] - fa_dic['auc_transfer']
        p_val = perm_p_val(val_auc, perm_lst)

        result_dic['pval_sa_test_fa_transfer'] = p_val
        result_dic['auc_sa_test_fa_transfer'] = val_auc

        np.save(os.path.join(save_path, f'{args.data_type}.npy'), result_dic)