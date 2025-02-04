"""
Compute the whole brain single cluster and lesion analysis 
The objective is to compute the cofidence interval of the whole brian in intra task and inter task 
"""
import numpy as np
import pandas as pd
import os
import time 
from scipy.stats import ttest_1samp
from algo import * 

from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc

BOOT_TIME = 10

root_path = '/data/home/attsig/attention_signature'
weight_path = os.path.join(root_path,'svc_analysis/prediction_results')  
data_path = os.path.join(root_path, 'train_test_data')

mask_NaN = np.load(os.path.join(root_path, 'train_test_data',
                                    'mask_NaN.npy'))
weights = np.load(os.path.join(weight_path, 'sa', 'beta.npy'), 
                        allow_pickle=True).item()
intercepts = np.load(os.path.join(weight_path, 'sa', 'intercept.npy'), 
                        allow_pickle=True).item()

test_X = np.load(os.path.join(data_path, f'test_X_sa.npy'))
test_Y = np.load(os.path.join(data_path, f'test_Y_sa.npy'))

transfer_X = np.load(os.path.join(data_path, f'test_X_fa.npy'))
transfer_Y = np.load(os.path.join(data_path, f'test_Y_fa.npy'))

test_X, transfer_X = scale(test_X), scale(transfer_X)

# mean weights and intercepts
avg_weights = np.zeros(weights[0].size)
for weight in weights.values():
    avg_weights = np.vstack((avg_weights, weight))     
avg_weights = avg_weights[1:, :]
avg_weights = np.mean(avg_weights, axis=0)

avg_intercepts = np.mean([i for i in intercepts.values()])

concat_data_test = np.c_[test_X, test_Y]
concat_data_transfer = np.c_[transfer_X, transfer_Y]

test_lst, transfer_lst = [], []
for i in range(BOOT_TIME):
    start = time.time()
    boot_matrix = bootstrap_by_subject(concat_data_test)
    X_test = boot_matrix[:,:-1]
    y_test = boot_matrix[:,-1]

    boot_matrix = bootstrap_by_subject(concat_data_transfer)
    X_transfer = boot_matrix[:,:-1]
    y_transfer = boot_matrix[:,-1]

    auc_test = compute_auc(avg_weights,
                           avg_intercepts,
                           X_test,
                           y_test)
    auc_transfer = compute_auc(avg_weights,
                               avg_intercepts,
                               X_transfer,
                               y_transfer)
    
    test_lst.append(auc_test)
    transfer_lst.append(auc_transfer)

    end = time.time()
    print(f'Bootstrapping {i} finished, time cost: {end-start}')

# statistical inferance against 0.5 
result_dic = {}
tasks = ['intra', 'inter']
# for 
task = tasks[0]
if task == 'intra':
    ci = CI(np.array(test_lst)) 
else:
    ci = CI(np.array(transfer_lst)) 
lb, hb = "%.2f" % ci[0][0], "%.2f" % ci[0][1]
point = "%.2f" % ci[0][2]
