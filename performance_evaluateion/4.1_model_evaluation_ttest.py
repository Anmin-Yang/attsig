"""
Statistival inference of Model Performance
1. FA: task0 VS task1
2. SA: task0 VS task1
3. difference of FA VS difference of SA 

match-sample two-sided t-test 

"""
import numpy as np
import os
from algo import *
from scipy.stats import ttest_rel, ttest_1samp, ttest_ind

def load_dic(name):
    arr = np.load(os.path.join(root_path, f'{name}.npy'),
                allow_pickle=True).item()['auc']
    return np.array(arr).reshape(-1,1)  

def print_result(arr1, arr2, task):
    print(f'-----{task}---------')
    print('ttest_result:')
    print(ttest_rel(arr1, arr2))
    print('--------------')
    print('cohen\'d:')
    print(compare_cohen_d(arr1, arr2))
    print('--------------')

if __name__ == '__main__':
    root_path = '/Users/anmin/Documents/attsig_data/model_evaluation/bootstrapped'

    fa_test = load_dic('fa_test')
    fa_transfer = load_dic('fa_transfer')
    sa_test = load_dic('sa_test')
    sa_transfer = load_dic('sa_transfer')

    print_result(fa_test, fa_transfer, 'fa_task')
    print_result(sa_test, sa_transfer, 'sa_task')
    print_result(fa_test - fa_transfer, 
                sa_test - sa_transfer, 
                'fa_sa_compare')