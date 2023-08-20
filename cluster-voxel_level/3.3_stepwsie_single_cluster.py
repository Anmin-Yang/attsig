"""
Gradual single cluster analysis 

Ramdomrize mask for every step, for X and test data 

Retrain modles 

Pararalized computation, each with 10 steps, totalling 100 steps

Dictionary Structure
--------------------
{{train_time: {cluster_idx : {step_num: [auc_test, auc_transfer]}
              }
    }
}
...
}
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

parser = argparse.ArgumentParser()
parser.add_argument("it_idx",
                    help="{0, 1, 2, 3, 4 ...}",
                    type=int)
parser.add_argument("data_type",
                    help="{'fa', 'sa'}",
                    type=str)

def gen_mask(step_num, series, data_type, mask_NaN):
    """
    Generate mask for model retrain
    """
    cluster_idx = series['Unnamed: 0']
    num_voxel = series['number_of_voxels']
    
    # load cluster mask 
    nii_arr = image.get_data(os.path.join(nii_path, 
                f'{cluster_idx}_{num_voxel}_{data_type}.nii.gz')).flaqtten()     
    nii_arr = nii_arr[mask_NaN]

    # pick out voxels of interest
    mask_cat = nii_arr == cluster_idx
    # get the index of the element that is True
    idx_cat = np.where(mask_cat)[0] 
    # randomly select elements from the index
    idx_cat = np.random.choice(idx_cat, step_num, replace=False)

    # generate single cluster mask, 
    # modify mask_cat where only the inx_cat is True
    mask_cat_single_cluster = np.zeros_like(mask_cat)  
    mask_cat_single_cluster[idx_cat] = True

    return mask_cat_single_cluster

def get_auc(model, X, y):
    """
    Return auc score of the model
    """
    y_pred = model.decision_function(X) 
    fpr, tpr, _ = roc_curve(y, y_pred)
    return auc(fpr, tpr)

def compute_auc(series,
                train_X, train_y,
                test_X, test_y,
                transfer_X, transfer_y,
                data_type, mask_NaN,
                master_dic):
    num_steps = int(series['number_of_voxels'] // 10 + 1)
    dic_temp = {}
    for step_num in range(num_steps):
        if step_num == (num_steps - 1):
            step = series['number_of_voxels']
        else:
            step = 10 + 10 * step_num
        # if step > 400, only contiune the for loop if step is a multiple of 100
        if step > 400 and step % 100 != 0 and step != series['number_of_voxels']:
            continue
        
        # get data mask 
        data_mask = gen_mask(step, series, data_type, mask_NaN)

        # get data
        train_X_temp = train_X.copy()
        train_X_temp = train_X_temp[:, data_mask]

        test_X_temp = test_X.copy()
        test_X_temp = test_X_temp[:, data_mask]

        transfer_X_temp = transfer_X.copy()
        transfer_X_temp = transfer_X_temp[:, data_mask]

        # train model
        svc = SVC(kernel='linear')
        svc.fit(train_X_temp, train_y)

        # test model and stroe 
        dic_temp[step] = [get_auc(svc, test_X_temp, test_y),
                          get_auc(svc, transfer_X_temp, transfer_y)]
        
        # store to the master dic 
        if series['Unnamed: 0'] not in master_dic:  
            master_dic[series['Unnamed: 0']] = dic_temp
        else:
            master_dic[series['Unnamed: 0']].update(dic_temp)
        
    return master_dic

if __name__ == '__main__':
    args = parser.parse_args()
    TRAIN_TIME = 100

    # Load data
    root_path = '/data/home/attsig/attention_signature'
    data_path = os.path.join(root_path, 'train_test_data')
    df_path = os.path.join(root_path, 'svc_analysis', 'cluster_table_small_version')
    nii_path = os.path.join(root_path, 'svc_analysis', 'nii_file', 'cluster_map_small_version',
                            f'{args.data_type}_screened')

    # load data of prediction 
    mask_NaN = np.load(os.path.join(root_path, 'train_test_data',
                                        'mask_NaN.npy'))

    train_X = np.load(os.path.join(data_path, f'train_X_{args.data_type}.npy'))
    train_y = np.load(os.path.join(data_path, f'train_Y_{args.data_type}.npy'))

    test_X = np.load(os.path.join(data_path, f'test_X_{args.data_type}.npy'))
    test_y = np.load(os.path.join(data_path, f'test_Y_{args.data_type}.npy'))

    names = ['fa', 'sa'] 
    names.remove(args.data_type)

    transfer_X = np.load(os.path.join(data_path, f'test_X_{names[0]}.npy'))
    transfer_y = np.load(os.path.join(data_path, f'test_Y_{names[0]}.npy'))

    train_X, test_X, transfer_X = scale(train_X), scale(test_X), scale(transfer_X)

    # dataframe 
    df_data_fa = pd.read_csv(os.path.join(df_path,
                        'fa_integrated.csv'))
    df_data_sa = pd.read_csv(os.path.join(df_path,
                            'sa_integrated.csv'))
    idx_lst = list(df_data_sa['Unnamed: 0'])
    if args.data_type == 'fa':
        df_data = df_data_fa[df_data_fa['Unnamed: 0'].isin(idx_lst)]
    else:
        df_data = df_data_sa

    loop_holder = {}
    # main loop 
    for i in range(TRAIN_TIME):
        start = time.time()
        master_dic = {}
        for index, series in df_data.iterrows():
            master_dic = compute_auc(series,
                                    train_X, train_y,
                                    test_X, test_y,
                                    transfer_X, transfer_y,
                                    args.data_type, mask_NaN,
                                    master_dic)
        loop_holder[i] = master_dic

        end = time.time()
        print('----------------------------------')
        print(f'sample time {i} finished, time used: {end-start}')

    # save the result   
    save_path = os.path.join(root_path, 'svc_analysis', 'cluster_stepwise_random_retrain', 
                            f'{args.data_type}')
    np.save(os.path.join(save_path, f'{args.it_idx}.npy'), loop_holder)