"""
Gradual single cluster analysis applied to whole brain 

Retrain models 
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
    idx_cat = np.random.choice(np.arange(np.sum(mask_NaN)), 
                               step_num, replace=False)

    mask_cat_single_cluster = np.zeros(np.sum(mask_NaN), dtype=bool)  
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
        # if step > 10000, only contiune the for loop if step is a multiple of 100
        #if step > 10000 and step % 100 != 0 and step != series['number_of_voxels']:
            #continue
        
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
        if series['type'] not in master_dic:  
            master_dic[series['type']] = dic_temp
        else:
            master_dic[series['type']].update(dic_temp)
        #print(f'finish step: {step} ')

        if step % 1000 == 0:
            print(f'step {step} is done')

    return master_dic

if __name__ == '__main__':
    args = parser.parse_args()
    TRAIN_TIME = 10

    # Load data
    root_path = '/data/home/attsig/attention_signature'
    data_path = os.path.join(root_path, 'train_test_data')
    df_path = os.path.join(root_path, 'svc_analysis', 'cluster_table_small_version')
    nii_path = os.path.join(root_path, 'svc_analysis', 'nii_file')

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

    # generate the cluster table
    df_data = pd.DataFrame(np.array([args.data_type, np.sum(mask_NaN)]).reshape(1, 2),
                        columns=['type', 'number_of_voxels'])
    df_data['number_of_voxels'] = df_data['number_of_voxels'].astype(int)

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
    save_path = os.path.join(root_path, 'svc_analysis', 
                            'cluster_stepwise_random_retrain',
                            'whole_brain_matched_step',
                            args.data_type)
    np.save(os.path.join(save_path, 
                        f'{args.it_idx}.npy'), 
            loop_holder)