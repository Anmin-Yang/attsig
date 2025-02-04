"""
Gradual prediction based on Yeo's 7 networks
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
from nilearn.image import load_img, resample_to_img

parser = argparse.ArgumentParser()
parser.add_argument("it_idx",
                    help="{0, 1, 2, 3, 4 ...}",
                    type=int)
parser.add_argument("data_type",
                    default='fa',
                    help="{'fa', 'sa'}",
                    type=str)

def gen_mask(step_num, yeo_atlas, mask_NaN, network_idx):
    """
    Generate mask for model retrain
    """    
    yeo_mask = yeo_atlas == network_idx
    idx_cat = np.where(yeo_mask)[0] 
    idx_cat = np.random.choice(idx_cat, step_num, replace=False)

    mask_cat_single_cluster = np.zeros_like(yeo_mask) 
    mask_cat_single_cluster[idx_cat] = True

    return mask_cat_single_cluster

def get_auc(model, X, y):
    """
    Return auc score of the model
    """
    y_pred = model.decision_function(X) 
    fpr, tpr, _ = roc_curve(y, y_pred)
    return auc(fpr, tpr)

def compute_auc(network_idx, total_num,
                train_X, train_y,
                test_X, test_y,
                transfer_X, transfer_y,
                mask_NaN,
                master_dic):
    num_steps = int(total_num // 10 + 1)
    dic_temp = {}
    for step_num in range(num_steps):
        if step_num == (num_steps - 1):
            step = total_num
        else:
            step = 10 + 10 * step_num
        # if step > 400, only contiune the for loop if step is a multiple of 100
        #if step > 400 and step % 1000 != 0 and step != total_num:
            #continue
        
        # get data mask 
        data_mask = gen_mask(step, yeo_atlas, mask_NaN, network_idx)

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
        if network_idx not in master_dic:  
            master_dic[network_idx] = dic_temp
        else:
            master_dic[network_idx].update(dic_temp)
        
        if step % 100 == 0:
            print(f'Network {network_idx}, step {step} is done')

    return master_dic

if __name__ == '__main__':
    args = parser.parse_args()
    TRAIN_TIME = 10

    # Load data
    root_path = '/data/home/attsig/attention_signature'
    data_path = os.path.join(root_path, 'train_test_data')
    nii_path = os.path.join(root_path, 
                            'svc_analysis', 
                            'nii_file',
                            'Yeo_JNeurophysiol11_MNI152',
                            'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz')

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

    # yeo atlas 
    yeo_atlas = load_img(nii_path)
    ref_nii = load_img(os.path.join(root_path, 
                                    'svc_analysis', 
                                    'nii_file', 
                                    'fa_fdr_corrected.nii.gz'))
    yeo_atlas = resample_to_img(yeo_atlas, ref_nii)

    yeo_atlas = np.array(yeo_atlas.dataobj).astype('int32')
    yeo_atlas = np.squeeze(yeo_atlas).flatten()[mask_NaN]

    # get the number of voxels in each network
    unique, counts = np.unique(yeo_atlas, return_counts=True)
    dic_yeo_keys = ['BG', 'VIS', 'SOM', 'DAN', 'VAN', 'LIM', 'FPN', 'DMN']
    dic_yeo = dict(zip(dic_yeo_keys, counts))

    loop_holder = {}
    # main loop 
    for i in range(TRAIN_TIME):
        print('----------------------------------')
        start = time.time()
        master_dic = {}
        for network_idx in range(1, 8): # 7 networks
            master_dic = compute_auc(network_idx, counts[network_idx],
                                    train_X, train_y,
                                    test_X, test_y,
                                    transfer_X, transfer_y,
                                    mask_NaN,
                                    master_dic)
        loop_holder[i] = master_dic

        end = time.time()
        print(f'sample time {i} finished, time used: {end-start}')

    # save the result   
    save_path = os.path.join(root_path, 'svc_analysis', 
                            'network_stepwise', 
                            'matched_step',
                            args.data_type)
    np.save(os.path.join(save_path, 
                        f'{args.it_idx}.npy'), 
            loop_holder)