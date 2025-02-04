import numpy as np
import os
import time
import argparse
from algo import *

from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedKFold

parser = argparse.ArgumentParser()
parser.add_argument("it_idx",
                    help="{0, 1, 2, 3, 4}",
                    type=int)

if __name__ == '__main__':
    args = parser.parse_args()

    # Load Data
    # define data path
    path = '/data/home/attsig/attention_signature/train_test_data'
    save_path = '/data/home/attsig/attention_signature/svc_analysis/beta_bootstrap'

    # load training data
    train_X_sa = np.load(os.path.join(path,'train_X_sa.npy'))
    train_Y_sa = np.load(os.path.join(path,'train_Y_sa.npy'))
    mask_NaN = np.load(os.path.join(path,'mask_NaN.npy'))

    # scale data
    train_X_sa = scale(train_X_sa)

    # Main Loop
    # set hpyer-paramters
    svc = SVC(kernel='linear') # default C=1.0

    # result holder
    concat_data = np.c_[train_X_sa,train_Y_sa]
    boot_size = 200
    beta = {}

    counter = 0
    for i in range(boot_size):
        print('--------------------------------------------')
        print(f'This is the {i}th main loop to {boot_size}')
        start_loop = time.time()

        # sample new data
        bootstraped_data = bootstrap_by_subject(concat_data)
        X = bootstraped_data[:,:-1]
        Y = bootstraped_data[:,-1]

        #random_state = 12883823
        rkf = RepeatedKFold(n_splits=10, n_repeats=10)

        # result holder
        beta_temp = {}
        for train_inx, test_inx in rkf.split(X):
            start = time.time()
            svc.fit(X[train_inx], Y[train_inx])

            beta_temp[counter] = svc.coef_[0]

            end = time.time()
            #print(f'The {counter}th iteration of is completed,\
                #using time {end - start}s.')
            counter += 1

        # average beta weights over one iteration time
        boot_beta = np.zeros(X[0].shape)
        for val in beta_temp.values():
            boot_beta = np.vstack((boot_beta, val))
        boot_beta = boot_beta[1:,:]

        beta_avg = np.mean(boot_beta, axis=0)

        beta[i] = beta_avg

        end_loop = time.time()
        print(f'The {i}th calculation is completed, using time {end_loop - start_loop}s')

    # save beta weights
    np.save(os.path.join(save_path,f'sa_boot_{args.it_idx}.npy'), beta)
