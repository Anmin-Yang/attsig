import numpy as np
import pandas as pd
import os
import time
import argparse

from algo import *

from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedKFold

parser = argparse.ArgumentParser()
parser.add_argument("task",
                    help="{'fa','sa'}",
                    type=str)


if __name__ == '__main__':
    args = parser.parse_args()

    # define data path
    path = '/data/home/attsig/attention_signature/train_test_data'

    # read train and test data, also test on sa
    train_X = np.load(os.path.join(path, f'train_X_{args.task}.npy'))
    test_X = np.load(os.path.join(path, f'test_X_{args.task}.npy'))
    train_Y = np.load(os.path.join(path, f'train_Y_{args.task}.npy'))
    test_Y = np.load(os.path.join(path, f'test_Y_{args.task}.npy'))

    mask_NaN = np.load(os.path.join(path,'mask_NaN.npy'))

    # num_example = train_X.shape[0]

    # sacle data
    # this is because pipeline is not compatible to .coef_
    train_X, test_X = scale(train_X), scale(test_X)

    svc = SVC(kernel='linear') # default C=1.0

    ############################################################################
    # the C will be fixed to 1.0 to maximize the accuray of the training set
    #params = {'svc__C': np.arange(0, 1.1, 0.1)}

    #gs_svc = GridSearchCV(svc_pipe,
    #                  param_grid=params,
    #                 scoring='accuracy',
    #                  cv=int(num_example/2),
    #                  n_jobs=10) # leave-one-out
    ############################################################################
    random_state = 12883823
    rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=random_state)

    # result holder
    prediction_result = {}
    beta = {}
    intercept = {}

    counter = 0
    for train_inx, test_inx in rkf.split(train_X):
        start = time.time()
        svc.fit(train_X[train_inx], train_Y[train_inx])

        prediction_result[counter] = (train_Y[test_inx],
                                      svc.predict(train_X[test_inx]),
                                      test_Y,
                                      svc.predict(test_X))
        #(truth,prediction,truth,prediction)
        beta[counter] = svc.coef_[0]
        intercept[counter] = svc.intercept_[0]

        end = time.time()
        print(f'The {counter}th iteration of {args.task} is completed,\
              using time {end - start}s.')
        counter += 1

    # save results
    save_path = '/data/home/attsig/attention_signature/svc_analysis/prediction_results'

    
    np.save(os.path.join(save_path, args.task, 'prediction_results.npy'),
            prediction_result)
    np.save(os.path.join(save_path, args.task, 'beta.npy'),
            beta)
    np.save(os.path.join(save_path, args.task, 'intercept.npy'),
        intercept)

    print(f'{args.task}: Saving Completed.')
