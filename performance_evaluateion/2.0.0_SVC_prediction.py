import numpy as np
import os
import time
import argparse
import sys

from algo import *

from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.model_selection import RepeatedKFold

#sys.argv = ['test_script', 'sa']
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

    train_X, test_X = scale(train_X), scale(test_X)

    svc = SVC(kernel='linear', random_state=42)  
    random_state = 12883823
    rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=random_state)

    # result holder
    prediction_result = {}
    beta = {}
    intercept = {}

    counter = 0
    for train_inx, valid_inx in rkf.split(train_X):
        start = time.time()
        svc.fit(train_X[train_inx], train_Y[train_inx])

        y_true_train = train_Y[train_inx]
        y_true_valid = train_Y[valid_inx]
        y_train_predict = svc.predict(train_X[train_inx])
        y_valid_predict = svc.predict(train_X[valid_inx])
        y_train_prob = svc.decision_function(train_X[train_inx])
        y_valid_prob = svc.decision_function(train_X[valid_inx])

        prediction_result[counter] = {'train': summarize_results(y_true_train, 
                                                                y_train_predict,
                                                                y_train_prob),
                                    'valid': summarize_results(y_true_valid, 
                                                                y_valid_predict,
                                                                y_valid_prob)}
        
        beta[counter] = svc.coef_[0]
        intercept[counter] = svc.intercept_[0]

        end = time.time()
        print(f'The {counter}th iteration of {args.task} is completed,\
                using time {end - start}s.')
        counter += 1

    # save results
    save_path = '/data/home/attsig/attention_signature/svc_analysis/prediction_results'


    np.save(os.path.join(save_path, args.task, 'prediction_results_summarized.npy'),
            prediction_result)
    np.save(os.path.join(save_path, args.task, 'beta.npy'),
            beta)
    np.save(os.path.join(save_path, args.task, 'intercept.npy'),
        intercept)

    print(f'{args.task}: Saving Completed.')
