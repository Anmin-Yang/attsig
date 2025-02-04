import numpy as np
import os
import sys 
import joblib
import argparse
from algo import *

#sys.argv = ['test_script', 'sa']
parser = argparse.ArgumentParser()
parser.add_argument("task",
                   help="{'fa','sa'}",
                   type=str)

if __name__ == '__main__':
    root_path = '/data/home/attsig/attention_signature'
    path = os.path.join(root_path, 'train_test_data')
    args = parser.parse_args()

    # read train and test data 
    train_X = np.load(os.path.join(path, f'train_X_{args.task}.npy'))
    test_X = np.load(os.path.join(path, f'test_X_{args.task}.npy'))
    train_Y = np.load(os.path.join(path, f'train_Y_{args.task}.npy'))
    test_Y = np.load(os.path.join(path, f'test_Y_{args.task}.npy'))

    save_path = os.path.join(root_path, 
                             'svc_analysis', 
                             'prediction_results_pipeline')

    # load model 
    best_model = joblib.load(os.path.join(save_path, f'best_model_{args.task}.pkl'))

    # get model kernal 
    model_kernel = best_model.best_params_['svc__kernel']
    print(f'best model kernel: {model_kernel}')

    # get model C
    model_C = best_model.best_params_['svc__C']
    print(f'best model C: {model_C}')

    # get model gamma
    model_gamma = best_model.best_params_['svc__gamma']
    print(f'best model gamma: {model_gamma}')

    # eavaluate training result with auc 
    train_pred = best_model.predict(train_X)
    train_auc = roc_auc_score(train_Y, train_pred)
    print(f'train_auc: {train_auc}')

    # eavaluate test result with auc
    test_pred = best_model.predict(test_X)
    test_auc = roc_auc_score(test_Y, test_pred)
    print(f'test_auc: {test_auc}')
