import numpy as np
import os
import time
import sys 
import joblib
import argparse

from algo import *

from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, scale

#sys.argv = ['test_script', 'sa']
parser = argparse.ArgumentParser()
parser.add_argument("task",
                   help="{'fa','sa'}",
                   type=str)

if __name__ == '__main__':
    path = '/data/home/attsig/attention_signature/train_test_data'
    save_path = '/data/home/attsig/attention_signature/svc_analysis/loop_prediciton_result'


    args = parser.parse_args()

    job_lst = {'fa', 'sa'}
    task_lst = job_lst - {args.task}
    transfer_task = task_lst.pop()

    # load data 
    X = np.load(os.path.join(path, f'X_{args.task}.npy'))
    y = np.load(os.path.join(path, f'Y_{args.task}.npy'))

    X_transfer = np.load(os.path.join(path, f'X_{transfer_task}.npy'))
    y_transfer = np.load(os.path.join(path, f'Y_{transfer_task}.npy'))

    rsfk = RepeatedStratifiedKFold(n_splits=5, 
                                    n_repeats=10, 
                                    random_state=42)


    prediction_result = {}
    avg_weight_dic = {}
    for i, (train_index, test_index) in enumerate(rsfk.split(X, y)):
        print(f"batch {i}:")

        prediction_result[f'batch_{i}'] = {}
        beta_dic = {}
        intercept_dic = {}

        train_temp_X = X[train_index]
        train_temp_y = y[train_index]

        test_temp_X = X[test_index]
        test_temp_Y = y[test_index]

        rskf_nested = RepeatedStratifiedKFold(n_splits=10, 
                                            n_repeats=10, 
                                            random_state=42)

        counter = 0
        for nested_train_inx, nested_valid_inx in rskf_nested.split(train_temp_X, train_temp_y):
            start = time.time()

            pipeline = Pipeline([
                ('scaler', StandardScaler()),   
                ('svc', SVC(kernel='linear', random_state=42))  
                ])

            pipeline.fit(train_temp_X[nested_train_inx], train_temp_y[nested_train_inx]) 

            y_true_train = train_temp_y[nested_train_inx]
            y_true_valid = train_temp_y[nested_valid_inx]
            y_train_predict = pipeline.predict(train_temp_X[nested_train_inx])
            y_valid_predict = pipeline.predict(train_temp_X[nested_valid_inx])
            y_train_prob = pipeline.decision_function(train_temp_X[nested_train_inx])
            y_valid_prob = pipeline.decision_function(train_temp_X[nested_valid_inx])

            if counter == 0: 
                    prediction_result[f'batch_{i}']['cross_validated'] = {}
            prediction_result[f'batch_{i}']['cross_validated'][counter] = {'train': summarize_results(y_true_train, 
                                                                                                    y_train_predict,
                                                                                                    y_train_prob),
                                                                            'valid': summarize_results(y_true_valid, 
                                                                                                        y_valid_predict,
                                                                                                        y_valid_prob)}
                
            svc = pipeline.named_steps['svc']
            beta_dic[counter] = svc.coef_[0]
            intercept_dic[counter] = svc.intercept_[0]

            end = time.time()
            print(f'The {counter}th iteration of {args.task} is completed,\
                    using time {end - start}s.')
            counter += 1
        
        # mean weights and intercepts
        avg_weights = np.zeros(beta_dic[0].size)
        for weight in beta_dic.values():
            avg_weights = np.vstack((avg_weights, weight))     
        avg_weights = avg_weights[1:, :]
        avg_weights = np.mean(avg_weights, axis=0)
        avg_weight_dic[f'batch_{i}'] = avg_weights

        avg_intercepts = np.mean([i for i in intercept_dic.values()])
        
        # evaluation on test and transfer set 
        y_test_predict, y_test_prob = decision_value(avg_weights,
                                                    avg_intercepts,
                                                    scale(test_temp_X))
        y_transfer_predict, y_transfer_prob = decision_value(avg_weights,
                                                            avg_intercepts,
                                                            X_transfer)

        buddled_reslut = {'test': summarize_results(test_temp_Y, 
                                                    y_test_predict,
                                                    y_test_prob),
                        'transfer': summarize_results(y_transfer, 
                                                    y_transfer_predict,
                                                    y_transfer_prob)}
        prediction_result[f'batch_{i}']['final'] = buddled_reslut

    # save result 
    np.save(os.path.join(save_path, f'{args.task}_prediction_results_rsfk.npy'),
            prediction_result)
    np.save(os.path.join(save_path, f'{args.task}_avg_weights_rsfk.npy'),
            avg_weight_dic)