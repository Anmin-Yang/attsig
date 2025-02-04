import numpy as np
import pandas as pd
import os
import sys 
import argparse
from algo import *

#sys.argv = ['test_script', 'sa']
parser = argparse.ArgumentParser()
parser.add_argument("task",
                   help="{'fa','sa'}",
                   type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    save_path = '/data/home/attsig/attention_signature/svc_analysis/loop_prediciton_result'

    prediction_result = np.load(os.path.join(save_path, 
                                            f'{args.task}_prediction_results_rsfk.npy',
                                            ),
                                allow_pickle=True).item()

    # train data
    summarized_result_train = {'accuracy': [],
                            'precision': [],
                            'recall': [],
                            'f1': [],
                            'roc_auc': []}
    for batch_id, batch_val in prediction_result.items():
        for it_id, it_val in batch_val['cross_validated'].items():
            summarized_result_train['accuracy'].append(it_val['train']['Accuracy'])
            summarized_result_train['precision'].append(it_val['train']['Precision'])
            summarized_result_train['recall'].append(it_val['train']['Recall'])
            summarized_result_train['f1'].append(it_val['train']['F1-Score'])
            summarized_result_train['roc_auc'].append(it_val['train']['AUC'])
    # convert to pd 
    summarized_result_train = pd.DataFrame(summarized_result_train)
    # get mean and sd and write into dataframe 
    mean = summarized_result_train.mean()
    std = summarized_result_train.std()

    df_train = pd.DataFrame({'train_accuracy': ["%.2f" % mean['accuracy'] + "±" + "%.2f" % std['accuracy']],
                            'train_precision': ["%.2f" % mean['precision'] + "±" + "%.2f" % std['precision']],
                            'train_recall': ["%.2f" % mean['recall'] + "±" + "%.2f" % std['recall']],
                            'train_f1': ["%.2f" % mean['f1'] + "±" + "%.2f" % std['f1']],
                            'train_roc_auc': ["%.2f" % mean['roc_auc'] + "±" + "%.2f" % std['roc_auc']]})

    # validation_data 
    summarized_result_valid = {'accuracy': [],
                            'precision': [],
                            'recall': [],
                            'f1': [],
                            'roc_auc': []}
    for batch_id, batch_val in prediction_result.items():
        for it_id, it_val in batch_val['cross_validated'].items():
            summarized_result_valid['accuracy'].append(it_val['valid']['Accuracy'])
            summarized_result_valid['precision'].append(it_val['valid']['Precision'])
            summarized_result_valid['recall'].append(it_val['valid']['Recall'])
            summarized_result_valid['f1'].append(it_val['valid']['F1-Score'])
            summarized_result_valid['roc_auc'].append(it_val['valid']['AUC'])

    # convert to pd
    summarized_result_valid = pd.DataFrame(summarized_result_valid)
    # get mean and sd and write into dataframe
    mean = summarized_result_valid.mean()
    std = summarized_result_valid.std()

    df_valid = pd.DataFrame({'valid_accuracy': ["%.2f" % mean['accuracy'] + "±" + "%.2f" % std['accuracy']],
                                'valid_precision': ["%.2f" % mean['precision'] + "±" + "%.2f" % std['precision']],
                                'valid_recall': ["%.2f" % mean['recall'] + "±" + "%.2f" % std['recall']],
                                'valid_f1': ["%.2f" % mean['f1'] + "±" + "%.2f" % std['f1']],
                                'valid_roc_auc': ["%.2f" % mean['roc_auc'] + "±" + "%.2f" % std['roc_auc']]})

    # test data
    summarized_result_test = {'accuracy': [],
                            'precision': [],
                            'recall': [],
                            'f1': [],
                            'roc_auc': []}
    for batch_id, batch_val in prediction_result.items():
        summarized_result_test['accuracy'].append(batch_val['final']['test']['Accuracy'])
        summarized_result_test['precision'].append(batch_val['final']['test']['Precision'])
        summarized_result_test['recall'].append(batch_val['final']['test']['Recall'])
        summarized_result_test['f1'].append(batch_val['final']['test']['F1-Score'])
        summarized_result_test['roc_auc'].append(batch_val['final']['test']['AUC'])
    # convert to df
    summarized_result_test = pd.DataFrame(summarized_result_test)

    # get mean and sd and write into dataframe
    mean = summarized_result_test.mean()
    std = summarized_result_test.std()

    df_test = pd.DataFrame({'test_accuracy': ["%.2f" % mean['accuracy'] + "±" + "%.2f" % std['accuracy']],
                            'test_precision': ["%.2f" % mean['precision'] + "±" + "%.2f" % std['precision']],
                            'test_recall': ["%.2f" % mean['recall'] + "±" + "%.2f" % std['recall']],
                            'test_f1': ["%.2f" % mean['f1'] + "±" + "%.2f" % std['f1']],
                            'test_roc_auc': ["%.2f" % mean['roc_auc'] + "±" + "%.2f" % std['roc_auc']]})

    # transfer data 
    summarized_result_transfer = {'accuracy': [],
                                'precision': [],
                                'recall': [],
                                'f1': [],
                                'roc_auc': []}
    for batch_id, batch_val in prediction_result.items():
        summarized_result_transfer['accuracy'].append(batch_val['final']['transfer']['Accuracy'])
        summarized_result_transfer['precision'].append(batch_val['final']['transfer']['Precision'])
        summarized_result_transfer['recall'].append(batch_val['final']['transfer']['Recall'])
        summarized_result_transfer['f1'].append(batch_val['final']['transfer']['F1-Score'])
        summarized_result_transfer['roc_auc'].append(batch_val['final']['transfer']['AUC'])
    # convert to df
    summarized_result_transfer = pd.DataFrame(summarized_result_transfer)

    # get mean and sd and write into dataframe
    mean = summarized_result_transfer.mean()
    std = summarized_result_transfer.std()

    df_transfer = pd.DataFrame({'transfer_accuracy': ["%.2f" % mean['accuracy'] + "±" + "%.2f" % std['accuracy']],
                                'transfer_precision': ["%.2f" % mean['precision'] + "±" + "%.2f" % std['precision']],
                                'transfer_recall': ["%.2f" % mean['recall'] + "±" + "%.2f" % std['recall']],
                                'transfer_f1': ["%.2f" % mean['f1'] + "±" + "%.2f" % std['f1']],
                                'transfer_roc_auc': ["%.2f" % mean['roc_auc'] + "±" + "%.2f" % std['roc_auc']]})
                                                
    # combine dfs
    df = pd.concat([df_train, df_valid, df_test, df_transfer], axis=1)
    df.to_csv(os.path.join(save_path, f'{args.task}_summarized_results_rsfk.csv'), index=False)
                            
