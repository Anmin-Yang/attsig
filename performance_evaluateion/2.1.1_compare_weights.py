import numpy as np
import os
import time
import sys 
import argparse
from sklearn.preprocessing import scale
from scipy.stats import pearsonr

from algo import *

#sys.argv = ['test_script', 'sa']
parser = argparse.ArgumentParser()
parser.add_argument("task",
                   help="{'fa','sa'}",
                   type=str)
if __name__ == '__main__':
    args = parser.parse_args()

    root_path = '/data/home/attsig/attention_signature'
    save_path = os.path.join(root_path,
                            'svc_analysis/loop_prediciton_result')
    weight_path = os.path.join(root_path,
                            'svc_analysis/prediction_results')  

    # load rsfk weights 
    rsfk_weights = np.load(os.path.join(save_path, 
                                        f'{args.task}_avg_weights_rsfk.npy'),
                        allow_pickle=True).item()
    batch_size = len(rsfk_weights)

    # load svc weights
    svr_weights = np.load(os.path.join(weight_path, 
                                        args.task, 
                                        'avg_beta.npy')) 

    r = []
    for i in range(batch_size):
        rsfk_weight = rsfk_weights[f'batch_{i}']
        rsfk_weight = scale(rsfk_weight)
        svr_weight = scale(svr_weights)
        res = pearsonr(rsfk_weight, svr_weight)
        r.append(res[0])

    print(f'Pearson correlation coefficient for {args.task}: {np.mean(r)}')
    print(f'std of Pearson correlation coefficient for {args.task}: {np.std(r)}')