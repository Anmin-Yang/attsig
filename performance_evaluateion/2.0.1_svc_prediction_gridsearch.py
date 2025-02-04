import numpy as np
import os
import time
import sys 
import joblib
import argparse

from algo import *

from sklearn.svm import SVC
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

    
    # Define the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),   
        ('svc', SVC(random_state=42))  
    ])

    # Define the hyperparameter grid
    param_grid = {
        'svc__kernel': ['linear', 'rbf', 'poly'],  
        'svc__C': [0.1, 1, 10, 100],               
        'svc__gamma': ['scale', 'auto']            
    }

    # Set up RepeatedKFold cross-validation
    cv = RepeatedKFold(n_splits=10, 
                       n_repeats=10, 
                       random_state=42)

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=5,   
        verbose=1    
    )

    # Fit the grid search on training data
    grid_search.fit(train_X, train_Y)

    # Evaluate on the test set
    best_model = grid_search.best_estimator_
    test_predictions = best_model.predict(test_X)
    test_accuracy = accuracy_score(test_Y, test_predictions)

    # Print results
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # save model 
    save_path = os.path.join(root_path, 
                             'svc_analysis', 
                             'prediction_results_pipeline')
    joblib.dump(grid_search, 
                os.path.join(save_path, 
                            f'best_model_{args.task}.pkl'))
