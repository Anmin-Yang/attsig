# Description

This file contains codes responsible for the analysis in **Fig 2.**

# Codes

- [1.0_behavioral_results.py](1.0_behavioral_results.py)
  - Behavioral performance of sbujects
    - d'
    - reaction time 

- [2.0_data_prepare.py](2.0_data_prepare.py)

  - Apply the union mask to select features across subjects

  - Prepares the features and targets for the machine learning algorithm

  - Training and test data split 

- [3.0_SVC_prediction.py](3.0_SVC_prediction.py)
  - Apply Support Vector Classification to both FA and SA 
- [4.0_model_evaluation_bootstrap.py](4.0_model_evaluation_bootstrap.py)
  - Bootstrap over 10,000 times to construct the confidence interval for metrics of model performance 
- [4.1_model_evaluation_ttest.py](4.1_model_evaluation_ttest.py)
  - Statistical inference for model performance 