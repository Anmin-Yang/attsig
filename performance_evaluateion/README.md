# Description

- [0_behavioral_results.py](0_behavioral_results.py)
  - Behavioral performance of sbujects
    - d'
    - reaction time 
- [1.0_data_prepare.py](1.0_data_prepare.py)
- Apply the union mask to select features across subjects
  
- Prepares the features and targets for the machine learning algorithm
  
- Training and test data split 
- [2.0.0_SVC_prediction.py](2.0.0_SVC_prediction.py)
  - Apply Support Vector Classification to both FA and SA 
- [2.0.1_svc_prediction_gridsearch.py](2.0.1_svc_prediction_gridsearch.py)
  - Apply gridsearch for best combination of parameters 
- [2.0.2_get_best_model_parameters.py](2.0.2_get_best_model_parameters.py)
  - print out best combination of parameters 
- [2.1.0_svc_cross_validated_rsfk.py](2.1.0_svc_cross_validated_rsfk.py)
  - nested cross-validation 
- [2.1.1_compare_weights.py](2.1.1_compare_weights.py)
  - compare weights between one-shot train-test split and nested crossp-validation training
- [2.1.2_rsfk_summarize.py](2.1.2_rsfk_summarize.py)
  - summarization of nested cross-validation results
- [3.0_model_evaluation_bootstrap.py](3.0_model_evaluation_bootstrap.py)
  - Bootstrap over 10,000 times to construct the confidence interval for metrics of model performance 
- [3.1_model_evaluation_permuation.py](3.1_model_evaluation_permuation.py)
  - permutation test for model performance 