# Description

This file contains codes responsible for the analysis in **Fig 4** and relevant supplimentary materials. 

# Codes

- [1.0_watershed.ipynb](1.0_watershed.ipynb)

  - Watershed algorithm to segment the FAS and SAS into clusters

- [2.0_cluster_table_whole_brain.py](2.0_cluster_table_whole_brain.py)

  - Single-cluster and 'virtual lesion' analysis on the whole brain level 

- [3.0_cluster_retrain.py](3.0_cluster_retrain.py)

  - Retrain predictive models based on clusters with single-cluster and 'virtual lesion' analysis	

- [3.1_cluster_retrain_bootstrap_for_table.py](3.1_cluster_retrain_bootstrap_for_table.py)

  - Bootstrap to construct the confidence interval of the AUC score for each cluster in single-cluster and 'virtual lesion' analysis 

- [3.2_cluster_statistics.py](3.2_cluster_statistics.py)

  - Statistical inference for cluster analysis 

- [3.3_stepwsie_single_cluster.py](3.3_stepwsie_single_cluster.py)

  - Stepwise single-cluster analysis with matched voxel size 

- [3.4_stepwise_single_cluster_whole_brain.py](3.4_stepwise_single_cluster_whole_brain.py)

  - Stepwise analysis in whole-brain signature with matched voxel size

- [4.0_radar_octants.py](4.0_radar_octants.py)

  - Voxel-level analysis

  