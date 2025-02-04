# Description

- [1.0_bootstrap_beta_fa.py](1.0_bootstrap_beta_fa.py) [1.0_bootstrap_beta_sa.py](1.0_bootstrap_beta_sa.py)
  - Bootstrap over 1,000 times to reconstruct the confidence interval for every voxel 

- [2.0_fa_bootstrap_significance.py](2.0_fa_bootstrap_significance.py) [2.0_sa_bootstrap_significance.py](2.0_sa_bootstrap_significance.py)
  - FAS and SAS were determined as the significant voxels that passed multi-comparison correction based on the bootstrapped distribution
- [2.1_compute_cluster_correlation.py](2.1_compute_cluster_correlation.py)
  - Pearson's correlation of the conjunctin areas of the FAS and SAS
- [3.0_graudual_prediction_network.py](3.0_graudual_prediction_network.py) 
  - Network-level analysis with matched voxel size based on Yeo's atlas 
- [4.0_univariate_analysis.py](4.0_univariate_analysis.py) 
  - Univariate analysis of the FAS and the SAS, significant voxels threholded with Cohen's d 
- [5.0_yeo_network.py](5.0_yeo_network.py)
  - The intensity of the predictive voxels in the FAS and SAS defined by Yeo's network 

