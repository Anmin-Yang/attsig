# Description

This file contains codes responsible for the analysis in **Fig 3** and relevant supplimentary materials.

# Codes

- [1.0_bootstrap_beta_fa.py](1.0_bootstrap_beta_fa.py) [1.0_bootstrap_beta_sa.py](1.0_bootstrap_beta_sa.py)
  - Bootstrap over 1,000 times to reconstruct the confidence interval for every voxel 

- [2.0_fa_bootstrap_significance.py](2.0_fa_bootstrap_significance.py) [2.0_sa_bootstrap_significance.py](2.0_sa_bootstrap_significance.py)
  - FAS and SAS were determined as the significant voxels that passed multi-comparison correction based on the bootstraped distribution