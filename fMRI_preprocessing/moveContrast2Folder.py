import os
from os.path import join as pj
import shutil
import pandas as pd

#fileDetection from anatomical and fMRI data
filename_path = '/nfs/s2/userhome/tianjinhua/workingdir/NSP_attention/nspSpm/fileDetection.xls'
subjName = pd.read_excel(filename_path,header=0,sheet_name='fileDetection')
a,b = subjName.shape

folder = '/nfs/s2/userhome/tianjinhua/workingdir/NSP_attention/nspSpm/data'

#fa file name and target folder
faContrastName = ['beta_0001.nii','beta_0002.nii','beta_0003.nii']#'con_0001.nii','con_0002.nii'

contrastFolder = '/nfs/s2/userhome/tianjinhua/workingdir/NSP_attention/nspSpm/betaresults/featureAttention'

contrast = ['feature','conjunction','conjunction_feature']#'conjunction_feature','feature_conjunction'

for i in range(a):
    #moveFolder = '/nfs/e2/workingshop/tianjinhua/NSP_attention/nspSpm/' + subj + '/1st_fa'
    subj = subjName.iloc[i,0]
    behavior = subjName.iloc[i,4]
    subjFolder = pj(folder,subj)
    if behavior == 1:
        for i in range(len(faContrastName)):
            tarName = faContrastName[i]
            targetFolder = pj(contrastFolder, contrast[i])
            if not os.path.exists(targetFolder):
                os.makedirs(targetFolder)

            fileName = pj(subjFolder,'1st_fa',tarName)
            newName = subj + '_' + tarName

            if os.path.isfile(fileName):
                lastName1 = pj(targetFolder,tarName)
                lastName2 = pj(targetFolder,newName)

                shutil.copyfile(fileName, lastName1)
                #lastName1 = pj(targetFolder,tarName)
                #lastName2 = pj(targetFolder,newName)
                os.rename(lastName1, lastName2)


#fa file name and target folder
saContrastName = ['beta_0001.nii','beta_0002.nii','beta_0003.nii']#'con_0001.nii','con_0002.nii'

contrastFolder = '/nfs/s2/userhome/tianjinhua/workingdir/NSP_attention/nspSpm/betaresults/spatialAttention'

contrast = ['central','peripheral','sa']#'conjunction_feature','feature_conjunction'

for i in range(a):
    #moveFolder = '/nfs/e2/workingshop/tianjinhua/NSP_attention/nspSpm/' + subj + '/1st_fa'
    subj = subjName.iloc[i,0]
    behavior = subjName.iloc[i,4]
    subjFolder = pj(folder,subj)
    if behavior == 1:
        for i in range(len(saContrastName)):
            tarName = saContrastName[i]
            targetFolder = pj(contrastFolder, contrast[i])
            if not os.path.exists(targetFolder):
                os.makedirs(targetFolder)

            fileName = pj(subjFolder,'1st_sa',tarName)
            newName = subj + '_' + tarName

            if os.path.isfile(fileName):
                lastName1 = pj(targetFolder,tarName)
                lastName2 = pj(targetFolder,newName)

                shutil.copyfile(fileName, lastName1)
                #lastName1 = pj(targetFolder,tarName)
                #lastName2 = pj(targetFolder,newName)
                os.rename(lastName1, lastName2)