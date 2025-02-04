#reorganize data according selected anat and func files
'''
Structure of preprocessing files:
project
    sub0001
        anat
            nii data ()
        sa
            rawdata (unzip and delete the raw data)
            para
        fa
            rawdata (unzip and delete the raw data)
            para

data selection:
1. file detection: anatomical data (later research would be base on this list)
2. mark the selected fa and sa file
3. makedir and move files
'''
import os
from os.path import join as pj
import numpy as np
import pandas as pd
import shutil
import gzip

#filter the fMRI data by anatomical data
funcDir = '/nfs/t1/nspnifti/nii'
newDir = '/nfs/s2/userhome/tianjinhua/workingdir/NSP_attention/nspSpm'

anatPrepDir = '/nfs/s2/userhome/tianjinhua/workingdir/NSP_attention/nspAna'

#fileDetection from anatomical and fMRI data
filename_path = '/nfs/s2/userhome/tianjinhua/workingdir/NSP_attention/nspSpm/fileDetection.xls'
subjName = pd.read_excel(filename_path,header=0,sheet_name='fileDetection')
#subjName  = subjName.iloc[:,0].values
a,b = subjName.shape

funcPath = '/nfs/t1/nspnifti/nii'

#uncompress the file
def un_gz(file_name):
    """ungz zip file"""
    #get file name and remove .gz
    f_name = file_name.replace(".gz", "")
    #create the object gzip
    g_file = gzip.GzipFile(file_name)
    #open and write
    open(f_name, "wb+").write(g_file.read())
    #close the file
    g_file.close()

for i in range(a): #[20:-1]
    subj = subjName.iloc[i, 0]
    behavior = subjName.iloc[i, 4]

    subPaths = pj(funcPath,subj)
    faPath = pj(subPaths,'fa')
    saPath = pj(subPaths,'sa')
    anatPath = pj(subPaths,'3danat')
    memPath1 = pj(subPaths,'mem','002')
    memPath2 = pj(subPaths, 'mem', '004')
    #judge if move file or not?
    #if behavior and fa/sa folder exit, move the fMRI file

    if behavior == 1 and os.path.exists(faPath) and os.path.exists(saPath) and os.path.exists(anatPath) and os.path.exists(memPath1) and os.path.exists(memPath2):
        # make the subj dir / working dir
        subjDir = pj(newDir, subj)
        if not os.path.exists(subjDir):
            os.makedirs(subjDir)

        # make anat dir and move file (003)
        anatDir = pj(subjDir, 'anat')
        if not os.path.exists(anatDir):
            os.makedirs(anatDir)

        # move anat file (preprocessed file)
        anatName = subj + '_ana.nii'
        oriAnatPrepDir = pj(anatPrepDir, anatName)
        shutil.copy(oriAnatPrepDir, anatDir)
        
        for dirs1 in os.listdir(faPath):
            walkPath1 = pj(faPath,dirs1)
            if os.path.isdir(walkPath1): # os.path.isdir and os.path.isflie
                for faFolder in os.listdir(walkPath1):
                    faFile = pj(walkPath1,'func.nii.gz')
                    if os.path.isfile(faFile):
                        # make the dir of raw fa epi
                        newFaDir = pj(subjDir, 'fa')
                        if not os.path.exists(newFaDir):
                            os.makedirs(newFaDir)
                        shutil.copy(faFile, newFaDir)
                        # move the par file
                        paraDir1 = pj(walkPath1, 'fa.par')
                        shutil.copy(paraDir1, newFaDir)

                        newFaFile = pj(newFaDir,'func.nii.gz')
                        un_gz(newFaFile)
                        os.remove(newFaFile)

        for dirs2 in os.listdir(saPath):
            walkPath2 = pj(saPath,dirs2)
            if os.path.isdir(walkPath2):
                for saFolder in os.listdir(walkPath2):
                    saFile = pj(walkPath2, 'func.nii.gz')
                    if os.path.isfile(saFile):
                        # make the dir of raw fa epi
                        newSaDir = pj(subjDir, 'sa')
                        if not os.path.exists(newSaDir):
                            os.makedirs(newSaDir)
                        shutil.copy(saFile, newSaDir)
                        # move the par file
                        paraDir2 = pj(walkPath2, 'sa.par')
                        shutil.copy(paraDir2, newSaDir)

                        newSaFile = pj(newSaDir,'func.nii.gz')
                        un_gz(newSaFile)
                        os.remove(newSaFile)
        '''
        #if both two memory file exist, move the file, unzip the file, remove unzip file, rename files
        memSubPath1 = pj(memPath1, 'func.nii.gz')
        memSubPath2 = pj(memPath2, 'func.nii.gz')
        if os.path.isfile(memSubPath1) and os.path.isfile(memSubPath2):
            newMemDir = pj(subjDir, 'mem')
            if not os.path.exists(newMemDir):
                os.makedirs(newMemDir)
            #define and move two nii files
            newMemFile = pj(newMemDir, 'func.nii.gz')
            newNewMenFile1 = pj(newMemDir,'func1.nii')
            newNewMenFile2 = pj(newMemDir,'func2.nii')
            newName = pj(newMemDir,'func.nii')

            shutil.copy(memSubPath1, newMemDir)
            un_gz(newMemFile)
            os.remove(newMemFile)
            os.rename(newName,newNewMenFile1)

                                                                                                                                                                                                                                                                                                                                                                                                                               shutil.copy(memSubPath2, newMemDir)
            un_gz(newMemFile)
            os.remove(newMemFile)
            os.rename(newName,newNewMenFile2)

            #define and moe two par files
            paraDir1 = pj(memPath1,'mem.par')
            paraDir2 = pj(memPath2,'mem.par')
            newParaDir1 = pj(newMemDir,'mem1.par')
            newParaDir2 = pj(newMemDir,'mem2.par')
            #move and rename file
            shutil.copyfile(paraDir1, newParaDir1)
            shutil.copyfile(paraDir2, newParaDir2)
        '''