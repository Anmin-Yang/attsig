# walk through folders and remove certain files
#work with the reNameflie.py

import os
from os.path import join as pj
import pandas as pd

import shutil
import subprocess
import gzip

#filter the fMRI data by anatomical data
rootDir = '/nfs/s2/userhome/tianjinhua/workingdir/NSP_attention/nspSpm'

#fileDetection from anatomical and fMRI data
filename_path = '/nfs/s2/userhome/tianjinhua/workingdir/NSP_attention/nspSpm/fileDetection.xls'
subjName = pd.read_excel(filename_path,header=0,sheet_name='fileDetection')
#subjName  = subjName.iloc[:,0].values
a,b = subjName.shape
#a = 10
#func1;func2 168 4,164

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

#fslroi /nfs/s2/userhome/tianjinhua/workingdir/NSP_attention/nspSpm/S0001/mem/func1.nii /nfs/s2/userhome/tianjinhua/workingdir/NSP_attention/nspSpm/S0001/mem/func1.nii.gz 4 164
'''
def callFsl(path,file,part1,part2):
    os.chdir(path)
    filepath = pj(path,file)
    filepath2 = pj(path,'func2.nii')
    cmdPrep = 'fslroi ' + filepath + ' ' + filepath2 + ' ' + str(part1) + ' ' + str(part2)
    #subprocess.call(cmdPrep)
    os.system(cmdPrep)
'''

def callFsl(file1,file2,part1,part2):
    cmdPrep = 'fslroi ' + file1 + ' ' + file2 + ' ' + str(part1) + ' ' + str(part2)
    #subprocess.call(cmdPrep)
    os.system(cmdPrep)

for i in range(a): #[20:-1]
    subj = subjName.iloc[i, 0]
    behavior = subjName.iloc[i, 4]

    subPaths = pj(rootDir,subj)
    faPath = pj(subPaths,'fa')
    saPath = pj(subPaths,'sa')
    memPath = pj(subPaths,'mem')
    '''
    #remove volumes and delete file
    faFile = pj(faPath,'func.nii')
    if behavior==1 and os.path.isfile(faFile):
        #remove volumes
        callFsl(faPath,'func.nii',4,160)
        # unzip new file
        newFaFile = pj(faPath,'func2.nii.gz')
        un_gz(newFaFile)
        #remove zipfile
        os.remove(newFaFile)
        os.remove(faFile)

    saFile = pj(saPath, 'func.nii')
    if behavior==1 and os.path.isfile(saFile):
        # make the dir of raw fa epi
        callFsl(saPath, 'func.nii',4,158)
        # unzip new file
        newSaFile = pj(saPath, 'func2.nii.gz')
        un_gz(newSaFile)
        # remove zipfile
        os.remove(newSaFile)
        os.remove(saFile)
    '''

    memFile1 = pj(memPath,'func1.nii')
    memFile2 = pj(memPath,'func2.nii')
    newMemFile1 = pj(memPath, 'func1.nii.gz')
    newMemFile2 = pj(memPath, 'func2.nii.gz')
    if os.path.isfile(memFile1) and os.path.isfile(memFile2):
        # make the dir of raw fa epi
        callFsl(memFile1, newMemFile1, 4,164)
        callFsl(memFile2, newMemFile2, 4,164)
        # remove original file
        os.remove(memFile1)
        os.remove(memFile2)

        un_gz(newMemFile1)
        un_gz(newMemFile2)
        # remove zip file
        os.remove(newMemFile1)
        os.remove(newMemFile2)


'''
# rename modified files
for i in range(a): #[20:-1]
    subj = subjName.iloc[i, 0]
    behavior = subjName.iloc[i, 4]

    subPaths = pj(rootDir,subj)
    faPath = pj(subPaths,'fa')
    saPath = pj(subPaths,'sa')
    #remove volumes and delete file

    newFaFile = pj(faPath,'func2.nii')
    newNewFafile = pj(faPath,'func.nii')
    #rename file
    if behavior==1 and os.path.isfile(newFaFile):
        os.rename(newFaFile,newNewFafile)

    newSaFile = pj(saPath, 'func2.nii')
    newNewSafile = pj(saPath,'func.nii')
    #rename file
    if behavior==1 and os.path.isfile(newSaFile):
        os.rename(newSaFile,newNewSafile)
'''