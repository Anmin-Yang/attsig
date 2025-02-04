import os
import numpy as np
import pandas as pd 
import nibabel as nib
from nilearn import image
from sklearn.preprocessing import scale,MinMaxScaler
from scipy.stats import norm
import time

import collections

from sklearn.metrics import accuracy_score, precision_score, \
                            recall_score, f1_score, roc_curve, auc, roc_auc_score

def X_matrix(sub_name,path,model_type,feature_num):
    """
    generate X input matrix for predictor
    2D array (subjects,features)

    Parameter
    ---------
    sub_name: list
        list of sujects' names
    path: str
        path containing files of every subject
    model_type: str
        'fa': model trained on feature-based attention
        'sa': model trained on spacial-based attention
    feature_num: int
        number of features, how many voxels in one subject

    Retrun
    ------
    X: ndarray
        2D array of feature matrix, (subjects,features)
    """

    X = np.zeros(feature_num)

    if model_type == 'fa':
        file_name = '1st_fa'
        for name in sub_name:
            path_temp = os.path.join(path,name,file_name)
            feature_array = image.get_data(os.path.join(path_temp,'beta_0001.nii')).flatten()
            conjunction_array = image.get_data(os.path.join(path_temp,'beta_0002.nii')).flatten()
            X = np.vstack((X,feature_array,conjunction_array))
    else:
        file_name = '1st_sa'
        for name in sub_name:
            path_temp = os.path.join(path,name,file_name)
            array1 = image.get_data(os.path.join(path_temp,'beta_0001.nii')).flatten()
            array2 = image.get_data(os.path.join(path_temp,'beta_0002.nii')).flatten()
            X = np.vstack((X,array1,array2))

    X = X[1:,:]

    return X

def Y_matrix(X):
    """
    generate Y input matrix for predictor
    1D array (labels)

    Parameter
    ---------
    X: int
        label number according to the shape of input X (how many subjects)

    Return
    ------
    Y: ndarray
        1D array of labels
        (0,1) periodically for fa where 0 denotes feature, 1 denotes conjunction
        (0,1) periodically for sa  where 0 denotes central, 1 denotes peripheral

    """
    length = X.shape[0]
    Y = np.array((0,1)*(int(length/2)))

    return Y

def nan_mask(matrix):
    """
    find features that contain NaN and return a NaN mask

    Parameter
    ---------
    matrix: ndarray
        2D matrix (subjects,features)(m,n)

    Retrun
    ------
    mask: ndarray
        1D array, if one value esistes in one feature, that feature is eliminated
        represented in bool value
    """
    is_nan = np.isnan(matrix)
    is_nan_array = is_nan.sum(axis=0) # is a NaN in this feature, then the sum is not 0
    mask = (is_nan_array == 0)

    return mask

def mask_back(mask_matrix,value_matrix,mask_type):
    """
    back project value matrix to original matrix
    the shape of the matrix returned shoud be like mask matrix
    used for reduced beta coefficients and mni space

    Parameter
    ---------
    mask_matrix: ndarray
        matrix with bool value, 1D array
    value_matrix: ndarray
        matrix with real value, the size of which is smaller than mask matrix
    mask_type: string
        the type of mask_matrix
        'beta': mask the present beta value back to full PC size,
            in which case the masked out value will be filled with 0
        'mni': mask the reverse transformed beta value to full mni space,
            in which case the masked out value will be set as nan
        'z_scored_beta': mask the useful voxels into full beta space after z-scored
        'z_scored_mni': after full beta space is z-scored, the beta space is converted into mni space

    Return
    ------
    reshaped_matrix: ndarray
        projecting real value to original matrix
        if the mask_type is z_scored in full beta size,
        the returned reshaped matrix is full-size matrix with z_scored value inside
        the max of all activation is (max(z-value)+1)
        
        if the mask_type is z_scored in full mni size.
        in the returned matrix, the nan value of voxels are converted into (min(z-value)-1)

    """
    lis_mask = list(mask_matrix.flatten())
    length_mask = len(lis_mask)
    lis_value = list(value_matrix.flatten())

    if 'z_scored' not in mask_type:
        counter = 0
        for i in range(length_mask):
            if lis_mask[i]:
                lis_mask[i] = lis_value[counter]
                counter += 1
            else:
                if mask_type == 'mni':
                    lis_mask[i] = np.nan
                else:
                    continue
    
    elif mask_type=='z_scored_beta': # when 'z_scored' is applied
        #lis_value = scale(lis_value) # z-scored are moved to the whole activation space
        insert_min = np.min(value_matrix[value_matrix>0])-0.1
        counter = 0
        for i in range(length_mask):
            if lis_mask[i]:
                lis_mask[i] = lis_value[counter]
                counter += 1
            else:
                lis_mask[i] = insert_min
    elif mask_type=='z_scored_mni':
        insert_min = np.min(value_matrix[value_matrix>0])
        counter = 0
        for i in range(length_mask):
            if lis_mask[i]:
                lis_mask[i] = lis_value[counter]
                counter += 1
            else:
                lis_mask[i] = insert_min
            
    reshaped_matrix = np.array(lis_mask)  
    
    return reshaped_matrix

def summarize_results(y_true, y_pred, y_prob=None, average='binary'):
    """
    Summarizes prediction results with key metrics.
    
    Parameters
    ----------
    y_true: Ground truth (true labels)
    y_pred: Predicted labels
    y_prob: Predicted probabilities (for AUC, optional)
    average: Averaging method for multiclass ('binary', 'macro', 'weighted', etc.)
    
    Returns
    -------
    metrics: Dictionary of computed metrics
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    if y_prob is not None:
        # Only compute AUC if probabilities are provided
        metrics["AUC"] = roc_auc_score(y_true, y_prob, 
                                       multi_class='ovr' if average != 'binary' \
                                        else 'raise')
    
    return metrics

def bootstrap_data(original_data):
    """
    bootstrap from orininal data to create a new data set, sample with replacement

    Parameter
    ---------
    original_data: ndarray
        2D matrix, (subjects,features) (m,n)

    Return
    ------
    new_matrix: ndarray
    
        new data set by bootstrap, sample with repalcement
        2D matrix, (subjects,features) (m,n)
    """
    select_range = original_data.shape[0]
    indexes = np.random.choice(a=select_range,size=select_range)

    new_matrix = np.zeros(original_data.shape[1])
    for index in indexes:
        new_matrix = np.vstack((new_matrix,original_data[index]))

    new_matrix = new_matrix[1:,:]

    return new_matrix

######################################################
# this function is faulty
# as what it really computing is the rank of real data among bootstrapped data,
# idexed by p-value
def p_transfer(data_matrix):
    """
    transfer beta map to p-map, from beta to p value
    return the p value of the observed array

    Parameter
    ---------
    data_matrix: ndarray
        beta map, 2D array, (m,n)
        m: number of subjects
        n: number of features, denoting number of voxels

    Return
    ------
    p_map: ndarray
        1D array, (n,)
        every element denotes the p value of that voxel
    """
    start = time.time()

    length = data_matrix.shape[1] # how many features
    data_matrix = scale(data_matrix).flatten() # convert to Z score
    data_matrix = data_matrix[-length:] # only the real z-score instead of bootstrapped scores are considered

    mid = time.time()
    print(f'Converting Z scores is complete, using time {mid-start} seconds.')

    p_map = [norm.cdf(value) for value in data_matrix]
    p_map = np.array(p_map)

    end = time.time()
    print(f'Calculation of p value is complete, using time {end-mid} seconds.')

    return p_map
########################################################

def CI(data_matrix,percentile=0.95):
    """
    calcuate the confidence interal of the bootstraped data
    
    Parameter:
    ----------
    data_matrix: ndarray
        matrix in shape (m,n),
        m: number of bootstrap time 
        n: number of features 
    percentile: float
        the percntile of confidence interval, set to two sides
        default to 0.95 
    
    Return:
    -------
        ci: ndarray
            (m,3), m is the feature, equaling the n in matrix shape
                   3 is the CI and mean, (lower bound, upper bound,mean)
    """
    mean = np.mean(data_matrix,axis=0)
    std = np.std(data_matrix,axis=0)
    l_ci = norm.ppf((1-percentile)/2,mean,std)
    h_ci = norm.ppf((percentile+(1-percentile)/2),mean,std)
    
    ci = np.c_[l_ci.T,h_ci.T,mean]
    
    return ci            
            
    
def p_ci(data_matrix,alpha,two_tail,null_hyphothesis):
    """
    compute the confidence interval of sampled data in every feature,
    give alpha level
    
    Parameter
    ---------
    data_matrix: ndarray
        bootstraped data, 2D array, (m,n)
        m: number of examplers
        n: number of features
        if the input is an array with shape(n,), it should be reshaped into(-1,1)
    alpha: float
        alpha level for statistical inference
    two_tail: bool
        whether the inference is two-tailed
    null_hyphothesis: float
        the null hyphothesis for statitical inference, invarient in every feature
        for beta value inference, null_hyphothesis=0
        for binary classification task, null_hyphothesis=0.5
    
    Return
    ------
    inference: dic
        'p_map': ndarray
            (n,), the p value of every feature
        'is_pass': ndarray
            (n,), with bool, whether the paticular feature passes inference
            True if passes
        'ci': ndarray
            (n,2), the two features are the lower bound and upper bound 
            of the confidence interval    
    """
    if two_tail: # two tail inference, percental on each tail is half of alpha 
        percentail = alpha/2
    else: 
        percentail = alpha

    # dertermine the confidence interval of bootstrapped data 
    ci_lower = np.percentile(data_matrix,percentail*100,axis=0) # shape(n,)
    ci_upper = np.percentile(data_matrix,(1-percentail)*100,axis=0)
    ci = np.c_[ci_lower,ci_upper] 
    
    # dertermine p map 
    null_array = np.ones(data_matrix.shape[1])*null_hyphothesis
    data_matrix = np.r_[data_matrix,null_array.reshape(1,-1)]
    
    sort_index = data_matrix.argsort(axis=0) # index from small to large 
    
    rank_index = np.argwhere(sort_index==(sort_index.shape[0]-1))
    rank_index = rank_index[:,0] # the index of every column, denoting the rank of each feature 
    p_map = rank_index/(data_matrix.shape[0]) # top k% of all data, denoting p value
    
    # statistical inference 
    
    ## left or right bias 
    is_right = (null_hyphothesis<np.median(data_matrix[:-1,:],axis=0)) # whether the distribution is at right of null hypothesis
    is_pass = []
    for i in range(is_right.size):
        if is_right[i]:
            if p_map[i]<percentail:
                is_pass.append(True)
            else:
                is_pass.append(False)
        else:
            if (1-p_map[i])<percentail:
                is_pass.append(True)
            else:
                is_pass.append(True)
    
    is_pass = np.array(is_pass)
    
    # generate dic for the returned results
    inference = {'p_map':p_map,
                'is_pass':is_pass,
                'ci':ci}
    return inference 
    
        

def to_mni(data_array):
    """
    convert array(beta) to mni space
    specific for the analysis

    Parameter
    ---------
    data_array: ndarray
        1D array, contraining beta value

    Return
    ------
    mni_data: Nifti1Image
        NiftiImage, with space and afine specific for this analysis
    """
    # acquire sample parameters
    sample_path = '/data/home/attsig/attention_signature/sample_data/S0001/1st_fa/beta_0001.nii'
    sample_nii = nib.load(sample_path)
    ori_shape = np.array(sample_nii.get_data()).shape
    affine = sample_nii.affine.copy()
    hdr = sample_nii.header.copy()

    del sample_nii

    mni_data = nib.Nifti1Image(data_array.reshape(ori_shape),affine,hdr)

    return mni_data

def p_threshold(path_beta,path_p,threshold):
    """
    define a p_threshold of activation
    the value below the threshold is set to 0

    Parameter
    ---------
    path_beta: str
        the path of nii file, beta value
    path_p: str
        the path of nii file, p value
    threshold: float
        the threshold of p value, range is [0,1]

    Return
    ------
    masked_array: ndarray
        the value below threshold is set to 0
    """
    # basic parameters of nii file
    nii_map_beta = nib.load(path_beta)
    activation_map = nii_map_beta.get_data()

    nii_map_p = nib.load(path_p)
    p_map = nii_map_p.get_data()

    mask1 = (p_map>(1-threshold/2))
    mask2 = (p_map<(threshold/2))
    is_above_t = np.logical_or(mask1,mask2)
    corrected_map = activation_map*is_above_t

    return corrected_map

def vectorized_back(ori_array,modified_array):
    """
    the modified_array is manipulated by abs,and hense a scalar, lossing orientation information as a vector
    the orientation information is preserved in ori_array, derterming whether a scalar is positive or negative
    this function change a scalar back to vector

    Parameter
    ---------
    ori_array: ndarray
        1-D array, the original array storing orientation information
    modified_array: ndarray
        1-D array, the modified array lossing orientation information

    Return
    ------
    vectorized_array: ndarray
        1-D array, each item is a vector, storing both orientaiton information and modified value
    """
    mask_minus = (ori_array<0)
    input_array = list(modified_array)

    for i in range(mask_minus.size):
        if mask_minus[i]: # the original value is negtive
            input_array[i] = -1*input_array[i]
    
    mask_zero = (ori_array==0)
    for i in range(mask_zero.size):
        if mask_zero[i]: # the original value is 0
            input_array[i] = 0

    vectorized_array = np.array(input_array)

    return vectorized_array

def stat_infer(data_matrix, alpha=0.05, is_two_tail=True):
    """
    statistical inference whether a specific voxel is statistically significant,
    given alpha level
    the inference is based on the bootstrapped data 
    
    Parameter:
    ------ ----
    data_matrix: ndarray
        shape (m,n)
        m: bootstrap time
        n: number of features
    alpha: float
        alpha level, 
        default at 0.05
    is_two_tail: bool
        whether the statistical inference is two-tail or one-tail
        default is two-tail 
    
    Return:
    -------
    bool_map: ndarray
        boolean value in each element
        shape (1,n)
    """
    if is_two_tail:
        percentile = 1 - alpha/2
    else:
        percentile = 1 - alpha
    
    ci = CI(data_matrix,percentile) # low_ci, high_ci, mean 
    
    bool_map = []
    for i in range(data_matrix.shape[1]):
        if (0>ci[i,0]) and (0<ci[i,1]):
            bool_map.append(False)
        else:
            bool_map.append(True)
    
    bool_map = np.array(bool_map).reshape(1,-1)
    return bool_map

def p_value_continue(data_matrix,null_hyphothesis=0): 
    """
    Parameters
    ----------
    data_matrix : ndarray
        the bootstraped data,
        in shape(m,n)
        m is the number of bootstrap time
        n is the number of features
    null_hyphothesis : float
        the h0 hyphothesis
        

    Returns
    -------
    p_val: ndarray
        in shape(n,)
        every feature carries a p-value

    """
    p_val = []
    mean = np.mean(data_matrix,axis=0)
    std = np.std(data_matrix,axis=0)
    
    for i in range(data_matrix.shape[1]):
        arr = data_matrix[:,i]
        
        if null_hyphothesis < mean[i]:
            p_temp = norm.cdf(null_hyphothesis,mean[i],std[i])
        else: 
            p_temp = 1 - norm.cdf(null_hyphothesis,mean[i],std[i])
         
        p_val.append(p_temp)
        
    
    p_val = np.array(p_val)
    
    return p_val

def p_value_discrete(boot_li,null_h=0):
    """
    Parameters
    ----------
    boot_li: list
        bootsrapped data as a distribution
    null_h: float
        null hypothesis
    
    Returns
    -------
    p_val: float
        p value
    """
    p_val = np.sum(null_h>np.array(boot_li))/len(boot_li)
    
    return p_val
    
def p_value_arr(arr, val, type):
    """
    Calculate the p value of a specific value in a distribution

    Parameters
    ----------
    arr: list
        the distribution
    val: float
        the value to be tested
    type: str
        'left' or 'right' or 'two_sided'
    
    Returns
    -------
    p_val: float
        p value
    """
    if type == 'left':
        p_val = np.sum(val>np.array(arr))/len(arr)
    elif type == 'right':
        p_val = np.sum(val<np.array(arr))/len(arr)
    elif type == 'two_sided':
        p_val = np.sum(val<np.array(arr))/len(arr) + np.sum(val>np.array(arr))/len(arr)
    else:
        raise ValueError('type should be left or right or two_sided')
    
    return p_val


def stat_infer_p(p_map,alpha_level,two_tail=True):
    """
    Parameters:
    -----------
    p_map: array_like
        (n,)
    alpha_level: float
        alpha level
    two_tail: bool
        whether the statistical inference is tow-tailed
        default at True 
    
    Return:
    -------
    mask: array_like
        (n,)
        bool_value whether a feature passed the statistical inference 
    """
    if two_tail:
        percentile = 1 - alpha_level/2
    else:
        percentile = 1 - alpha_level
    
    mask = []
    for value in p_map:
        if (value<percentile) or (value>(1-percentile)):
            mask.append(True)
        else:
            mask.append(False)
    mask = np.array(mask)
    
    return mask 

def generate_com_diff_mask(fa_array,sa_array,criteria=0.5):
    """
    generate mask of common and diff map
    the rank of the voxels in each condition is determined by the absolute value of beta 
    
    Parameters:
    -----------
    fa_array: ndarray
        the nan_masked beta value of FA map 
        in shape(1,n)
    sa_array: ndarray
        the nan_masked beta value of SA map 
        in shape(1,n)
    criteria: float
        the top k in both map will be integrated into common map 
        n is in range (0,1), and default at 0.5, which is also the medium 
    
    Return:
    -------
    common_map_fa: ndarray
        bool value in each element  
        in shape(1,n)
    common_map_sa: ndarray 
        bool value in each element  
        in shape(1,n)
    diff_map_fa: ndarray
        bool value in each element 
        in shape(1,n)
    diff_map_sa: ndarray
        bool value in each element 
        in shape(1,n)
    """
    # min_max sacle to range (0,1)
    scaler = MinMaxScaler()
    fa_array = scaler.fit_transform(abs(fa_array).T).T # the scale operates within features 
    sa_array = scaler.fit_transform(abs(sa_array).T).T
    
    # cut_off value 
    fa_cut_val = np.percentile(fa_array,criteria*100)
    sa_cut_val = np.percentile(sa_array,criteria*100)
    
    # loop for integrated maps
    common_map_fa = []
    common_map_sa = []
    diff_map_fa = []
    diff_map_sa = []
    
    for i in range(fa_array.size):
        fa_val = abs(fa_array[0,i])
        sa_val = abs(sa_array[0,i])
        
        if (fa_val>fa_cut_val) and (sa_val>sa_cut_val): # common 
            common_map_fa.append(True)
            common_map_sa.append(True)
            diff_map_fa.append(False)
            diff_map_sa.append(False)
        elif (fa_val>fa_cut_val) and (sa_val<=sa_cut_val): # diff biased to FA 
            common_map_fa.append(False)
            common_map_sa.append(False)
            diff_map_fa.append(True)
            diff_map_sa.append(False)
        elif (fa_val<=fa_cut_val) and (sa_val>sa_cut_val): # diff biased to SA 
            common_map_fa.append(False)
            common_map_sa.append(False)
            diff_map_sa.append(True)
            diff_map_fa.append(False)
        else: # none-selective 
            common_map_fa.append(False)
            common_map_sa.append(False)
            diff_map_fa.append(False)
            diff_map_sa.append(False)
    
    common_map_fa,common_map_sa = np.array(common_map_fa).reshape(1,-1),np.array(common_map_sa).reshape(1,-1)
    diff_map_fa,diff_map_sa = np.array(diff_map_fa).reshape(1,-1),np.array(diff_map_sa).reshape(1,-1)
    
    return common_map_fa,common_map_sa,diff_map_fa,diff_map_sa

def get_threshold(data):
    """
    get the second least absolute value of the given data 
    the returned threshold is used as the threshold of the brain surface plot
    
    Parameters:
    ----------
    data: ndarray 
        data used for brain plot 
    
    Return:
    ------
    threshold: float
        the threshold that functions in the latter brain plot 
    """
    data = data.flatten()
    data = abs(data)
    data = data[data!=0] # fill out zero 
    threshold = np.min(data)
    
    return threshold 

def bootstrap_by_subject(data_matrix):
    """
    bootstrap by subject, on the assumption that data matrix is based on the 
    binaray discrimination task 
    
    Parameters:
    -----------
    data_matrix: ndarray
        (m,n+1): m, observation number; n, feature number; 1, prediction label  
    
    Return:
    -------
    data_matrix_bootstrapped: ndarray
        (m,n+1): m, observation number; n, feature number; 1, prediction label
    """
    select_range = data_matrix.shape[0]/2
    indexes = np.random.choice(a=int(select_range),size=int(select_range))

    data_matrix_bootstrapped = np.zeros((1,data_matrix.shape[1]))
    for index in indexes:
        data_matrix_bootstrapped = np.r_[data_matrix_bootstrapped,data_matrix[2*index,:].reshape(1,-1),data_matrix[2*index+1,:].reshape(1,-1)]
         
    data_matrix_bootstrapped = data_matrix_bootstrapped[1:,:]
    
    return data_matrix_bootstrapped
    

def index_to_mask(beta_array,num_preserve):
    """
    according to the value of beta, preserve top k number of beta
    return a boolean mask 

    Parameters:
    -----------
    beta_array: ndarray
        (n,), n is the number of feature
        beta value 
    num_preserve: int 
        number that is preserved

    Return:
    -------
    mask: list
        (n,)
        boolean mask 
        the number of True value is k 
    """
    sorted_index = np.argsort(beta_array)
    mask_temp = [True]*num_preserve + [False]*(sorted_index.size-num_preserve)

    mask = []
    for i in range(sorted_index.size):
        index = np.where(sorted_index==i)[0][0]
        mask.append(mask_temp[index])
    mask = np.array(mask)
    
    return mask   

def stat_infer_perm(data_matrix,weight_map,alpha=0.05,is_two_tail=True):
    """
    statistical inference of the permutation result,
    return the boolean map of weight_map
    if the true weight falls outside the distribution threshold, then this weight is statistically sifnificant 
    
    Parameters:
    -----------
    data_matrix: ndarray
        shape(m,n)
        m: number of bootstrap observations 
        n: number of features(on pc-level or voxel-level)
    weight_map: ndarray
        shape(m,)
        m: number of features(on pc-level or voxel-level)
    alpha: float
        significance threshold 
    is_two_tail: bool 
        whether the statistical inference is performed in two-trail or one tail
    
    Return:
    -------
    mask: ndarray
        shape(m,)
        m: number of features
        boolean value inside
        if a specific feature passed the statistical inference, then the boolean value is True 
    """
    if is_two_tail:
        percentile = 1 - alpha/2
    else:
        percentile = 1 - alpha
    
    ci = CI(data_matrix,percentile) # low_ci, high_ci, mean 
    
    mask = []
    
    for i in range(weight_map.size):
        weight = weight_map[i]
        
        if (weight>ci[i,0]) and (weight<ci[i,1]):
            mask.append(False)
        else:
            mask.append(True)
            
    mask = np.array(mask)
    
    return mask 

def sort_prediction(prediction_arr,label_arr):
    """
    summarize prediciton results based on the true labels
    the true lables are taylord specific for this binary training
    the summarized results are mean and standard deviation
    
    Parameters:
    -----------
    prediction_arr: ndarray
        shape(n,)
            n: number of samples 
        the prediciton from the classifier
    label_arr: ndarray
        shape(n,)
            n: number of samples
        the true label
        in the binary form of zero and one 
    
    Return:
    -------
        summary: tuple
            ((mean,sd),(mean,sd))
            the first tuple is the summary of label zero
            the second tuple is the summary of label one
    """
    label_0 = []
    label_1 = []
    for i in range(len(label_arr)):
        if label_arr[i]==0:
            label_0.append(prediction_arr[i])
        else:
            label_1.append(prediction_arr[i])
    
    # flag val condition where nan value could appear
    if len(label_0)==0: # empty list in lable zero
        summary = ((np.nan,np.nan),
                  (np.mean(label_1),np.std(label_1)))
    elif len(label_1)==0:
        summary = ((np.mean(label_0),np.std(label_0)),
                   (np.nan,np.nan))
    else:
        summary = ((np.mean(label_0),np.std(label_0)),
              (np.mean(label_1),np.std(label_1)))
    
    return summary 

def extract_weights(nii_file_path,mask_nan):
    """
    extract weights from nii file 
    return beta value in meaningful voxels 
    
    Parameters:
    -----------
    nii_file: str
        absolute adress of nii file 
    mask_nan: ndarray
        mask out residual voxels 
    
    Return:
    -------
        beta_map: ndarray
        beta map containing weights in meaningful voxels
    """
    beta_map = nib.load(nii_file_path)
    beta_map = beta_map.get_data().flatten()
    beta_map = beta_map[mask_nan]
    
    return beta_map 

def convert_prediction_label(prediction_dic):
    """
    convert prediciton label results into DataFrame
    
    the prediction dic applies to the old version of prediciton label,
    where inside dictionary, the (mean,sd) are stored for each model instead of 
    the complete prediction and truth for each item 
    """
    dataset_list = list(prediction_dic.keys())
    iter_num = len(prediction_dic[dataset_list[0]])
    
    true_label_arr = ([0]*iter_num + [1]*iter_num)*len(dataset_list)
    
    prediction_arr = []
    for dataset in dataset_list:
        results = prediction_dic[dataset]
        for i in range(2):
            for iter in range(len(results)):
                prediction_arr.append(results[iter][i][0]) # here zero takes the mean, while 'one' takes sd
    
    dataset_arr = []
    for dataset in dataset_list:
        dataset_arr += [dataset]*(iter_num*2)
    
    prediction_matrix = np.c_[true_label_arr,prediction_arr,dataset_arr]
    prediciton_df = pd.DataFrame(prediction_matrix,columns=['true_label','prediction','dataset_type'])\
    
    return prediciton_df

def concat_boot_data(file_path,boot_time,data_type,predict_type):
    """
    concatnate bootstrap data based on the data type:
        prediction label
        weights
        r-square 
    Parameters:
    -----------
    file_path: str
        the absolute path of data file 
    boot_time: int
        the total time that the bootstrap code has runded
        should each bootstrap contains 200 loops, for 1000 observations,
        boot_time = 5
    data_tpye: str
        the type of concatenated file
        ['predict_label','weight','r-square']
    predict_type: str
        ['fa','sa']
    
    Return:
    -------
    data: dic
    """
    
    # file name 
    file_name = []
    for i in range(boot_time):
        file_name.append(predict_type+f'_boot_{200*(i+1)}.npy')
    
    total_boot_time = 200*(boot_time)
    
    # dertermine data pool
    if data_type == 'predict_label':
        train_predict,test_predict,transfer_predict,train_truth = np.array([]),np.array([]),np.array([]),np.array([])
    elif data_type == 'weight':
        weight_pool = np.array([])
    elif data_type == 'r-square':
         data = {'train':[],
               'test':[],
               'transfer':[]}
    
    if data_type == 'predict_label':
        for i in range(boot_time):
            file = np.load(os.path.join(file_path,file_name[i]),allow_pickle=True)
            file = file[:,0] # when 'predict_label'
            keys = list(file[0].keys())
            
            for j in range(file.size):
                temp_dic = file[j]
                train_predict = np.append(train_predict,temp_dic[keys[0]][0])
                test_predict = np.append(test_predict,temp_dic[keys[1]][0])
                transfer_predict = np.append(transfer_predict,temp_dic[keys[2]][0])
                train_truth = np.append(train_truth,temp_dic[keys[3]][0])
                
        # out boot loop
        train_predict = train_predict.reshape(total_boot_time,-1)
        test_predict = test_predict.reshape(total_boot_time,-1)
        transfer_predict = transfer_predict.reshape(total_boot_time,-1)
        train_truth = train_truth.reshape(total_boot_time,-1)
        
        data = {'train': train_predict,
               'test': test_predict,
               'transfer': transfer_predict,
               'truth': train_truth}
    
    elif data_type == 'weight':
        for i in range(boot_time):
            file = np.load(os.path.join(file_path,file_name[i]),allow_pickle=True)
            file = file[:,1]
            for j in range(file.size):
                weight_pool = np.append(weight_pool,file[j])
        weight_pool = weight_pool.reshape(total_boot_time,-1)
        
        data = {'weight': weight_pool}
    
    elif data_type == 'r-square':
         for i in range(boot_time):
            file =  np.load(os.path.join(file_path,file_name[i]),allow_pickle=True)
            file = file[:,2]
            key_names = list(file[0].keys())
            
            for j in range(file.size):
                data['train'].append(file[j][key_names[0]])
                data['test'].append(file[j][key_names[1]])
                data['transfer'].append(file[j][key_names[2]])
                                
    return data  

def concat_boot_data(file_path,boot_time,data_type,predict_type):
    """
    concatnate bootstrap data based on the data type:
        prediction label
        weights
        r-square 
    Parameters:
    -----------
    file_path: str
        the absolute path of data file 
    boot_time: int
        the total time that the bootstrap code has runded
        should each bootstrap contains 200 loops, for 1000 observations,
        boot_time = 5
    data_tpye: str
        the type of concatenated file
        ['predict_label','weight','r-square']
    predict_type: str
        ['fa','sa']
    
    Return:
    -------
    data: dic
    """
    
    # file name 
    file_name = []
    for i in range(boot_time):
        file_name.append(predict_type+f'_boot_{200*(i+1)}.npy')
    
    total_boot_time = 200*(boot_time)
    
    # dertermine data pool
    if data_type == 'predict_label':
        train_predict,test_predict,transfer_predict,train_truth = np.array([]),np.array([]),np.array([]),np.array([])
    elif data_type == 'weight':
        weight_pool = np.array([])
    elif data_type == 'r-square':
         data = {'train':[],
               'test':[],
               'transfer':[]}
    
    if data_type == 'predict_label':
        for i in range(boot_time):
            file = np.load(os.path.join(file_path,file_name[i]),allow_pickle=True)
            file = file[:,0] # when 'predict_label'
            keys = list(file[0].keys())
            
            for j in range(file.size):
                temp_dic = file[j]
                train_predict = np.append(train_predict,temp_dic[keys[0]][0])
                test_predict = np.append(test_predict,temp_dic[keys[1]][0])
                transfer_predict = np.append(transfer_predict,temp_dic[keys[2]][0])
                train_truth = np.append(train_truth,temp_dic[keys[3]][0])
                
        # out boot loop
        train_predict = train_predict.reshape(total_boot_time,-1)
        test_predict = test_predict.reshape(total_boot_time,-1)
        transfer_predict = transfer_predict.reshape(total_boot_time,-1)
        train_truth = train_truth.reshape(total_boot_time,-1)
        
        data = {'train': train_predict,
               'test': test_predict,
               'transfer': transfer_predict,
               'truth': train_truth}
    
    elif data_type == 'weight':
        for i in range(boot_time):
            file = np.load(os.path.join(file_path,file_name[i]),allow_pickle=True)
            file = file[:,1]
            for j in range(file.size):
                weight_pool = np.append(weight_pool,file[j])
        weight_pool = weight_pool.reshape(total_boot_time,-1)
        
        data = {'weight': weight_pool}
    
    elif data_type == 'r-square':
         for i in range(boot_time):
            file =  np.load(os.path.join(file_path,file_name[i]),allow_pickle=True)
            file = file[:,2]
            key_names = list(file[0].keys())
            
            for j in range(file.size):
                data['train'].append(file[j][key_names[0]])
                data['test'].append(file[j][key_names[1]])
                data['transfer'].append(file[j][key_names[2]])
                                
    return data       

def cal_prediction(activation,weights,intercept,truth):
    """
    Calculate prediction from activation using one final model,
    Return DataFram for plots
    
    Parameters:
    -----------
    activation : ndarray
        activation map for each subject
        (n,m), n samples, m features
    weights : ndarray
        weights stored form each leave-one-out model 
        (n,m), n: leave-one-out times; m: feature num
    intercept : ndarray
        intercept for each model 
        (n,), n: leave-one-out times
    truth : ndarray
        ture label
        (n,), n samples
    
    Returns:
    --------
    df : DataFrame
        DataFrame ready for plots
    """
    weights = np.average(weights,axis=0)
    intercept = np.mean(intercept)
    
    prediction = np.dot(activation,weights) + intercept
    
    df = pd.DataFrame(np.c_[prediction,truth],
                      columns=['prediction','truth'])
    
    return df 

def force_choice_acc(df,data_type):
    """
    Calculate the two-alternative forced-choiced accuracy in each data_type
    the correct trial is defined as :
        in the pair of prediction (a,b) with truth (0,1), a > b
    
    Parameters:
    -----------
    df : DataFrame
        columns = ['prediction','truth','data_type']
        the turth is periodically arranged as (0,1)
    
    data_type: str
        ['test',transfer']
    
    Returns:
    --------
    acc : float 
        total accuracy 
    """
    df = df[df['data_type']==data_type]
    arr_temp = np.array(df[['prediction','truth']])
    is_true = []
    for i in range(int(arr_temp.shape[0]/2)):
        arr = arr_temp[i*2:i*2+2,0] # for I know truth is always (0,1)
        if arr[0]<arr[1]:
            is_true.append(1)
        else:
            is_true.append(0)
    
    acc = np.average(is_true)
    return acc

def cal_corr(df,data_type):
    """
    Calcuate the correlation between truth and prediction 
    
    Parameters:
    -----------
    df : DataFrame
        columns = ['prediction','truth','data_type']
        the turth is periodically arranged as (0,1)
    
    data_type : str
        ['test','transfer']
    
    Return:
    -------
    val : float 
        the correlation value
    """
    df = df[df['data_type']==data_type]
    val = np.corrcoef(df.prediction, df.truth)[0,1]
    
    return val

def performance_drop(partition_map,
                     full_partition_map,
                     activation_map,
                     significance_mask,
                     weights,
                     intercept,
                     mute_type,
                     prediction_label,
                     prediction_type,
                     analysis_type,
                     partition_min=50,
                     partition_step=10):
    """
    Step by step mute partition areas in SVR prediction,
    the model performance is evaluated by cohen'd, forced-choice accuracy,correlation
    
    Parameters:
    -----------
    partition_map: ndarray
        flattened,each element indicates region label
        this map is masked out with nan mask
    full_partition_map: ndarray
        unmasked partition map 
    activation_map: ndarray
        (m,n),
        m: number of subjects
        n: number of voxels
    significance_mask: ndarray
        the mask that shink weights into significance voxels
    weights: ndarray
        flattened, each elemennt indicates weight obtained from original linear SVR
    intercept: float
        obtained from original linear SVR
        no need of average
    partition_min: int
        the minimum size of a cluster, those clusters whose size are smaller than the 
        thehold are omitted 
    mute_type: str
        {'balanced': the number of muted voxles is determined by the partition_min,
                    where this number is equal across partition areas,
        'max': each partition area is muted by all of its voxels }
    analysis_type: str
        whether compute the residual or coupute this cluster
        {'leision','cluster'}
    prediciton_label: ndarray
        this label could be either test set or transfer set, according to activation map
        {'test_set','transfer'}
    prediction_type: str
        {'test','transfer'}
    partition_step: int 
        the step to test each cluster, default at 10
    
    Returns:
    --------
    acc,cohen,corr: dict
                    three idexes as the measurement for modle performance as it is 
                    Note NOT as the drop of performance 
                    
                    For the case that a cluster is larger than the minimux threshold, 
                    for a cluster key, the ndarray shape is (m,n), m is the number of voxels 
                    the this cluster afford, starting at min threshold and grows with a step
                    of 5 voxels, 
                    n is the repitation time, default setting is 100 
    """
    count = collections.Counter(partition_map)
    
    repitation_time = 100 # should mute type is balanced, the mask is radomly drawn for n times
    
    acc,cohen,corr = {},{},{}
    
    for key in count.keys():
        import time
        start = time.time()
        
        if mute_type == 'max':
            if analysis_type=='leision':
                mask = (partition_map!=key) # mask out the ROI, reserve all other voxels
            else:
                mask = (partition_map==key) # test this cluster 
            
            # mask back to full size
            counter = 0
            mask_full = []
            for k in range(significance_mask.size):
                if significance_mask[k]:
                    mask_full.append(mask[counter])
                    counter+=1
                else:
                    mask_full.append(True)
                
            weights_temp = weights*np.array(mask_full) 
            
            df_prediction = cal_prediction(activation_map,weights_temp.reshape(1,-1),intercept,prediction_label)
            df_prediction['data_type'] = prediction_type
        
            # forced-choice accuracy
            acc_temp = force_choice_acc(df_prediction,prediction_type)
            acc[key] = acc_temp
        
            # cohen'd
            cohen_d_temp = cohen_d(df_prediction,prediction_type)
            cohen[key] = cohen_d_temp
        
            # correlation 
            corr_temp = cal_corr(df_prediction,prediction_type)
            corr[key] = corr_temp
            
        else: # mask out by random minmum voxels
            num_voxel = np.sum(partition_map==key)
            if num_voxel < partition_min: # this region is smaller than the minimum threshold
                continue
            else:
                mother_mask = (partition_map!=key) # mask out ROI
                step = partition_step
                
                acc_loop,cohen_loop,corr_loop = [],[],[]

                for r in range(num_voxel//step+2):
                    if r!=(num_voxel//step+1):
                        num_voxel_p = r*step
                    else:
                        num_voxel_p = num_voxel
                     
                    mask_region = np.r_[np.ones(num_voxel_p),np.zeros(num_voxel-num_voxel_p)]

                    for rep in range(repitation_time):
                        np.random.shuffle(mask_region)
                        count_rep = 0
                        mother_mask_temp = mother_mask.copy()
                        if analysis_type=='leision':
                            for i in range(mother_mask_temp.size):
                                if not mother_mask_temp[i]: # this is ROI
                                    mother_mask_temp[i] = mask_region[count_rep] 
                                    count_rep+=1
                        else:
                            for i in range(mother_mask_temp.size):
                                if mother_mask_temp[i]:
                                    mother_mask_temp[i]=False
                                else:
                                    mother_mask_temp[i] = mask_region[count_rep] 
                                    count_rep+=1
                        
                        # mask back to full space
                        mother_mask_temp_full = []
                        counter = 0
                        for k in range(significance_mask.size):
                            if significance_mask[k]:
                                mother_mask_temp_full.append(mother_mask_temp[counter])
                                counter+=1
                            else:
                                if analysis_type=='leision':
                                    mother_mask_temp_full.append(True)
                                else:
                                    mother_mask_temp_full.append(False)
 
 
                         
                        weights_temp = weights*np.array(mother_mask_temp_full)
                        
                         
                        weights_roi = weights_temp
                        df_prediction = cal_prediction(activation_map,weights_roi.reshape(1,-1),
                                                       intercept,
                                                       prediction_label)
                        df_prediction['data_type'] = prediction_type
                
                        # forced-choice accuracy 
                        acc_temp = force_choice_acc(df_prediction,prediction_type)
                        acc_loop.append(acc_temp)
                
                        # cohen'd
                        cohen_d_temp = cohen_d(df_prediction,prediction_type)
                        cohen_loop.append(cohen_d_temp)
                
                        # correlation 
                        corr_temp = cal_corr(df_prediction,prediction_type)
                        corr_loop.append(corr_temp)
                            
                row_num = int(len(acc_loop)/repitation_time)
                acc[key] = np.array(acc_loop).reshape(row_num,-1)
                cohen[key] = np.array(cohen_loop).reshape(row_num,-1)
                corr[key] = np.array(corr_loop).reshape(row_num,-1)
                    
                end = time.time()
                print(f'The {key} cluster is completed, using time {end-start} s.')
    
    return acc,cohen,corr


def mask_to_mask(mask1,mask2):
    """
    mask back a smaller mask to a bigger mask
    
    mask1: ndarray
        1D mask with a smaller size
    mask2: ndarray 
        1D mask with a larger size
    """
    new_mask = []
    counter = 0
    
    for i in range(mask2.size):
        if mask2[i]:
            new_mask.append(mask1[counter])
            counter+=1
        else:
            new_mask.append(False)
    new_mask = np.array(new_mask)
    
    return new_mask

def performance_drop_select(partition_map, \
                            full_partition_map,\
                            activation_map,\
                            significance_mask,\
                            weights,\
                            intercept,\
                            prediction_label,\
                            prediction_type,\
                            analysis_type,\
                            partition_min=50,\
                            partition_step=10):
    """
    Step by step mute partition areas in SVR prediction,
    the model performance is evaluated by cohen'd, forced-choice accuracy,correlation
    
    Parameters:
    -----------
    partition_map: ndarray
        flattened,each element indicates region label
        this map is masked out with nan mask
    full_partition_map: ndarray
        unmasked partition map 
    activation_map: ndarray
        (m,n),
        m: number of subjects
        n: number of voxels
    significance_mask: ndarray
        the mask that shink weights into significance voxels
    weights: ndarray
        flattened, each elemennt indicates weight obtained from original linear SVR
    intercept: float
        obtained from original linear SVR
        no need of average
    partition_min: int
        the minimum size of a cluster, those clusters whose size are smaller than the 
        thehold are omitted 
    analysis_type: str
        whether compute the residual or coupute this cluster
        {'leision','cluster'}
    prediciton_label: ndarray
        this label could be either test set or transfer set, according to activation map
        {'test_set','transfer'}
    prediction_type: str
        {'test','transfer'}
    partition_step: int 
        the step to test each cluster, default at 10
    
    Returns:
    --------
    acc,cohen,corr,alter_weights: dict
                    three idexes as the measurement for modle performance as it is 
                    Note NOT as the drop of performance 
                    
                    For the case that a cluster is larger than the minimux threshold, 
                    for a cluster key, the ndarray shape is (m,n), m is the number of voxels 
                    the this cluster afford, starting at min threshold and grows with a step
                    of k voxels
                     
    """
    count = collections.Counter(partition_map)
    
    acc,cohen,corr,alter_weights = {},{},{},{}
    
    for key in count.keys():
        import time
        start = time.time()  
        
        num_voxel = np.sum(partition_map==key)
        if num_voxel < partition_min: # this region is smaller than the minimum threshold
            continue
        else:
            acc_loop,cohen_loop,corr_loop = [],[],[]
            weights_k = np.zeros(significance_mask.shape)
            
            roi_mask = (partition_map==key)
            roi_value = weights[significance_mask][roi_mask]
            roi_value_sorted_arg = np.argsort(-roi_value) # the position of vals, decrease fashion
                
            for r in range(num_voxel//partition_step+2):
                num_voxel_del = r*partition_step  
                roi_value_sorted_arg_temp = roi_value_sorted_arg[:num_voxel_del]
                    
                # selected voxel mask
                mask_selected_voxel = []
                for i in range(roi_value.size):
                    if i in roi_value_sorted_arg_temp:
                        mask_selected_voxel.append(True)
                    else:
                        mask_selected_voxel.append(False)
                mask_selected_voxel = np.array(mask_selected_voxel)
                    
                # mask back to partition_map sapce 
                roi_mask_r = mask_to_mask(mask_selected_voxel,roi_mask)
                # mask back to significance weight space
                significance_mask_r = mask_to_mask(roi_mask_r,significance_mask)
                    
                if analysis_type=='cluster': 
                    weights_r = weights*significance_mask_r
                else:
                    weights_r = weights*np.array(abs(significance_mask_r-1),dtype=bool) # reverse Ture and False as lesion condition 
                    
                weights_k = np.vstack((weights_k,weights_r))
                
                df_prediction = cal_prediction(activation_map,weights_r.reshape(1,-1),intercept,prediction_label)
                df_prediction['data_type'] = prediction_type
                
                # forced-choice accuracy 
                acc_temp = force_choice_acc(df_prediction,prediction_type)
                acc_loop.append(acc_temp)
                
                # cohen'd
                cohen_d_temp = cohen_d(df_prediction,prediction_type)
                cohen_loop.append(cohen_d_temp)
                
                # correlation 
                corr_temp = cal_corr(df_prediction,prediction_type)
                corr_loop.append(corr_temp)
                
            weights_k = weights_k[1:,:]
                
        acc[key] = np.array(acc_loop)
        cohen[key] = np.array(cohen_loop) 
        corr[key] = np.array(corr_loop) 
        alter_weights[key] = weights_k
                    
        end = time.time()
        print(f'The {key} cluster is completed, using time {end-start} s.')
    
    return acc,cohen,corr,alter_weights

def dic_to_dataframe(dic,weights,X,Y,intercept,prediction_type,analysis_type):
    """
    convert partition analysis results into DataFrame ready for analysis
    including confidence interval 
    
    Parameters:
    -----------
    dic: dictionary
        index dictionary 
    weights: dictionary 
        weights dicitonary 
    X: ndarray
        1D, X for prediction
    Y: ndarray
        1D, Y for prediction
    intercept: float
        the intercept for prediction
    prediction_type: str
        {'acc','cohen','corr'}
    analysis_type: str
        {'cluster','lesion'}
    
    Returns:
    --------
    df: DataFrame
        DF ready for plot
        columns = {'cluster_name',
                    'step_number',
                    'val',
                    'low_ci',
                    'high_ci',
                    'p_val'}
    """
    
    boot_num = 1000
    keys = dic.keys()
    
    # define null hypothesis
    if prediction_type=='acc':
        null_hyphothesis=0.5
    else:
        null_hyphothesis=0
    
    cluster_name_col,step_num_col,val_col,low_ci,high_ci,p_val = [],[],[],[],[],[]
    
    for key in keys:
        start = time.time()
        arr_temp = dic[key]
        weights_temp = weights[key]
        
        for t in range(arr_temp.shape[0]):
            if t==0:
                if analysis_type=='cluster':
                    continue
            
            cluster_name_col.append(key)
            step_num_col.append(t)
            val_col.append(arr_temp[t])
            
            val_distri = []
            for boot in range(boot_num):
                # bootstrap to form new X 
                boot_data = bootstrap_by_subject(np.c_[X,Y])
                X_temp = boot_data[:,:-1]
                Y_temp = boot_data[:,-1]
                    
                df_prediction = cal_prediction(X_temp,weights_temp[t].reshape(1,-1),intercept,Y_temp)
                df_prediction['data_type'] = prediction_type
                if prediction_type=='acc':
                    val_distri.append(force_choice_acc(df_prediction,prediction_type))
                elif prediction_type=='cohen':
                    val_distri.append(cohen_d(df_prediction,prediction_type))
                else:
                    val_distri.append(cal_corr(df_prediction,prediction_type))
                    
            # compute ci and p-value
            ci_val = CI(val_distri)
            low_ci.append(ci_val[0][0])
            high_ci.append(ci_val[0][1])
                
            ## p_val
            p_val.append(p_value_discrete(val_distri,null_h=null_hyphothesis))
            
        end = time.time()
        print(f'Key {key} is completed, using time {end-statr}s.')
    
    
    
    df = pd.DataFrame(np.c_[cluster_name_col,
                            step_num_col,
                            val_col,
                            low_ci,
                           high_ci,
                           p_val],
                      columns=['cluster_name',
                               'step_number',
                               'value',
                               'low_ci',
                              'high_ci',
                              'p_value'])
    
    return df    

def avg_weights(data_matrix):
    """
    Compute the average weight of multiple models

    Parameters
    ----------
    data_matrix : dictionary
        with key as index of model,
        value as ndarray with shape (n, )
    
    Returns
    -------
    avg_weight : ndarray with shape (n, )
    """
    avg_weight = np.zeros(data_matrix[0].size)
    for weight in data_matrix.values():
        avg_weight = np.vstack((avg_weight, weight))     
    avg_weight = avg_weight[1:, :]
    avg_weight = np.mean(avg_weight, axis=0)

    return avg_weight

def cohen_d(arr1, arr2):
    """
    Compute cohen's d of two arrays 

    Parameters:
    -----------
    arr1, arr2 : ndarray
        two arrays with the same shape (n,)

    Returns:
        val : float
            effect size 
    """
    miu_1 = np.mean(arr1)
    miu_2 = np.mean(arr2)
    
    sigma_1 = np.std(arr1)
    sigma_2 = np.std(arr2)

    sd_pooled = ((sigma_1 ** 2 + sigma_2 ** 2) / 2) ** 0.5
    
    val = (miu_2 - miu_1) / sd_pooled 
    
    return val 

def cohen_d_one_sample(arr, mu):
    """
    Compute cohen's d of one sample and a constant value 

    Parameters:
    -----------
    arr : ndarray
        array with shape (n,)
    mu : float
        constant value 

    Returns:
    --------
    val : float
        effect size 
    """
    miu = np.mean(arr)
    sigma = np.std(arr)
    
    val = (mu - miu) / sigma
    
    return val

def compare_cohen_d(arr1, arr2):
    """
    Generate an array of cohen'd value 
    
    Parameters
    ----------
    arr1, arr2 : ndarray with shape (m, n)
        m : number of observants
        n : number of features
    
    Returns
    -------
    cohen_arr : ndarray with shape (n, )
        array of effect size 
    """
    cohen_arr = []

    for i in range(arr1.shape[1]):
        temp1 = arr1[:,i]
        temp2 = arr2[:,i]
        cohen_arr.append(cohen_d(temp1, temp2))
    
    return np.array(cohen_arr)

def decision_value(weight, intercept, feature_val):
    """
    Bypass the SVC framework, equivalent to decision function
    
    Parameters 
    ----------
    weight: ndarray with shape (n, )
        mean weights 
    intercept: float
        mean intercept 
    feature_val: ndarray with shape (m, n)
        scaled feature matrix
    
    Returns
    -------
    predict_cat : ndarray with shape (m, )
        the predicted category, {0, 1}
        
    decision_val : ndarray with shape (m, )

    """
    decision_val = np.dot(feature_val, weight) + intercept

    predict_cat = []
    for i in decision_val:
        if i < 0:
            predict_cat.append(0)
        else:
            predict_cat.append(1)
    predict_cat = np.array(predict_cat)

    return predict_cat, decision_val 

def compute_auc(weight, intercept, feature_val, truth):
    """
    compute the auc value 

    Parameters
    ----------
    weight: ndarray with shape (n, )
        mean weights 
    intercept: float
        mean intercept 
    feature_val: ndarray with shape (m, n)
        scaled feature matrix
    truth : ndarray with shape (n,)
        the true lable of each observation 
    
    Returns
    -------
    auc_val : float
        auc value of the testing observation 
    """

    _, decision = decision_value(weight, 
                            intercept, 
                            feature_val)
    
    fpr, tpr, _ = roc_curve(truth, decision)
    auc_val = auc(fpr, tpr)
    
    return auc_val