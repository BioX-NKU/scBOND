import scanpy as sc
import anndata as ad
import pandas as pd
import torch
import numpy as np
import random
import episcanpy.api as epi
import scipy
from scipy.sparse import csr_matrix
from scipy.stats.mstats import gmean
from scBridge.logger import *


def imputation_met(adata, number_cell_covered=10, imputation_value='median', save=None, copy=False):
    """
    Impute missing values in methyaltion level matrices. The imputsation is based on the average
    methylation value of the given variable.
    It also filter out variables that are covered in an unsufficient number of cells in order to 
    reduce the feature space to meaningful variables and discard potential coverage biases. 

    Parameters
    ----------
    adata: AnnData object containing 'nan'

    number_cell_covered: minimum number of cells to be covered in order to retain a variable

    imputation_value: imputation of the missing value can be made either on the mean or the median

    Return
    ------
    Return a new AnnData object
    """
    old_features = adata.var_names.tolist()
    
    new_matrix = []
    new_features_name = []
    means = []
    medians = []
    feat_nb = 0

    length1 = len(adata.X[0,:])
    length2 = len(adata.X[:,0])
    adata.obs['coverage_cells'] = [length1 - np.isnan(line).sum() for line in adata.X]
    adata.obs['mean_cell_methylation'] = [np.nansum(line)/length1 for line in adata.X]
    adata.var['coverage_feature'] = [length2 - np.isnan(line).sum() for line in adata.X.T]
    adata.var['mean_feature_methylation'] = [np.nansum(line)/length2 for line in adata.X.T]
    adata.var['median_feature_methylation'] = [np.nanmedian(line) for line in adata.X.T]

    adata2 = adata[:, adata.var['coverage_feature']>=number_cell_covered].copy()

    if imputation_value == 'mean':
        for index in range(len(adata2.var_names.tolist())):
            adata2.X[:,index] = np.nan_to_num(adata2.X[:,index], nan=adata2.var['mean_feature_methylation'][index])
            
    elif imputation_value == 'median':
        for index in range(len(adata2.var_names.tolist())):
            adata2.X[:,index] = np.nan_to_num(adata2.X[:,index], nan=adata2.var['median_feature_methylation'][index])

    elif imputation_value =='zero':
        for index in range(len(adata2.var_names.tolist())):
            adata2.X[:,index] = np.nan_to_num(adata2.X[:,index], nan=0)

    if save is not None:
        filename = save.rstrip('.h5ad') if save.endswith('.h5ad') else save + '.h5ad'
        adata2.write_h5ad(filename)
    if not copy:
        adata = adata2.copy()
    else:
        return adata2

def add_methylation_noise(
    MET_data, 
    noise_rate=0.05,
    seed=None
):
    """
    Simulates measurement errors in methylation sequencing by randomly flipping a proportion of sites between 0 and 1.
    
    Parameters
    ----------
    MET_data: AnnData
        Methylation data for processing.
        
    noise_rate: float
        Proportion of sites to flip, range 0.01-0.10, default 0.05.
        
    seed: int
        Random seed for reproducibility, default None.
        
    Returns
    ---------
    noisy_met_data: AnnData
        Methylation data with added noise.
    """
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    noisy_MET_data = MET_data.copy()
    
    n_cells, n_features = noisy_MET_data.shape
    
    for cell_idx in range(n_cells):
        cell_flips = int(n_features * noise_rate)        
        flip_indices = random.sample(range(n_features), cell_flips)        
        for feat_idx in flip_indices:
            if isinstance(noisy_MET_data.X, np.ndarray):
                noisy_MET_data.X[cell_idx, feat_idx] = 1 - noisy_MET_data.X[cell_idx, feat_idx]
            else:
                # 处理稀疏矩阵
                current_value = noisy_MET_data.X[cell_idx, feat_idx]
                noisy_MET_data.X[cell_idx, feat_idx] = 1 - current_value
    
    return noisy_MET_data


def RNA_data_preprocessing(
    RNA_data,
    normalize_total = True,
    log1p = True,
    use_hvg = True,
    n_top_genes = 3000,
    save_data = False,
    file_path = None,
    logging_path = None
):
    """
    Preprocessing for RNA data, we choose normalization, log transformation and highly variable genes, using scanpy.
    
    Parameters
    ----------
    RNA_data: Anndata
        RNA anndata for processing.
        
    normalize_total: bool
        choose use normalization or not, default True.
        
    log1p: bool
        choose use log transformation or not, default True.
        
    use_hvg: bool
        choose use highly variable genes or not, default True.
        
    n_top_genes: int
        the count of highly variable genes, if not use highly variable, set use_hvg = False and n_top_genes = None, default 3000.
        
    save_data: bool
        choose save the processed data or not, default False.
        
    file_path: str
        the path for saving processed data, only used if save_data is True, default None.
   
    logging_path: str
        the path for output process logging, if not save, set it None, default None.

    Returns
    ---------
    RNA_data_processed: Anndata
        RNA data with normalization, log transformation and highly variable genes selection preprocessed.
    """
    RNA_data_processed = RNA_data.copy()
    
    my_logger = create_logger(name='RNA preprocessing', ch=True, fh=False, levelname=logging.INFO, overwrite=False)
    if not logging_path is None:
        file_handle=open(logging_path + '/Parameters_Record.txt',mode='w')
        file_handle.writelines([
            '------------------------------\n',
            'RNA Preprocessing\n',
            'normalize_total: '+str(normalize_total)+'\n', 
            'log1p: '+str(log1p)+'\n',
            'use_hvg: '+str(use_hvg)+'\n', 
            'n_top_genes: '+str(n_top_genes)+'\n'
        ])
        file_handle.close()
    
    RNA_data_processed.var_names_make_unique()
    
    if not normalize_total or not log1p or not use_hvg:
        my_logger.warning('prefered to process data with default settings to keep the best result.')
    
    if normalize_total:
        my_logger.info('normalize size factor.')
        sc.pp.normalize_total(RNA_data_processed)
        
    if log1p:
        my_logger.info('log transform RNA data.')
        sc.pp.log1p(RNA_data_processed)
    
    if use_hvg:
        my_logger.info('choose top '+str(n_top_genes)+' genes for following training.')
        sc.pp.highly_variable_genes(RNA_data_processed, n_top_genes=n_top_genes)
        RNA_data_processed = RNA_data_processed[:, RNA_data_processed.var['highly_variable']]
    
    if save_data:
        my_logger.warning('writing processed RNA data to target file.')
        if use_hvg:
            RNA_data_processed.write_h5ad(file_path + '/normalize_' + str(normalize_total) + '_log1p_' + str(log1p) + '_hvg_' + str(use_hvg) + '_' + str(n_top_genes) + '_RNA_processed_data.h5ad')
        else:
            RNA_data_processed.write_h5ad(file_path + '/normalize_' + str(normalize_total) + '_log1p_' + str(log1p) + '_hvg_' + str(use_hvg) + '_RNA_processed_data.h5ad')
            
    return RNA_data_processed


def MET_data_preprocessing(
    MET_data,
    imputation = 'median',
    min_cells = 0.007,
    normalize = 'scale',
    add_noise = False,
    noise_rate = 0.05,
    noise_seed = 42,
    save_data = False,
    file_path = None,
    logging_path = None
):
    """
    Preprocessing for methylation data, including filtering features, missing value imputation, 
    optional noise addition, and normalization.
    
    Parameters
    ----------
    MET_data: AnnData
        Methylation data for processing.
        
    imputation: str
        Method for missing value imputation: 'mean', 'median' or 'zero', default 'median'.
        
    min_cells: float
        Minimum proportion of cells where a feature must be present to be retained, default 0.007.
        
    normalize: str
        Normalization method: 'normalize_total' for total count normalization or 'scale' for min-max scaling, default 'scale'.
        
    add_noise: bool
        Whether to add methylation data measurement error noise, default False.
        
    noise_rate: float
        If adding noise, the flip rate of the methylation data, range 0.01-0.10, default 0.05.
        
    noise_seed: int
        Random seed for noise generation, default 42.
        
    save_data: bool
        Choose to save the processed data or not, default False.
        
    file_path: str
        The path for saving processed data, only used if save_data is True, default None.
   
    logging_path: str
        The path for output process logging, if not provided, set to None, default None.

    Returns
    ---------
    MET_data_processed: AnnData
        Methylation data with feature filtering, imputation, add_noise(optional), normalization processed.
        
    """
    MET_data_processed = MET_data.copy()
    max_temp = None
    my_logger = create_logger(name='MET preprocessing', ch=True, fh=False, levelname=logging.INFO, overwrite=False)
    if not logging_path is None:
        file_handle=open(logging_path + '/Parameters_Record.txt',mode='a')
        file_handle.writelines([
            '------------------------------\n'
            'MET Preprocessing\n'
            'normalize: '+str(normalize)+'\n',
            'add_noise: '+str(add_noise)+'\n',
            'noise_rate: '+str(noise_rate)+'\n'
        ])
        file_handle.close()
    
    min_cells_count = np.ceil(min_cells * MET_data_processed.shape[0])
    my_logger.info(f'Filtering features: min {min_cells_count} cells ({min_cells:.3%}) per feature')
    epi.pp.filter_features(MET_data_processed, min_cells=np.ceil(min_cells * MET_data_processed.shape[0]))
    
    my_logger.info(f'Performing missing value imputation using method: {imputation}')
    MET_data_processed = imputation_met(MET_data_processed, number_cell_covered = 1, imputation_value = imputation, save = None, copy = True)
    
    if add_noise:
        my_logger.info(f'Adding noise to methylation data with flip rate: {noise_rate:.2%}')
        MET_data_processed = add_methylation_noise(MET_data_processed, noise_rate=noise_rate, seed=noise_seed)
    
    if normalize == 'normalize_total':
        my_logger.info('Performing total count normalization')
        sc.pp.normalize_total(MET_data_processed)
    elif normalize == 'scale':
        max_temp = np.max(MET_data_processed.X)
        min_temp = np.min(MET_data_processed.X)
        my_logger.info(f'Performing min-max scaling.')
        MET_data_processed.X = (MET_data_processed.X - min_temp) / (max_temp - min_temp)
    
    if save_data:
        my_logger.warning('writing processed methylation data to target file.')
        MET_data.write_h5ad(file_path + '/imputation_' + str(imputation) + '_min_cells_' + str(min_cells) + '_normalize_' + str(normalize) + '_noise_' + str(add_noise) + '_MET_processed_data.h5ad')

    return MET_data_processed