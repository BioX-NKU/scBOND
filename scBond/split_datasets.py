import scanpy as sc
import anndata as ad
import pandas as pd
import torch
import numpy as np
import random


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    

def three_fold_split_dataset(
    RNA_data, 
    MET_data, 
    seed = 19193
):
    if not seed is None:
        setup_seed(seed)
    
    temp = [i for i in range(len(RNA_data.obs_names))]
    random.shuffle(temp)
    
    id_list = []
    
    test_count = int(0.33 * len(temp)) 
    validation_count = int(0.11 * len(temp))  
    
    for i in range(3):
        test_id = temp[:test_count]
        validation_id = temp[test_count:test_count + validation_count]
        train_id = temp[test_count + validation_count:]
        
        temp = temp[test_count:] + temp[:test_count]
        
        id_list.append([train_id, validation_id, test_id])
    
    return id_list

def five_fold_split_dataset(
    RNA_data, 
    MET_data, 
    seed = 19193
):
    
    if not seed is None:
        setup_seed(seed)
    
    temp = [i for i in range(len(RNA_data.obs_names))]
    random.shuffle(temp)
    
    id_list = []
    
    test_count = int(0.2 * len(temp))
    validation_count = int(0.16 * len(temp))
    
    for i in range(5):
        test_id = temp[: test_count]
        validation_id = temp[test_count: test_count + validation_count]
        train_id = temp[test_count + validation_count:]
        temp.extend(test_id)
        temp = temp[test_count: ]

        id_list.append([train_id, validation_id, test_id])
    
    return id_list


def cell_type_split_dataset(
    RNA_data, 
    ATAC_data, 
    seed = 19193
):
    
    if not seed is None:
        setup_seed(seed)
    
    cell_type_list = list(RNA_data.obs.cell_type)
    cell_type = list(RNA_data.obs.cell_type.cat.categories)

    random.shuffle(cell_type)
    
    id_list = []
    
    test_count = int(len(cell_type) / 3)
    train_count = int(len(cell_type) / 2)
    validation_count = len(cell_type) - test_count - train_count
    
    for i in range(3):
        test_type = cell_type[: test_count]
        validation_type = cell_type[test_count: test_count + validation_count]
        train_type = cell_type[test_count + validation_count:]
        cell_type.extend(test_type)
        cell_type = cell_type[test_count: ]

        train_id = [i for i in range(len(cell_type_list)) if cell_type_list[i] in train_type]
        validation_id = [i for i in range(len(cell_type_list)) if cell_type_list[i] in validation_type]
        test_id = [i for i in range(len(cell_type_list)) if cell_type_list[i] in test_type]

        id_list.append([train_id, validation_id, test_id])
    
    return id_list


