import os, sys, json, shutil
import os.path as osp
import numpy as np

from importlib import import_module


file_path = osp.dirname(osp.realpath(__file__))



def list_experiments():
    experiment_folder = osp.join(file_path, "..", "experiments") 
    experiment_files  = os.listdir(experiment_folder)
    experiment_files.remove("done")
    experiment_files.remove("failed")
    return experiment_folder, experiment_files


def get_A_func(name):
    import scripts.A_func_list as A_func_list
    A_func = getattr(A_func_list, name)   # Import A_func from the file
    return A_func


def split_events(ids, data_splits = [0.8, 0.1, 0.1], seed = 25):
    # Setup seed
    np.random.seed(seed)
    
    # permutate the indices of event numers
    N = len(ids)
    idxs = np.random.permutation(N)
    train_split = int(data_splits[0] * N)
    val_split   = int(data_splits[1] * N) + train_split

    # Split indices
    idx_tr, idx_val, idx_test  = np.split(idxs, [train_split, val_split])

    # Split events
    train_events = ids[idx_tr]
    val_events   = ids[idx_val]
    test_events  = ids[idx_test]

    return train_events, val_events, test_events


def instructions_to_dataset_name(construction_dict):
    Data         = construction_dict['Data']
    GraphType    = construction_dict['GraphType']
    GraphParam   = construction_dict['GraphParam']

    name = Data + "_" + GraphType + str(GraphParam)

    return name 


def remove_dataset(Data, GraphType, GraphParam = None):
    # function to remove data_folder and adjacency completely to restart

    # Remove data
    Data_folder = osp.join(file_path, "..", "data", "features")
    if Data in os.listdir(Data_folder):
        shutil.rmtree(osp.join(Data_folder, Data))

    # Remove adjacancy folder
    Graph_Name = Data + "_" + GraphType + str(GraphParam)
    Adjancancy_folder = osp.join(file_path, "..", "data", "adjacency")

    if Graph_Name  in os.listdir(Adjancancy_folder):
        shutil.rmtree(osp.join(Adjancancy_folder, Graph_Name))


def check_dataset(Data, GraphType, GraphParam = None):
    """
    Check if a given dataset is generated, else initiate the process
    Return data_exists, as_exists
    Boolean determing if x data file and as data file are constructed
    """
    Data_folder = osp.join(file_path, "..", "data", "features")
    if Data not in os.listdir(Data_folder):
        os.mkdir(osp.join(Data_folder, Data))
        data_exists = False
    else:
        data_exists = True
    
    Graph_Name = Data + "_" + GraphType + str(GraphParam)

    Adjancancy_folder = osp.join(file_path, "..", "data", "adjacency")

    if Graph_Name not in os.listdir(Adjancancy_folder):
        os.mkdir(osp.join(Adjancancy_folder, Graph_Name))
        as_exists    = False
    else:
        as_exists    = True

    return data_exists, as_exists




