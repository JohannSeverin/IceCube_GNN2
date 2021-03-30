import sqlite3, os, sys, pickle, tqdm

import os.path as osp
import tensorflow as tf
import numpy as np

from pandas import read_sql

from spektral.data import Dataset, Graph
from scipy.sparse import csr_matrix

verbose = True
from scripts.utils import instructions_to_dataset_name, check_dataset, split_events, get_A_func, remove_dataset


file_path = osp.dirname(osp.realpath(__file__))



class graph_dataset(Dataset):
    """
    General Dataset Graph given arguments in json file
    """


    def __init__(self, construct_dict, type = "train", initialize = False):
        # Initialize the Dataset, mostly just unpack the construction dictionairy
        self.type        = type

        self.name        = instructions_to_dataset_name(construct_dict)

        self.Data        = construct_dict['Data']
        self.GraphType   = construct_dict['GraphType']
        self.GraphParam  = construct_dict['GraphParam']

        self.raw_path    = construct_dict['raw_path']

        self.event_lims  = construct_dict['event_lims']
        self.node_lims   = construct_dict['node_lims']

        self.graph_batch = construct_dict['graph_batch']
        self.buffer_size = construct_dict['buffer_size']
        self.data_split  = construct_dict['data_split'] 
        self.seed        = 25
        
        self.features    = construct_dict["features"]
        self.targets     = construct_dict["targets"]
        self.transform   = construct_dict["transforms"]

        self.verbose     = construct_dict["verbose"]

        if construct_dict["clear_dataset"] and initialize:
            if construct_dict['verbose']:
                print("Redoing dataset")
            remove_dataset(self.Data, self.GraphType, self.GraphParam)

        super().__init__()


    @property
    def path(self):
        return osp.join(file_path, "..", "data", "adjacency", self.name)


    def download(self):
        if self.verbose:
            print("Preparing dataset")

        # Check if the data exists, if not create directories
        xs_exists, as_exists = check_dataset(self.Data, self.GraphType, self.GraphParam)

        if not self.raw_path:
            self.raw_path = osp.join(file_path, "..", "data", "raw", self.Data + ".db")
        
        x_path = osp.join(file_path, "..", "data", "features", self.Data)
        a_path = self.path

        A_func = get_A_func(self.GraphType)

        if self.transform:
            transformers = pickle.load(open(self.transform, "rb"))

        with sqlite3.connect(self.raw_path) as conn:    # Connect to raw database    
            if self.verbose:
                print(f"Connected to {self.raw_path}")
            # Gather ids from sql file
            event_query = "select event_no from truth"
            if self.event_lims:
                event_query += " where " + self.event_lims
            event_ids = np.array(read_sql(event_query, conn)).flatten()
            # print(event_ids)
            
            # Split event_numbers in train/test
            train_events, val_events, test_events = split_events(event_ids, self.data_split, self.seed)

            del event_ids # Remove unecessary ram usage

            # Generate x features if they do not exist
            if not xs_exists:
                
                # Loop over train, validation and test
                for type, events in zip(["train", "val", "test"], [train_events, val_events, test_events]):
                    
                    if verbose:
                        print(f"Extracting features for {type}")

                    # generate x features loop
                    for i in tqdm.tqdm(range(0, len(events), self.graph_batch)):
                        get_ids = events[i: i + self.graph_batch]

                        # print(get_ids)

                        feature_query = f"select event_no, {', '.join(self.features)} from features where event_no in {tuple(get_ids)}"
                        if self.node_lims:
                            feature_query += " and " + self.node_lims # Add further restrictions

                        features      = read_sql(feature_query, conn)

                        target_query = f"select {'event_no, ' + ', '.join(self.targets)} from truth where event_no in {tuple(get_ids)}"

                        targets      = read_sql(target_query, conn)

                        # Convert to np arrays and split xs in list
                        f_event      = np.array(features['event_no'])
                        x_long       = np.array(features[self.features])
                        ys           = np.array(targets)

                        # Add inverse transforms (FIX LATER???)
                        if self.transform:
                            for col, trans in enumerate(self.features):
                                if trans in list(transformers['features'].keys()):
                                    x_long[:, col] = transformers["features"][trans].inverse_transform(x_long[:, col].reshape(-1, 1)).flatten()
                            for col, trans in enumerate(self.targets):
                                if trans in list(transformers['truth'].keys()):
                                    ys[:, col]     = transformers["truth"][trans].inverse_transform(ys[:, col].reshape(-1, 1)).flatten()
                                

                        _, counts    = np.unique(f_event.flatten(), return_counts = True)

                        xs           = np.split(x_long, np.cumsum(counts[: -1]))

                        
                        # Save in folder
                        with open(osp.join(x_path, type + str(i) + ".dat"), "wb") as xy_file:
                            pickle.dump([xs, ys], xy_file)

            if not as_exists:
                # Load data from the xs and generate appropiate adjacency matrices in the a - folder
                
                if verbose:
                    print("Making adjacency matrices")

                for xy_file in tqdm.tqdm(os.listdir(x_path)):
                    with open(osp.join(x_path, xy_file), "rb") as file:
                        xs, ys = pickle.load(file)
                    As = []

                    for x in xs:
                        try:
                            a = A_func(x[:, :3], self.GraphParam)
                        except:
                             a = csr_matrix(np.ones(shape = (x.shape[0], x.shape[0])) - np.eye(x.shape[0]))
                        As.append(a)
                    
                    with open(osp.join(a_path, xy_file), "wb") as a_file:
                        pickle.dump(As, a_file)


    def generator(self):
        # Define paths
        x_path  = osp.join(file_path, "..", "data", "features", self.Data)
        a_path  = self.path


        file_names = [f for f in os.listdir(x_path) if self.type in f] 

        if self.type == "train":
            n_files = max(self.buffer_size // self.graph_batch, 1)

            np.random.shuffle(file_names)
            file_names = file_names[:n_files]

        x_files = [osp.join(x_path, f) for f in file_names] 
        a_files = [osp.join(a_path, f) for f in file_names] 

        # Define generator for data loading
        def graph_generator():
            
            # Loop over files
            for xy_path, a_path in zip(x_files, a_files):
                
                xy_file = pickle.load(open(xy_path, "rb"))
                # print(xy_file)
                xs, ys = xy_file
                
                As  = pickle.load(open(a_path,  "rb"))

                # Loop over data
                for x, y, a in zip(xs, ys, As):
                    yield Graph(x = x, a = a, y = y)

        return graph_generator()



    def read(self):
        graph_generator = self.generator()
        return [i for i in graph_generator]










