import sqlite3, os, sys, pickle, tqdm

import os.path as osp
import tensorflow as tf
import numpy as np

from pandas import read_sql

from spektral.data import Dataset, Graph

verbose = True
from utils import instructions_to_dataset_name, check_dataset, split_events, get_A_func


file_path = osp.dirname(osp.realpath(__file__))



class graph_dataset(Dataset):
    """
    General Dataset Graph given arguments in json file
    """


    def __init__(self, construct_dict, type = "train"):
        # Initialize the Dataset, mostly just unpack the construction dictionairy
        self.type        = type

        self.name        = instructions_to_dataset_name(construct_dict)
        self.per_file    = 10000

        self.Data        = construct_dict['Data']
        self.GraphType   = construct_dict['GraphType']
        self.GraphParam  = construct_dict['GraphParam']

        self.raw_path    = construct_dict['raw_path']

        self.event_lims  = construct_dict['event_lims']
        self.node_lims   = construct_dict['node_lims']

        self.data_split  = construct_dict['data_split'] 
        self.seed        = 25
        
        self.features    = construct_dict["features"]
        self.targets     = construct_dict["targets"]



        super().__init__()


    @property
    def path(self):
        return osp.join(file_path, "..", "data", "adjacency", self.name)


    def download(self):
        # Check if the data exists, if not create directories
        xs_exists, as_exists = check_dataset(self.Data, self.GraphType, self.GraphParam)

        if not self.raw_path:
            raw_path = osp.join(file_path, "..", "data", "raw", self.Data + ".db")
        
        x_path = osp.join(file_path, "..", "data", "features", self.Data)
        a_path = self.path

        A_func = get_A_func(self.GraphType)

        with sqlite3.connect(raw_path) as conn:    # Connect to raw database    
            # Gather ids from sql file
            event_query = "select event__no from truth"
            if self.event_lims:
                event_query += " where " + self.event_lims
            event_ids = read_sql(event_query)
            
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
                    for i in tqdm(range(0, len(events), self.graph_batch)):
                        get_ids = train_events[i: i + self.graph_batch]

                        feature_query = f"select event_no, {', '.join(self.features)} from features where event_no in {tuple(get_ids)}"
                        if self.node_lims:
                            feature_query += " and " + self.node_lims # Add further restrictions

                        features      = read_sql(conn, feature_query)

                        target_query = f"select {', '.join(self.targets)} from truth where event_no in {tuple(get_ids)}"

                        targets      = read_sql(conn, target_query)

                        # Convert to np arrays and split xs in list
                        f_event      = np.array(features['event_no'])
                        x_long       = np.array(features[self.features])

                        _, counts    = np.unique(f_event.flatten, return_counts = True)

                        xs           = np.split(x_long, np.cumsum(counts[: -1]))
                        ys           = np.array(targets)

                        
                        # Save in folder
                        with open(osp.join(x_path, type + str(i) + ".dat"), "wb") as xy_file:
                            pickle.dump(xy_file, [xs, ys])

            if not as_exists:
                # Load data from the xs and generate appropiate adjacency matrices in the a - folder
                
                if verbose:
                    print("Making adjacency matrices")

                for xy_file in tqdm(os.listdir(x_path)):

                    xs, ys = pickle.load(open(xy_file, "rb"))

                    as = []

                    for x in xs:
                        a = A_func(x, self.GraphParam)
                        as.append(a)
                    
                    head, tail = osp.split(xy_file)
                    with open(osp.join(a_path, head), "wb") as a_file:
                        pickle.dump(a_file, as)


    def read(self):

        # Define paths
        x_path  = osp.join(file_path, "..", "data", "features", self.Data)
        a_path  = self.path

        x_files = [osp.join(x_path, f) for f in os.listdir(x_path) if self.type in f] 
        a_files = [osp.join(a_path, f) for f in os.listdir(a_path) if self.type in f] 

        # Define generator for data loading
        def graph_generator():
            
            # Loop over files
            for xy_path, a_path in zip(x_files, a_files):
                
                xy_file = pickle.load(open(xy_path, "rb"))
                xs, ys = xy_file
                
                as  = pickle.load(open(a_path,  "rb"))

                # Loop over data
                for x, y, a in zip(xs, ys, as):
                    yield Graph(x = x, a = a, y = y)

        return graph_generator










