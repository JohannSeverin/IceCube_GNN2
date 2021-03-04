import os, sys
import os.path as osp


def setup_folders():
    file_dir      = osp.dirname(osp.realpath(__file__))
    parent_dir    = osp.split(file_dir)[0] 
    folder = ['data', 'models', 'experiments']

    for f in folder:
        os.mkdir(osp.join(file_dir, f))
