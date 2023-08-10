import pandas as pd
import os

import torch
from torch_geometric.data import InMemoryDataset

from src.utils import smiles2geodata, get_atom_features

class TempDataset(InMemoryDataset):
    def __init__(self, path, root='../csv_data',raw_name='clean_smile_dataset.csv',transform=None, pre_transform=None):
        
        self.filename = os.path.join(root,raw_name)
        #self.processed_filename = os.path.join(root,processed_name)
        
        # read a csv from that path:
        self.df = pd.read_csv(path)
        
        # assign dataset attribute "input_vectors" to be the 2048 bit vector representing each molecule:
        self.x = self.df[self.df.columns[0]].values

        # assign dataset attribute "output_targets" to be the scalar representing binding strength (last column):
        self.y = self.df[self.df.columns[-1]].values   
        
        
        super(TempDataset, self).__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        
        node_features_dict, edge_features_dict = get_atom_features(self.x)

        data_list = [smiles2geodata(x,y,node_features_dict, edge_features_dict) for x,y in zip(self.x,self.y)]
        
        data, slices = self.collate(data_list)
        
        torch.save((data, slices), self.processed_paths[0])
        
        
