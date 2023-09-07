import pandas as pd
import os

import torch
from torch_geometric.data import InMemoryDataset


from src.utils import smiles2geodata, get_atom_features

class TempDataset(InMemoryDataset):
    # message to future vanessa:
    # next time- review the InMemoryDataset tutorial and figure out how to not have 
    # raw_name be defaulted in the constructor, or how to override
    # the default when calling the constructor in T_V_main.py
    def __init__(self, root='/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv',transform=None, pre_transform=None, pre_filter=None, log=True):
        
        # gives TempDataset class the attributes from the InMemoryDataset class (inheritance)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def  raw_file_names(self):
         return ['train_full.csv', 'val_full', 'test_full']
        # self.filename = os.path.join(root,raw_name)
        # self.processed_filename = os.path.join(root,processed_name)
        
        # # read a csv from that path:
        # self.df = pd.read_csv(self.filename)
    @property
    def processed_file_names(self):
        return ['data.pt']
        
        # assign dataset attribute "input_vectors" to be the 2048 bit vector representing each molecule:
        self.x = self.df[self.df.columns[0]].values

        # assign dataset attribute "output_targets" to be the scalar representing binding strength (last column):
        self.y = self.df[self.df.columns[-1]].values   
        
        
        # super(TempDataset, self).__init__(root, transform, pre_transform)
        # ##################################################################################    what does this do
        # self.data, self.slices = torch.load(self.processed_paths[0])
        

    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        
        node_features_dict, edge_features_dict = get_atom_features(self.x)

        data_list = [smiles2geodata(x,y,node_features_dict, edge_features_dict) for x,y in zip(self.x,self.y)]
        
        data, slices = self.collate(data_list)
        
        torch.save((data, slices), self.processed_paths[0])
        
        
# test_set = TempDataset()
# print(len(test_set))
