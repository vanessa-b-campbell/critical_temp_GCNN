import pandas as pd
import os

import torch
from torch_geometric.data import InMemoryDataset


from src.utils import smiles2geodata, get_atom_features

class TempDataset(InMemoryDataset):
            
                                                                                            #no default raw_name- input when creating dataset object
    def __init__(self, root=None, raw_name = None, transform=None, pre_transform=None, pre_filter=None, log=True):
        self.filename = os.path.join(root,raw_name)
        #self.processed_filename = os.path.join(root,processed_name)
        # make root gerneral
        # 3 diferent folders in chemprop_splits_csv train/testing/val
        # get rid of processed name it's doing nothing
        # track down data.pt
        ########################################################################- do I need line 15? it is commmented out in lipofilicity_PyGeo
        
        # read a csv from that path:
        self.df = pd.read_csv(self.filename)
        
        # assign dataset attribute "input_vectors" to be the 2048 bit vector representing each molecule:
        self.x = self.df[self.df.columns[0]].values

        # assign dataset attribute "output_targets" this is the critial temperature (last column):
        self.y = self.df[self.df.columns[-1]].values   
        
        
        super(TempDataset, self).__init__(root, transform, pre_transform)
        
        #                                    #self.processed_paths[0] -> attribute of the InMemoryDataset class -> is's the first column of the list of file paths to processed data-
        #                                    # but where is it coming from. What is the first column? I didn't initalize this anywhere. 
        #                       # torch.load() used to load the processed data stored in the file specified by self.processed_paths[0]
        self.data, self.slices = torch.load(self.processed_paths[0])
        # Attributes of the dataset object- must be the InMemoryDataset
        # will be populated by loaded data and used later during training or inferencing? 


    def __len__(self):
        return len(self.y)
        
    
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        
        node_features_dict, edge_features_dict = get_atom_features(self.x)

        data_list = [smiles2geodata(x,y,node_features_dict, edge_features_dict) for x,y in zip(self.x,self.y)]
        
        data, slices = self.collate(data_list)
        
        torch.save((data, slices), self.processed_paths[0])
        
        
            
            
# testing prints      
# data = TempDataset(raw_name = 'train_full.csv', processed_name = 'training_processed.pt')
# print(len(data))
# data.process()
# should be 921
