import pandas as pd
import os

import torch
from torch_geometric.data import InMemoryDataset


from src.utils import smiles2geodata, get_atom_features

class TempDataset(InMemoryDataset):
            
                                                                                            #no default raw_name- input when creating dataset object
    def __init__(self, root, path, features_dict, edge_features_dict, transform=None, pre_transform=None): # take in the dictionaries pulled from T_V_main.py slef.edge_feature dict 
        
        self.features_dict = features_dict
        self.edge_features_dict = edge_features_dict

        self.filename = os.path.join(path)
        #self.processed_filename = os.path.join(root,processed_name)
        
        
        # read a csv from that path:
        self.df = pd.read_csv(self.filename)
        
        # assign dataset attribute "input_vectors" to be the 2048 bit vector representing each molecule:
        self.x = self.df[self.df.columns[0]].values

        # assign dataset attribute "output_targets" this is the critial temperature (last column):
        self.y = self.df[self.df.columns[-1]].values   
        
        
        super(TempDataset, self).__init__(root,  transform, pre_transform)
        
        #                                    #self.processed_paths[0] -> attribute of the InMemoryDataset class -> is's the first column of the list of file paths to processed data-
        #                                    # but where is it coming from. What is the first column? I didn't initalize this anywhere. 
        #                       # torch.load() used to load the processed data stored in the file specified by self.processed_paths[0]
        self.data, self.slices = torch.load(self.processed_paths[0])
        # Attributes of the dataset object- must be the InMemoryDataset
        # will be populated by loaded data and used later during training or inferencing? 


    def processed_file_names(self):
        return ['new_data.pt']
    
    
    def process(self):
        # comment 48 out- in datalist change self.edge features, sme.nodefeatures
        #node_features_dict, edge_features_dict = get_atom_features(self.x)
        
        data_list = [smiles2geodata(x,y,self.features_dict, self.edge_features_dict) for x,y in zip(self.x,self.y)]
        
        data, slices = self.collate(data_list)
        
        torch.save((data, slices), self.processed_paths[0])
        
    
    def __len__(self):
        return len(self.y)
        
    

        
# testing prints      
# processed_name_train = 'proccess_train.pt'
# raw_name_train = 'train_full.csv'
# root_train = '/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv/training'
# train_set = TempDataset(root =root_train , raw_name = raw_name_train)
# print(len(train_set))
# train_set.process()
# # print(train_set.data)
# print(len(train_set.slices))
# print(train_set.data_list)
#should be 921
