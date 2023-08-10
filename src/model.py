import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  GATConv, NNConv
from torch_geometric.nn import aggr

class GCN_Temp(torch.nn.Module):
    def __init__(self, initial_dim_gcn, edge_dim_feature,
                hidden_dim_nn_1=2000,
                p1 = 0.5,
                hidden_dim_nn_2=500,
                p2 = 0.4 ,
                hidden_dim_nn_3=100,
                p3 = 0.3 ,
                
                
                hidden_dim_fcn_1=1000,
                hidden_dim_fcn_2=100,
                hidden_dim_fcn_3=50,
                ):
        super(GCN_Temp, self).__init__()

        self.nn_conv_1 = NNConv(initial_dim_gcn, hidden_dim_nn_1,
                                nn=torch.nn.Sequential(torch.nn.Linear(edge_dim_feature, initial_dim_gcn * hidden_dim_nn_1)), 
                                aggr='add' )
        self.dropout_1 = nn.Dropout(p=p1)
        
        self.nn_conv_2 = NNConv(hidden_dim_nn_1, hidden_dim_nn_2,
                                nn=torch.nn.Sequential(torch.nn.Linear(edge_dim_feature, hidden_dim_nn_1 * hidden_dim_nn_2)), 
                                aggr='add')
        self.dropout_2 = nn.Dropout(p=p2)
        
        self.nn_conv_3 = NNConv(hidden_dim_nn_2, hidden_dim_nn_3,
                                nn=torch.nn.Sequential(torch.nn.Linear(edge_dim_feature, hidden_dim_nn_2 * hidden_dim_nn_3)), 
                                aggr='add')
        self.dropout_3 = nn.Dropout(p=p3)
                
        

        self.readout = aggr.SumAggregation()

        self.linear1 = nn.Linear(hidden_dim_nn_3, hidden_dim_fcn_1)
        self.linear2 = nn.Linear(hidden_dim_fcn_1, hidden_dim_fcn_2)
        self.linear3 = nn.Linear(hidden_dim_fcn_2, hidden_dim_fcn_3)
        self.linear4 = nn.Linear(hidden_dim_fcn_3, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.nn_conv_1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout_1(x)
        
        x = self.nn_conv_2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout_2(x)
        
        x = self.nn_conv_3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout_3(x)
        
        x = self.readout(x, data.batch)

        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        
        return x.view(-1,)
    
    
