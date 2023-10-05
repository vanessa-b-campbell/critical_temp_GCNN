#%%
import time
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from src.device import device_info
from src.data import TempDataset
from src.model import GCN_Temp 
import pandas as pd
from src.process import predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import numpy as np
from rdkit import Chem
from T_V_main import features_dict_fullset, edge_features_dict_fullset, initial_dim_gcn, edge_dim_feature
import os

# # # # # # # # # # # # # # # # # # update date before running

weights_file = "best_model_weights_10_05.pth"



# device information
device_information = device_info()
print(device_information)
device = device_information.device

# Set up model:
model =  GCN_Temp(initial_dim_gcn, edge_dim_feature).to(device)






## SET UP testing DATALOADERS: ---
root = '/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv'
path =  '/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv/Testing/test_full.csv'
test_set = TempDataset(root, path, features_dict_fullset, edge_features_dict_fullset)

print('length of testing set: ', len(test_set)) # should be 116



batch_size = 10
test_dataloader = DataLoader(test_set, batch_size, shuffle=False)

file_path_name = '/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv/Testing/test_predict.csv'
input_all_test, target_all_test, pred_prob_all_test = predict(model, test_dataloader, device, weights_file, file_path_name)

print(target_all_test)


r2_test = r2_score(target_all_test, pred_prob_all_test)
mae_test = mean_absolute_error(target_all_test, pred_prob_all_test)
rmse_test = mean_squared_error(target_all_test, pred_prob_all_test, squared=False)
r_test, _ = pearsonr(target_all_test, pred_prob_all_test)

#testing stats
legend_text = "R2 Score: {:.4f}\nR Pearson: {:.4f}\nMAE: {:.4f}\nRMSE: {:.4f}".format(
    r2_test, r_test , mae_test, rmse_test
)

plt.figure(figsize=(4, 4), dpi=100)
plt.scatter(target_all_test, pred_prob_all_test, alpha=0.3)
plt.plot([min(target_all_test), max(target_all_test)], [min(target_all_test),
                                                            max(target_all_test)], color="k", ls="--")
plt.xlim([min(target_all_test), max(target_all_test)])
plt.title('Testing')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend([legend_text], loc="lower right")
plt.show()



data = {
    "Metric": [
        "initial_dim_gcn ",
        "edge_dim_feature",
        "testing split ",
        "batch_size", 
        "learning_rate",
        "number_of_epochs",
        "r2_test",
        "r_test",
        "mae_test",
        "rmse_test", 
        "weights_file"
    ],
    "Value": [
        initial_dim_gcn,
        edge_dim_feature,
        path,
        batch_size,
        r2_test, 
        r_test, 
        mae_test, 
        rmse_test,
        weights_file
    ],
    
}



df = pd.DataFrame(data)
path = '/home/jbd3qn/Downloads/critical_temp_GCNN/results'


filename = os.path.join(root,results_file)
df.to_csv(filename,  index=False)

