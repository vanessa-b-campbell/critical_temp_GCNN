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
from src.process import train, validation, predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from src.utils import get_atom_features
import os
import csv
from math import sqrt


# FIRST- BEFORE RUNNING!!!!!!
# 1. delete all processed files in ../chemprop_splits_csv
# 2. update dates on model folder

model_folder = './model_11_03_IV/'
os.makedirs(model_folder)


# device information
device_information = device_info()
print(device_information)
device = device_information.device

start_time = time.time()

# full set path and root for features/edge dictionaries
root = '/home/jbd3qn/Downloads/critical_temp_GCNN'
path = '/home/jbd3qn/Downloads/critical_temp_GCNN/train_val_test_full.csv' 

# Read in smiles list from full set
df = pd.read_csv(path)
fullset_smiles_list = df[df.columns[0]].values

# use get_atoms_features from utils.py to get features/edge dictionaries from full_set
features_dict_fullset, edge_features_dict_fullset = get_atom_features(fullset_smiles_list)



# save features dict as .csv file
feat_dict_name = 'features_dict.csv'                               #name of features_dict file
model_f_dict_path = os.path.join(model_folder + feat_dict_name)     #features_dict save path


with open(model_f_dict_path, 'w', newline='') as file:
    writer = csv.writer(file)
    header = features_dict_fullset.keys()
    writer.writerow(header)
    for row in zip(*features_dict_fullset.values()):
        writer.writerow(row)
file.close()


# save egde features dict as .csv file
edge_dict_name = 'edge_features_dict.csv'                          #name of edge_features_dict file
model_e_dict_path = os.path.join(model_folder + edge_dict_name)     #edge_features_dict save path

with open(model_e_dict_path, 'w', newline='') as file:
    writer = csv.writer(file)
    header = edge_features_dict_fullset.keys()
    writer.writerow(header)
    for row in zip(*edge_features_dict_fullset.values()):
        writer.writerow(row)
file.close()



# making full set into a TempDataset object to use as initial_dim_gcn and edge_dim_feature for 
# model parameters
full_set = TempDataset(root, path, features_dict_fullset, edge_features_dict_fullset)



## SET UP DATALOADERS: ---

# Training set
root_train = '/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv/training'
path_train = '/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv/training/train_full.csv'
train_set = TempDataset(root_train, path_train, features_dict_fullset, edge_features_dict_fullset)

print('length of training set: ', len(train_set))


# Validatiion set
root_val = '/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv/validation'
path_val = '/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv/validation/val_full.csv'
val_set = TempDataset(root_val, path_val, features_dict_fullset, edge_features_dict_fullset)

print('length of validation set: ', len(val_set))

# # # # # # # # # # # # # # # # # Testing # # # # # # # # # # # # # # # # # # # # # # # #

## SET UP testing DATALOADERS: ---
root_test = '/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv/Testing'
path_test = '/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv/Testing/test_full.csv'
test_set = TempDataset(root_test, path_test, features_dict_fullset, edge_features_dict_fullset)

print('length of testing set: ', len(test_set)) # should be 116

# using full set to get number of features/ number of edge festures
initial_dim_gcn = full_set.num_features         #45
edge_dim_feature = full_set.num_edge_features   #11

print('Number of NODES features: ', initial_dim_gcn)
print('Number of EDGES features: ', edge_dim_feature)



finish_time_preprocessing = time.time()
time_preprocessing = (finish_time_preprocessing - start_time) / 60

# Build pytorch training and validation set dataloaders:

## RUN TRAINING LOOP: ---

# Train with a random seed to initialize weights:
torch.manual_seed(0)

## SET UP MODEL


# hidden_dim_nn_1=500
# p1 = 0 
# hidden_dim_nn_2=250
# p2 = 0 
# hidden_dim_nn_3=100
# p3 = 0

# hidden_dim_fcn_1=100
# hidden_dim_fcn_2=50
# hidden_dim_fcn_3=5
# hidden_dim_gat_0 = 15

## default
hidden_dim_nn_1=2000
p1 = 0 
hidden_dim_nn_2=500
p2 = 0
hidden_dim_nn_3=100
p3 = 0
# if with drop out number of layers needs to be larger 
# try no drop out with smaller layers

hidden_dim_fcn_1=1000
hidden_dim_fcn_2=100
hidden_dim_fcn_3=50
#ARMA layer
hidden_dim_gat_0 = 15
# change number- 

model =  GCN_Temp(initial_dim_gcn, edge_dim_feature,
                hidden_dim_nn_1, 
                p1, 
                hidden_dim_nn_2, 
                p2, 
                hidden_dim_nn_3, 
                p3,
                hidden_dim_gat_0,
                hidden_dim_fcn_1,
                hidden_dim_fcn_2,
                hidden_dim_fcn_3).to(device)

model_weights_name = "best_model_weights.pth"
model_weights_path = os.path.join(model_folder + model_weights_name)

########################################################################################################


num_of_epochs = 100
learning_rate = 0.001
batch_size = 10

# Set up optimizer:

train_dataloader = DataLoader(train_set, batch_size, shuffle=False)
val_dataloader = DataLoader(val_set, batch_size, shuffle=False)

optimizer = optim.Adam(model.parameters(), learning_rate)

train_losses = []
val_losses = []

best_val_loss = float('inf')  # infinite

start_time_training = time.time()
for epoch in range(1, num_of_epochs): #TODO

    train_loss = train(model, device, train_dataloader, optimizer, epoch)
    train_losses.append(train_loss)

    val_loss = validation(model, device, val_dataloader, epoch)
    val_losses.append(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        
        torch.save(model.state_dict(), model_weights_path)


finish_time_training = time.time()
time_training = finish_time_training -start_time_training


#Testing:
weights_file = model_weights_path


#%%
# Training:
train_predict_file = 'train_predict.csv' 
train_predict_file_path_name= os.path.join(model_folder + train_predict_file)

input_all_train, target_all_train, pred_prob_all_train = predict(model, train_dataloader, device, weights_file, train_predict_file_path_name)

r2_train = r2_score(target_all_train, pred_prob_all_train)
mae_train = mean_absolute_error(target_all_train, pred_prob_all_train)
mse_train = mean_squared_error(target_all_train, pred_prob_all_train, squared=False)
r_train, _ = pearsonr(target_all_train, pred_prob_all_train)

# Validation:
val_predict_file = 'val_predict.csv'
val_predict_file_path_name = os.path.join(model_folder + val_predict_file)

input_all_val, target_all_val, pred_prob_all_val = predict(model, val_dataloader, device, weights_file, val_predict_file_path_name )

r2_val = r2_score(target_all_val, pred_prob_all_val)
mae_val = mean_absolute_error(target_all_val, pred_prob_all_val)
mse_val = mean_squared_error(target_all_val, pred_prob_all_val, squared=False)
r_val, _ = pearsonr(target_all_val, pred_prob_all_val)


######## plots
# training vs validation losses
plt.plot(train_losses, label='Train losses')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Train losses')

plt.plot(val_losses, label='Validation losses')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Losses')
plt.savefig(os.path.join(model_folder + 'train_val_loss.png'))
plt.show()



# training stats
legend_text = "R2 Score: {:.4f}\nR Pearson: {:.4f}\nMAE: {:.4f}\nRMSE: {:.4f}".format(
    r2_train, r_train , mae_train, mse_train
)

plt.figure(figsize=(4, 4), dpi=100)
plt.scatter(target_all_train, pred_prob_all_train, alpha=0.3)
plt.plot([min(target_all_train), max(target_all_train)], [min(target_all_train),
                                                            max(target_all_train)], color="k", ls="--")
plt.xlim([min(target_all_train), max(target_all_train)])
plt.title('Training')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend([legend_text], loc="lower right")
plt.savefig(os.path.join(model_folder + 'train_parody.png'))
plt.show()


# Validation stats
legend_text = "R2 Score: {:.4f}\nR Pearson: {:.4f}\nMAE: {:.4f}\nRMSE: {:.4f}".format(
    r2_val, r_val , mae_val, mse_val
)

plt.figure(figsize=(4, 4), dpi=100)
plt.scatter(target_all_val, pred_prob_all_val, alpha=0.3)
plt.plot([min(target_all_val), max(target_all_val)], [min(target_all_val),
        max(target_all_val)], color="k", ls="--")
plt.xlim([min(target_all_val), max(target_all_val)])
plt.title('Validation')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend([legend_text], loc="lower right")
plt.savefig(os.path.join(model_folder + 'val_parody.png'))
plt.show()


print('hola mundo')



#Times
finish_time = time.time()
time_prediction = (finish_time - finish_time_training) / 60
total_time = (finish_time - start_time) / 60
print("\n //// Preprocessing time: {:3f} minutes ////".format(time_preprocessing))
print("\n //// Training time: {:3f} minutes ////".format(time_training))
print("\n //// Prediction time: {:3f} minutes ////".format(time_prediction))
print("\n //// Total time: {:3f} minutes ////".format(total_time))
#results DataFrame
layers = ''

if (hidden_dim_nn_1 == 2000 and p1 == 0.5 and 
    hidden_dim_nn_2==500 and p2 == 0.4 and 
    hidden_dim_nn_3==100 and p3 == 0.3 and 
    hidden_dim_fcn_1==1000 and hidden_dim_fcn_2==100 and 
    hidden_dim_fcn_3==50 and hidden_dim_gat_0== 45):
    
    layers = ('default_ARMA: ' + str(hidden_dim_nn_1) + '-' + str(p1) + '-' + 
                str(hidden_dim_nn_2) + '-' + str(p2) + '-' + 
                str(hidden_dim_nn_3)+ '-' + str(p3) + '-' + 
                str(hidden_dim_gat_0) + '-' +
                str(hidden_dim_fcn_1) + '-' + str(hidden_dim_fcn_2) + '-' + 
                str(hidden_dim_fcn_3))
    
elif (hidden_dim_nn_1 == 2000 and p1 == 0.5 and 
    hidden_dim_nn_2==500 and p2 == 0.4 and 
    hidden_dim_nn_3==100 and p3 == 0.3 and 
    hidden_dim_fcn_1==1000 and hidden_dim_fcn_2==100 and 
    hidden_dim_fcn_3==50):

    layers = ('default: ' + str(hidden_dim_nn_1) + '-' + str(p1) + '-' + 
                str(hidden_dim_nn_2) + '-' + str(p2) + '-' + 
                str(hidden_dim_nn_3)+ '-' + str(p3) + '-' + 
                str(hidden_dim_fcn_1) + '-' + str(hidden_dim_fcn_2) + '-' + 
                str(hidden_dim_fcn_3))

else: 
    layers = (str(hidden_dim_nn_1) + '-' + str(p1) + '-' + 
                str(hidden_dim_nn_2) + '-' + str(p2) + '-' + 
                str(hidden_dim_nn_3)+ '-' + str(p3) + '-' + 
                str(hidden_dim_gat_0) + '-' +
                str(hidden_dim_fcn_1) + '-' + str(hidden_dim_fcn_2) + '-' + 
                str(hidden_dim_fcn_3))


data = {
    "Metric": [
        "number_features",
        "num_edge_features",
        "initial_dim_gcn ",
        "edge_dim_feature",
        "layers",
        "training split ",
        "validation split ",
        "batch_size", 
        "learning_rate",
        "number_of_epochs",
        "r2_train",
        "r_train",
        "mae_train",
        "mse_train", 
        "time_training",
        "r2_validation",
        "r_validation",
        "mae_validation",
        "mse_validation",
        'time_processing(min)',
        "time_prediction",
        "total_time",
        "weights_file",
    ],
    "Value": [
        full_set.num_features,
        full_set.num_edge_features,
        initial_dim_gcn,
        edge_dim_feature ,
        layers,
        path_train,
        path_val,
        batch_size,
        learning_rate,
        num_of_epochs,
        r2_train, 
        r_train, 
        mae_train, 
        mse_train,
        time_training,
        r2_val,
        r_val,
        mae_val, 
        mse_val,
        time_preprocessing,
        time_prediction,
        total_time,
        weights_file
    ],
    
}

df = pd.DataFrame(data)

results_file_name = 'T_V_results.csv'   
results_file = os.path.join(model_folder + results_file_name)

df.to_csv(results_file, index=False)




#%%
# # # # # # # # # # # # # # # # # Testing # # # # # # # # # # # # # # # # # # # # # # # #

test_dataloader = DataLoader(test_set, batch_size, shuffle=False)

test_predict_file = 'test_predict.csv'
test_predict_file_path_name = os.path.join(model_folder + test_predict_file)

input_all_test, target_all_test, pred_prob_all_test = predict(model, test_dataloader, device, weights_file, test_predict_file_path_name)


r2_test = r2_score(target_all_test, pred_prob_all_test)
mae_test = mean_absolute_error(target_all_test, pred_prob_all_test)
mse_test = mean_squared_error(target_all_test, pred_prob_all_test, squared=False)
r_test, _ = pearsonr(target_all_test, pred_prob_all_test)

#testing stats
legend_text = "R2 Score: {:.4f}\nR Pearson: {:.4f}\nMAE: {:.4f}\nRMSE: {:.4f}".format(
    r2_test, r_test , mae_test, mse_test
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
plt.savefig(os.path.join(model_folder + 'test_parody.png'))
plt.show()



test_data = {
    "Metric": [
        "initial_dim_gcn ",
        "edge_dim_feature",
        "testing split ",
        "batch_size", 
        "learning_rate",
        "number_of_epochs",
        "layers",
        "r2_test",
        "r_test",
        "mae_test",
        "mse_test", 
        "weights_file"
    ],
    "Value": [
        initial_dim_gcn,
        edge_dim_feature,
        path_test,
        batch_size,
        learning_rate,
        num_of_epochs,
        layers,
        r2_test, 
        r_test, 
        mae_test, 
        mse_test,
        weights_file
    ],
    
}


df_2 = pd.DataFrame(test_data)

test_results_file_name = 'test_results.csv'         
test_results_file = os.path.join(model_folder + test_results_file_name)


# name of test_results_file is at the beginnning of file
df_2.to_csv(test_results_file,  index=False)