#%%
# # # # # # # # # #  replaces main.py # # # # # # # # # # # 
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


# # # # # # # # # # # # # # # # # # update date before running
results_file = 'T_V_results10_05.csv'
test_results_file = 'test_results10_05.csv'
model_weights_name = "best_model_weights_10_05.pth"




# device information
device_information = device_info()
print(device_information)
device = device_information.device

start_time = time.time()

# full set path and root for features/edge dictionaries
root = '/home/jbd3qn/Downloads/critical_temp_GCNN'
path = '/home/jbd3qn/Downloads/critical_temp_GCNN/train_val_full.csv' 

# Read in smiles list from full set
df = pd.read_csv(path)
fullset_smiles_list = df[df.columns[0]].values

# use get_atoms_features from utils.py to get features/edge dictionaries from full_set
features_dict_fullset, edge_features_dict_fullset = get_atom_features(fullset_smiles_list)

full_set = TempDataset(root, path, features_dict_fullset, edge_features_dict_fullset)
# making full set into a TempDataset object to use as initial_dim_gcn and edge_dim_feature for 
# model parameters

finish_time_preprocessing = time.time()
time_preprocessing = (finish_time_preprocessing - finish_time_preprocessing) / 60


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



# Build pytorch training and validation set dataloaders:
batch_size = 10

train_dataloader = DataLoader(train_set, batch_size, shuffle=False)
val_dataloader = DataLoader(val_set, batch_size, shuffle=False)




## RUN TRAINING LOOP: ---

# Train with a random seed to initialize weights:
torch.manual_seed(0)

## SET UP MODEL

# using full set to get number of features/ number of edge festures
initial_dim_gcn = full_set.num_features         #45
edge_dim_feature = full_set.num_edge_features   #11

print('Number of NODES features: ', initial_dim_gcn)
print('Number of EDGES features: ', edge_dim_feature)


model =  GCN_Temp(initial_dim_gcn, edge_dim_feature).to(device)


# Set up optimizer:
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), learning_rate)

train_losses = []
val_losses = []

best_val_loss = float('inf')  # infinite

start_time_training = time.time()
num_of_epochs = 100
for epoch in range(1, num_of_epochs): #TODO

    train_loss = train(model, device, train_dataloader, optimizer, epoch)
    train_losses.append(train_loss)

    val_loss = validation(model, device, val_dataloader, epoch)
    val_losses.append(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        
        torch.save(model.state_dict(), model_weights_name)


finish_time_training = time.time()
time_training = finish_time_training -start_time_training


#Testing:
weights_file = model_weights_name


#%%
# Training:
input_all_train, target_all_train, pred_prob_all_train = predict(model, train_dataloader, device, weights_file, 
                                        file_path_name= '/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv/training/training_predict.csv')



r2_train = r2_score(target_all_train, pred_prob_all_train)
mae_train = mean_absolute_error(target_all_train, pred_prob_all_train)
rmse_train = mean_squared_error(target_all_train, pred_prob_all_train, squared=False)
r_train, _ = pearsonr(target_all_train, pred_prob_all_train)

# Validation:

input_all_val, target_all_val, pred_prob_all_val = predict(model, val_dataloader, device, weights_file,
                                        file_path_name = '/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv/validation/val_predict.csv' )

r2_val = r2_score(target_all_val, pred_prob_all_val)
mae_val = mean_absolute_error(target_all_val, pred_prob_all_val)
rmse_val = mean_squared_error(target_all_val, pred_prob_all_val, squared=False)
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
plt.show()



# training stats
legend_text = "R2 Score: {:.4f}\nR Pearson: {:.4f}\nMAE: {:.4f}\nRMSE: {:.4f}".format(
    r2_train, r_train , mae_train, rmse_train
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
plt.show()


# Validation stats
legend_text = "R2 Score: {:.4f}\nR Pearson: {:.4f}\nMAE: {:.4f}\nRMSE: {:.4f}".format(
    r2_val, r_val , mae_val, rmse_val
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

data = {
    "Metric": [
        "number_features",
        "num_edge_features",
        "initial_dim_gcn ",
        "edge_dim_feature",
        "training split ",
        "validation split ",
        "batch_size", 
        "learning_rate",
        "number_of_epochs",
        "r2_train",
        "r_train",
        "mae_train",
        "rmse_train", 
        "r2_validation",
        "r_validation",
        "mae_validation",
        "rmse_validation",
        "time_preprocessing", 
        "time_training",
        "time_prediction",
        "total_time",
        "weights_file"
    ],
    "Value": [
        full_set.num_features,
        full_set.num_edge_features,
        initial_dim_gcn,
        edge_dim_feature ,
        path_train,
        path_val,
        batch_size,
        learning_rate,
        num_of_epochs,
        r2_train, 
        r_train, 
        mae_train, 
        rmse_train,
        r2_val,
        r_val,
        mae_val, 
        rmse_val,
        time_preprocessing, 
        time_training,
        time_prediction,
        total_time,
        weights_file
    ],
    
}

df = pd.DataFrame(data)

path = '/home/jbd3qn/Downloads/critical_temp_GCNN/results'


filename = os.path.join(root,results_file)
df.to_csv(filename, index=False)




#%%
# # # # # # # # # # # # # # # # # Testing # # # # # # # # # # # # # # # # # # # # # # # #

## SET UP testing DATALOADERS: ---
root = '/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv'
path =  '/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv/Testing/test_full.csv'
test_set = TempDataset(root, path, features_dict_fullset, edge_features_dict_fullset)

print('length of testing set: ', len(test_set)) # should be 116

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



test_data = {
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


df = pd.DataFrame(test_data)
path = '/home/jbd3qn/Downloads/critical_temp_GCNN/results'


filename = os.path.join(root,test_results_file)
df.to_csv(filename,  index=False)