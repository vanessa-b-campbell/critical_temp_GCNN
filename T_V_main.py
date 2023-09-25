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

# device information
device_information = device_info()
print(device_information)
device = device_information.device

start_time = time.time()

## SET UP DATALOADERS: ---
data = TempDataset(root = '/home/jbd3qn/Downloads/critical_temp_GCNN', raw_name = 'train_val_full.csv')

print('Number of NODES features: ', data.num_features)
print('Number of EDGES features: ', data.num_edge_features)

finish_time_preprocessing = time.time()
time_preprocessing = (finish_time_preprocessing - finish_time_preprocessing) / 60

# checking

# Build pytorch training and validation set dataloaders:
batch_size = 10


raw_name_val = 'val_full.csv'
val_set = TempDataset(root = '/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv/validation', raw_name= raw_name_val)

raw_name_train = 'train_full.csv'
train_set = TempDataset(root = '/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv/training', raw_name = raw_name_train)

print(len(val_set))
print(len(train_set))

train_dataloader = DataLoader(train_set, batch_size, shuffle=False)
val_dataloader = DataLoader(val_set, batch_size, shuffle=False)




## RUN TRAINING LOOP: ---

# Train with a random seed to initialize weights:
torch.manual_seed(0)

## SET UP MODEL

initial_dim_gcn = data.num_features         #45?
edge_dim_feature = data.num_edge_features   #11?

print(initial_dim_gcn)
print(edge_dim_feature)

model =  GCN_Temp(initial_dim_gcn, edge_dim_feature).to(device)
# print(x.shape)

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
        
        torch.save(model.state_dict(), "best_model_weights_09_25.pth")

finish_time_training = time.time()
time_training = finish_time_training -start_time_training


#Testing:
weights_file = "best_model_weights_09_25.pth"


#%%
# Training:
input_all_train, target_all_train, pred_prob_all_train = predict(model, train_dataloader, device, weights_file, 
                                        file_path_name= '/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv/Testing/training_predict.csv')



r2_train = r2_score(target_all_train.cpu(), pred_prob_all_train.cpu())
mae_train = mean_absolute_error(target_all_train.cpu(), pred_prob_all_train.cpu())
rmse_train = mean_squared_error(target_all_train.cpu(), pred_prob_all_train.cpu(), squared=False)
r_train, _ = pearsonr(target_all_train.cpu(), pred_prob_all_train.cpu())

# Validation:

input_all_val, target_all_val, pred_prob_all_val = predict(model, val_dataloader, device, weights_file,
                                        file_path_name = '/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv/validation/val_predict.csv' )

r2_val = r2_score(target_all_val.cpu(), pred_prob_all_val.cpu())
mae_val = mean_absolute_error(target_all_val.cpu(), pred_prob_all_val.cpu())
rmse_val = mean_squared_error(target_all_val.cpu(), pred_prob_all_val.cpu(), squared=False)
r_val, _ = pearsonr(target_all_val.cpu(), pred_prob_all_val.cpu())


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
plt.scatter(target_all_train.cpu(), pred_prob_all_train.cpu(), alpha=0.3)
plt.plot([min(target_all_train.cpu()), max(target_all_train.cpu())], [min(target_all_train.cpu()),
                                                            max(target_all_train.cpu())], color="k", ls="--")
plt.xlim([min(target_all_train.cpu()), max(target_all_train.cpu())])
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
plt.scatter(target_all_val.cpu(), pred_prob_all_val.cpu(), alpha=0.3)
plt.plot([min(target_all_val.cpu()), max(target_all_val.cpu())], [min(target_all_val.cpu()),
        max(target_all_val.cpu())], color="k", ls="--")
plt.xlim([min(target_all_val.cpu()), max(target_all_val.cpu())])
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
        data.num_features,
        data.num_edge_features,
        initial_dim_gcn,
        edge_dim_feature ,
        raw_name_train,
        raw_name_val,
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
df.to_csv('/home/jbd3qn/Downloads/critical_temp_GCNN/results.csv', index=False)


# # for validation
# target_all_val = target_all_val.cpu()
# pred_prob_all_val = pred_prob_all_val.cpu()

# target_all_val = target_all_val.numpy()
# pred_prob_all_val = pred_prob_all_val.numpy()

# # creating csv file of true critical temps and predicted temps for validation
# df3 = pd.DataFrame(target_all_val)
# df4 = pd.DataFrame(pred_prob_all_val)

# combined_df_1 = pd.concat([df3, df4], ignore_index=True, axis=1)
# combined_df_1.columns = ['critical_temp_val', 'predicted_temp_val']
# combined_df_1.to_csv('/home/jbd3qn/Downloads/critical_temp_GCNN/predicted_temp_val.csv', index=False)



# # for training
# target_all_train = target_all_train.cpu()
# pred_prob_all_train = pred_prob_all_train.cpu()

# target_all_train = target_all_train.numpy()
# pred_prob_all_train = pred_prob_all_train.numpy()

# # creating a csv file of the true critical temps entering the model and true critical temps exiting the model
# # further comparison analysis in matching_in_out.py
# df = pd.DataFrame(target_all_train)
# dfa = pd.DataFrame(train_set.y)
# combine_df = pd.concat([dfa, df], ignore_index=True, axis=1)
# combine_df.columns = ['Pre_model_true_temp', 'post_model_true_temp']
# combine_df.to_csv('/home/jbd3qn/Downloads/critical_temp_GCNN/train_pre_post_critical_T.csv', index=False)


# # creating csv file of true critical temps and predicted temps for training
# df1 = pd.DataFrame(target_all_train)
# df2 = pd.DataFrame(pred_prob_all_train)

# combined_df = pd.concat([df1, df2], ignore_index=True, axis=1)
# combined_df.columns = ['critical_temp_train', 'predicted_temp_train']
# combined_df.to_csv('/home/jbd3qn/Downloads/critical_temp_GCNN/predicted_temp_training.csv', index=False)
