
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

data = TempDataset('/home/jbd3qn/Downloads/critical_temp_GCNN/csv_data/No_outliers_smile_dataset.csv')
print('Number of NODES features: ', data.num_features)
print('Number of EDGES features: ', data.num_edge_features)
#################################


finish_time_preprocessing = time.time()
time_preprocessing = (finish_time_preprocessing - finish_time_preprocessing) / 60


# Set up model:
val_set = TempDataset('/home/jbd3qn/Downloads/critical_temp_GCNN/val_full.csv')
train_set = TempDataset('/home/jbd3qn/Downloads/critical_temp_GCNN/train_full.csv')


# Build pytorch training and validation set dataloaders:
batch_size = 10
dataloader = DataLoader(data, batch_size, shuffle=True) ################################# like 80% sure this line does nothing


train_dataloader = DataLoader(train_set, batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size, shuffle=True)



## RUN TRAINING LOOP: ---

# Train with a random seed to initialize weights:
torch.manual_seed(0)


## SET UP MODEL
# note to self- the number of features and the number of edge features is the same whether
# using whole data for '.num_features' or just a combination of train/val data
# Ask daniel what these are actually doing- how are these values aquired from the dataset
initial_dim_gcn = data.num_features         #45
edge_dim_feature = data.num_edge_features   #11


model =  GCN_Temp(initial_dim_gcn, edge_dim_feature).to(device)


# Set up optimizer:
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), learning_rate)

train_losses = []
val_losses = []

best_val_loss = float('inf')  # infinite

start_time_training = time.time()
num_of_epochs = 200
for epoch in range(1, num_of_epochs): #TODO

    train_loss = train(model, device, train_dataloader, optimizer, epoch)
    train_losses.append(train_loss)

    val_loss = validation(model, device, val_dataloader, epoch)
    val_losses.append(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        
        torch.save(model.state_dict(), "best_model_weights_new.pth")

finish_time_training = time.time()
time_training = finish_time_training -start_time_training


#Testing:
weights_file = "best_model_weights_new.pth"

test_set = TempDataset('/home/jbd3qn/Downloads/critical_temp_GCNN/test_full.csv')

test_dataloader = DataLoader(test_set, batch_size, shuffle=True)
input_all, target_all, pred_prob_all = predict(model, test_dataloader, device, weights_file)


r2_test = r2_score(target_all.cpu(), pred_prob_all.cpu())
mae_test = mean_absolute_error(target_all.cpu(), pred_prob_all.cpu())
rmse_test = mean_squared_error(target_all.cpu(), pred_prob_all.cpu(), squared=False)
r_test, _ = pearsonr(target_all.cpu(), pred_prob_all.cpu())

#testing stats
legend_text = "R2 Score: {:.4f}\nR Pearson: {:.4f}\nMAE: {:.4f}\nRMSE: {:.4f}".format(
    r2_test, r_test , mae_test, rmse_test
)

plt.figure(figsize=(4, 4), dpi=100)
plt.scatter(target_all.cpu(), pred_prob_all.cpu(), alpha=0.3)
plt.plot([min(target_all.cpu()), max(target_all.cpu())], [min(target_all.cpu()),
                                                            max(target_all.cpu())], color="k", ls="--")
plt.xlim([min(target_all.cpu()), max(target_all.cpu())])
plt.title('Testing')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend([legend_text], loc="lower right")
plt.show()











# #%%
# # Training:
# input_all, target_all, pred_prob_all = predict(model, train_dataloader, device, weights_file)


# r2_train = r2_score(target_all.cpu(), pred_prob_all.cpu())
# mae_train = mean_absolute_error(target_all.cpu(), pred_prob_all.cpu())
# rmse_train = mean_squared_error(target_all.cpu(), pred_prob_all.cpu(), squared=False)
# r_train, _ = pearsonr(target_all.cpu(), pred_prob_all.cpu())

# # Validation:

# input_all, target_all, pred_prob_all = predict(model, val_dataloader, device, weights_file)

# r2_val = r2_score(target_all.cpu(), pred_prob_all.cpu())
# mae_val = mean_absolute_error(target_all.cpu(), pred_prob_all.cpu())
# rmse_val = mean_squared_error(target_all.cpu(), pred_prob_all.cpu(), squared=False)
# r_val, _ = pearsonr(target_all.cpu(), pred_prob_all.cpu())


# ######## plots
# # training vs validation losses
# plt.plot(train_losses, label='Train losses')
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('Train losses')

# plt.plot(val_losses, label='Validation losses')
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('Losses')
# plt.show()



# # training stats
# legend_text = "R2 Score: {:.4f}\nR Pearson: {:.4f}\nMAE: {:.4f}\nRMSE: {:.4f}".format(
#     r2_train, r_train , mae_train, rmse_train
# )

# plt.figure(figsize=(4, 4), dpi=100)
# plt.scatter(target_all.cpu(), pred_prob_all.cpu(), alpha=0.3)
# plt.plot([min(target_all.cpu()), max(target_all.cpu())], [min(target_all.cpu()),
#                                                             max(target_all.cpu())], color="k", ls="--")
# plt.xlim([min(target_all.cpu()), max(target_all.cpu())])
# plt.title('Training')
# plt.xlabel("True Values")
# plt.ylabel("Predicted Values")
# plt.legend([legend_text], loc="lower right")
# plt.show()


# # Validation stats
# legend_text = "R2 Score: {:.4f}\nR Pearson: {:.4f}\nMAE: {:.4f}\nRMSE: {:.4f}".format(
#     r2_val, r_val , mae_val, rmse_val
# )

# plt.figure(figsize=(4, 4), dpi=100)
# plt.scatter(target_all.cpu(), pred_prob_all.cpu(), alpha=0.3)
# plt.plot([min(target_all.cpu()), max(target_all.cpu())], [min(target_all.cpu()),
#         max(target_all.cpu())], color="k", ls="--")
# plt.xlim([min(target_all.cpu()), max(target_all.cpu())])
# plt.title('Validation')
# plt.xlabel("True Values")
# plt.ylabel("Predicted Values")
# plt.legend([legend_text], loc="lower right")
# plt.show()


# print('hola mundo')



# #Times
# finish_time = time.time()
# time_prediction = (finish_time - finish_time_training) / 60
# total_time = (finish_time - start_time) / 60
# print("\n //// Preprocessing time: {:3f} minutes ////".format(time_preprocessing))
# print("\n //// Training time: {:3f} minutes ////".format(time_training))
# print("\n //// Prediction time: {:3f} minutes ////".format(time_prediction))
# print("\n //// Total time: {:3f} minutes ////".format(total_time))
# #results DataFrame

# data = {
#     "Metric": [
#         "number_features",
#         "num_edge_features",
#         "initial_dim_gcn ",
#         "edge_dim_feature",
#         # "hidden_dim_nn_1 ",
#         # "p1 ",
#         # "hidden_dim_nn_2 ",
#         # "p2 ",
#         # "hidden_dim_nn_3 ",
#         # "p3 ",
#         # "hidden_dim_fcn_1 ",
#         # "hidden_dim_fcn_2 ",
#         # "hidden_dim_fcn_3 ",
#         "training split ",
#         "validation split ",
#         "batch_size", 
#         "learning_rate",
#         "number_of_epochs",
#         "r2_train",
#         "r_train",
#         "mae_train",
#         "rmse_train", 
#         "r2_validation",
#         "r_validation",
#         "mae_validation",
#         "rmse_validation",
#         "time_preprocessing", 
#         "time_training",
#         "time_prediction",
#         "total_time"
#     ],
#     "Value": [
#         train_val_dataset.num_features,
#         train_val_dataset.num_edge_features,
#         initial_dim_gcn,
#         edge_dim_feature ,
#         # hidden_dim_nn_1 ,
#         # p1 ,
#         # hidden_dim_nn_2 ,
#         # p2 ,
#         # hidden_dim_nn_3,
#         # p3 ,
#         # hidden_dim_fcn_1 ,
#         # hidden_dim_fcn_2 ,
#         # hidden_dim_fcn_3 ,
#         "Chemprop",
#         "Chemprop",
#         batch_size,
#         learning_rate,
#         num_of_epochs,
#         r2_train, 
#         r_train, 
#         mae_train, 
#         rmse_train,
#         r2_val,
#         r_val,
#         mae_val, 
#         rmse_val,
#         time_preprocessing, 
#         time_training,
#         time_prediction,
#         total_time
#     ],
    
# }

# df = pd.DataFrame(data)
# df.to_csv('/home/jbd3qn/Downloads/critical_temp_GCNN/results.csv', index=False)