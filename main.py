
#%% 

import time
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from src.device import device_info
from src.data import TempDataset
from src.model import GCN_Temp 
from src.process import train, validation, predict

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr


device_information = device_info()
print(device_information)
device = device_information.device

start_time = time.time()

## SET UP DATALOADERS: ---


# Build starting dataset: 
# dataset = TempDataset('C:\\Users\\color\\Documents\\Bilodeau_Research_Python\\Critical_Temp_Research\\critical_temp_GCNN\\csv_data\\No_outliers_smile_dataset.csv')
dataset = TempDataset('/home/jbd3qn/Downloads/critical_temp_GCNN/csv_data/No_outliers_smile_dataset.csv')
print('Number of NODES features: ', dataset.num_features)
print('Number of EDGES features: ', dataset.num_edge_features)


# # Number of datapoints in the training set:
n_train = int(len(dataset) * 0.8)

# # Number of datapoints in the validation set:
n_val = len(dataset) - n_train

# # Define pytorch training and validation set objects:
train_set, val_set = torch.utils.data.random_split(
    dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42) #TODO 
)

# # Build pytorch training and validation set dataloaders:
batch_size = 10
dataloader = DataLoader(dataset, batch_size, shuffle=True)


train_dataloader = DataLoader(train_set, batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size, shuffle=True)

## RUN TRAINING LOOP: ---

# Train with a random seed to initialize weights:
torch.manual_seed(0)

# Set up model:
model =  GCN_Temp(initial_dim_gcn = dataset.num_features
                , edge_dim_feature = dataset.num_edge_features
                ).to(device)

# Set up optimizer:
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), learning_rate)

train_losses = []
val_losses = []

Sstart_time = time.time()
for epoch in range(1, 200): #TODO

    train_loss = train(model, device, train_dataloader, optimizer, epoch)
    train_losses.append(train_loss)

    val_loss = validation(model, device, val_dataloader, epoch)
    val_losses.append(val_loss)

finish_time = time.time()
time = finish_time -start_time
print("\n -Training finished: {:3f} seconds".format(time))

plt.plot(train_losses, label='Train losses')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Train losses')

plt.plot(val_losses, label='Validation losses')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Losses')
plt.show()

# Training:
input_all, target_all, pred_prob_all = predict(model, train_dataloader, device)

r2 = r2_score(target_all.cpu(), pred_prob_all.cpu())
mae = mean_absolute_error(target_all.cpu(), pred_prob_all.cpu())
rmse = mean_squared_error(target_all.cpu(), pred_prob_all.cpu(), squared=False)
r, _ = pearsonr(target_all.cpu(), pred_prob_all.cpu())

legend_text = "R2 Score: {:.4f}\nR Pearson: {:.4f}\nMAE: {:.4f}\nRMSE: {:.4f}".format(r2, r , mae, rmse)

plt.figure(figsize=(4, 4), dpi=100)
plt.scatter(target_all.cpu(), pred_prob_all.cpu(), alpha=0.3)
plt.plot([min(target_all.cpu()), max(target_all.cpu())], [min(target_all.cpu()),
                                                            max(target_all.cpu())], color="k", ls="--")
plt.xlim([min(target_all.cpu()), max(target_all.cpu())])
plt.title('Training')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend([legend_text], loc="lower right")
plt.show()

# Validation:

input_all, target_all, pred_prob_all = predict(model, val_dataloader, device)

r2 = r2_score(target_all.cpu(), pred_prob_all.cpu())
mae = mean_absolute_error(target_all.cpu(), pred_prob_all.cpu())
rmse = mean_squared_error(target_all.cpu(), pred_prob_all.cpu(), squared=False)
r, _ = pearsonr(target_all.cpu(), pred_prob_all.cpu())

legend_text = "R2 Score: {:.4f}\nR Pearson: {:.4f}\nMAE: {:.4f}\nRMSE: {:.4f}".format(r2, r , mae, rmse)

plt.figure(figsize=(4, 4), dpi=100)
plt.scatter(target_all.cpu(), pred_prob_all.cpu(), alpha=0.3)
plt.plot([min(target_all.cpu()), max(target_all.cpu())], [min(target_all.cpu()),
        max(target_all.cpu())], color="k", ls="--")
plt.xlim([min(target_all.cpu()), max(target_all.cpu())])
plt.title('Validation')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend([legend_text], loc="lower right")
plt.show()


print('hola mundo')


# %%
