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



# device information
device_information = device_info()
print(device_information)
device = device_information.device

# Set up model:

initial_dim_gcn = 45
edge_dim_feature = 11

model =  GCN_Temp(initial_dim_gcn, edge_dim_feature).to(device)


batch_size = 10

weights_file = "best_model_weights_09_07.pth"


## SET UP testing DATALOADERS: ---
test_set = TempDataset(raw_name ='test_full.csv', processed_name='test_processed.pt')
print(len(test_set)) # should be 116


test_dataloader = DataLoader(test_set, batch_size, shuffle=False)

input_all_test, target_all_test, pred_prob_all_test = predict(model, test_dataloader, device, weights_file)


# input_all_test = input_all_test.cpu()
target_all_test = target_all_test.cpu()
print(target_all_test)

pred_prob_all_test = pred_prob_all_test.cpu()



r2_test = r2_score(test_set.y, pred_prob_all_test.cpu())
mae_test = mean_absolute_error(test_set.y, pred_prob_all_test.cpu())
rmse_test = mean_squared_error(test_set.y, pred_prob_all_test.cpu(), squared=False)
r_test, _ = pearsonr(test_set.y, pred_prob_all_test.cpu())

#testing stats
legend_text = "R2 Score: {:.4f}\nR Pearson: {:.4f}\nMAE: {:.4f}\nRMSE: {:.4f}".format(
    r2_test, r_test , mae_test, rmse_test
)

plt.figure(figsize=(4, 4), dpi=100)
plt.scatter(test_set.y, pred_prob_all_test.cpu(), alpha=0.3)
plt.plot([min(test_set.y), max(test_set.y)], [min(test_set.y),
                                                            max(test_set.y)], color="k", ls="--")
plt.xlim([min(test_set.y), max(test_set.y)])
plt.title('Testing')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend([legend_text], loc="lower right")
plt.show()









# r2_test = r2_score(target_all_test.cpu(), pred_prob_all_test.cpu())
# mae_test = mean_absolute_error(target_all_test.cpu(), pred_prob_all_test.cpu())
# rmse_test = mean_squared_error(target_all_test.cpu(), pred_prob_all_test.cpu(), squared=False)
# r_test, _ = pearsonr(target_all_test.cpu(), pred_prob_all_test.cpu())

# #testing stats
# legend_text = "R2 Score: {:.4f}\nR Pearson: {:.4f}\nMAE: {:.4f}\nRMSE: {:.4f}".format(
#     r2_test, r_test , mae_test, rmse_test
# )

# plt.figure(figsize=(4, 4), dpi=100)
# plt.scatter(target_all_test.cpu(), pred_prob_all_test.cpu(), alpha=0.3)
# plt.plot([min(target_all_test.cpu()), max(target_all_test.cpu())], [min(target_all_test.cpu()),
#                                                             max(target_all_test.cpu())], color="k", ls="--")
# plt.xlim([min(target_all_test.cpu()), max(target_all_test.cpu())])
# plt.title('Testing')
# plt.xlabel("True Values")
# plt.ylabel("Predicted Values")
# plt.legend([legend_text], loc="lower right")
# plt.show()



plt.figure(figsize=(4, 4), dpi=100)
plt.scatter(test_set.y, target_all_test.cpu(), alpha=0.3)
plt.plot([min(test_set.y), max(test_set.y)], [min(test_set.y),
                                                            max(test_set.y)], color="k", ls="--")
plt.xlim([min(test_set.y), max(test_set.y)])
plt.title('Testing')
plt.xlabel("test_set")
plt.ylabel("target_all_test")
plt.legend([legend_text], loc="lower right")
plt.show()




target_all_test = target_all_test.numpy()
pred_prob_all_test = pred_prob_all_test.numpy()

df = pd.DataFrame(target_all_test)


df1 = pd.DataFrame(test_set.x)
df2 = pd.DataFrame(test_set.y)
df3 = pd.DataFrame(pred_prob_all_test)

combine_df = pd.concat([df2, df], ignore_index=True, axis=1)
combine_df.columns = ['Pre_model_true_temp', 'post_model_true_temp']
combine_df.to_csv('/home/jbd3qn/Downloads/critical_temp_GCNN/pre_post_critical_T.csv', index=False)

combined_df = pd.concat([df1, df2, df3], ignore_index=True, axis=1)
combined_df.columns = ['SMILES', 'critical_temp_test', 'predicted_temp_test']
combined_df.to_csv('/home/jbd3qn/Downloads/critical_temp_GCNN/predicted_temp_test.csv', index=False)