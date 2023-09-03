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

weights_file = "best_model_weights_new.pth"


## SET UP testing DATALOADERS: ---
test_set = TempDataset('/home/jbd3qn/Downloads/critical_temp_GCNN/test_full.csv')

test_dataloader = DataLoader(test_set, batch_size, shuffle=True)

input_all, target_all, pred_prob_all = predict(model, test_dataloader, device, weights_file)


input_all = input_all.cpu()
target_all = target_all.cpu()
pred_prob_all = pred_prob_all.cpu()



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



input_all = input_all.numpy()
target_all = target_all.numpy()
pred_prob_all = pred_prob_all.numpy()


data_smile = {
    "Metric": [
        "SMILEs"
    ],
    "Value": [
        input_all,
    ],
    
}


data_temp = {
    "Metric": [
        "critical_temp",
    ],
    "Value": [
        target_all,
    ],
    
}

data_p_temp = {
    "Metric": [
        "predicted temp"
    ],
    "Value": [
        pred_prob_all
    ],
    
}



smile_input = []

for each in input_all:
    mol = Chem.MolFromFingerprint(each)
    if mol is not None:
        # 2. Generate a SMILES string
        smiles = Chem.MolToSmiles(mol)
        smile_input.append(smiles)
    else:
        print("Could not convert the fingerprint to a molecule.")

df1 = pd.DataFrame(smile_input)

df2 = pd.DataFrame(target_all)
df3 = pd.DataFrame(pred_prob_all)

combined_df = pd.concat([df1, df2, df3], ignore_index=True, axis=1)
combined_df.columns = ['SMILEs', 'critical_temp', 'predicted_temp']
# ok need to make them into comlumns also to make the fingerprint back into SMILEs 
# is that possible? 
combined_df.to_csv('/home/jbd3qn/Downloads/critical_temp_GCNN/predicted_temp.csv', index=False)