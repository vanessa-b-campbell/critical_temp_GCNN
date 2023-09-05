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
test_set = TempDataset()
print(test_set)

test_dataloader = DataLoader(test_set, batch_size, shuffle=True)

input_all_test, target_all_test, pred_prob_all_test = predict(model, test_dataloader, device, weights_file)


input_all_test = input_all_test.cpu()
target_all_test = target_all_test.cpu()
pred_prob_all_test = pred_prob_all_test.cpu()



r2_test = r2_score(target_all_test.cpu(), pred_prob_all_test.cpu())
mae_test = mean_absolute_error(target_all_test.cpu(), pred_prob_all_test.cpu())
rmse_test = mean_squared_error(target_all_test.cpu(), pred_prob_all_test.cpu(), squared=False)
r_test, _ = pearsonr(target_all_test.cpu(), pred_prob_all_test.cpu())

#testing stats
legend_text = "R2 Score: {:.4f}\nR Pearson: {:.4f}\nMAE: {:.4f}\nRMSE: {:.4f}".format(
    r2_test, r_test , mae_test, rmse_test
)

plt.figure(figsize=(4, 4), dpi=100)
plt.scatter(target_all_test.cpu(), pred_prob_all_test.cpu(), alpha=0.3)
plt.plot([min(target_all_test.cpu()), max(target_all_test.cpu())], [min(target_all_test.cpu()),
                                                            max(target_all_test.cpu())], color="k", ls="--")
plt.xlim([min(target_all_test.cpu()), max(target_all_test.cpu())])
plt.title('Testing')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend([legend_text], loc="lower right")
plt.show()



input_all_test = input_all_test.numpy()
target_all_test = target_all_test.numpy()
pred_prob_all_test = pred_prob_all_test.numpy()


data_smile = {
    "Metric": [
        "SMILEs"
    ],
    "Value": [
        input_all_test,
    ],
    
}


data_temp = {
    "Metric": [
        "critical_temp",
    ],
    "Value": [
        target_all_test,
    ],
    
}

data_p_temp = {
    "Metric": [
        "predicted temp"
    ],
    "Value": [
        pred_prob_all_test
    ],
    
}



# smile_input = []

# for each in input_all_test:
#     mol = Chem.MolFromSmiles(each)
#     if mol is not None:
#         # 2. Generate a SMILES string
#         smiles = Chem.MolToSmiles(mol)
#         smile_input.append(smiles)
#     else:
#         print("Could not convert the fingerprint to a molecule.")

# df1 = pd.DataFrame(smile_input)
# df1 = pd.DataFrame(input_all_test)

# print(df1)
#why on earth is it 11355 rows long? 

df2 = pd.DataFrame(target_all_test)
df3 = pd.DataFrame(pred_prob_all_test)

combined_df = pd.concat([df2, df3], ignore_index=True, axis=1)
combined_df.columns = ['critical_temp', 'predicted_temp']
# ok need to make them into comlumns also to make the fingerprint back into SMILEs 
# is that possible? 
combined_df.to_csv('/home/jbd3qn/Downloads/critical_temp_GCNN/predicted_temp.csv', index=False)