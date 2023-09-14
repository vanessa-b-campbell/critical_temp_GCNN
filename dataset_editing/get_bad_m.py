import pandas as pd

###### bring in the testing data
testing_data = pd.read_csv('/home/jbd3qn/Downloads/critical_temp_GCNN/predicted_temp_test.csv')
smiles = testing_data.iloc[:,0].tolist()
temp_column = testing_data.iloc[:,1].tolist()
predict_column = testing_data.iloc[:,2].tolist()
test_list_GCNN = list(zip(smiles, temp_column, predict_column))

###### bring in the validation data
val_data = pd.read_csv('/home/jbd3qn/Downloads/critical_temp_GCNN/predicted_temp_val.csv')
# smiles = val_data.iloc[:,0].tolist()
temp_column = val_data.iloc[:,0].tolist()
predict_column = val_data.iloc[:,1].tolist()
val_list_GCNN = list(zip(temp_column, predict_column))





mol_val = []
for index in range(0,len(val_list_GCNN)):
    if val_list_GCNN[index][1] <= 250:
        mol_val.append(val_list_GCNN[index])
    
    
mol_test = []
for index in range(0,len(test_list_GCNN)):    
    if  test_list_GCNN[index][2] <= 250:
        mol_test.append(test_list_GCNN[index])


print("from validation")
for each in mol_val:
    print(each)
    
print('\n from testing')
for each in mol_test:
    print(each)
    
    
    
# # for testing 
# bad_mol= []
# for index in range(0,len(test_list_GCNN)):
#     temp_difference = abs(test_list_GCNN[index][1]-test_list_GCNN[index][2])
#     if temp_difference >= 60:
#         bad_mol.append(test_list_GCNN[index])
#         bad_mol.append(temp_difference)
        
        
# for each in bad_mol:
#     print(each)
 
# print(len(bad_mol)/2)

# percent =( (len(bad_mol)/2)/ len(smiles) )* 100
# print(percent)