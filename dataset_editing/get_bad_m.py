import pandas as pd


###### bring in the testing data
predict_test_data = pd.read_csv('/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv/Testing/test_predict_10_05.csv')

og_test_data = pd.read_csv('/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv/Testing/test_full.csv')

smiles = og_test_data.iloc[:,0].tolist()
temp_column = og_test_data.iloc[:,1].tolist()
predict_column = predict_test_data.iloc[:,1].tolist()
test_list_GCNN = list(zip(smiles, temp_column, predict_column))



test_GCNN = pd.DataFrame(test_list_GCNN)
test_GCNN.columns = ['SMILES', 'critical_temp', 'predict_temp']

test_GCNN.to_csv('/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv/Testing/SmiCriPre_test_full.csv', index=False)



bad_mols = []
for index in range(0,len(test_list_GCNN)):
    diff = abs(test_list_GCNN[index][1] - test_list_GCNN[index][2])
    if diff >= 100:
        bad_mols.append(test_list_GCNN[index][:])
    
print('bad mols from testing split:')
for each in bad_mols:
    print(each)
    
    
# for index in range(0,len(bad_mols)):  
#     mol = Chem.MolFromSmiles(bad_mols[index][0])
#     img = Draw.MolToImage(mol)
#     print(img.show())
    
    
# mol_test = []
# for index in range(0,len(test_list_GCNN)):    
#     if  test_list_GCNN[index][2] <= 250:
#         mol_test.append(test_list_GCNN[index])


# print("from validation")
# for each in mol_val:
#     print(each)
    
    
    
    
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