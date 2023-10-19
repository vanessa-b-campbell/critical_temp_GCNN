import functional_group_lists as fglist
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import csv
import os

# for now assuming SMILES strings are already clean

# is_clean? || do_clean? test if clean or maybe actually clean- 
# this will be hard to soft code
# substructure searched throughout whole dataset + statisitc created
# get functional group stat summary
# convert to fingerprint

class DeltaMolData:
    def __init__(self, path):

        #### 1. read in csv file using pandas
        self.raw_data = pd.read_csv(path)

        # first convert the first and second column into list
        self.input_column = self.raw_data.iloc[:,0].tolist()
        self.output_column = self.raw_data.iloc[:,1].tolist()

        # zip input_smiles and cTemp into a list of tuples
        self.raw_data_list = list(zip(self.input_column, self.output_column))

    def make_clean(self):
        pass 



    def get_functioanl_group(self):
        
        for func_i in range(0, len(fglist.func_group_list)):
            for smile in self.input_column:

                mol = Chem.MolFromSmiles(smile)
                pattern = Chem.MolFromSmiles(fglist.func_group_list[func_i][0])
                match = mol.HasSubstructMatch(pattern) # will return a true or false
                if match: 
                    (fglist.func_group_list[func_i][1]).append(smile)
                    
            
                

            # Statistics
            # still inside the loop- percentage of molecules that are a specific group is calculated and printed 
            # for each fnctional group type
            percent = (len(fglist.func_group_list[func_i][1])/len(self.input_column))*100
            message = ("{:.2f}% of molecules have {} group".format(percent,fglist.func_group_list[func_i][2]))
            fglist.full_func_stats.append(message)
            
            if len(fglist.func_group_list[func_i][1]) != 0:
                print('\n')
                print('Smiles in {} group:'.format(fglist.func_group_list[func_i][2]))
                print(fglist.func_group_list[func_i][1])

        # second for loop is identical to first except this pulls from SMARTS functional group list
        # and converts from SMARTs to mols for functional groups
        for func_i in range(0, len(fglist.metalloid_group_list)):
            for smile in self.input_column:
                mol = Chem.MolFromSmiles(smile)
                pattern = Chem.MolFromSmarts(fglist.metalloid_group_list[func_i][0])
                match = mol.HasSubstructMatch(pattern)
                if match:
                    (fglist.metalloid_group_list[func_i][1]).append(smile)
            # Statistic
            percent = (len(fglist.metalloid_group_list[func_i][1])/ len(self.input_column)) * 100
            message = ("{:.2f}% of molecules have {} group".format(percent,fglist.metalloid_group_list[func_i][2]))
            fglist.full_func_stats.append(message)

        fglist.func_group_list.extend(fglist.metalloid_group_list)

        gutter_list = []

        for smile in self.input_column:
            used = False
            for functional_group in fglist.func_group_list:
                if smile in functional_group[1]:
                    used = True
    
            if not used:
                gutter_list.append(smile)
        
        inorganic_list =[]      
        for smile in self.input_column:
            if smile not in fglist.carbon_list:
                inorganic_list.append(smile)
        
                
        print('\n')
        print('inorganic smiles')
        print(inorganic_list)
        

        # calculate what percentage of molecules are leftover
        leftover = ( len(gutter_list) / len(self.input_column) ) * 100

        # if no molecules are leftover then print a confirmation
        if leftover == 0:
            print('\n')
            print('All molecules are accounted for')
            print('There are {} molecuels in dataset'.format(len(self.input_column)))

        # if there are reming molecules print the percentage 
        # remaining and each in gutter_list
        else:
            print('\n')
            print("{:.2f}% of molecules are in no group".format(leftover))
            for each in gutter_list:
                print('\n')
                print(each)


        self.inorganic_percent = "inorganic percent: {:.2f}%".format(((len(inorganic_list) / len(self.input_column) ) ))
        self.full_func_stats = ("\n".join(fglist.full_func_stats))



def to_fingerprints(path, filename, save_file_path):
#### 1. read in csv file using pandas
    raw_data = pd.read_csv(path)

    # first convert the first and second column into list
    input_column = raw_data.iloc[:,0].tolist()
    output_column = raw_data.iloc[:,1].tolist()

    # zip input_smiles and cTemp into a list of tuples
    raw_data_list = list(zip(input_column,output_column))


    fingerprints = []
    fingerprints_strings = []

    fingerprint_radius = 2
    fingerprint_size = 2048

    for smile in input_column:
        mol = Chem.MolFromSmiles(smile) 
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, fingerprint_radius, nBits=fingerprint_size)
        fingerprints_strings.append(fingerprint.ToList())

    # adding each to clean_fingerprint list
    for each in fingerprints_strings:
        fingerprints.append(each)

    for index in range(0,len(output_column)):
        fingerprints[index].append(output_column[index])


    specific_path = save_file_path + filename

    with open(specific_path, 'w', newline = '') as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows(fingerprints)
    print(f"Data saved {filename} successfully")





########### how to use
data = DeltaMolData("/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv/Testing/test_full.csv")
data.get_functioanl_group()
print(data.inorganic_percent)
print(data.full_func_stats)
#to_fingerprints("C:\\Users\\color\\Documents\\Bilodeau_Research_Python\\critical-Temp-LNN\\csv_data\\no_outliers_smile_dataset.csv", 'no_outliers_fgprnt_data.csv', 'C:\\Users\\color\\Documents\\Bilodeau_Research_Python\\critical-Temp-LNN\\csv_data')

