import pandas as pd


testing_data = pd.read_csv('/home/jbd3qn/Downloads/critical_temp_GCNN/test_full.csv')

for index in range(0, len(testing_data)):
    print(testing_data[index])