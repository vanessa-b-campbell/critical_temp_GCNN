# need to line up the pre training real temps with the post training real temps
import pandas as pd

#first isolate the pre-training-real-temps and save as a list
Pre_data = pd.read_csv('/home/jbd3qn/Downloads/critical_temp_GCNN/chemprop_splits_csv/train_full.csv')
Pre_list = Pre_data.iloc[:,1].tolist()


# next isolate the post-training-real-temps and save as a list
Post_data = pd.read_csv('/home/jbd3qn/Downloads/critical_temp_GCNN/predicted_temp_training.csv')
Post_list = Post_data.iloc[:,0].tolist()

print(len(Pre_list)) 
print(len(Post_list)) 

# create a new empty list called matches
matches = []

# make a loop 
for temp_pre in Pre_list:
    for temp_post in Post_list:
        if temp_pre == temp_post:
            new_tuple = (temp_pre, temp_post)
            matches.append(new_tuple)
            Post_list.remove(temp_post)
            Pre_list.remove(temp_pre)



# print out the list of matches
for each in matches:
    print(each)
    
print(len(matches))
print(Pre_list)
print(len(Pre_list))