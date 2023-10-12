hidden_dim_nn_1=500
p1 = 0 #drop out prob. for each layer
hidden_dim_nn_2=250
p2 = 0 
hidden_dim_nn_3=100
p3 = 0

hidden_dim_fcn_1=100
hidden_dim_fcn_2=50
hidden_dim_fcn_3=5



if hidden_dim_nn_1 == 2000 and p1 == 0.5 and hidden_dim_nn_2==500 and p2 == 0.4 and hidden_dim_nn_3==100 and p3 == 0.3 and hidden_dim_fcn_1==1000 and hidden_dim_fcn_2==100 and hidden_dim_fcn_3==5:
    layers = 'deafult'
else: 
    
    layers = (str(hidden_dim_nn_1) + '-' + str(p1) + '-' + 
                str(hidden_dim_nn_2) + '-' + str(p2) + '-' + 
                str(hidden_dim_nn_3)+ '-' + str(p3) + '-' + 
                str(hidden_dim_fcn_1) + '-' + str(hidden_dim_fcn_2) + '-' + 
                str(hidden_dim_fcn_3))

print(layers)
