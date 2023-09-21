import torch


def train(model, device, dataloader, optim, epoch):
    
    model.train()
    
    loss_func = torch.nn.MSELoss(reduction='sum') 
    loss_collect = 0

    # Looping over the dataloader allows us to pull out or input/output data:
    # Enumerate allows us to also get the batch number:
    for batch in dataloader:

        # Zero out the optimizer:
        optim.zero_grad()

        # Make a prediction:
        pred = model(batch.to(device))

        # Calculate the loss:
        loss = loss_func(pred.double(), batch.y.double())

        # Backpropagation:
        loss.backward()
        optim.step()

        # Calculate the loss and add it to our total loss
        loss_collect += loss.item()  # loss summed across the batch
        
        # Print out our Training loss so we know how things are going


    # Return our normalized losses so we can analyze them later:
    loss_collect /= len(dataloader.dataset)
    
    print(
        "Epoch:{}   Training dataset:   Loss per Datapoint: {:.3f}%".format(
            epoch, loss_collect*100
        )
    )  
    return loss_collect    

def validation(model, device, dataloader, epoch):

    model.eval()
    loss_collect = 0
    loss_func = torch.nn.MSELoss(reduction='sum') 
        
    # Remove gradients:
    with torch.no_grad():

        # Looping over the dataloader allows us to pull out or input/output data:
        for batch in dataloader:
            # Make a prediction:
            pred = model(batch.to(device))
            # Calculate the loss:
            loss = loss_func(pred.double(), batch.y.double())
            # Calculate the loss and add it to our total loss
            loss_collect += loss.item()  # loss summed across the batch

    loss_collect /= len(dataloader.dataset)
    
    # Print out our test loss so we know how things are going
    print(
        "Epoch:{}   Validation dataset: Loss per Datapoint: {:.3f}%".format(
            epoch, loss_collect*100
        )
    )  
    print('---------------------------------------')     
    # Return our normalized losses so we can analyze them later:
    return loss_collect



def predict(model, dataloader, device, weights_file):

    # Set our model to evaluation mode:
    model.eval()
    model.load_state_dict(torch.load(weights_file))

    X_all = []
    y_all = []
    pred_all = []

    # Remove gradients:
    with torch.no_grad():

        # Looping over the dataloader allows us to pull out or input/output data:
        for batch in dataloader:

            # Make a prediction:
            pred = model(batch.to(device))

            X_all.append(batch.x.to(device))
            y_all.append(batch.y.to(device))
            pred_all.append(pred.to(device))
            # dir(graph(x)) = smile
            # append(smile)
            # create a csv (smile, real value (y), pred())

    X_all = torch.concat(X_all)
    y_all = torch.concat(y_all)
    pred_all = torch.concat(pred_all)
    
    # dictionary of SMILES and fingerprints
    ########################################################################### right here create csv

    return X_all, y_all, pred_all

######### create a dicitionary in utils.py to keep together graphs and SMILES
######### do this instead of converting to csv in model.py