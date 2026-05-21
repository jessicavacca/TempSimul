from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F



# Function to train the model
def regression_train_loop(accelerator,
                          model,
                          train,
                          val,
                          optimizer,
                          scheduler=None,
                          patience=5,
                          epochs=100,
                          lossf=F.mse_loss,
                          tqdm_on=True):
    """_Training loop for the model_

    Args:
        model: model to train
        optimizer: pytorch optimizer, for example torch.optim.Adam
        train: training data
        val: validation data
        epochs: number of epochs

    Returns:
        _type_: training history, a dictionary with the training and validation loss for each epoch
    """

    def epoch_loss(dataset):
        data_loss = 0.0
        for i, (lookback_data, horizon_data) in enumerate(dataset):
            inputs = lookback_data.to('cuda')
            y = horizon_data.to('cuda')
            outputs = model(inputs).squeeze()
            loss = lossf(y, outputs)
            data_loss += loss.item()
        return data_loss / (i + 1)

    def early_stopping(val_loss, patience=5):
        if len(val_loss) > patience:
            if val_loss[-1] > np.mean(val_loss[-(patience + 1):-1]):
                return True

    hist_loss = {'train': [], 'val': []}
    pbar = tqdm(range(epochs)) if tqdm_on else range(epochs)
    for epoch in pbar:  # loop for all the epochs
        for i, (lookback_data, horizon_data) in enumerate(train):
            # take the data and upload it to the GPU
            inputs = lookback_data.to('cuda')
            y = horizon_data.to('cuda')

            # Reset the gradients
            optimizer.zero_grad()

            # Apply the data to the model
            outputs = model(inputs).squeeze()
            # Calculate the loss
            loss = lossf(y, outputs)

            # Make the backward pass
            accelerator.backward(loss)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Calculate the loss in the training and validation sets
        with torch.no_grad():
            hist_loss['train'].append(epoch_loss(train))
            hist_loss['val'].append(epoch_loss(val))

        # Show the loss in the training and validation sets
        if tqdm_on:
            pbar.set_postfix({
                'train': hist_loss['train'][-1],
                'val': hist_loss['val'][-1],
                'lr': optimizer.param_groups[0]['lr']
            })

        # If the loss in the validation set does not decrease, stop the training
        if early_stopping(hist_loss['val'], patience):
            break

    return hist_loss
