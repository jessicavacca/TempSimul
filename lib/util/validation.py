import torch.nn.functional as F

def dataset_loss(model, dataset, lossf=F.mse_loss):
    data_loss = 0.0
    for i, (lookback_data, horizon_data) in enumerate(dataset):
        inputs = lookback_data.to('cuda')
        y = horizon_data.to('cuda')
        outputs = model(inputs).squeeze()
        loss = lossf(y, outputs)
        data_loss += loss.item()
    return data_loss / (i + 1)