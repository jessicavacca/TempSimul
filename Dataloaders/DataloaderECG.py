from glob import glob
from torch.utils.data import Dataset
import numpy as np
import torch


class ECGDataset(Dataset):
    """ Loads ECG datasets
    
        Loads a dataset with series in a numpy array format.

        Return as dataloader of the form (X, y) where X is a 
        tensor of shape (batch_size, n_channels, n_samples) 
        and y is a tensor of shape (batch_size,) containing 
        the forecasting horizon.

        Parameters:
        - dir: directory where the dataset is stored. The dataset is expected to be stored in .npz files, with the last column containing the class labels and the preceding columns containing the ECG data.
        - dataset: which dataset to load (train, val, test)
        - norm: which normalization to apply (minmax, robust, zscore, izscore, iminmax)
        - channel: which channel to use as target (0-11, None for all channels)
        - past: number of past samples to use as input
        - horizon: forecasting horizon
        - stride: stride between samples (default 1)
    """

    def __init__(
        self,
        dir,
        dataset='train',
        norm=None,
        past=1,
        horizon=1,
        stride=1,
        channel=0,
    ):
        self.dir = dir
        self.dataset = dataset
        self.horizon = horizon
        self.past = past
        self.stride = stride
        self.channel = channel
        print(f"Loading data from {self.dir}")
        datafiles = sorted(glob(f"{self.dir}/*_{dataset}.npz"))
        data = np.load(datafiles[0])

        try:
            self.X_train = data['data']
        except:
            raise ValueError("Data files must contain a 'data' array")

        self.weights = None
        self.max_val = np.max(self.X_train)
        self.min_val = np.min(self.X_train)

        if norm is not None:
            if norm == 'minmax':
                self.mean = np.min(self.X_train, axis=(0, 2))
                self.std = np.max(self.X_train, axis=(0, 2)) - np.min(
                    self.X_train, axis=(0, 2))
                self.X_train = (np.swapaxes(self.X_train, 1, 2) -
                                self.mean) / self.std
                self.X_train = np.swapaxes(self.X_train, 1, 2)
            elif norm == 'robust':
                q1 = np.quantile(self.X_train, 0.01, axis=(0, 2))
                q3 = np.quantile(self.X_train, 0.99, axis=(0, 2))
                self.mean = q1
                self.std = q3 - q1
                self.X_train = (np.swapaxes(self.X_train, 1, 2) -
                                self.mean) / self.std
                self.X_train = np.swapaxes(self.X_train, 1, 2)
            elif norm == 'zscore':
                self.mean = np.mean(self.X_train, axis=(0, 2))
                self.std = np.std(self.X_train, axis=(0, 2))
                self.X_train = (np.swapaxes(self.X_train, 1, 2) -
                                self.mean) / self.std
                self.X_train = np.swapaxes(self.X_train, 1, 2)
            elif norm == 'izscore':
                self.mean = np.mean(self.X_train, axis=2, keepdims=True)
                self.std = np.std(self.X_train, axis=2, keepdims=True)
                self.X_train = (self.X_train - self.mean) / (self.std + 1e-8)
            elif norm == 'iminmax':
                self.mean = np.min(self.X_train, axis=2, keepdims=True)
                self.std = np.max(self.X_train, axis=2,
                                  keepdims=True) - np.min(
                                      self.X_train, axis=2, keepdims=True)
                self.X_train = (self.X_train - self.mean) / (self.std + 1e-8)

        window_size = self.past + self.horizon
        n_samples = (self.X_train.shape[2] - window_size) // self.stride + 1
        self.X_train = np.array([
            self.X_train[:, :, i * self.stride:i * self.stride + window_size]
            for i in range(n_samples)
        ])
        self.y_train = self.X_train[:, :, -self.horizon:, channel]
        self.X_train = self.X_train[:, :, :self.past, :]

        print(f'{self.dataset}')
        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.y_train.shape}')
        print(f'NORM={norm}')

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


class ECGDatasetTokens(Dataset):
    """ Loads ECG datasets tokenized with a VQ-VAE codebook. 
        The token sequences are expected to be stored in .npy files, 
        with the last column containing the class labels and 
        the preceding columns containing the token sequences."""

    def __init__(
        self,
        dir,
        dataset='train',
        vocab_size=512,
    ):
        self.dir = dir
        self.dataset = dataset
        self.vocab_size = vocab_size
        print(f"Loading data from {self.dir}")
        datafiles = sorted(glob(f"{self.dir}/*_{dataset}.npz"))
        data = np.load(datafiles[0])
        self.X_train = data['data']
        self.y_train = data['labels']

        print(f'{self.dataset}')
        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.y_train.shape}')
        print("NClasses:",
              np.unique(self.y_train).shape[0], np.unique(self.y_train))

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return torch.tensor(self.X_train[idx]), self.y_train[idx]


if __name__ == "__main__":
    train = ECGDataset("/home/bejar/bsc/Data/PTBXL", dataset="train")
    # train_loader = DataLoader(train, batch_size=64, shuffle=True)
    # for i, (X, y) in enumerate(train_loader):
    #     print(X.shape, y.shape)
    #     if i == 10:
    #         break
