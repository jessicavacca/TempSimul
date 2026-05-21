from ai4ha.util import instantiate_from_config
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data import WeightedRandomSampler, random_split


def load_dataset(config):
    """ 
     loads the train, validation and test datasets
     if there is more than one train the datasets are concatenated
        Args:
            config: dict
        Returns: list
    """
    datasets = {'train': [], 'val': [], 'test': []}
    for key in config['dataset']:
        if 'train' in key:
            print(f"Loading {key} dataset")
            datasets['train'].append(
                instantiate_from_config(config['dataset'][key]))
            train_data = datasets['train'][-1]
            if 'sampling_smoothing' in config['dataset'][key]['params']:
                sampler = WeightedRandomSampler(train_data.weights,
                                                len(train_data),
                                                replacement=True)
                config["dataloader"]["sampler"] = sampler
                config["dataloader"]["shuffle"] = False
                print(f"Using weighted sampler")
                print(f"Weights: {train_data.weights}")
            else:
                config["dataloader"]["shuffle"] = True
                print("Using normal sampler")
        elif 'val' in key:
            print(f"Loading {key} dataset")
            datasets['val'].append(
                instantiate_from_config(config['dataset'][key]))
        elif 'test' in key:
            print(f"Loading {key} dataset")
            datasets['test'].append(
                instantiate_from_config(config['dataset'][key]))

    if 'split' in config['dataset']:
        d1, d2, d3 = random_split(datasets['train'][0],
                                  config['dataset']['split'])
        datasets['train'], datasets['val'], datasets['test'] = [d1], [d2], [d3]

    for key in datasets.keys():
        if datasets[key] == []:
            datasets[key] = None
        else:
            print(f"Loading {key} dataset")
            if len(datasets[key]) > 1:
                datasets[key] = ConcatDataset(datasets[key])
            else:
                datasets[key] = datasets[key][0]
            datasets[key] = DataLoader(datasets[key], **config["dataloader"])

    print(config["dataloader"])

    return datasets
