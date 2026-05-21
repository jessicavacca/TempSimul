import inspect
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, \
    MultiplicativeLR
from lib.util.optimisation import get_scheduler



def get_optimizer(model, accelerator, config):
    """_Generate the optimizer object_
    """
    if config['optimizer']['opt'] == 'adam':
        return Adam(model.parameters(),
                    lr=config['optimizer']['learning_rate'] * accelerator.num_processes,
                    betas=(config['optimizer']['beta1'],
                           config['optimizer']['beta2']))
    elif config['optimizer']['opt'] == 'adamw':
        return AdamW(model.parameters(),
                     lr=config['optimizer']['learning_rate'] * accelerator.num_processes,
                     betas=(config['optimizer']['beta1'],
                            config['optimizer']['beta2']),
                     weight_decay=config['optimizer']['weight_decay'],
                     eps=config['optimizer']['epsilon'])


def get_lr_scheduler(config, optimizer, dataloader, accparams):
    """_Get the learning rate scheduler_

    Args:
        config (_type_): _description_
        optimizer (_type_): _description_
    Returns:
        _type_: _description_
    """

    if config['lr_scheduler']['type'] == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode=config['lr_scheduler']['mode'],
            factor=config['lr_scheduler']['factor'],
            patience=config['lr_scheduler']['patience'],
            threshold=config['lr_scheduler']['threshold'],
            threshold_mode=config['lr_scheduler']['threshold_mode'],
            cooldown=config['lr_scheduler']['cooldown'],
            min_lr=config['lr_scheduler']['min_lr'],
            eps=config['lr_scheduler']['eps'])
    elif config['lr_scheduler']['type'] == 'mutiplicative':
        return MultiplicativeLR(
            optimizer, lr_lambda=lambda x: config['lr_scheduler']['factor'])
    else:
        return get_scheduler(
            config['lr_scheduler']['type'],
            optimizer=optimizer,
            num_warmup_steps=config["lr_scheduler"]["lr_warmup_steps"] *
            accparams["gradient_accumulation_steps"],
            num_training_steps=config["train"]["num_epochs"]
            #(len(dataloader) * config["train"]["num_epochs"]
        )
