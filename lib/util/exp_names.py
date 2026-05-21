from lib.Forecast.SimpleTransformerForecast import SimpleTransformerForecast_exp_name
from lib.Forecast.PatchTransformerForecast import PatchTransformerForecast_exp_name, PatchTransformerForecast2_exp_name

def experiment_name_regression(cfg):
    """
    Returns the experiment name for regression models. The name is based on the dataset, 
    batch size, model type, optimizer, learning rate, and loss function.
    

    DATASET-batch_size-CL-regression-OPT-optimizer-lr-learning_rate-LS-loss
    """

    def loss_name(cfg):
        if cfg["loss"]["loss"] == "CE":
            return "CE"
        elif cfg["loss"]["loss"] == "BCE":
            return "BCE"
        elif cfg["loss"]["loss"] == "Ordinal":
            return "OR"
        elif cfg["loss"]["loss"] == "Focal":
            return "FC"
        else:
            return "l"

    ae_name = {"SimpleTransformerForecast": SimpleTransformerForecast_exp_name,
               "PatchTransformerForecast": PatchTransformerForecast_exp_name,
               "PatchTransformerForecast2": PatchTransformerForecast2_exp_name}

    name = f"{cfg['dataset']['name']}"
    name += f"-B{cfg['dataloader']['batch_size']}"
    name += f"-CL-{ae_name[cfg['model']['modeltype']](cfg)}"
    name += f"-OP-{cfg['optimizer']['opt']}"
    name += f"-LR{cfg['optimizer']['learning_rate']}"
    name += f"-SC{cfg['lr_scheduler']['type']}"
    if "lr_warmup_steps" in cfg["lr_scheduler"]:
        name += f"-w{cfg['lr_scheduler']['lr_warmup_steps']}"
    if "factor" in cfg["lr_scheduler"]:
        name += f"-f{cfg['lr_scheduler']['factor']}"
    name += f"-LS-{loss_name(cfg)}"

    return name
