from time import time
import importlib
import numpy as np
import torch
import os
from omegaconf import OmegaConf
from lib.paths import LOCAL_REMOTE_PATHS_CONFIG

def tof(x):
    return 'T' if x else 'F'


def time_management(last_time, qe_time, time_budget, logger):
    """
    Keeps the training time
    """
    epoch_time = (time() - last_time) / 60.0
    last_time = time()
    if len(qe_time) > 10:
        qe_time.pop(0)
    qe_time.append(epoch_time)
    time_budget -= epoch_time
    hours = int(time_budget // 60)
    mins = time_budget - (time_budget // 60) * 60
    logger.info(
        f"** Remaining time budget: {hours:02d}h {mins:3.2f}m - mean iteration time {np.mean(qe_time):3.2f}m **"
    )

    return last_time, qe_time, time_budget


def print_memory(logger, accelerator, where):
    logger.info(
        f"MEM: memory allocated: {torch.cuda.memory_allocated(device=accelerator.device)/(1014*1024)}"
    )
    logger.info(f"MEM: --------- {where} ------------------\n")
    logger.info(
        f"MEM: \n {torch.cuda.memory_summary(device=accelerator.device)}")
    logger.info("MEM: ---------------------------")


def initialize_dirs(base, dirlist):
    """
    Create directories in a base directory
    """
    for d in dirlist:
        if not os.path.exists(os.path.join(base, d)):
            os.makedirs(os.path.join(base, d), exist_ok=True)


# Functions taken from Latent Diffusion code
def instantiate_from_config(config):
    if "class" not in config:
        raise KeyError("Expected key `class` to instantiate.")
    return get_obj_from_str(config["class"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def log_cond(logger, att, struc):
    if att in struc:
        logger.info(f"{att}: {struc[att]}")

def save_config(config, nfile):
    """
    Saves configuration file
    """
    f = open(nfile, "w")
    OmegaConf.save(config, f=f)
    f.close()


def fix_paths(config, local=False):
    """Fixes the paths in the configuration file according to the local or remote execution
     Args:
        config: dict, configuration file
        local: bool, if True, fixes the paths for local execution, otherwise for remote execution
     Returns:
        config: dict, configuration file with fixed paths
    """
    def fix_dataset_paths(dataset, config):
        for key in config[dataset].keys():
            if ('train' in key) or ('test' in key) or ('val' in key):
                path_par = []
                if 'filename' in config[dataset][key]['params']:
                    path_par.append('filename')
                if 'dir' in config[dataset][key]['params']:
                    path_par.append('dir')
                if 'data_root' in config[dataset][key]['params']:
                    path_par.append('data_root')
                if 'dirreal' in config[dataset][key]['params']:
                    path_par.append('dirreal')
                if 'dirgen' in config[dataset][key]['params']:
                    path_par.append('dirgen')

                if path_par is not None:
                    for opath in path_par:
                        config[dataset][key]['params'][
                            opath] = LOCAL_REMOTE_PATHS_CONFIG['Data'][
                                path] + config[dataset][key]['params'][opath]

    # Fix experiment path
    path = 'local_path' if local else 'remote_path'
    config['exp_dir'] = LOCAL_REMOTE_PATHS_CONFIG['Experiment'][path] + config[
        'exp_dir']
    # Fix dataset paths
    if 'dataset' in config:
        fix_dataset_paths('dataset', config)

    return config
