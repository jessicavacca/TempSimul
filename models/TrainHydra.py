import hydra
from omegaconf import DictConfig, OmegaConf, SCMode
from accelerate.logging import get_logger
from lib.util.exp_names import experiment_name_regression
from lib.util.misc import adapt_hydra_optuna_sweep, fix_paths
from lib.Train.Regression import regression_loop


logger = get_logger(__name__, log_level="INFO")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def RegressorTrain(cfg: DictConfig) -> None:
    # Convert to dictionary so we can modify it
    cfg = OmegaConf.to_container(cfg,
                                 structured_config_mode=SCMode.DICT_CONFIG)
    cfg = fix_paths(cfg, cfg['local'])
    cfg = adapt_hydra_optuna_sweep(cfg)
    cfg['name'] = experiment_name_regression(cfg)
    return regression_loop(cfg, logger)


if __name__ == "__main__":
    RegressorTrain()
