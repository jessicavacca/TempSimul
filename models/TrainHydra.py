import hydra
from omegaconf import DictConfig, OmegaConf, SCMode
from accelerate.logging import get_logger
from ai4ha.util import experiment_name_regressor
from ai4ha.util.config import adapt_hydra_optuna_sweep
from ai4ha.Train import regression_loop


logger = get_logger(__name__, log_level="INFO")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def RegressorTrain(cfg: DictConfig) -> None:
    # Convert to dictionary so we can modify it
    cfg = OmegaConf.to_container(cfg,
                                 structured_config_mode=SCMode.DICT_CONFIG)
    cfg = fix_paths(cfg, cfg['local'])
    cfg = adapt_hydra_optuna_sweep(cfg)
    cfg['name'] = experiment_name_regressor(cfg)
    return regression_loop(cfg, logger)


if __name__ == "__main__":
    RegressorTrain()
