import logging
import math
import torch

from lib.Forecast.SimpleTransformerForecast import SimpleTransformerForecast
from lib.Forecast.PatchTransformerForecast import PatchTransformerForecast, PatchTransformerForecast2
from lib.util.misc import initialize_dirs, save_config
from lib.util.data import load_dataset
from lib.util.train import get_optimizer, get_scheduler
from lib.util.textlog import textlog
from lib.util.validation import dataset_loss
from lib.util.trainloop import regression_train_loop

from accelerate import Accelerator, DistributedDataParallelKwargs #, ProjectConfiguration



def regression_loop(config, logger):
    BASE_DIR = f"{config['exp_dir']}/logs/{config['name']}"
    initialize_dirs(
        BASE_DIR, ["checkpoints", "logs", "samples", "final", "model", "best"])

    save_config(config, f"{BASE_DIR}/config.yaml")
    accparams = config["accelerator"].copy()
    accparams["project_dir"] = BASE_DIR

    # if "projectconf" in config:
    #     accparams["project_config"] = ProjectConfiguration(
    #         **config["projectconf"])

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=accparams["gradient_accumulation_steps"] > 1)
    accelerator = Accelerator(**accparams, kwargs_handlers=[ddp_kwargs])

    logger.info(f"{config['name']}")
    dataloaders = load_dataset(config)
    len_train_data = len(dataloaders["train"])
    # logger.info(f"Dataset size: {len(train_data)}")

    # global_step = 0
    # first_epoch = 0

    total_batch_size = (config["dataloader"]["batch_size"] *
                        accelerator.num_processes *
                        accparams["gradient_accumulation_steps"])
    num_update_steps_per_epoch = math.ceil(
        len_train_data / accparams["gradient_accumulation_steps"])
    max_train_steps = config["train"]["num_epochs"] * num_update_steps_per_epoch
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    # tlog = textlog(f"{BASE_DIR}/logs/losslog.csv",
    #                ["CEloss", "ACCTrain", "ACCVal"])

    if config["model"]["modeltype"] == "SimpleTransformerForecast":
        model = SimpleTransformerForecast(**config["model"]["params"])
    elif config["model"]["modeltype"] == "PatchTransformerForecast":
        model = PatchTransformerForecast(**config["model"]["params"])
    elif config["model"]["modeltype"] == "PatchTransformerForecast2":
        model = PatchTransformerForecast2(**config["model"]["params"])
    else:
        raise ValueError(
            f"Unsupported model type: {config['model']['modeltype']}")

    model.log_parameters(logger)

    # Initialize the optimizer
    optimizer = get_optimizer(model, accelerator, config)

    lr_scheduler = get_scheduler(
        config["lr_scheduler"]["type"],
        optimizer=optimizer,
        num_warmup_steps=config["lr_scheduler"]["lr_warmup_steps"] *
        accparams["gradient_accumulation_steps"],
        num_training_steps=(len_train_data * config["train"]["num_epochs"]),
    )

    if config["loss"]["loss"] == "L2":
        lossf = torch.nn.MSELoss()
    elif config["loss"]["loss"] == "L1":
        lossf = torch.nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss type: {config['loss']['loss']}")

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloaders["train"], lr_scheduler)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len_train_data}")
    logger.info(f"  Num Epochs = {config['train']['num_epochs']}")
    logger.info(
        f"  Instantaneous batch size per device = {config['dataloader']['batch_size']}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {accparams['gradient_accumulation_steps']}"
    )
    logger.info(f"  Total optimization steps = {max_train_steps}")

    logger.info(f"MEM: {torch.cuda.max_memory_allocated()}")

    train_history = regression_train_loop(
        accelerator,
        model,
        train_dataloader,
        dataloaders["val"],
        optimizer,
        lr_scheduler,
        config["train"]["num_epochs"],
        config["train"]["patience"],
        lossf=lossf,
        tqdm_on=accelerator.is_main_process,
    )

    logger.info("***** Training complete *****")
    logger.info(f"Best validation loss: {min(train_history['val'])}")
    logger.info(
        f"Best epoch: {train_history['val'].index(min(train_history['val']))}")
    logger.info(f"Final epoch: {len(train_history['val'])}")

    logger.info("*** Validation Results ***")
    val_loss = dataset_loss(model, dataloaders['val'])
    logger.info(f"Validation {config['loss']['loss']} = {val_loss}")

    test_loss = dataset_loss(model, dataloaders['test'])
    logger.info(f"Test {config['loss']['loss']} = {test_loss}")

    return val_loss
